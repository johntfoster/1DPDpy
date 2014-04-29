#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import numpy.ma as ma
import numpy.random
import scipy.spatial
import scipy.optimize
import matplotlib.pyplot as plt



class PD_Problem():
    '''
       This class initializes a 1D peridynamic problem.  This problem is essentially
       a long bar with displacement boundary conditions applied to boundary regions
       equal to 1-horizon at each end.

       Initialization parameters are as follows:

       + ``bar_length`` - length of bar, it will be centered at the origin

       + ``number_of_elements`` - the discretization level

       + ``bulk_modulus``

       + ``randomization_factor`` - if this optional parameter is set to anything other than 0.0, a set of discretization points interior to the bar are randomly perturbed by the amount set by this parameter.  Regions near the boundaries are not perturbed in order to maintain a consistency in the application of the boundary conditions such that comparisons can be made between perturbed and uniformly spaced models.  A reasonable setting for the parameter will be in the range of 0.0-0.3

       + ``constitutive_model_flag`` - this parameter defaults to ``LPS`` but could also be set to ``correspondece``.  If set to ``LPS`` the linear peridynamic solid model of Silling et al. 2007 will be used in solving for the internal force.  If set to `correspondence` an elastic stress-strain law is converted to force vector-states.  Both formulations will yield the same result.

    '''

    def __init__(self,bar_length=20,number_of_elements=20,
            bulk_modulus=100,horizon=None,randomization_factor=0.0,
            constitutive_model_flag='LPS'):
        '''
           Initialization function
        '''

        #Problem data
        self.bulk_modulus = bulk_modulus
        self.constitutive_model_flag=constitutive_model_flag


        self.bar_length = bar_length
        self.number_of_elements = number_of_elements

        delta_x = bar_length / number_of_elements

        #:This array contains the *element* node locations.  i.e., they define the discrete regions along the bar. The peridynamic node locations will be at the centroid of these regions.
        self.nodes = np.linspace(-bar_length / 2.0, bar_length / 2.0, num=number_of_elements + 1)

        #Set horizon from parameter list or as default
        if horizon != None:
            self.horizon = horizon
        else:
            self.horizon = delta_x * 3.015

        #If the randomization_factor is non-zero, the interior nodes (not where the
        #boundary conditions are applied), will be randomly perturbed
        if randomization_factor != 0.0:
            #Don't perturb the nodes on the boundary
            number_of_boundary_nodes = int(np.round(2.0*horizon/delta_x)) + 1
            
            #Create random perturbations to be applied on the boundary
            random_perturbations = numpy.random.uniform(-randomization_factor*delta_x,
                    randomization_factor*delta_x,len(self.nodes) - int(2*number_of_boundary_nodes))

            #Add random perturbations to original node positions
            self.nodes[number_of_boundary_nodes:-number_of_boundary_nodes] += random_perturbations

        #Compute the pd_node locations, kdtree, nns, and reference_position_state
        self.__setup_discretization()


    def __setup_discretization(self):
        
        nodes = self.nodes

        #:The lengths of the *elements*
        self.lengths = (nodes[1:] - nodes[0:-1]) 

        #:The PD nodes are the centroids of the elements
        self.pd_nodes = nodes[0:-1] + self.lengths / 2.0

        #:Create's a kdtree to do nearest neighbor search
        self.tree = scipy.spatial.cKDTree(self.pd_nodes[:,None])

        #Get PD nodes in the neighborhood of support + largest node spacing, this will
        #find all potential partial volume nodes as well. The distances returned from the
        #search turn out to be the reference_magnitude_state, so we'll store them now
        #to avoid needed to calculate later.
        self.reference_magnitude_state, self.neighborhood = self.tree.query(self.pd_nodes[:,None], 
                k=100, p=2, eps=0.0, distance_upper_bound=(self.horizon + np.max(self.lengths)/2))

        #Remove node indices from their own neighborhood and pad the array with -1's 
        #where the nearest neighbor search padded with the tree length
        self.neighborhood = np.delete(np.where(self.neighborhood == self.tree.n, -1, self.neighborhood),0,1)
        self.reference_magnitude_state = np.delete(self.reference_magnitude_state,0,1)


        #Compute the reference_position_state.  Using the terminology of Silling et al. 2007
        self.reference_position_state = np.array(self.pd_nodes[self.neighborhood] - self.pd_nodes[:,None])

        #Compute the partial volumes
        self.__compute_partial_volumes()

        #Define a few local (to function) convenience variables from data returned by
        #__compute_partial_volumes
        vol_state = self.volume_state
        max_neigh = self.max_neighbors
        ref_pos_state = self.reference_position_state
        ref_mag_state = self.reference_magnitude_state

        #:A Numpy masked array of indices that include all other nodes inside the support neighborhood
        self.neighborhood = ma.masked_array(self.neighborhood[:,:max_neigh],mask=vol_state.mask)
        #This array should never change so we will set it to read only
        self.neighborhood.setflags(write=False)
        #:A Numpy masked array containing the *reference position vector-state* as defined in Silling et al. 2007
        self.reference_position_state = ma.masked_array(ref_pos_state[:,:max_neigh],mask=vol_state.mask)
        #This array should never change so we will set it to read only
        self.reference_position_state.setflags(write=False)
        #:A Numpy masked array containing the *reference magnitude vector-state* as defined in Silling et al. 2007
        self.reference_magnitude_state = ma.masked_array(ref_mag_state[:,:max_neigh],mask=vol_state.mask)
        self.reference_magnitude_state.setflags(write=False)

        #:A Numpy masked array containing the *influence vector-state* as defined in Silling et al. 2007
        ratio = self.reference_magnitude_state / self.horizon
        self.influence_state = np.ones_like(vol_state) - 35.0 * ratio ** 4.0 + 84.0 * ratio ** 5.0 - 70 * ratio ** 6.0 + 20 * ratio ** 7.0

        #Compute the shape tensor (really a scalar because this is 1d, just want to use 
        #consistent terminology)
        #:A Numpy masked array containing the *shape tensor* as defined in Silling et al. 2007, in 1D this is identical to the *weighted volume*, this would not be true in higher dimensions
        self.shape_tensor = (self.influence_state * self.reference_position_state * 
                self.reference_position_state * vol_state).sum(axis=1)
        self.shape_tensor.setflags(write=False)

        #:The displacement field
        self.displacement = np.zeros_like(self.pd_nodes)

        return


    def __compute_partial_volumes(self):

        #Setup some local (to function) convenience variables
        neigh = self.neighborhood
        lens = self.lengths
        ref_mag_state = self.reference_magnitude_state
        ref_pos_state = self.reference_position_state
        horiz = self.horizon

        #Compute the volume_state, where the nodal volume = length * area, where the area is 
        #implicitly equal to 1 and the length calculation takes into account the partially
        #covered distances on either side of the horizon. This is the key to patch test
        #consistency.

        #Initialize the volume_state to the lengths
        vol_state = lens[neigh]
        #Place dummy -1's in node locations that are not fully inside the support neighborhood nor have a partial volume
        vol_state = np.where(ref_mag_state < horiz + lens[neigh] / 2.0, vol_state, -1)

        #Check to see if the neighboring node has a partial volume
        is_partial_volume = np.abs(horiz - ref_mag_state) < lens[neigh] / 2.0
        #Two different scenarios:
        is_partial_volume_case1 = np.all([is_partial_volume, 
                                         ref_mag_state >= horiz],axis=0)
        is_partial_volume_case2 = np.all([is_partial_volume, 
                                          ref_mag_state < horiz],axis=0) 

        #Compute the partial volumes conditionally
        vol_state = np.where(is_partial_volume_case1, lens[neigh] / 2.0 - (ref_mag_state - horiz), vol_state)
        vol_state = np.where(is_partial_volume_case2, lens[neigh] / 2.0 + (horiz - ref_mag_state), vol_state)
        #If the partial volume is predicted to be larger than the real volume, set it back
        vol_state = np.where(vol_state > lens[neigh], lens[neigh], vol_state)

        #Trim down the arrays to the minimum and create masked arrays for the upcoming
        #internal force calculation
        self.max_neighbors = np.max((vol_state != -1).sum(axis=1))
        #:The *volume scalar-state* is a masked array containing the partial volumes of the support neighborhood
        self.volume_state = ma.masked_equal(vol_state[:,:self.max_neighbors],-1)
        self.volume_state.harden_mask()

        #Now compute the "reverse volume state", this is the partial volume of the "source" node, i.e. node i,
        #as seen from node j.  This doesn't show up anywhere in any papers, it's just used here for computational
        #convenience
        rev_vol_state = np.ones_like(vol_state) * lens[:,None]
        rev_vol_state = np.where(is_partial_volume_case1, lens[:, None] / 2.0 - (ref_mag_state - horiz), rev_vol_state)
        rev_vol_state = np.where(is_partial_volume_case2, lens[:, None] / 2.0 + (horiz - ref_mag_state), rev_vol_state)
        #If the partial volume is predicted to be larger than the real volume, set it back
        rev_vol_state = np.where(rev_vol_state > lens[:, None], lens[:, None], rev_vol_state)
        #:The *reverse volume scalar-state* is defined for computational convenience, it is the partial volume of the *source* node as seen from the support neighborhood nodes
        self.reverse_volume_state = ma.masked_array(rev_vol_state[:,:self.max_neighbors], mask=self.volume_state.mask)

        return

    #Compute the force vector-state using a LPS peridynamic formulation
    def __compute_force_state_LPS(self, disp):
            
        #Define some local convenience variables     
        ref_pos = self.pd_nodes 
        inf_state = self.influence_state
        ref_pos_state = self.reference_position_state
        ref_mag_state = self.reference_magnitude_state
        shape_tens = self.shape_tensor
        vol_state = self.volume_state
        rev_vol_state = self.reverse_volume_state
        neigh = self.neighborhood

        #Compute the deformed positions of the nodes
        def_pos = ref_pos + disp

        #Compute deformation state
        def_state = ma.masked_array(def_pos[neigh] - def_pos[:,None], mask=neigh.mask)

        #Compute approximate deformation gradient, this is only for plotting purposes, it's
        #not needed in this constitutive model
        def_grad = (inf_state * def_state * ref_pos_state * vol_state).sum(axis=1) / shape_tens
        self.deformation_gradient = def_grad

        #Compute deformation magnitude state
        def_mag_state = (def_state * def_state)**0.5 

        #Compute deformation unit state
        def_unit_state = def_state / def_mag_state

        #Compute scalar extension state
        exten_state = def_mag_state - ref_mag_state

        #Compute scalar force state for a elastic constitutive model
        scalar_force_state = 9.0 * self.bulk_modulus / shape_tens[:, None] * exten_state
       
        #Compute the force state
        force_state = scalar_force_state * def_unit_state

        return force_state

    #Compute the force vector-state using a correspondence, i.e. stress-strain
    #formulation through approximation of the deformation gradient
    def __compute_force_state_correspondence(self,disp):

        #Define some local convenience variables     
        ref_pos = self.pd_nodes 
        inf_state = self.influence_state
        ref_pos_state = self.reference_position_state
        shape_tens = self.shape_tensor
        neigh = self.neighborhood
        vol_state = self.volume_state

        #Compute the deformed positions of the nodes
        def_pos = ref_pos + disp

        #Compute deformation state
        def_state = ma.masked_array(def_pos[neigh] - def_pos[:,None], mask=neigh.mask)

        #Compute approximate deformation gradient
        def_grad = (inf_state * def_state * ref_pos_state * vol_state).sum(axis=1) / shape_tens
        #:The deformation gradient field
        self.deformation_gradient = def_grad

        #Compute 1d strain
        strain = def_grad - 1.0

        #Compute 1d stress
        stress = self.bulk_modulus * strain

        #In 1d the Cauchy stress and 1st Piola-Kirchoff stress coincide, so there
        #is no need to convert, but there would be in higher dimensions
       
        #Compute the force state
        force_state = inf_state * stress[:,None] / shape_tens[:,None] * ref_pos_state

        return force_state



   
    # Internal force calculation
    def __compute_internal_force(self, displacement, force):
            
        #Define some local convenience variables     
        vol_state = self.volume_state
        rev_vol_state = self.reverse_volume_state
        neigh = self.neighborhood
        
        #Compute the force vector-state according to the choice of constitutive
        #model  
        if self.constitutive_model_flag == 'LPS':
            force_state = self.__compute_force_state_LPS(displacement)

        elif self.constitutive_model_flag == 'correspondence':
            force_state = self.__compute_force_state_correspondence(displacement)

        #Integrate nodal forces 
        force = np.zeros_like(force)
        force[:] += (force_state * vol_state).sum(axis=1)

        tmp = np.bincount(neigh.compressed(), (force_state * rev_vol_state).compressed())
        force[:len(tmp)] -= tmp

        return


    #The residual is objective function for the minimization problem
    def __compute_residual(self, disp):

        #Apply displacements
        disp[self.left_boundary_region] = (self.prescribed_displacement
                                           / self.bar_length / 0.5
                                           * self.pd_nodes[self.left_boundary_region])
        disp[self.right_boundary_region] = (self.prescribed_displacement
                                           / self.bar_length / 0.5
                                           * self.pd_nodes[self.right_boundary_region])
                                           

        #Initialize residual
        residual_vector = np.zeros_like(self.displacement)

        #Compute the out-of-balance force vector
        self.__compute_internal_force(disp, residual_vector)

        #Zero out the residual in the kinematic boundary condition region
        residual_vector[self.left_boundary_region] = 0.0
        residual_vector[self.right_boundary_region] = 0.0

        return residual_vector


    def solve(self,prescribed_displacement=0.1):
        '''
           The method will solve the instantiated problem object given an initial
           prescribed displacement.  The solution method utilizes a Newton-Krylov
           minimization technique.  This choice of method avoids the need to explicitly
           specify a tangent-stiffness matrix  and works well for large problems as 
           well as provides convergent solutions even in the presence of geometric
           nonlinearities that may arise when due to large prescribed displacements. 

           Initialization parameters are as follows:

           + ``prescribed_displacement`` - The initial displacement prescribed to the bar
        '''
    
        self.prescribed_displacement = prescribed_displacement

        #Find the nodes within 1 horizon of each end to apply the boundary conditions on.
        #:The node indices of the boundary region at the left end of the bar
        self.left_boundary_region = self.tree.query_ball_point(self.pd_nodes[0, None], r=self.horizon, p=2, eps=0.0)
        #:The node indices of the boundary region at the right end of the bar
        self.right_boundary_region = self.tree.query_ball_point(self.pd_nodes[-1, None], r=self.horizon, p=2, eps=0.0)

        #Initial guess is linear between endpoint displacements
        guess = np.linspace(-prescribed_displacement, prescribed_displacement, len(self.displacement))

        #Solve
        self.displacement = scipy.optimize.newton_krylov(self.__compute_residual, guess, x_rtol=1.0e-12)


    #Public get functions
    def get_solution(self):
        ''' Convenience function for retrieving displacement'''
        return self.displacement

    def get_nodes(self):
        ''' Convenience function for retrieving peridynamic node locations'''
        return self.pd_nodes

    def get_deformation_gradient(self):
        ''' Convenience function for retrieving deformation gradient'''
        return self.deformation_gradient


### Main Program ####
if __name__ == "__main__":

    #Define problem size
    fixed_horizon = 3.5
    fixed_length = 40
    delta_x = 0.5

    #Instantiate a 1d peridynamic problem with equally spaced nodes
    problem1 = PD_Problem(bar_length=fixed_length, number_of_elements=(fixed_length/delta_x), horizon=fixed_horizon)
    #Solve the problem
    problem1.solve()
    #Get the node locations and displacement solution
    disp1 = problem1.get_solution()
    nodes1 = problem1.get_nodes()

    #Instantiate a 1d peridynamic problem with randomly spaced nodes, the interior nodes
    #are perturbed randomly by randomization_factor
    #problem2 = PD_Problem(bar_length=fixed_length, number_of_elements=(fixed_length/delta_x), horizon=fixed_horizon, randomization_factor=0.3)
    #Solve the problem
    #problem2.solve()
    #Get the node locations and displacement solution
    #disp2 = problem2.get_solution()
    #nodes2 = problem2.get_nodes()

    #Instantiate a 1d peridynamic problem with equally spaced nodes using a 
    #correspondence constitutive model
    #problem3 = PD_Problem(bar_length=fixed_length, 
            #number_of_elements=(fixed_length/delta_x), 
            #horizon=fixed_horizon,
            #constitutive_model_flag='correspondence')
    #Solve the problem
    #problem3.solve()
    #problem1.solve(absolute_tolerence=1.0e-6,number_of_iterations=1000)
    #Get the node locations and displacement solution
    #disp3 = problem3.get_solution()
    #nodes3 = problem3.get_nodes()

    #Instantiate a 1d peridynamic problem with randomly spaced nodes, the interior nodes
    #are perturbed randomly by randomization_factor. Uses the correspondence constitutive
    #model
    #problem4 = PD_Problem(bar_length=fixed_length, 
            #number_of_elements=(fixed_length/delta_x), 
            #horizon=fixed_horizon, 
            #randomization_factor=0.3, 
            #constitutive_model_flag='correspondence')
    #Solve the problem
    #problem4.solve()
    #Get the node locations and displacement solution
    #disp4 = problem4.get_solution()
    #nodes4 = problem4.get_nodes()

    plt.plot(nodes1, disp1, 'k-', label='LPS')
    #plt.plot(nodes2, disp2, 'r-', label='LPS - random')
    #plt.plot(nodes3, disp3, 'b-', label='Corr. - equal')
    #plt.plot(nodes4, disp4, 'g-', label='Corr. - random')
    plt.legend(loc='upper left')
    plt.show()



