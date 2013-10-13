#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import numpy.ma as ma
import numpy.random
import scipy.spatial
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import scipy.optimize
import matplotlib.pyplot as plt

np.set_printoptions(precision=4)


class PD_Problem():

    def __init__(self,bar_length=20,number_of_elements=20,
            bulk_modulus=100,horizon=None,randomization_factor=0.0):


        #Problem data
        self.bulk_modulus = bulk_modulus


        self.bar_length = bar_length
        self.number_of_elements = number_of_elements

        delta_x = bar_length / number_of_elements

        #This array are the ``element'' node locations.  i.e., they define the discrete
        #regions along the bar. The peridynamic node locations will be at the centroid
        #of these regions.
        self.nodes = np.linspace(0.0, bar_length, num=number_of_elements + 1)

        #Set horizon from parameter list or as default
        if horizon != None:
            self.horizon = horizon
        else:
            self.horizon = delta_x * 3.015

        #If the randomization_factor is non-zero, the interior nodes (not where the
        #boundary conditions are applied), will be randomly perturbed
        if randomization_factor != 0.0:
            #Don't perturb the nodes on the boundary
            number_of_boundary_nodes = int(np.round(horizon/delta_x)) + 1
            
            #Create random perturbations to be applied on the boundary
            random_perturbations = numpy.random.uniform(-randomization_factor*delta_x,
                    randomization_factor*delta_x,len(self.nodes) - int(2*number_of_boundary_nodes))

            #Add random perturbations to original node positions
            self.nodes[number_of_boundary_nodes:-number_of_boundary_nodes] += random_perturbations

        #Compute the pd_node locations, kdtree, nns, and reference_position_state
        self.__setup_discretization()


    def __setup_discretization(self):
        
        nodes = self.nodes

        #The lengths of the ``elements''
        self.lengths = (nodes[1:] - nodes[0:-1]) 

        #The PD nodes are the centroids of the elements
        self.pd_nodes = nodes[0:-1] + self.lengths / 2.0

        #Create a kdtree to do nearest neighbor search
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

        self.neighborhood = ma.masked_array(self.neighborhood[:,:max_neigh],mask=vol_state.mask)
        self.reference_position_state = ma.masked_array(ref_pos_state[:,:max_neigh],mask=vol_state.mask)
        self.reference_magnitude_state = ma.masked_array(ref_mag_state[:,:max_neigh],mask=vol_state.mask)

        #Initialize influence state
        self.influence_state = np.ones_like(vol_state)

        #Compute the shape tensor (really a scalar because this is 1d, just want to use 
        #consistent terminology)
        self.shape_tensor = (self.influence_state * self.reference_position_state * 
                self.reference_position_state * vol_state).sum(axis=1)

        #Initialize the displacement
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
        #vol_state = np.where(ref_mag_state <= horiz, vol_state, -1)
        vol_state = np.where(ref_mag_state < horiz + lens[neigh] / 2.0, vol_state, -1)

        #Check to see if the neighboring node has a partial volume
        is_partial_volume = np.abs(horiz - ref_mag_state) < lens[neigh] / 2.0
        #Four different scenarios,
        is_partial_volume_case1 = np.all([is_partial_volume, 
                                          ref_mag_state > horiz],axis=0)
        is_partial_volume_case2 = np.all([is_partial_volume, 
                                          ref_mag_state < horiz],axis=0) 
        #Positive ref_pos_state and node inside horiz
        #Positive ref_pos_state and node outside horizon
        #is_partial_volume_case1 = np.all([is_partial_volume, 
                                          #ref_pos_state > 0, 
                                          #0 < ref_mag_state - horiz, 
                                          #ref_mag_state - horiz < lens[neigh] / 2], axis=0)
        #Positive ref_pos_state and node inside horiz
        #is_partial_volume_case2 = np.all([is_partial_volume,
                                          #ref_pos_state > 0, 
                                          #0 > ref_mag_state - horiz, 
                                          #ref_mag_state - horiz > -lens[neigh] / 2], axis=0)
        #Negative ref_pos_state and node outside horiz
        #is_partial_volume_case3 = np.all([is_partial_volume,
                                          #ref_pos_state < 0, 
                                          #0 < ref_mag_state - horiz, 
                                          #ref_mag_state - horiz < lens[neigh] / 2], axis=0)
        #Negative ref_pos_state and node inside horiz
        #is_partial_volume_case4 = np.all([is_partial_volume,
                                          #ref_pos_state < 0, 
                                          #0 > ref_mag_state - horiz, 
                                          #ref_mag_state - horiz > -lens[neigh] / 2], axis=0)

        #This is basically a series of nested if-statement in numpy
        vol_state = np.where(is_partial_volume_case1, lens[neigh] / 2.0 - (ref_mag_state - horiz), vol_state)
        vol_state = np.where(is_partial_volume_case2, lens[neigh] / 2.0 + (horiz - ref_mag_state), vol_state)
        #vol_state = np.where(is_partial_volume_case1, horiz - (ref_pos_state - lens[neigh] / 2.0), vol_state)
        #vol_state = np.where(is_partial_volume_case2, horiz - (ref_pos_state - lens[neigh] / 2.0), vol_state)
        #vol_state = np.where(is_partial_volume_case3, horiz - np.abs(ref_pos_state + lens[neigh] / 2.0), vol_state)
        #vol_state = np.where(is_partial_volume_case4, horiz - np.abs(ref_pos_state + lens[neigh] / 2.0), vol_state)

        #Trim down the arrays to the minimum and create masked arrays for the upcoming
        #internal force and tangent stiffness calculations
        self.max_neighbors = np.max((vol_state != -1).sum(axis=1))
        self.volume_state = ma.masked_equal(vol_state[:,:self.max_neighbors],-1)
        self.volume_state.harden_mask()

        #Now compute the "reverse volume state", this is the partial volume of the "source" node, i.e. node i,
        #as seen from node j
        #rev_vol_state = vol_state.copy()
        #rev_vol_state = np.where(is_partial_volume_case1, lens[:, None] / 2.0 - (ref_mag_state - horiz), rev_vol_state)
        #rev_vol_state = np.where(is_partial_volume_case2, lens[:, None] / 2.0 + (horiz - ref_mag_state), rev_vol_state)
        #self.reverse_volume_state = ma.masked_array(rev_vol_state[:,:self.max_neighbors], mask=self.volume_state.mask)

        return

   
        # Internal force calculation
    def __compute_internal_force(self, displacement, force):
            
        #Define some local convenience variables     
        ref_pos = self.pd_nodes 
        inf_state = self.influence_state
        ref_pos_state = self.reference_position_state
        ref_mag_state = self.reference_magnitude_state
        shape_tens = self.shape_tensor
        vol_state = self.volume_state
        #rev_vol_state = self.reverse_volume_state
        lens = self.lengths
        neigh = self.neighborhood

        #Compute the deformed positions of the nodes
        def_pos = ref_pos + displacement

        #Compute deformation state
        def_state = ma.masked_array(def_pos[neigh] - def_pos[:,None], mask=neigh.mask)

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

        
        #Integrate nodal forces 
        force[:] += 0.5 * (force_state * vol_state).sum(axis=1)

        #tmp = np.bincount(neigh.compressed(), (force_state * rev_vol_state).compressed())
        #force[:len(tmp)] -= tmp

        return

    def __compute_internal_force_correspondence(self):

        #Compute approximate deformation gradient
        #deformation_gradient = (deformation_state * ref_pos_state * vol_state).sum(axis=1)
        #Compute 1d strain
        #strain = deformation_gradient - 1.0
        #Compute 1d stress
        #stress = bulk_modulus * strain

        #In 1d the Cauchy stress and 1st Piola-Kirchoff stress coincide, so there
        #is no need to convert, but there would be in higher dimensions
       
        #Compute the force state
        #force_state = inf_state * stress[:,None] / shape_tens[:,None] * ref_pos_state

        return


    def __compute_force_difference(self, probed_displacement, probe_length, probed_force, force):


        probed_force[:] = 0.0     

        self.__compute_internal_force(probed_displacement, probed_force)                

        #Forward difference formula
        return (probed_force - force) / probe_length


    def __compute_tangent_stiffness(self, alpha=1.e-4, beta=1.e-6):

        disp = self.displacement

        #Compute the internal force based on the reference displacement
        force = np.zeros_like(disp)

        self.__compute_internal_force(disp, force)

        #Construct the probe matrix
        probe_lengths = alpha * np.abs(disp) + beta

        probed_disp_matrix = np.identity(len(probe_lengths)) * probe_lengths[:, None]

        #Return the tangent stiffness matrix from forward difference calculation
        probed_force = np.zeros_like(force)

        tangent_stiffness = np.array([ self.__compute_force_difference(args[0], args[1], 
            probed_force, force) 
            for args in zip(probed_disp_matrix, probe_lengths)])

        return -tangent_stiffness.T

    def __compute_residual(self, disp):

        #Apply displacements
        disp[self.left_boundary_region] = -self.prescribed_displacement
        disp[self.right_boundary_region] = self.prescribed_displacement

        #Initialize residual
        residual_vector = np.zeros_like(self.displacement)

        #Compute the out-of-balance force vector
        residual_vector[:] = 0.0
        self.__compute_internal_force(disp, residual_vector)

        #Zero out the residual in the kinematic boundary condition region
        residual_vector[self.left_boundary_region] = 0.0
        residual_vector[self.right_boundary_region] = 0.0

        #return scipy.linalg.norm(residual_vector)
        return residual_vector



    def solve(self,prescribed_displacement=0.01, number_of_iterations=100, absolute_tolerence=0.1):

        #Find the nodes within 1 horizon of each end to apply the boundary conditions on.
        self.left_boundary_region = self.tree.query_ball_point(self.pd_nodes[0, None], r=self.horizon, p=2, eps=0.0)
        self.right_boundary_region = self.tree.query_ball_point(self.pd_nodes[-1, None], r=self.horizon, p=2, eps=0.0)

        #Apply displacements
        self.displacement[self.left_boundary_region] = -prescribed_displacement
        self.displacement[self.right_boundary_region] = prescribed_displacement

        #Initialize residual
        residual_vector = np.zeros_like(self.displacement)

        print "Starting Newton's method solve...\n"
        
        #Newton's method solve
        for iteration in range(number_of_iterations):

            #Compute the out-of-balance force vector
            residual_vector[:] = 0.0
            self.__compute_internal_force(self.displacement, residual_vector)

            #Zero out the residual in the kinematic boundary condition region
            residual_vector[self.left_boundary_region] = 0.0
            residual_vector[self.right_boundary_region] = 0.0

            #Compute the residual
            residual = scipy.linalg.norm(residual_vector)
 
            print "Residual = " + str(residual)

            #Break if converged
            if residual < absolute_tolerence: break

            #Compute the tangent stiffness
            tangent_stiffness = self.__compute_tangent_stiffness()


            #Apply the boundary conditions to the tangent stiffness matrix
            for index in self.left_boundary_region:
                tangent_stiffness[index,:] = 0.0
                tangent_stiffness[index,index] = 1.0
            for index in self.right_boundary_region:
                tangent_stiffness[index,:] = 0.0
                tangent_stiffness[index,index] = 1.0

            self.displacement += scipy.sparse.linalg.spsolve(
                    scipy.sparse.csr_matrix(tangent_stiffness), residual_vector)
            #End for loop

        return 

    def solve2(self,prescribed_displacement=0.1):

        self.prescribed_displacement = prescribed_displacement

        #Find the nodes within 1 horizon of each end to apply the boundary conditions on.
        self.left_boundary_region = self.tree.query_ball_point(self.pd_nodes[0, None], r=self.horizon, p=2, eps=0.0)
        self.right_boundary_region = self.tree.query_ball_point(self.pd_nodes[-1, None], r=self.horizon, p=2, eps=0.0)

        #Initial guess is linear between endpoint displacements
        #guess = np.linspace(-prescribed_displacement, 0.0, len(self.displacement))
        guess = np.zeros_like(self.displacement)

        #Solve
        self.displacement = scipy.optimize.newton_krylov(self.__compute_residual, guess)

    #Public get functions
    def get_solution(self):
        return self.displacement

    def get_nodes(self):
        return self.pd_nodes


### Main Program ####
if __name__ == "__main__":

    fixed_horizon = 3.5
    fixed_length = 20
    delta_x = 1.0

    #Instantiate a 1d peridynamic problem with equally spaced nodes
    problem1 = PD_Problem(bar_length=fixed_length, number_of_elements=(fixed_length/delta_x), horizon=fixed_horizon)
    #Solve the problem via Newton's method
    problem1.solve2()
    #problem1.solve(absolute_tolerence=1.0e-6,number_of_iterations=1000)
    #Get the node locations and displacement solution
    disp1 = problem1.get_solution()
    nodes1 = problem1.get_nodes()

    #Instantiate a 1d peridynamic problem with randomly spaced nodes
    problem2 = PD_Problem(bar_length=fixed_length, number_of_elements=(fixed_length/delta_x), horizon=fixed_horizon, randomization_factor=0.3)
    #Solve the problem via Newton's method
    problem2.solve2()
    #Get the node locations and displacement solution
    disp2 = problem2.get_solution()
    nodes2 = problem2.get_nodes()
    
    #plt.plot(nodes1, disp1, 'k*-')
    plt.plot(nodes1, disp1, 'k*-', nodes2, disp2, 'r+-')
    plt.show()



