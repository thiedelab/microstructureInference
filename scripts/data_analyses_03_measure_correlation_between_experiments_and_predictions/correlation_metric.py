#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 15:19:35 2025

@author: kwang
"""

import freud
import numpy as np
from skimage.metrics import hausdorff_distance, structural_similarity, mean_squared_error
from scipy import ndimage
from Map_Bragg_disks_to_image import map_sim_BDs_to_images, map_exp_BDs_to_images
from scipy.ndimage import gaussian_filter
import ot
from scipy.spatial.distance import cdist
import scipy.sparse as sp
from scipy.optimize import linprog
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import NearestNeighbors

def generate_2D_IMG_array(qx,
                          qy,
                          intensity,
                          k_max,
                          pixel_numbers,
                          corner_centered = False,
                          remove_direct_beam = False,
                             ):
    """
    Map py4DSTEM pointList object to 2D diffraction pattern array

    Args:            
        corner_centered: (boolean)
            boolean variable that determines whether to set the location of
            direct beam (qx, qy) = (0, 0) to corner of 2D numpy array of 
            diffraction image (diff2D_img) at index [0][0]. If True, 
            direct beam (qx, qy) = (0, 0) is located at the corner of 
            diff2D_img at index[0][0]. If False, direct beam (qx, qy) = (0, 0)
            is located at the center of 2D numpy array
            default: False

        remove_direct_beam: (boolean)
            boolean variable that directs whether to remove
            direct beam or not.
            Default is False

    Additional attributes:
        indices_of_qx_in_diff2d_img:
            indices of pixels of qx of Bragg peaks    
            
        indices_of_qy_in_diff2d_img:
            indices of pixels of qy of Bragg peaks

    returns:
          diff2D_img: 2D array of Bragg peaks 
          (shape: ($pixel_numbers$, $pixel_numbers$))
    """

    qx_refined = qx
    qy_refined = qy
    intensity_refined = intensity
    mask = None

    if remove_direct_beam:            
        mask = np.full((len(intensity)), True)
        index_of_direct_beam = np.argmax(intensity)
        mask[index_of_direct_beam] = False
        qx_refined = qx_refined[mask]
        qy_refined = qy_refined[mask]
        intensity_refined = intensity_refined[mask]
        
    qx_refined[np.where(np.abs(qx_refined[:]) < (1e-14))[0]] = 0.0
    qy_refined[np.where(np.abs(qy_refined[:]) < (1e-14))[0]] = 0.0

    stack_qxqy = np.stack((qx_refined, qy_refined), axis = 1)
    width_of_diff2D_img = k_max * 2.0
    pixel_size = width_of_diff2D_img / pixel_numbers

    if pixel_numbers % 2 == 0:
        pixel_bins = np.arange(-k_max, k_max + pixel_size, pixel_size) # change
        pixel_bins[np.where(np.abs(pixel_bins[:]) < (1e-14))[0]] = 0.0 # change
        
        
        digitized_bin = np.digitize(stack_qxqy, pixel_bins) - 1
    else:
        pixel_bins = np.linspace(-k_max - pixel_size/2.0, k_max + pixel_size/2.0, pixel_numbers + 1, endpoint = True)
        digitized_bin = np.digitize(stack_qxqy, pixel_bins) - 1


    diff2D_img = np.zeros((pixel_numbers, pixel_numbers))
    diff2D_img[digitized_bin[:,0],digitized_bin[:,1]] = intensity_refined

    if corner_centered:
        diff2D_img = np.roll(diff2D_img, (-int(pixel_numbers / 2), -int(pixel_numbers / 2)), axis=(0, 1))

    return diff2D_img

def assignment_cost_pairwise_distance(input_exp_BD_features, input_sim_BD_features, maximum_distance_cutoff = 0.47, power_of_intensity = 1):


    exp_BD_features = np.copy(input_exp_BD_features)
    sim_BD_features = np.copy(input_sim_BD_features)
    

    exp_n = exp_BD_features.shape[0]
    sim_n = sim_BD_features.shape[0]
    exp_BD_feature_dim = exp_BD_features.shape[1]
    sim_BD_feature_dim = sim_BD_features.shape[1]
    
    assert exp_BD_feature_dim == sim_BD_feature_dim, "feature dimension (axis 1) of experimental BraggDisk vector set and simulated BraggDisk vector set do not match"
    
    exp_BDs = exp_BD_features[:,:2]
    sim_BDs = sim_BD_features[:,:2]

    if exp_BD_features.shape[1] == 3:
        exp_BD_features[:,2] = exp_BD_features[:,2] ** power_of_intensity
        sim_BD_features[:,2] = sim_BD_features[:,2] ** power_of_intensity
    
    if exp_n < sim_n:
        
        pairDist_based_on_positions = cdist(exp_BDs, sim_BDs)
        
        reshaped_pairDist_based_on_positions = pairDist_based_on_positions.reshape((exp_n * sim_n,1))   # reshape the cost matrix to a column vector 

        constraint_ub_matrix = sp.kron(np.ones((1, exp_n)), sp.eye_array(sim_n)).toarray()
        constraint_ub_val = np.ones(sim_n)
        
        constraint_eq_matrix = sp.kron(sp.eye_array(exp_n), np.ones((1, sim_n))).toarray()
        constraint_eq_val = np.ones(exp_n)
        
        res = linprog(reshaped_pairDist_based_on_positions, constraint_ub_matrix, constraint_ub_val, constraint_eq_matrix, constraint_eq_val, (0, 1))
        indices_of_paired_BDs = np.where(res.x > 0.0)[0]

        pairDist_based_on_features = cdist(exp_BD_features, sim_BD_features)
        reshaped_pairDist_based_on_features = pairDist_based_on_features.reshape((exp_n * sim_n,1))

        pairFeatureDist = reshaped_pairDist_based_on_features[indices_of_paired_BDs]
        pairFeatureDist = pairFeatureDist.reshape(pairFeatureDist.shape[0])        
        pairFeatureDist = np.where(pairFeatureDist < maximum_distance_cutoff, pairFeatureDist, maximum_distance_cutoff)

        final_assignment_cost = np.average(pairFeatureDist)

    elif exp_n == sim_n:

        pairDist_based_on_positions = cdist(exp_BDs, sim_BDs)
        
        reshaped_pairDist_based_on_positions = pairDist_based_on_positions.reshape((exp_n * sim_n,1))   # reshape the cost matrix to a column vector 

        constraint_eq_matrix_1 = sp.kron(np.ones((1, exp_n)), sp.eye_array(sim_n)).toarray()
        constraint_eq_val_1 = np.ones(sim_n)
        
        constraint_eq_matrix_2 = sp.kron(sp.eye_array(exp_n), np.ones((1, sim_n))).toarray()
        constraint_eq_val_2 = np.ones(exp_n)
        
        constraint_eq_matrix_stacked = np.vstack((constraint_eq_matrix_1, constraint_eq_matrix_2))
        constraint_eq_val_stacked = np.hstack((constraint_eq_val_1, constraint_eq_val_2)).T
                
        res = linprog(reshaped_pairDist_based_on_positions, None, None, constraint_eq_matrix_stacked, constraint_eq_val_stacked, (0, 1))
        
        indices_of_paired_BDs = np.where(res.x > 0.0)[0]
        pairDist_based_on_features = cdist(exp_BD_features, sim_BD_features)
        reshaped_pairDist_based_on_features = pairDist_based_on_features.reshape((exp_n * sim_n,1))

        pairFeatureDist = reshaped_pairDist_based_on_features[indices_of_paired_BDs]
        pairFeatureDist = pairFeatureDist.reshape(pairFeatureDist.shape[0])

        pairFeatureDist = np.where(pairFeatureDist < maximum_distance_cutoff, pairFeatureDist, maximum_distance_cutoff)

        final_assignment_cost = np.average(pairFeatureDist)

    else:

        pairDist_based_on_positions = cdist(exp_BDs, sim_BDs)
        
        reshaped_pairDist_based_on_positions = pairDist_based_on_positions.reshape((exp_n * sim_n,1))   # reshape the cost matrix to a column vector 
                
        
        
        constraint_eq_matrix = sp.kron(sp.eye_array(exp_n), np.ones((sim_n,1))).toarray().T
        constraint_eq_val = np.ones(exp_n)
        
        constraint_ub_matrix = sp.kron(np.ones((1, exp_n)), sp.eye_array(sim_n)).toarray()
        constraint_ub_val = np.ones(sim_n) * exp_n        
        
        
        res = linprog(reshaped_pairDist_based_on_positions, constraint_ub_matrix, constraint_ub_val, constraint_eq_matrix, constraint_eq_val, (0, 1))
        
        indices_of_paired_BDs = np.where(res.x > 0.0)[0]
        pairDist_based_on_features = cdist(exp_BD_features, sim_BD_features)
        reshaped_pairDist_based_on_features = pairDist_based_on_features.reshape((exp_n * sim_n,1))

        pairFeatureDist = reshaped_pairDist_based_on_features[indices_of_paired_BDs]
        pairFeatureDist = pairFeatureDist.reshape(pairFeatureDist.shape[0])

        pairFeatureDist = np.where(pairFeatureDist < maximum_distance_cutoff, pairFeatureDist, maximum_distance_cutoff)
        final_assignment_cost = np.average(pairFeatureDist)
        
        
    return final_assignment_cost

def pyxem_correlation_metric(image, py4DSTEM_bragg_vecto_list, k_max, pixel_numbers, intensity_power_gamma = None):
    
    qx = py4DSTEM_bragg_vecto_list.data['qx']
    qy = py4DSTEM_bragg_vecto_list.data['qy']
    intensity = py4DSTEM_bragg_vecto_list.data['intensity']
    
    if intensity_power_gamma is not None:
        intensity = intensity ** intensity_power_gamma;
    
    kinematic_diffraction_pattern_2D = generate_2D_IMG_array(
                                                qx,
                                                qy,
                                                intensity,
                                                k_max,
                                                pixel_numbers,
                                                corner_centered = False,
                                                remove_direct_beam = True,
                                                )
    
    kinematic_diffraction_pattern_2D = kinematic_diffraction_pattern_2D / np.sqrt(np.sum(np.square(kinematic_diffraction_pattern_2D)))
    
    experimen_diffraction_pattern_2D = image / np.sqrt(np.sum(np.square(image)))
    
    correlation_score_Q = np.sum(np.multiply(kinematic_diffraction_pattern_2D,experimen_diffraction_pattern_2D))
    
    return correlation_score_Q

# def tanh_distance(distances, length_scaling_factor = 1):
#     exponent = np.exp((distances / length_scaling_factor) * 2)
#     return np.divide(exponent - 1, exponent + 1)

# def pairwise_distances(A, B):
#     """
#     Compute the pairwise Euclidean distances between points in A and points in B.
    
#     Parameters:
#         A (np.ndarray): Array of shape (N, D)
#             N is number of points, D is the spatial dimension
#         B (np.ndarray): Array of shape (M, D)
#             M is number of points, D is the spatial dimension
    
#     Returns:
#         np.ndarray: A distance matrix of shape (N, M) where element (i, j)
#                     is the distance between A[i] and B[j].
#     """
#     # Ensure input is numpy array
#     A = np.asarray(A)
#     B = np.asarray(B)
    
#     # Compute squared differences using broadcasting
#     diff = A[:, np.newaxis, :] - B[np.newaxis, :, :]  # Shape: (N, M, 3)
#     dist_squared = np.sum(diff ** 2, axis=2)          # Shape: (N, M)
#     distances = np.sqrt(dist_squared)                 # Euclidean distance
    
#     return distances
    
# def assignment_cost_tanh_dist(input_exp_BD_features, input_sim_BD_features, pixel_size = 0.0328, scale_based_on_pixel_size = 16, power_of_intensity = 0.4):

#     def tanh_distance(distances, length_scaling_factor = 1):
#         exponent = np.exp((distances / length_scaling_factor) * 2)
#         return np.divide(exponent - 1, exponent + 1)

#     scaling_factor_for_division = pixel_size * scale_based_on_pixel_size

#     exp_BD_features = np.copy(input_exp_BD_features)
#     sim_BD_features = np.copy(input_sim_BD_features)
    

#     exp_n = exp_BD_features.shape[0]
#     sim_n = sim_BD_features.shape[0]
#     exp_BD_feature_dim = exp_BD_features.shape[1]
#     sim_BD_feature_dim = sim_BD_features.shape[1]
    
#     assert exp_BD_feature_dim == sim_BD_feature_dim, "feature dimension (axis 1) of experimental BraggDisk vector set and simulated BraggDisk vector set do not match"
    
#     exp_BDs = exp_BD_features[:,:2]
#     sim_BDs = sim_BD_features[:,:2]

#     if exp_BD_features.shape[1] == 3:
#         exp_BD_features[:,2] = exp_BD_features[:,2] ** power_of_intensity
#         sim_BD_features[:,2] = sim_BD_features[:,2] ** power_of_intensity
    
#     if exp_n < sim_n:
        
#         pairDist_based_on_positions = cdist(exp_BDs, sim_BDs)
        
#         reshaped_pairDist_based_on_positions = pairDist_based_on_positions.reshape((exp_n * sim_n,1))   # reshape the cost matrix to a column vector 

#         constraint_ub_matrix = sp.kron(np.ones((1, exp_n)), sp.eye_array(sim_n)).toarray()
#         constraint_ub_val = np.ones(sim_n)
        
#         constraint_eq_matrix = sp.kron(sp.eye_array(exp_n), np.ones((1, sim_n))).toarray()
#         constraint_eq_val = np.ones(exp_n)
        
#         res = linprog(reshaped_pairDist_based_on_positions, constraint_ub_matrix, constraint_ub_val, constraint_eq_matrix, constraint_eq_val, (0, 1))
#         indices_of_paired_BDs = np.where(res.x > 0.0)[0]

#         pairDist_based_on_features = cdist(exp_BD_features, sim_BD_features)
#         reshaped_pairDist_based_on_features = pairDist_based_on_features.reshape((exp_n * sim_n,1))

#         pairFeatureDist = reshaped_pairDist_based_on_features[indices_of_paired_BDs]
#         pairFeatureDist = pairFeatureDist.reshape(pairFeatureDist.shape[0])

#         tanh_pairFeatureDist = tanh_distance(pairFeatureDist, scaling_factor_for_division)        
#         final_assignment_cost = np.average(tanh_pairFeatureDist)

#     elif exp_n == sim_n:

#         pairDist_based_on_positions = cdist(exp_BDs, sim_BDs)
        
#         reshaped_pairDist_based_on_positions = pairDist_based_on_positions.reshape((exp_n * sim_n,1))   # reshape the cost matrix to a column vector 

#         constraint_eq_matrix_1 = sp.kron(np.ones((1, exp_n)), sp.eye_array(sim_n)).toarray()
#         constraint_eq_val_1 = np.ones(sim_n)
        
#         constraint_eq_matrix_2 = sp.kron(sp.eye_array(exp_n), np.ones((1, sim_n))).toarray()
#         constraint_eq_val_2 = np.ones(exp_n)
        
#         constraint_eq_matrix_stacked = np.vstack((constraint_eq_matrix_1, constraint_eq_matrix_2))
#         constraint_eq_val_stacked = np.hstack((constraint_eq_val_1, constraint_eq_val_2)).T
                
#         res = linprog(reshaped_pairDist_based_on_positions, None, None, constraint_eq_matrix_stacked, constraint_eq_val_stacked, (0, 1))
        
#         indices_of_paired_BDs = np.where(res.x > 0.0)[0]
#         pairDist_based_on_features = cdist(exp_BD_features, sim_BD_features)
#         reshaped_pairDist_based_on_features = pairDist_based_on_features.reshape((exp_n * sim_n,1))

#         pairFeatureDist = reshaped_pairDist_based_on_features[indices_of_paired_BDs]
#         pairFeatureDist = pairFeatureDist.reshape(pairFeatureDist.shape[0])

#         tanh_pairFeatureDist = tanh_distance(pairFeatureDist, scaling_factor_for_division)
#         final_assignment_cost = np.average(tanh_pairFeatureDist)

#     else:

#         pairDist_based_on_positions = cdist(exp_BDs, sim_BDs)
        
#         reshaped_pairDist_based_on_positions = pairDist_based_on_positions.reshape((exp_n * sim_n,1))   # reshape the cost matrix to a column vector 
                
        
        
#         constraint_eq_matrix = sp.kron(sp.eye_array(exp_n), np.ones((sim_n,1))).toarray().T
#         constraint_eq_val = np.ones(exp_n)
        
#         constraint_ub_matrix = sp.kron(np.ones((1, exp_n)), sp.eye_array(sim_n)).toarray()
#         constraint_ub_val = np.ones(sim_n) * exp_n        
        
        
#         res = linprog(reshaped_pairDist_based_on_positions, constraint_ub_matrix, constraint_ub_val, constraint_eq_matrix, constraint_eq_val, (0, 1))
        
#         indices_of_paired_BDs = np.where(res.x > 0.0)[0]
#         pairDist_based_on_features = cdist(exp_BD_features, sim_BD_features)
#         reshaped_pairDist_based_on_features = pairDist_based_on_features.reshape((exp_n * sim_n,1))

#         pairFeatureDist = reshaped_pairDist_based_on_features[indices_of_paired_BDs]
#         pairFeatureDist = pairFeatureDist.reshape(pairFeatureDist.shape[0])

#         tanh_pairFeatureDist = tanh_distance(pairFeatureDist, scaling_factor_for_division)
#         final_assignment_cost = np.average(tanh_pairFeatureDist)


#     return final_assignment_cost, res.x.reshape(exp_n, sim_n)
    
# def assignment_cost_divide_by_num_matchedPair(exp_BD_features, sim_BD_features, pixel_size = 0.0328, pixel_numbers = 128, pairing_rmax = None):
    
#     if pairing_rmax is None:
#         pairing_rmax = pixel_size * 5
    

#     exp_n = exp_BD_features.shape[0]
#     sim_n = sim_BD_features.shape[0]
#     exp_BD_feature_dim = exp_BD_features.shape[1]
#     sim_BD_feature_dim = sim_BD_features.shape[1]
    
#     assert exp_BD_feature_dim == sim_BD_feature_dim, "feature dimension (axis 1) of experimental BraggDisk vector set and simulated BraggDisk vector set do not match"
    
#     exp_BDs = exp_BD_features[:,:2]
#     sim_BDs = sim_BD_features[:,:2]
    
#     # print("exp_BD_features\n", exp_BD_features, "\n")
#     # print("sim_BD_features\n", sim_BD_features, "\n")
    
#     # print("exp_BDs\n", exp_BDs, "\n")
#     # print("sim_BDs\n", sim_BDs, "\n")

    

#     if exp_n < sim_n:
        
#         #############################################################################
#         ######### Assign using the distance between Bragg Disks in reciprocal space

#         pairDist = cdist(exp_BDs, sim_BDs)
        
#         # print("cost before reshaping\n", cost, "\n")
        
#         reshaped_pairDist = pairDist.reshape((exp_n * sim_n,1))   # reshape the cost matrix to a column vector 

#         # print("reshaped_pairDist after reshaping\n", reshaped_pairDist, "\n")
#         # print("reshaped_pairDist.shape", reshaped_pairDist.shape, "\n")

#         constraint_ub_matrix = sp.kron(np.ones((1, exp_n)), sp.eye_array(sim_n)).toarray()
#         constraint_ub_val = np.ones(sim_n)
        
#         constraint_eq_matrix = sp.kron(sp.eye_array(exp_n), np.ones((1, sim_n))).toarray()
#         constraint_eq_val = np.ones(exp_n)
        
#         # print("constraint_ub_matrix\n", constraint_ub_matrix, "\n")
#         # print("constraint_eq_matrix \n", constraint_eq_matrix, "\n")
        
#         # print("constraint_ub_val\n", constraint_ub_val, "\n")
#         # print("constraint_eq_val\n", constraint_eq_val, "\n")
        
#         res = linprog(reshaped_pairDist, constraint_ub_matrix, constraint_ub_val, constraint_eq_matrix, constraint_eq_val, (0, 1))
#         indices_of_paired_BDs = np.where(res.x > 0.0)[0]
#         pairDist_of_paired_BDs = reshaped_pairDist[indices_of_paired_BDs]
#         indices_of_paired_BDs_within_r_max = np.where(pairDist_of_paired_BDs < pairing_rmax)[0]        
#         number_of_paired_BDs_within_r_max = indices_of_paired_BDs_within_r_max.shape[0]
        
        
#         pairFeatureCost = (exp_BD_features, sim_BD_features)
#         reshaped_pairFeatureCost = pairFeatureCost.reshape((exp_n * sim_n,1))
        
#         final_assignment_cost = np.sum(reshaped_pairFeatureCost[indices_of_paired_BDs])
#         if number_of_paired_BDs_within_r_max > 0:
#             final_assignment_cost = final_assignment_cost / number_of_paired_BDs_within_r_max
        
#         # print("np.sum(reshaped_pairFeatureCost[indices_of_paired_BDs])", np.sum(reshaped_pairFeatureCost[indices_of_paired_BDs]))
#         # print("final_assignment_cost\n", final_assignment_cost, "\n")
        
        
#         # print("reshaped_pairFeatureCost.shape", reshaped_pairFeatureCost.shape)
#         # print("")
        
#         # print("np.sum(pairDist_of_paired_BDs)", np.sum(pairDist_of_paired_BDs))        
#         # print("Minimum cost = ", res.fun)
        
#         # print("np.where(pairDist_of_paired_BDs < pairing_rmax)\n", np.where(pairDist_of_paired_BDs < pairing_rmax), "\n")
#         # print("indices_of_paired_BDs_within_r_max\n", indices_of_paired_BDs_within_r_max, "\n")
        
        
#         # # print("res.message\n", res.message, "\n")
#         # print("res.x\n", res.x, "\n")
        
#         # print("np.where(res.x > 0.0)[0]\n", np.where(res.x > 0.0)[0], "\n")
#         # print("indices_of_exp_BDs_with_sim_BD_neighbor\n", indices_of_paired_BDs, "\n")
#         # print("res.x.reshape(exp_n, sim_n)\n", res.x.reshape(exp_n, sim_n), "\n")
#         # print("Minimum cost = ", res.fun)
        
        
#         #############################################################################
#         ######### Assign using the distance between Bragg Disks in reciprocal space
        


#     elif exp_n == sim_n:

#         # print("exp_n == sim_n")

#         pairDist = cdist(exp_BDs, sim_BDs)


#         reshaped_pairDist = pairDist.reshape((exp_n * sim_n,1))   # reshape the cost matrix to a column vector   
#         # print("cost after reshaping\n", reshaped_cost, "\n")
#         # print("reshaped_cost.shape", reshaped_cost.shape, "\n")

#         constraint_eq_matrix_1 = sp.kron(np.ones((1, exp_n)), sp.eye_array(sim_n)).toarray()
#         constraint_eq_val_1 = np.ones(sim_n)
        
#         constraint_eq_matrix_2 = sp.kron(sp.eye_array(exp_n), np.ones((1, sim_n))).toarray()
#         constraint_eq_val_2 = np.ones(exp_n)
        
#         constraint_eq_matrix_stacked = np.vstack((constraint_eq_matrix_1, constraint_eq_matrix_2))
#         constraint_eq_val_stacked = np.hstack((constraint_eq_val_1, constraint_eq_val_2)).T
        
#         # print("constraint_eq_matrix_stacked\n", constraint_eq_matrix_stacked, "\n")
#         # print("constraint_eq_val_stacked\n", constraint_eq_val_stacked, "\n")
        
        
#         res = linprog(reshaped_pairDist, None, None, constraint_eq_matrix_stacked, constraint_eq_val_stacked, (0, 1))
#         # print("res.message\n", res.message, "\n")
#         # print("res.x\n", res.x, "\n")
#         # print("res.x.reshape(exp_n, sim_n)\n", res.x.reshape(exp_n, sim_n), "\n")
#         # print("Minimum cost = ", res.fun)
        
#         indices_of_paired_BDs = np.where(res.x > 0.0)[0]
#         pairDist_of_paired_BDs = reshaped_pairDist[indices_of_paired_BDs]
#         indices_of_paired_BDs_within_r_max = np.where(pairDist_of_paired_BDs < pairing_rmax)[0]        
#         number_of_paired_BDs_within_r_max = indices_of_paired_BDs_within_r_max.shape[0]
        
        
#         pairFeatureCost = cdist(exp_BD_features, sim_BD_features)
#         reshaped_pairFeatureCost = pairFeatureCost.reshape((exp_n * sim_n,1))
        
#         final_assignment_cost = np.sum(reshaped_pairFeatureCost[indices_of_paired_BDs])
#         if number_of_paired_BDs_within_r_max > 0:
#             final_assignment_cost = final_assignment_cost / number_of_paired_BDs_within_r_max

#     else:

#         # print("exp_n > sim_n")

#         pairDist = cdist(exp_BDs, sim_BDs)

#         # print("cost before reshaping\n", cost, "\n")
        
#         reshaped_pairDist = pairDist.reshape((sim_n * exp_n,1))   # reshape the cost matrix to a column vector   
        
#         # print("cost after reshaping\n", reshaped_cost, "\n")
#         # print("reshaped_cost.shape", reshaped_cost.shape, "\n")
        
        
        
#         constraint_eq_matrix = sp.kron(sp.eye_array(exp_n), np.ones((sim_n,1))).toarray().T
#         constraint_eq_val = np.ones(exp_n)
        
#         constraint_ub_matrix = sp.kron(np.ones((1, exp_n)), sp.eye_array(sim_n)).toarray()
#         constraint_ub_val = np.ones(sim_n) * exp_n
        
        
#         # print("constraint_eq_matrix\n", constraint_eq_matrix, "\n")
#         # print("constraint_eq_val\n", constraint_eq_val, "\n")
#         # print("constraint_ub_matrix\n", constraint_ub_matrix, "\n")
#         # print("constraint_ub_val\n", constraint_ub_val, "\n")
        
        
        
#         res = linprog(reshaped_pairDist, constraint_ub_matrix, constraint_ub_val, constraint_eq_matrix, constraint_eq_val, (0, 1))
#         # print("res.message\n", res.message, "\n")
#         # print("res.x\n", res.x, "\n")
#         # print("res.x.reshape(exp_n, sim_n)\n", res.x.reshape(exp_n, sim_n), "\n")
#         # print("Minimum cost = ", res.fun)
        
#         indices_of_paired_BDs = np.where(res.x > 0.0)[0]
#         pairDist_of_paired_BDs = reshaped_pairDist[indices_of_paired_BDs]
#         indices_of_paired_BDs_within_r_max = np.where(pairDist_of_paired_BDs < pairing_rmax)[0]        
#         number_of_paired_BDs_within_r_max = indices_of_paired_BDs_within_r_max.shape[0]
        
        
#         pairFeatureCost = cdist(exp_BD_features, sim_BD_features)
#         reshaped_pairFeatureCost = pairFeatureCost.reshape((exp_n * sim_n,1))
        
#         final_assignment_cost = np.sum(reshaped_pairFeatureCost[indices_of_paired_BDs])
#         if number_of_paired_BDs_within_r_max > 0:
#             final_assignment_cost = final_assignment_cost / number_of_paired_BDs_within_r_max
    


#     return final_assignment_cost, res.x.reshape(exp_n, sim_n)


# def assignment_cost(exp_BDs, 
#                     sim_BDs, 
#                     pixel_size = 0.0327, 
#                     pixel_numbers = 128,
#                     weighting='kde', 
#                     reg=0.01,
#                     distance_metric='euclidean', 
#                     sigma=1.0, 
#                     knn_k=5,
#                     kde_bandwidth = 0.5,
#                     ):

#     def compute_gaussian_weights(X, Y, sigma):
#         centroid = np.mean(X, axis=0)
#         dists = np.linalg.norm(X - centroid, axis=1)
#         weights = np.exp(-dists**2 / (2 * sigma**2))
#         return weights

#     def compute_kde_weights(X, Y):
#         kde = KernelDensity(kernel='gaussian', bandwidth=kde_bandwidth).fit(Y)
#         log_density = kde.score_samples(X)
#         return np.exp(log_density)

#     def compute_knn_density_weights(X, Y, k):
#         nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
#         dists, _ = nbrs.kneighbors(X)
#         avg_dist = np.mean(dists[:, 1:], axis=1)
#         return 1 / (avg_dist + 1e-8)

#     def get_weights(X, Y, method):
#         if method == 'uniform':
#             return np.ones(len(X))
#         elif method == 'gaussian_centroid':
#             return compute_gaussian_weights(X, sigma)
#         elif method == 'kde':
#             return compute_kde_weights(X, Y)
#         elif method == 'knn':
#             return compute_knn_density_weights(X, knn_k)
#         else:
#             raise ValueError(f"Unknown weighting method: {method}")

#     exp_n = exp_BDs.shape[0]
#     sim_n = sim_BDs.shape[0]
#     exp_BD_feature_dim = exp_BDs.shape[1]
#     sim_BD_feature_dim = sim_BDs.shape[1]
    
#     # print("exp_BDs\n", exp_BDs, "\n")
#     # print("sim_BDs\n", sim_BDs, "\n")

#     assert exp_BD_feature_dim == sim_BD_feature_dim, "feature dimension (axis 1) of experimental BraggDisk vector set and simulated BraggDisk vector set do not match"


#     weights_A = get_weights(exp_BDs, sim_BDs, weighting)
#     weights_A /= weights_A.sum()

#     # print("weights_A\n", weights_A, "\n")

#     if exp_n < sim_n:

#         # print("exp_n < sim_n")

#         # cost = cdist(exp_BDs, sim_BDs)
#         cost = cdist(exp_BDs, sim_BDs, 'euclidean')
        
#         # print("cost before reshaping\n", cost, "\n")
        
#         reshaped_cost = cost.reshape((exp_n * sim_n,1))   # reshape the cost matrix to a column vector 

#         # print("cost after reshaping\n", reshaped_cost, "\n")
#         # print("reshaped_cost.shape", reshaped_cost.shape, "\n")

#         constraint_ub_matrix = sp.kron(np.ones((1, exp_n)), sp.eye_array(sim_n)).toarray()
#         constraint_ub_val = np.ones(sim_n)
        
#         constraint_eq_matrix = sp.kron(sp.eye_array(exp_n), np.ones((1, sim_n))).toarray()
#         constraint_eq_val = np.ones(exp_n)
        
#         # print("constraint_ub_matrix\n", constraint_ub_matrix, "\n")
#         # print("constraint_eq_matrix \n", constraint_eq_matrix, "\n")
        
#         # print("constraint_ub_val\n", constraint_ub_val, "\n")
#         # print("constraint_eq_val\n", constraint_eq_val, "\n")
        
#         res = linprog(reshaped_cost, constraint_ub_matrix, constraint_ub_val, constraint_eq_matrix, constraint_eq_val, (0, 1))
#         # print("res.message\n", res.message, "\n")
#         # print("res.x\n", res.x, "\n")
#         indices_of_paired_BDs = np.where(res.x > 0.0)[0]
#         # print("indices_of_paired_BDs\n", indices_of_paired_BDs, "\n")
#         # print("res.x.reshape(exp_n, sim_n)\n", res.x.reshape(exp_n, sim_n), "\n")
#         # print("Minimum cost = ", res.fun)
#         cost_of_matched_exp_BDs = reshaped_cost[indices_of_paired_BDs]
#         cost_of_matched_exp_BDs = cost_of_matched_exp_BDs.reshape(cost_of_matched_exp_BDs.shape[0])

#         final_cost = np.dot(weights_A, cost_of_matched_exp_BDs)


#         # print("cost_of_matched_exp_BDs\n", cost_of_matched_exp_BDs, "\n")

#         # print("weights_A @ reshaped_cost[indices_of_paired_BDs]\n", weights_A @ reshaped_cost[indices_of_paired_BDs], "\n")
#         # print("np.dot(weights_A, cost_of_matched_exp_BDs)\n", np.dot(weights_A, cost_of_matched_exp_BDs), "\n")

#         # res.x.reshape(exp_n, sim_n)

#     elif exp_n == sim_n:

#         # print("exp_n == sim_n")

#         # cost = cdist(exp_BDs, sim_BDs)
#         cost = cdist(exp_BDs, sim_BDs, 'euclidean')


#         reshaped_cost = cost.reshape((exp_n * sim_n,1))   # reshape the cost matrix to a column vector   
#         # print("cost after reshaping\n", reshaped_cost, "\n")
#         # print("reshaped_cost.shape", reshaped_cost.shape, "\n")

#         constraint_eq_matrix_1 = sp.kron(np.ones((1, exp_n)), sp.eye_array(sim_n)).toarray()
#         constraint_eq_val_1 = np.ones(sim_n)
        
#         constraint_eq_matrix_2 = sp.kron(sp.eye_array(exp_n), np.ones((1, sim_n))).toarray()
#         constraint_eq_val_2 = np.ones(exp_n)
        
#         constraint_eq_matrix_stacked = np.vstack((constraint_eq_matrix_1, constraint_eq_matrix_2))
#         constraint_eq_val_stacked = np.hstack((constraint_eq_val_1, constraint_eq_val_2)).T
        
#         # print("constraint_eq_matrix_stacked\n", constraint_eq_matrix_stacked, "\n")
#         # print("constraint_eq_val_stacked\n", constraint_eq_val_stacked, "\n")
        
        
#         res = linprog(reshaped_cost, None, None, constraint_eq_matrix_stacked, constraint_eq_val_stacked, (0, 1))
#         # print("res.message\n", res.message, "\n")
#         # print("res.x\n", res.x, "\n")
#         # print("res.x.reshape(exp_n, sim_n)\n", res.x.reshape(exp_n, sim_n), "\n")
#         # print("Minimum cost = ", res.fun)
#         indices_of_paired_BDs = np.where(res.x > 0.0)[0]
#         # print("indices_of_paired_BDs\n", indices_of_paired_BDs, "\n")
#         # print("res.x.reshape(exp_n, sim_n)\n", res.x.reshape(exp_n, sim_n), "\n")
#         # print("Minimum cost = ", res.fun)
#         cost_of_matched_exp_BDs = reshaped_cost[indices_of_paired_BDs]
#         cost_of_matched_exp_BDs = cost_of_matched_exp_BDs.reshape(cost_of_matched_exp_BDs.shape[0])

#         final_cost = np.dot(weights_A, cost_of_matched_exp_BDs)

#     else:

#         # print("exp_n > sim_n")

#         # cost = cdist(exp_BDs, sim_BDs)
#         cost = cdist(exp_BDs, sim_BDs, 'euclidean')

#         # print("cost before reshaping\n", cost, "\n")
        
#         reshaped_cost = cost.reshape((sim_n * exp_n,1))   # reshape the cost matrix to a column vector   
        
#         # print("cost after reshaping\n", reshaped_cost, "\n")
#         # print("reshaped_cost.shape", reshaped_cost.shape, "\n")
        
        
        
#         constraint_eq_matrix = sp.kron(sp.eye_array(exp_n), np.ones((sim_n,1))).toarray().T
#         constraint_eq_val = np.ones(exp_n)
        
#         constraint_ub_matrix = sp.kron(np.ones((1, exp_n)), sp.eye_array(sim_n)).toarray()
#         constraint_ub_val = np.ones(sim_n) * exp_n
        
        
#         # print("constraint_eq_matrix\n", constraint_eq_matrix, "\n")
#         # print("constraint_eq_val\n", constraint_eq_val, "\n")
#         # print("constraint_ub_matrix\n", constraint_ub_matrix, "\n")
#         # print("constraint_ub_val\n", constraint_ub_val, "\n")
        
        
        
#         res = linprog(reshaped_cost, constraint_ub_matrix, constraint_ub_val, constraint_eq_matrix, constraint_eq_val, (0, 1))
#         # print("res.message\n", res.message, "\n")
#         # print("res.x\n", res.x, "\n")
#         # print("res.x.reshape(exp_n, sim_n)\n", res.x.reshape(exp_n, sim_n), "\n")
#         # print("Minimum cost = ", res.fun)
#         indices_of_paired_BDs = np.where(res.x > 0.0)[0]
#         # print("indices_of_paired_BDs\n", indices_of_paired_BDs, "\n")
#         # print("res.x.reshape(exp_n, sim_n)\n", res.x.reshape(exp_n, sim_n), "\n")
#         # print("Minimum cost = ", res.fun)
#         cost_of_matched_exp_BDs = reshaped_cost[indices_of_paired_BDs]
#         cost_of_matched_exp_BDs = cost_of_matched_exp_BDs.reshape(cost_of_matched_exp_BDs.shape[0])

#         final_cost = np.dot(weights_A, cost_of_matched_exp_BDs)

#     return final_cost, res.x.reshape(exp_n, sim_n)



# def unbalanced_sinkhorn_distance(
#                                 A, 
#                                 B, 
#                                 epsilon=0.05, 
#                                 tau=0.1,
#                                 weighting='uniform', 
#                                 reg=0.01, 
#                                 distance_metric='euclidean', 
#                                 sigma=1.0, 
#                                 knn_k=5
#                                 ):
#     """
#     Computes unbalanced optimal transport (partial matching) distance
#     between point sets A and B using Sinkhorn with KL marginal relaxation.
    
#     Parameters:
#     - A, B: (N, d) and (M, d) arrays of point sets.
#     - epsilon: entropic regularization term.
#     - tau: marginal relaxation strength (smaller = stricter match).
    
#     Returns:
#     - Transport cost (scalar)
#     - Transport matrix T (N x M)
#     """

#     def compute_gaussian_weights(X, Y, sigma):
#         centroid = np.mean(X, axis=0)
#         dists = np.linalg.norm(X - centroid, axis=1)
#         weights = np.exp(-dists**2 / (2 * sigma**2))
#         return weights

#     def compute_kde_weights(X, Y):
#         kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(Y)
#         log_density = kde.score_samples(X)
#         return np.exp(log_density)

#     def compute_knn_density_weights(X, Y, k):
#         nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
#         dists, _ = nbrs.kneighbors(X)
#         avg_dist = np.mean(dists[:, 1:], axis=1)
#         return 1 / (avg_dist + 1e-8)

#     def get_weights(X, Y, method):
#         if method == 'uniform':
#             return np.ones(len(X))
#         elif method == 'gaussian_centroid':
#             return compute_gaussian_weights(X, sigma)
#         elif method == 'kde':
#             return compute_kde_weights(X, Y)
#         elif method == 'knn':
#             return compute_knn_density_weights(X, knn_k)
#         else:
#             raise ValueError(f"Unknown weighting method: {method}")


#     # N, M = len(A), len(B)
    
#     weights_A = get_weights(A, B, weighting)
#     weights_B = get_weights(B, A, weighting)

#     weights_A /= weights_A.sum()
#     weights_B /= weights_B.sum()
    
#     C = ot.dist(A, B, metric='euclidean')  # Cost matrix
    
#     T = ot.unbalanced.sinkhorn_unbalanced(weights_A, weights_B, C, reg=epsilon, reg_m=tau)
#     cost = np.sum(T * C)

#     matched_indices = []
#     for i in range(len(A)):
#         j = np.argmax(T[i])  # Find the index in Q that has the highest weight for P[i]
#         matched_indices.append((i, j))
    
#     return cost, T



# def _subtract_dog(frame, min_sigma = 2, max_sigma = 8):
#     """Background removal using difference of Gaussians.

#     Parameters
#     ----------
#     frame : NumPy 2D array
#     min_sigma : float
#     max_sigma : float

#     Returns
#     -------
#     background_removed : Numpy 2D array

#     Examples
#     --------
#     >>> import pyxem.utils._dask as dt
#     >>> s = pxm.data.dummy_data.dummy_data.get_cbed_signal()
#     >>> s_rem = dt._background_removal_single_frame_dog(s.data[0, 0])

#     """
#     blur_max = gaussian_filter(frame, max_sigma)
#     blur_min = gaussian_filter(frame, min_sigma)
#     return np.maximum(np.where(blur_min > blur_max, frame, 0) - blur_max, 0)

# def remove_artifacts_at_corner(image, distance_from_corner = 10):
#     new_image = np.copy(image)
#     for i in range(0, image.shape[0], image.shape[0] - 1):
#         for j in range(0, image.shape[1], image.shape[1] - 1):

#             for neighbors_of_i in range(i - distance_from_corner, i + distance_from_corner + 1):
#                 if neighbors_of_i > -1 and neighbors_of_i < image.shape[0]:
#                     for neighbors_of_j in range(j - distance_from_corner, j + distance_from_corner + 1):
#                         if neighbors_of_j > -1 and neighbors_of_j < image.shape[1]:
#                             new_image[neighbors_of_i][neighbors_of_j] = np.min(image)
            
#     return new_image

# def rotation_wrt_zAxis(angle_in_rad):
#     return np.array(
#                     [
#                         [np.cos(angle_in_rad), np.sin(angle_in_rad), 0],
#                         [-np.sin(angle_in_rad), np.cos(angle_in_rad), 0],
#                         [0, 0, 1],
#                     ]
#                     )


# def rotation_wrt_xAxis(angle_in_rad):
#     return np.array(
#                     [
#                         [1, 0, 0],
#                         [0, np.cos(angle_in_rad), np.sin(angle_in_rad)],
#                         [0, -np.sin(angle_in_rad), np.cos(angle_in_rad)],
#                     ]
#                     )


# def rotMatrix_from_eulerAngles_ZXZ(
#                                     angle_z1,
#                                     angle_x1, 
#                                     angle_z2,
#                                     ):
    
#     rotationMatrix =    rotation_wrt_zAxis(angle_z1) @ \
#                         rotation_wrt_xAxis(angle_x1) @ \
#                         rotation_wrt_zAxis(angle_z2)
        
#     return rotationMatrix

# def remove_direct_beam(qx, qy, intensity):
#     index_of_direct_beam = np.argmax(intensity)
#     masked_qx = np.delete(qx, index_of_direct_beam)
#     masked_qy = np.delete(qy, index_of_direct_beam)
#     masked_intensity = np.delete(intensity, index_of_direct_beam)
#     return masked_qx, masked_qy, masked_intensity
    
# def cost_c_between_two_sets_of_Bragg_disks_disregard_sim_outlier(
#                                         experimental_Bragg_vectors, 
#                                         crystal, 
#                                         orientation, 
#                                         excitation_error, 
#                                         pixel_numbers,
#                                         pixel_size,
#                                         correlation_kernel_size = None,
#                                         removeDirectBeam = True,                                        
#                                         relative_intensity_scale = 1.0,
#                             ):

#     ## NOTE: if intensity values of simulated bragg vectors are unreliable, then I think this correlation score is unreliable   

#     ############################################################################################################################
#     ## Prepare Bragg Disk vector data set for experimental image and simulated image given an orientaiton

#     if correlation_kernel_size is None:
#         correlation_kernel_size = pixel_size * 3.
    
#     py4DSTEM_kinematic_Diffraction = map_sim_BDs_to_images(
#                                                             crystal, 
#                                                             orientation_matrix = orientation, 
#                                                             excitation_error = excitation_error, 
#                                                             pixel_numbers = pixel_numbers,
#                                                             )

#     sim_qx = np.copy(py4DSTEM_kinematic_Diffraction.qx)
#     sim_qy = np.copy(py4DSTEM_kinematic_Diffraction.qy)
#     sim_intensity = np.copy(py4DSTEM_kinematic_Diffraction.intensity)

#     exp_qx = np.copy(experimental_Bragg_vectors.data['qx'])
#     exp_qy = np.copy(experimental_Bragg_vectors.data['qy'])
#     exp_intensity = experimental_Bragg_vectors.data['intensity']

    
#     if removeDirectBeam:
#         sim_qx, sim_qy, sim_intensity = remove_direct_beam(sim_qx, sim_qy, sim_intensity)
#         exp_qx, exp_qy, exp_intensity = remove_direct_beam(exp_qx, exp_qy, exp_intensity)


    

#     ############################################################################################################################
#     # Query neighbors using freud AABBQuery

#     sim_BD_2D_pos = np.stack((sim_qx, sim_qy, np.zeros_like(sim_qx))).T    
#     exp_BD_2D_pos = np.stack((exp_qx, exp_qy, np.zeros_like(exp_qx))).T

    

#     image_width = pixel_size * pixel_numbers
#     box = freud.box.Box.cube(image_width * 2.0)
#     aq = freud.locality.AABBQuery(box, exp_BD_2D_pos)

#     query_result = aq.query(sim_BD_2D_pos, dict(num_neighbors=1, exclude_ii=False, r_max = correlation_kernel_size))
#     nlist = query_result.toNeighborList()
#     index_of_sim_pos_with_expNeighbors = nlist.query_point_indices
#     index_of_exp_pos_with_simNeighbors = nlist.point_indices
#     print("index_of_exp_pos_with_simNeighbors", index_of_exp_pos_with_simNeighbors)
    
#     exp_intensity = exp_intensity / np.max(exp_intensity)
#     sim_intensity = sim_intensity / np.max(sim_intensity)

#     cost_function_pairs = 0.0
#     if len(index_of_sim_pos_with_expNeighbors) > 0:
#         for (i, j) in nlist:
#             distance = np.linalg.norm(sim_BD_2D_pos[i] - exp_BD_2D_pos[j])
#             intensity_of_a_sim_BD = sim_intensity[i]
#             intensity_of_a_exp_BD = exp_intensity[j]
    
#             cost_function_pairs += (np.abs(intensity_of_a_exp_BD - (relative_intensity_scale * intensity_of_a_sim_BD)) * (distance)) / correlation_kernel_size
    
#         # cost_function_pairs = cost_function_pairs / len(index_of_sim_pos_with_expNeighbors)


#     cost_function_false_positive_exp = 0.0

#     for index in range(len(exp_BD_2D_pos)):
#         if index not in index_of_exp_pos_with_simNeighbors:
#             cost_function_false_positive_exp += exp_intensity[index]
    
#     # if cost_function_false_positive_exp > 0.0:
    
#     #     cost_function_false_positive_exp = cost_function_false_positive_exp / (len(exp_BD_2D_pos) - len(index_of_exp_pos_with_simNeighbors))

#     cost_function = cost_function_pairs + cost_function_false_positive_exp
    

#     return cost_function

# ## TODO
# def Q_index_between_ExpImg_and_convolvedSimImg_disregard_sim_outlier(
#                             experimental_image, 
#                             experimental_Bragg_vectors,
#                             py4DSTEM_crystal_object, 
#                             orientation, 
#                             excitation_error, 
#                             pixel_numbers,
#                             pixel_size,
#                             correlation_kernel,
#                             removeDirectBeam = True,
#                             considerEntireExpImage = True,
#                             BD_pairing_distance = None,
#                             ):
#     # liimt:
#     # if diffraction pattern has one perfect Bragg disk match this would give large correlation score
#     # not good..
    
#     if BD_pairing_distance is None:
#         BD_pairing_distance = pixel_size * 3
    
#     py4DSTEM_kinematic_Diffraction = map_sim_BDs_to_images(
#                                                             py4DSTEM_crystal_object, 
#                                                             orientation_matrix = orientation, 
#                                                             excitation_error = excitation_error, 
#                                                             pixel_numbers = pixel_numbers,
#                                                             )
    
#     sim_qx = np.copy(py4DSTEM_kinematic_Diffraction.qx)
#     sim_qy = np.copy(py4DSTEM_kinematic_Diffraction.qy)
#     sim_intensity = np.copy(py4DSTEM_kinematic_Diffraction.intensity)

#     exp_qx = np.copy(experimental_Bragg_vectors.data['qx'])
#     exp_qy = np.copy(experimental_Bragg_vectors.data['qy'])
#     exp_intensity = experimental_Bragg_vectors.data['intensity']
    


#     if removeDirectBeam:
#         sim_qx, sim_qy, sim_intensity = remove_direct_beam(sim_qx, sim_qy, sim_intensity)
#         exp_qx, exp_qy, exp_intensity = remove_direct_beam(exp_qx, exp_qy, exp_intensity)
            
    
    
#     sim_BD_2D_pos = np.stack((sim_qx, sim_qy, np.zeros_like(sim_qx))).T    
#     exp_BD_2D_pos = np.stack((exp_qx, exp_qy, np.zeros_like(exp_qx))).T

#     image_width = pixel_size * pixel_numbers
#     box = freud.box.Box.cube(image_width * 2.5)
#     aq = freud.locality.AABBQuery(box, exp_BD_2D_pos)

#     query_result = aq.query(sim_BD_2D_pos, dict(num_neighbors=1, exclude_ii=False, r_max = BD_pairing_distance))
#     nlist = query_result.toNeighborList()
#     index_of_sim_pos_with_expNeighbors = nlist.query_point_indices
#     index_of_exp_pos_with_simNeighbors = nlist.point_indices
    
#     if len(index_of_sim_pos_with_expNeighbors) > 0:
    
#         sim_qx = sim_qx[index_of_sim_pos_with_expNeighbors]
#         sim_qy = sim_qy[index_of_sim_pos_with_expNeighbors]
#         sim_intensity = sim_intensity[index_of_sim_pos_with_expNeighbors]
#         sim_intensity = sim_intensity / np.max(sim_intensity)
        
#         py4DSTEM_kinematic_Diffraction.qx, py4DSTEM_kinematic_Diffraction.qy, py4DSTEM_kinematic_Diffraction.intensity = sim_qx, sim_qy, sim_intensity    
        
            
#         simulated_DP_image = py4DSTEM_kinematic_Diffraction.generate_2D_IMG_array(
#                                                                     remove_direct_beam = False,
#                                                                     )
    
#         convolved_sim_image = ndimage.convolve(simulated_DP_image, correlation_kernel, mode='reflect')
#         convolved_sim_image_normalized = convolved_sim_image / np.linalg.norm(convolved_sim_image)
        
#         k_max = py4DSTEM_crystal_object.k_max
        

    
#         if considerEntireExpImage:            
#             experimental_image_normalized = experimental_image / np.linalg.norm(experimental_image)    
#             correlation_val = np.sum(np.multiply(experimental_image_normalized, convolved_sim_image_normalized))
    
#         else:    
#             indices_0, indices_1 = np.where(simulated_DP_image == 0.0)
#             experimental_image_maksed = np.copy(experimental_image)
#             experimental_image_maksed[indices_0,indices_1] = 0.0
#             indices_0_with_signals, indices_1_with_signals = np.where(experimental_image_maksed > 0.0)
#             experimental_image_maksed[indices_0_with_signals, indices_1_with_signals] = experimental_image_maksed[indices_0_with_signals, indices_1_with_signals] / np.sqrt(np.sum(np.square(experimental_image_maksed[indices_0_with_signals, indices_1_with_signals])))    
#             correlation_val = np.sum(np.multiply(experimental_image_maksed, convolved_sim_image_normalized))
    
#     else:
#         correlation_val = 0.0
    

#     return correlation_val


# def ssim_and_mse_metric_disregard_sim_outlier(
#                         experimental_image, 
#                         experimental_Bragg_vectors,
#                         py4DSTEM_crystal_object, 
#                         orientation, 
#                         excitation_error, 
#                         pixel_numbers,
#                         pixel_size,
#                         correlation_kernel,
#                         removeDirectBeam = True,
#                         considerEntireExpImage = True,
#                         BD_pairing_distance = None,
#                         ):

#     if BD_pairing_distance is None:
#         BD_pairing_distance = pixel_size * 3
    
#     py4DSTEM_kinematic_Diffraction = map_sim_BDs_to_images(
#                                                             py4DSTEM_crystal_object, 
#                                                             orientation_matrix = orientation, 
#                                                             excitation_error = excitation_error, 
#                                                             pixel_numbers = pixel_numbers,
#                                                             )
    
#     sim_qx = np.copy(py4DSTEM_kinematic_Diffraction.qx)
#     sim_qy = np.copy(py4DSTEM_kinematic_Diffraction.qy)
#     sim_intensity = np.copy(py4DSTEM_kinematic_Diffraction.intensity)

#     exp_qx = np.copy(experimental_Bragg_vectors.data['qx'])
#     exp_qy = np.copy(experimental_Bragg_vectors.data['qy'])
#     exp_intensity = experimental_Bragg_vectors.data['intensity']
    

#     if removeDirectBeam:
#         sim_qx, sim_qy, sim_intensity = remove_direct_beam(sim_qx, sim_qy, sim_intensity)
#         exp_qx, exp_qy, exp_intensity = remove_direct_beam(exp_qx, exp_qy, exp_intensity)
    
            
    
    
#     sim_BD_2D_pos = np.stack((sim_qx, sim_qy, np.zeros_like(sim_qx))).T    
#     exp_BD_2D_pos = np.stack((exp_qx, exp_qy, np.zeros_like(exp_qx))).T

#     image_width = pixel_size * pixel_numbers
#     box = freud.box.Box.cube(image_width * 2.5)
#     aq = freud.locality.AABBQuery(box, exp_BD_2D_pos)

#     query_result = aq.query(sim_BD_2D_pos, dict(num_neighbors=1, exclude_ii=False, r_max = BD_pairing_distance))
#     nlist = query_result.toNeighborList()
#     index_of_sim_pos_with_expNeighbors = nlist.query_point_indices
    
#     sim_qx = sim_qx[index_of_sim_pos_with_expNeighbors]
#     sim_qy = sim_qy[index_of_sim_pos_with_expNeighbors]
#     sim_intensity = sim_intensity[index_of_sim_pos_with_expNeighbors]
    
#     if len(index_of_sim_pos_with_expNeighbors) > 0:
    
#         sim_intensity = sim_intensity / np.max(sim_intensity)
#         exp_intensity = exp_intensity / np.max(exp_intensity)
        
#         py4DSTEM_kinematic_Diffraction.qx, py4DSTEM_kinematic_Diffraction.qy, py4DSTEM_kinematic_Diffraction.intensity = sim_qx, sim_qy, sim_intensity    
        
            
#         simulated_DP_image = py4DSTEM_kinematic_Diffraction.generate_2D_IMG_array(
#                                                                     remove_direct_beam = False,
#                                                                     )
    
#         convolved_sim_image = ndimage.convolve(simulated_DP_image, correlation_kernel, mode='reflect')
#         convolved_sim_image_normalized = convolved_sim_image / np.linalg.norm(convolved_sim_image)
        
#         ### EXPERIMENTAL CONVOLED IMAGES
        
#         exp_Brag_Disks_list = map_exp_BDs_to_images(exp_qx, exp_qy, exp_intensity, py4DSTEM_crystal_object.k_max, pixel_numbers)
#         expeirmental_DP_image = exp_Brag_Disks_list.generate_2D_IMG_array()
        
        
        
        
        
#         convoled_exp_image = ndimage.convolve(expeirmental_DP_image, correlation_kernel, mode='reflect')
#         convoled_exp_image_normalized = convoled_exp_image / np.linalg.norm(convoled_exp_image)
        
#         k_max = py4DSTEM_crystal_object.k_max
        
#         # fig, ax = plt.subplots(1, 1, figsize=(6, 6))
#         # ax.set_title("conv Exp (DP)")
#         # im = ax.imshow(convoled_exp_image_normalized,  norm = 'log', cmap = 'gray', extent=[-k_max, k_max, k_max, -k_max])
#         # ax.set_xlabel(r"$q_y$", fontsize = 16)
#         # ax.set_ylabel(r"$q_x$", fontsize = 16)
#         # plt.colorbar(im, ax = ax, fraction=0.0468, pad=0.02)
#         # plt.show()
        
#         # fig, ax = plt.subplots(1, 1, figsize=(6, 6))
#         # ax.set_title("conv Sim (DP)")
#         # im = ax.imshow(convolved_sim_image_normalized,  norm = 'log', cmap = 'gray', extent=[-k_max, k_max, k_max, -k_max])
#         # ax.set_xlabel(r"$q_y$", fontsize = 16)
#         # ax.set_ylabel(r"$q_x$", fontsize = 16)
#         # plt.colorbar(im, ax = ax, fraction=0.0468, pad=0.02)
#         # plt.show()
    
#         ssim = structural_similarity(convoled_exp_image_normalized , convolved_sim_image_normalized, data_range = convolved_sim_image_normalized.max() - convolved_sim_image_normalized.min())
#         mse = mean_squared_error(convoled_exp_image_normalized , convolved_sim_image_normalized)
#     else:
#         ssim, mse = 0, 1000

#     return ssim, mse


# def remove_direct_beam(qx, qy, intensity):
#     index_of_direct_beam = np.argmax(intensity)
#     masked_qx = np.delete(qx, index_of_direct_beam)
#     masked_qy = np.delete(qy, index_of_direct_beam)
#     masked_intensity = np.delete(intensity, index_of_direct_beam)
#     return masked_qx, masked_qy, masked_intensity
    
# def cost_c_between_two_sets_of_Bragg_disks(
#                                         experimental_Bragg_vectors, 
#                                         crystal, 
#                                         orientation, 
#                                         excitation_error, 
#                                         pixel_numbers,
#                                         pixel_size,
#                                         correlation_kernel_size = None,
#                                         removeDirectBeam = True,                                        
#                                         relative_intensity_scale = 1.0,
#                             ):

#     ## NOTE: if intensity values of simulated bragg vectors are unreliable, then I think this correlation score is unreliable   

#     ############################################################################################################################
#     ## Prepare Bragg Disk vector data set for experimental image and simulated image given an orientaiton

#     if correlation_kernel_size is None:
#         correlation_kernel_size = pixel_size * 3.
    
#     py4DSTEM_kinematic_Diffraction = map_sim_BDs_to_images(
#                                                             crystal, 
#                                                             orientation_matrix = orientation, 
#                                                             excitation_error = excitation_error, 
#                                                             pixel_numbers = pixel_numbers,
#                                                             )

#     sim_qx = np.copy(py4DSTEM_kinematic_Diffraction.qx)
#     sim_qy = np.copy(py4DSTEM_kinematic_Diffraction.qy)
#     sim_intensity = np.copy(py4DSTEM_kinematic_Diffraction.intensity)

#     exp_qx = np.copy(experimental_Bragg_vectors.data['qx'])
#     exp_qy = np.copy(experimental_Bragg_vectors.data['qy'])
#     exp_intensity = experimental_Bragg_vectors.data['intensity']

    
#     if removeDirectBeam:
#         sim_qx, sim_qy, sim_intensity = remove_direct_beam(sim_qx, sim_qy, sim_intensity)
#         exp_qx, exp_qy, exp_intensity = remove_direct_beam(exp_qx, exp_qy, exp_intensity)

#     exp_intensity = exp_intensity / np.max(exp_intensity)
#     sim_intensity = sim_intensity / np.max(sim_intensity)

    

#     ############################################################################################################################
#     # Query neighbors using freud AABBQuery

#     sim_BD_2D_pos = np.stack((sim_qx, sim_qy, np.zeros_like(sim_qx))).T
    
#     exp_BD_2D_pos = np.stack((exp_qx, exp_qy, np.zeros_like(exp_qx))).T

    

#     image_width = pixel_size * pixel_numbers
#     box = freud.box.Box.cube(image_width * 2.0)
#     aq = freud.locality.AABBQuery(box, exp_BD_2D_pos)

#     query_result = aq.query(sim_BD_2D_pos, dict(num_neighbors=1, exclude_ii=False, r_max = correlation_kernel_size))
#     nlist = query_result.toNeighborList()
#     index_of_sim_pos_with_expNeighbors = nlist.query_point_indices
#     index_of_exp_pos_with_simNeighbors = nlist.point_indices
#     distances = []

#     cost_function_pairs = 0.0
#     if len(index_of_sim_pos_with_expNeighbors) > 0:
#         for (i, j) in nlist:
#             distance = np.linalg.norm(sim_BD_2D_pos[i] - exp_BD_2D_pos[j])
#             intensity_of_a_sim_BD = sim_intensity[i]
#             intensity_of_a_exp_BD = exp_intensity[j]
    
#             cost_function_pairs += ((np.abs(intensity_of_a_exp_BD - (relative_intensity_scale * intensity_of_a_sim_BD)) * (1.0 - distance)) + (relative_intensity_scale * intensity_of_a_sim_BD * distance)) / correlation_kernel_size
    
#         cost_function_pairs = cost_function_pairs / len(index_of_sim_pos_with_expNeighbors)


#     cost_function_false_positive_exp = 0.0

#     for index in range(len(exp_BD_2D_pos)):
#         if index not in index_of_exp_pos_with_simNeighbors:
#             cost_function_false_positive_exp += 0.5 * exp_intensity[index]

    
#     cost_function_false_positive_sim = 0.0
#     for index in range(len(sim_BD_2D_pos)):
#         if index not in index_of_sim_pos_with_expNeighbors:
#             cost_function_false_positive_sim += 0.5 * sim_intensity[index] * relative_intensity_scale


#     cost_function = cost_function_pairs + cost_function_false_positive_exp + cost_function_false_positive_sim
    


#     return cost_function

# def image_correlation_Q_index_between_ExpImg_and_convolvedSimImg(
#                             experimental_image, 
#                             py4DSTEM_crystal_object, 
#                             orientation, 
#                             excitation_error, 
#                             pixel_numbers,
#                             correlation_kernel,
#                             removeDirectBeam = True,
#                             considerEntireExpImage = True,
#                             ):

#     ## NOTE: if intensity values of simulated bragg vectors are unreliable, then I think this correlation score is unreliable    
    
#     ## Well. There is another approach using dynamic simulation.
#     ## for given orientation, run dynamic scattering simulation, 
#     ## and find the thickness that leads to maximum correlation.
#     ## Still suffer from exciation errror though..

#     ## TO DO LIST:
#     ## WRITE DYNAMIC SCATTERING, and optimize it by thickness
#     ## MINIMIZE THE RELATIVE INTENSITY

#     if removeDirectBeam:

#         py4DSTEM_kinematic_Diffraction = map_sim_BDs_to_images(
#                                                                 py4DSTEM_crystal_object, 
#                                                                 orientation_matrix = orientation, 
#                                                                 excitation_error = excitation_error, 
#                                                                 pixel_numbers = pixel_numbers,
#                                                                 )

#         # print("py4DSTEM_kinematic_Diffraction.intensity before", py4DSTEM_kinematic_Diffraction.intensity)
#         py4DSTEM_kinematic_Diffraction.qx, py4DSTEM_kinematic_Diffraction.qy, py4DSTEM_kinematic_Diffraction.intensity = remove_direct_beam(py4DSTEM_kinematic_Diffraction.qx, py4DSTEM_kinematic_Diffraction.qy, py4DSTEM_kinematic_Diffraction.intensity)
#         # print("np.stack((py4DSTEM_kinematic_Diffraction.qx, py4DSTEM_kinematic_Diffraction.qy)).T\n",  np.stack((py4DSTEM_kinematic_Diffraction.qx, py4DSTEM_kinematic_Diffraction.qy)).T)
#         # py4DSTEM_kinematic_Diffraction.intensity = py4DSTEM_kinematic_Diffraction.intensity / np.sqrt(np.sum(np.square(py4DSTEM_kinematic_Diffraction.intensity)))
#         # print("py4DSTEM_kinematic_Diffraction.intensity before", py4DSTEM_kinematic_Diffraction.intensity)
#         simulated_DP_image = py4DSTEM_kinematic_Diffraction.generate_2D_IMG_array(
#                                                                     get_broadened = False, 
#                                                                     normalize_2D_IMG_array = False, 
#                                                                     remove_direct_beam = False,
#                                                                     )

#         convolved_sim_image = ndimage.convolve(simulated_DP_image, correlation_kernel, mode='reflect')
#         convolved_sim_image_normalized = convolved_sim_image / np.linalg.norm(convolved_sim_image)

   

#         if considerEntireExpImage:            
#             experimental_image_normalized = experimental_image / np.linalg.norm(experimental_image)    
#             correlation_val = np.sum(np.multiply(experimental_image_normalized, convolved_sim_image_normalized))

#         else:    
#             indices_0, indices_1 = np.where(simulated_DP_image == 0.0)
    
#             experimental_image_maksed = np.copy(experimental_image)
#             experimental_image_maksed[indices_0,indices_1] = 0.0
#             indices_0_with_signals, indices_1_with_signals = np.where(experimental_image_maksed > 0.0)
#             experimental_image_maksed[indices_0_with_signals, indices_1_with_signals] = experimental_image_maksed[indices_0_with_signals, indices_1_with_signals] / np.sqrt(np.sum(np.square(experimental_image_maksed[indices_0_with_signals, indices_1_with_signals])))    
#             correlation_val = np.sum(np.multiply(experimental_image_maksed, convolved_sim_image_normalized))


#     return correlation_val

# def image_correlation_Q_index(
#                             experimental_image, 
#                             py4DSTEM_crystal_object, 
#                             orientation, 
#                             excitation_error, 
#                             pixel_numbers,
#                             removeDirectBeam = True,
#                             considerEntireExpImage = True,
#                             ):

#     ## NOTE: if intensity values of simulated bragg vectors are unreliable, then I think this correlation score is unreliable    
    
#     ## Well. There is another approach using dynamic simulation.
#     ## for given orientation, run dynamic scattering simulation, 
#     ## and find the thickness that leads to maximum correlation.
#     ## Still suffer from exciation errror though..

#     ## TO DO LIST:
#     ## WRITE DYNAMIC SCATTERING, and optimize it by thickness
#     ## MINIMIZE THE RELATIVE INTENSITY

    

#     if removeDirectBeam:

#         py4DSTEM_kinematic_Diffraction = map_sim_BDs_to_images(
#                                                                 py4DSTEM_crystal_object, 
#                                                                 orientation_matrix = orientation, 
#                                                                 excitation_error = excitation_error, 
#                                                                 pixel_numbers = pixel_numbers,
#                                                                 )

#         # print("py4DSTEM_kinematic_Diffraction.intensity before", py4DSTEM_kinematic_Diffraction.intensity)
#         py4DSTEM_kinematic_Diffraction.qx, py4DSTEM_kinematic_Diffraction.qy, py4DSTEM_kinematic_Diffraction.intensity = remove_direct_beam(py4DSTEM_kinematic_Diffraction.qx, py4DSTEM_kinematic_Diffraction.qy, py4DSTEM_kinematic_Diffraction.intensity)
#         # print("np.stack((py4DSTEM_kinematic_Diffraction.qx, py4DSTEM_kinematic_Diffraction.qy)).T\n",  np.stack((py4DSTEM_kinematic_Diffraction.qx, py4DSTEM_kinematic_Diffraction.qy)).T)
#         py4DSTEM_kinematic_Diffraction.intensity = py4DSTEM_kinematic_Diffraction.intensity / np.sqrt(np.sum(np.square(py4DSTEM_kinematic_Diffraction.intensity)))
#         # print("py4DSTEM_kinematic_Diffraction.intensity before", py4DSTEM_kinematic_Diffraction.intensity)
#         simulated_DP_image = py4DSTEM_kinematic_Diffraction.generate_2D_IMG_array(
#                                                                     get_broadened = False, 
#                                                                     normalize_2D_IMG_array = False, 
#                                                                     remove_direct_beam = False,
#                                                                     )


#         if considerEntireExpImage:            
#             experimental_image_normalized = experimental_image / np.linalg.norm(experimental_image)    
#             correlation_val = np.sum(np.multiply(experimental_image_normalized, simulated_DP_image))

#         else:    
#             indices_0, indices_1 = np.where(simulated_DP_image == 0.0)
    
#             experimental_image_maksed = np.copy(experimental_image)
#             experimental_image_maksed[indices_0,indices_1] = 0.0
#             indices_0_with_signals, indices_1_with_signals = np.where(experimental_image_maksed > 0.0)
#             experimental_image_maksed[indices_0_with_signals, indices_1_with_signals] = experimental_image_maksed[indices_0_with_signals, indices_1_with_signals] / np.sqrt(np.sum(np.square(experimental_image_maksed[indices_0_with_signals, indices_1_with_signals])))    
#             correlation_val = np.sum(np.multiply(experimental_image_maksed, simulated_DP_image))
#     else:
    
#         py4DSTEM_kinematic_Diffraction = map_sim_BDs_to_images(
#                                                                 py4DSTEM_crystal_object, 
#                                                                 orientation_matrix = orientation, 
#                                                                 excitation_error = excitation_error, 
#                                                                 pixel_numbers = pixel_numbers,
#                                                                 )
#         py4DSTEM_kinematic_Diffraction.intensity = py4DSTEM_kinematic_Diffraction.intensity / np.sqrt(np.sum(np.square(py4DSTEM_kinematic_Diffraction.intensity)))
        
#         simulated_DP_image = py4DSTEM_kinematic_Diffraction.generate_2D_IMG_array(
#                                                                 get_broadened = False, 
#                                                                 normalize_2D_IMG_array = False, 
#                                                                 remove_direct_beam = False
#                                                                 )
        
    
#         experimental_image_normalized = experimental_image / np.linalg.norm(experimental_image)
#         # simulated_DP_image_normalized = simulated_DP_image / np.linalg.norm(simulated_DP_image)
#         correlation_val = np.sum(np.multiply(experimental_image_normalized, simulated_DP_image))

#     return correlation_val

# def ssim_and_mse_metric(
#                         experimental_image, 
#                         py4DSTEM_crystal_object, 
#                         orientation, 
#                         excitation_error, 
#                         pixel_numbers,
#                         correlation_kernel,
#                         removeDirectBeam = True,
#                         considerEntireExpImage = True,
#                         ):

#     ## NOTE: if intensity values of simulated bragg vectors are unreliable, then I think this correlation score is unreliable    
    
#     ## Well. There is another approach using dynamic simulation.
#     ## for given orientation, run dynamic scattering simulation, 
#     ## and find the thickness that leads to maximum correlation.
#     ## Still suffer from exciation errror though..

#     ## TO DO LIST:
#     ## WRITE DYNAMIC SCATTERING, and optimize it by thickness
#     ## MINIMIZE THE RELATIVE INTENSITY

    

#     if removeDirectBeam:

#         py4DSTEM_kinematic_Diffraction = map_sim_BDs_to_images(
#                                                                 py4DSTEM_crystal_object, 
#                                                                 orientation_matrix = orientation, 
#                                                                 excitation_error = excitation_error, 
#                                                                 pixel_numbers = pixel_numbers,
#                                                                 )

#         # print("py4DSTEM_kinematic_Diffraction.intensity before", py4DSTEM_kinematic_Diffraction.intensity)
#         py4DSTEM_kinematic_Diffraction.qx, py4DSTEM_kinematic_Diffraction.qy, py4DSTEM_kinematic_Diffraction.intensity = remove_direct_beam(py4DSTEM_kinematic_Diffraction.qx, py4DSTEM_kinematic_Diffraction.qy, py4DSTEM_kinematic_Diffraction.intensity)
#         # print("np.stack((py4DSTEM_kinematic_Diffraction.qx, py4DSTEM_kinematic_Diffraction.qy)).T\n",  np.stack((py4DSTEM_kinematic_Diffraction.qx, py4DSTEM_kinematic_Diffraction.qy)).T)
#         # py4DSTEM_kinematic_Diffraction.intensity = py4DSTEM_kinematic_Diffraction.intensity / np.sqrt(np.sum(np.square(py4DSTEM_kinematic_Diffraction.intensity)))
#         # print("py4DSTEM_kinematic_Diffraction.intensity before", py4DSTEM_kinematic_Diffraction.intensity)
#         simulated_DP_image = py4DSTEM_kinematic_Diffraction.generate_2D_IMG_array(
#                                                                     get_broadened = False, 
#                                                                     normalize_2D_IMG_array = False, 
#                                                                     remove_direct_beam = False,
#                                                                     )

#         convolved_sim_image = ndimage.convolve(simulated_DP_image, correlation_kernel, mode='reflect')
#         convolved_sim_image_normalized = convolved_sim_image / np.linalg.norm(convolved_sim_image)
#         hausdorff_distance, structural_similarity
#         experimental_image_normalized = experimental_image / np.linalg.norm(experimental_image)
#         # h_distance = hausdorff_distance(experimental_image_normalized, convolved_sim_image_normalized)
#         ssim = structural_similarity(experimental_image_normalized, convolved_sim_image_normalized, data_range = convolved_sim_image_normalized.max() - convolved_sim_image_normalized.min())
#         mse = mean_squared_error(experimental_image_normalized, convolved_sim_image_normalized)

#     return ssim, mse
