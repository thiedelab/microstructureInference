#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 08:20:54 2025

@author: kwang
"""
import torch.nn.functional as F
import torch
import torch.nn as nn
import h5py
from skimage import transform
import sys
import numpy as np
from emdfile import tqdmnd
from orientationMapping.dataModules import cubic_proper_point_group_operations, digitize_radial_distance, digitize_polarAngle, digitize_braggIntensity
from orientationMapping.LossFunctions import pointGroup_map_rotation_prediction, pointGroup_map_rotation_prediction_return_geodesic_distance_stack, pointGroup_map_rotation_and_phase_prediction, symmetric_orthogonalization
from scipy.spatial.distance import cdist
import scipy.sparse as sp
from scipy.optimize import linprog

def get_attention_matrix(model, dataloader, device, PAD = 0):
    
    attention_layer_0_stack = []
    attention_layer_1_stack = []
    attention_layer_2_stack = []
        
    with torch.no_grad():
        model.eval()
        for x in dataloader:
            features = x.to(device)
            pad_mask = (torch.sum(features, dim = 2) == PAD).view(features.size(0), 1, 1, features.size(1))
            
            _ = model(features, pad_mask, True)
            
            attn_layer0 = model.encoder.encoder_blocks[0].last_attn
            attn_layer1 = model.encoder.encoder_blocks[1].last_attn
            attn_layer2 = model.encoder.encoder_blocks[2].last_attn
            
            attention_layer_0_stack.append(attn_layer0.clone().detach())
            attention_layer_1_stack.append(attn_layer1.clone().detach())
            attention_layer_2_stack.append(attn_layer2.clone().detach())
            
    return attention_layer_0_stack, attention_layer_1_stack, attention_layer_2_stack

def predict_rotation_sim_data_with_labels(model, dataloader, device, PAD = 0):
    point_group_op_matrices = cubic_proper_point_group_operations()
    point_group_op_matrices = point_group_op_matrices.to(device)
    
    predicted_rotation_matrices = []
    geodesic_distances_stack = []
    losses, count = [], 0
        

    with torch.no_grad():
        model.eval()
        for x, y in dataloader:
            features = x.to(device)
            labels_r  = y.to(device)
            pad_mask = (torch.sum(features, dim = 2) == PAD).view(features.size(0), 1, 1, features.size(1))
            
            
            
            pred = model(features, pad_mask)
            
            # print("pred", pred)
            loss, geodesic_distances = pointGroup_map_rotation_prediction_return_geodesic_distance_stack(pred, labels_r, point_group_op_matrices)
            
            predicted_rotation_matrix = symmetric_orthogonalization(pred)
            predicted_rotation_matrices.append(predicted_rotation_matrix)
            geodesic_distances_stack.append(geodesic_distances)
            
            losses.append(loss.item())
            count += 1
            
    return torch.vstack(predicted_rotation_matrices), torch.hstack(geodesic_distances_stack), np.mean(losses)


def predict_rotation_and_phases_sim_data_with_labels(model, dataloader, device, PAD = 0):
    point_group_op_matrices = cubic_proper_point_group_operations()
    point_group_op_matrices = point_group_op_matrices.to(device)
    
    predicted_rotation_matrices = []
    geo_error, phase_prediction_error, count = 0, 0, 0
        

    with torch.no_grad():
        model.eval()
        for x, y, z in dataloader:
            features = x.to(device)
            labels_r  = y.to(device)
            labels_phase  = z.to(device)
            pad_mask = (torch.sum(features, dim = 2) == PAD).view(features.size(0), 1, 1, features.size(1))
            
            
            
            pred = model(features, pad_mask)
            
            # print("pred", pred)
            loss, rotation_prediction_loss, phase_prediction_loss = pointGroup_map_rotation_and_phase_prediction(pred, labels_r, labels_phase, point_group_op_matrices)
            
            predicted_rotation_matrix = symmetric_orthogonalization(pred)
            predicted_rotation_matrices.append(predicted_rotation_matrix)
            
            phase_prediction_error += phase_prediction_loss
            geo_error += rotation_prediction_loss
            count += 1
            
    return torch.vstack(predicted_rotation_matrices), geo_error/count, phase_prediction_error/count

def predict_rotation_experimental_data(model, exp_dataloader, device, PAD = 0):
    
    predicted_rotation_matrices = []

    with torch.no_grad():
        model.eval()
        for x in exp_dataloader:
            features = x.to(device)
            pad_mask = (torch.sum(features, dim = 2) == PAD).view(features.size(0), 1, 1, features.size(1))            
            
            pred = model(features, pad_mask)
                        
            predicted_rotation_matrix = symmetric_orthogonalization(pred)
            predicted_rotation_matrices.append(predicted_rotation_matrix)
            
    return torch.vstack(predicted_rotation_matrices)

def predict_rotation_and_phase_experimental_data(model, exp_dataloader, device, PAD = 0):
    
    predicted_rotation_matrices = []
    predicted_phase_labels = []

    with torch.no_grad():
        model.eval()
        for x in exp_dataloader:
            features = x.to(device)
            pad_mask = (torch.sum(features, dim = 2) == PAD).view(features.size(0), 1, 1, features.size(1))            
            
            pred = model(features, pad_mask)
                        
            predicted_rotation_matrix = symmetric_orthogonalization(pred[0])
            probabilities = torch.sigmoid(pred[1])
            predicted_phase_label = (probabilities >= 0.5).float()
            predicted_rotation_matrices.append(predicted_rotation_matrix)
            predicted_phase_labels.append(predicted_phase_label)
            
    return torch.vstack(predicted_rotation_matrices), torch.vstack(predicted_phase_labels)

def point_in_spherical_triangle_oriented(p, a, b, c, tol=1e-10):
    """
    Robustly tests if a point p lies inside a spherical triangle on the unit sphere.
    
    The spherical triangle is defined by three vertices `a`, `b`, and `c`
    , each a 3D vector on the unit sphere. The function determines whether the
    test vector `p` lies inside the spherical triangle, on its edge, or 
    or outside, considering spherical geometry and orientation.
    
    Parameters
    ----------
    p : array-like, shape (3,)
        The query point vector on the unit sphere to test for inclusion in the 
        spherical triangle.
        
    a, b, c : array-like, shape (3,)
        The three vertices of the spherical triangle, each normalized to 
        unit length.
        
    tol : float, optional (default=1e-10)
        Numerical tolerance for floating-point comparisons to handle points 
        near edges or vertices.
        
    Returns
    -------
    inside : bool
        True if `p` lies inside the spherical triangle 
        (strictly inside, on boundary, or antipodal),
        otherwise False.
    status : int
        +1  : Point lies strictly inside the original spherical triangle 
              (all dot products > tol).
        0   : Point lies exactly on an edge or vertex within tolerance.
        -1  : Point lies strictly inside the antipodal triangle.
        None : point does not lie in the triangle 
    
    Variables
    ---------
    normalize : function
        Nested helper function to normalize 3D vectors to unit length.
    n_face : ndarray, shape (3,)
        Approximate normal vector to the spherical triangle's surface, 
        computed as normalized sum of vertices.
        
    n_ab, n_bc, n_ca : ndarray, shape (3,)
        Oriented great-circle normal vectors for edges AB, BC, and CA, pointing
        inward toward triangle interior.
        
    s_ab, s_bc, s_ca : float
        Dot products between the test point `p` and each oriented edge normal, 
        indicating side relative to edges.
        
    signs : ndarray, shape (3,)
        Signs of the dot products used to determine point position relative to 
        all edges.
        
    """
    def normalize(v):
        v = np.asarray(v, dtype=float)
        return v / np.linalg.norm(v)

    a, b, c, p = map(normalize, [a, b, c, p])

    # Compute face normal (roughly pointing toward triangle center)
    n_face = normalize(a + b + c)

    # Oriented great-circle normals
    n_ab = np.cross(a, b)
    if np.dot(n_ab, n_face) < 0:
        n_ab = -n_ab

    n_bc = np.cross(b, c)
    if np.dot(n_bc, n_face) < 0:
        n_bc = -n_bc

    n_ca = np.cross(c, a)
    if np.dot(n_ca, n_face) < 0:
        n_ca = -n_ca

    # Dot tests (same hemisphere as triangle interior)
    s_ab = np.dot(p, n_ab)
    s_bc = np.dot(p, n_bc)
    s_ca = np.dot(p, n_ca)

    signs = np.sign([s_ab, s_bc, s_ca])
    if np.all(signs > tol):
        # strictly inside
        return True, +1
    elif np.any(np.abs(signs) <= tol) and np.all(signs >= -tol):
        # on edge or vertex (within tolerance)
        return True, 0  # boundary
    elif np.all(signs < -tol):
        # strictly inside antipode
        return True, -1
    else:
        return False, None


def find_zone_axis_misalignment(rotation_matrices_1_str, rotation_matrices_2_str):
    
    zone_axis_001 = np.array([0., 0., 1.])

    zone_axis_011 = np.array([0., 1., 1.])
    zone_axis_011 = zone_axis_011 / np.linalg.norm(zone_axis_011)
    
    zone_axis_111 = np.array([1., 1., 1.])
    zone_axis_111 = zone_axis_111 / np.linalg.norm(zone_axis_111)


    rotation_matrices_1 = np.load(rotation_matrices_1_str, mmap_mode='r')
    rotation_matrices_2 = np.load(rotation_matrices_2_str, mmap_mode='r')
    
    point_group_op_matrices = cubic_proper_point_group_operations()
    point_group_op_matrices_np = point_group_op_matrices.detach().cpu().numpy()

    if rotation_matrices_1.shape != rotation_matrices_2.shape:
        raise ValueError("rotation_matrices_1 and rotation_matrices_2 must have the same shape")
    if rotation_matrices_1.shape[1:] != (3, 3):
        raise ValueError("Each rotation matrix must be of shape (3,3)")

    total_zone_axis_misalignment = []

    for rotIdx, rotation_1 in enumerate(rotation_matrices_1):
        rotation_2 = rotation_matrices_2[rotIdx]

        rotation_1_symmEquiv = point_group_op_matrices_np @ rotation_1
        rotation_2_symmEquiv = point_group_op_matrices_np @ rotation_2

        zone_axis_for_rotation_1 = None
        zone_axis_for_rotation_2 = None
    
        for RotationMatrix in rotation_1_symmEquiv:
            zone_axis = RotationMatrix[:,2]
            IsCanonical_1, sign_1 = point_in_spherical_triangle_oriented(zone_axis, zone_axis_001, zone_axis_011, zone_axis_111)
            if IsCanonical_1:
                # real_label_canonical = RM
                zone_axis_for_rotation_1 = RotationMatrix[:,2]
                # print("rotation_1 rotation_matrix\n", RotationMatrix, "\n")
                # print("sign_1", sign_1)
                break
    
        # print("")
    
        for RotationMatrix in rotation_2_symmEquiv:
            zone_axis = RotationMatrix[:,2]
            IsCanonical_2, sign_2 = point_in_spherical_triangle_oriented(zone_axis, zone_axis_001, zone_axis_011, zone_axis_111)
            if IsCanonical_2:
                # real_label_canonical = RM
                zone_axis_for_rotation_2 = RotationMatrix[:,2]
                # print("rotation_2 rotation_matrix\n", RotationMatrix, "\n")
                # print("sign_2", sign_2)
                break

        # --- Error handling ---
        if zone_axis_for_rotation_1 is None:
            raise RuntimeError(f"No canonical zone axis found for rotation_1 at index {rotIdx}")
        if zone_axis_for_rotation_2 is None:
            raise RuntimeError(f"No canonical zone axis found for rotation_2 at index {rotIdx}")


        dot_product_between_two_zone_axes = np.clip(np.dot(zone_axis_for_rotation_1, zone_axis_for_rotation_2), -1.0, 1.0)
        zone_axis_misalignment = np.arccos(dot_product_between_two_zone_axes)
        # print("zone_axis_misalignment\n", zone_axis_misalignment)


        # zone_axis_misalignment = zone_axis_mis_alignment = np.arccos(np.dot(zone_axis_for_rotation_1, zone_axis_for_rotation_2))
        total_zone_axis_misalignment.append(zone_axis_misalignment)

    return np.array(total_zone_axis_misalignment)
    

def assignment_cost_pairwise_distance(input_BD_set_1, input_BD_set_2, assignment_tol = 1e-12):

    """
    Compute the weighted assignment cost between two sets of Bragg disks 
    based on pairwise Euclidean distance.

    Parameters
    ----------
    input_BD_set_1 : np.ndarray, shape (N1, 3)
        First set of Bragg disks. Each row corresponds to a Bragg disk with 
        features [x, y, intensity].
        
    input_BD_set_2 : np.ndarray, shape (N2, 3)
        Second set of Bragg disks. Each row corresponds to a Bragg disk with 
        features [x, y, intensity].
    
    N1 and N2 can be different.

    Returns
    -------
    final_assignment_cost : float
        Weighted average distance between optimally assigned Bragg disks 
        in the two sets. Weights are calculated as the mean intensity of 
        each matched pair.

    Notes
    -----
    1. The function automatically removes the direct beam 
        (point closest to (0, 0)) from each set.
    
    2. Uses linear programming to solve the optimal assignment problem that 
        minimizes total Euclidean distance between matched points.
        Also note that the distance is based on position [x,y] and does not 
        include intensity.
    
    3. The distance of each assigned pair is weighted by the average intensity 
        of the two disks.
    
    4. If the two sets have different sizes, the smaller set is matched to the 
        larger set with no point repeated.
    """


    def get_row_col(indices,  col_num):
        row_indices = []
        col_indices = []
        for index in indices:
            row_index = index // col_num
            col_index = index % col_num
            row_indices.append(row_index)
            col_indices.append(col_index)
        return np.array(row_indices), np.array(col_indices)



    # Copy Bragg disk sets and remove direct beam

    input_BD_set_1_features = np.copy(input_BD_set_1)
    input_BD_set_2_features = np.copy(input_BD_set_2)
    
    assert input_BD_set_1_features.shape[1] == input_BD_set_2_features.shape[1], "feature dimension (axis 1) of the first Bragg disk list and that of the second Bragg disk list do not match."
    assert input_BD_set_1_features.shape[1] == int(3), "feature dimension of the two pointlists should be 3."
    
    input_BD_set_1_positions_only = np.copy(input_BD_set_1_features[:,:2])
    input_BD_set_2_positions_only = np.copy(input_BD_set_2_features[:,:2])

    input_BD_set_1_intensity_only = np.copy(input_BD_set_1_features[:,2])
    input_BD_set_2_intensity_only = np.copy(input_BD_set_2_features[:,2])

    radial_distance_of_BD_set_1 = np.linalg.norm(input_BD_set_1_positions_only, axis = 1)
    radial_distance_of_BD_set_2 = np.linalg.norm(input_BD_set_2_positions_only, axis = 1)

    BD_set_1_direct_beam_index = np.argmin(radial_distance_of_BD_set_1)
    BD_set_2_direct_beam_index = np.argmin(radial_distance_of_BD_set_2)

    input_BD_set_1_positions_only = np.delete(input_BD_set_1_positions_only, BD_set_1_direct_beam_index, axis = 0)
    input_BD_set_2_positions_only = np.delete(input_BD_set_2_positions_only, BD_set_2_direct_beam_index, axis = 0)

    # normalize intensity by sum; the normalized intensity would be used as weights below.

    input_BD_set_1_intensity_only = np.delete(input_BD_set_1_intensity_only, BD_set_1_direct_beam_index, axis = 0)
    input_BD_set_2_intensity_only = np.delete(input_BD_set_2_intensity_only, BD_set_2_direct_beam_index, axis = 0)

    input_BD_set_1_intensity_only = input_BD_set_1_intensity_only / np.sum(input_BD_set_1_intensity_only)
    input_BD_set_2_intensity_only = input_BD_set_2_intensity_only / np.sum(input_BD_set_2_intensity_only)

    input_BD_set_1_number = input_BD_set_1_positions_only.shape[0]
    input_BD_set_2_number = input_BD_set_2_positions_only.shape[0]

    # In case the number of Bragg disks in set 1 and that in set 2 are not the same. 

    if input_BD_set_1_number != input_BD_set_2_number:

        if input_BD_set_1_number < input_BD_set_2_number:

            POINTS_positions_only = input_BD_set_1_positions_only
            POINTS_intensity = input_BD_set_1_intensity_only
    
            QUERYPOINTS_positions_only = input_BD_set_2_positions_only
            QUERYPOINTS_intensity = input_BD_set_2_intensity_only

        elif input_BD_set_1_number > input_BD_set_2_number:

            POINTS_positions_only = input_BD_set_2_positions_only
            POINTS_intensity = input_BD_set_2_intensity_only
    
            QUERYPOINTS_positions_only = input_BD_set_1_positions_only
            QUERYPOINTS_intensity = input_BD_set_1_intensity_only

                
        pairDist_based_on_positions = cdist(POINTS_positions_only, QUERYPOINTS_positions_only)
        
        reshaped_pairDist_based_on_positions = pairDist_based_on_positions.reshape((POINTS_positions_only.shape[0] * QUERYPOINTS_positions_only.shape[0], 1))   # reshape the cost matrix to a column vector 

        constraint_ub_matrix = sp.kron(np.ones((1, POINTS_positions_only.shape[0])), sp.eye_array(QUERYPOINTS_positions_only.shape[0])).toarray()
        constraint_ub_val = np.ones(QUERYPOINTS_positions_only.shape[0])
        
        constraint_eq_matrix = sp.kron(sp.eye_array(POINTS_positions_only.shape[0]), np.ones((1, QUERYPOINTS_positions_only.shape[0]))).toarray()
        constraint_eq_val = np.ones(POINTS_positions_only.shape[0])
        
        res = linprog(reshaped_pairDist_based_on_positions, constraint_ub_matrix, constraint_ub_val, constraint_eq_matrix, constraint_eq_val, (0, 1))
        indices_of_paired_BDs = np.where(res.x > assignment_tol)[0]

        row_idx, col_idx = get_row_col(indices_of_paired_BDs, pairDist_based_on_positions.shape[1])
        pairedDiskDist = reshaped_pairDist_based_on_positions[indices_of_paired_BDs]
        
        pairedDiskDist = pairedDiskDist.reshape(pairedDiskDist.shape[0])

        cost = pairedDiskDist * (POINTS_intensity[row_idx] + QUERYPOINTS_intensity[col_idx]) * 0.5
        final_assignment_cost = np.average(cost)

    
    # In case the number of Bragg disks in set 1 and that in set 2 are the same.
    else:

        POINTS_positions_only = input_BD_set_1_positions_only
        POINTS_intensity = input_BD_set_1_intensity_only

        QUERYPOINTS_positions_only = input_BD_set_2_positions_only
        QUERYPOINTS_intensity = input_BD_set_2_intensity_only

        pairDist_based_on_positions = cdist(POINTS_positions_only, QUERYPOINTS_positions_only)
        
        reshaped_pairDist_based_on_positions = pairDist_based_on_positions.reshape((POINTS_positions_only.shape[0] * QUERYPOINTS_positions_only.shape[0],1))   # reshape the cost matrix to a column vector 

        constraint_eq_matrix_1 = sp.kron(np.ones((1, POINTS_positions_only.shape[0])), sp.eye_array(QUERYPOINTS_positions_only.shape[0])).toarray()
        constraint_eq_val_1 = np.ones(QUERYPOINTS_positions_only.shape[0])
        
        constraint_eq_matrix_2 = sp.kron(sp.eye_array(POINTS_positions_only.shape[0]), np.ones((1, QUERYPOINTS_positions_only.shape[0]))).toarray()
        constraint_eq_val_2 = np.ones(POINTS_positions_only.shape[0])
        
        constraint_eq_matrix_stacked = np.vstack((constraint_eq_matrix_1, constraint_eq_matrix_2))
        constraint_eq_val_stacked = np.hstack((constraint_eq_val_1, constraint_eq_val_2)).T
                
        res = linprog(reshaped_pairDist_based_on_positions, None, None, constraint_eq_matrix_stacked, constraint_eq_val_stacked, (0, 1))
        
        indices_of_paired_BDs = np.where(res.x > assignment_tol)[0]
        row_idx, col_idx = get_row_col(indices_of_paired_BDs,  pairDist_based_on_positions.shape[1])
        

        pairedDiskDist = reshaped_pairDist_based_on_positions[indices_of_paired_BDs]
        
        pairedDiskDist = pairedDiskDist.reshape(pairedDiskDist.shape[0])

        cost = pairedDiskDist * (POINTS_intensity[row_idx] + QUERYPOINTS_intensity[col_idx]) * 0.5
        final_assignment_cost = np.average(cost)
        
    return final_assignment_cost


def sample_rotation_at_rand_geodesic_distance(R1, geodesic_distance, random_seed = 555):
    """
    Sample a random rotation R2 such that the geodesic distance
    d(R1, R2) = geodesic_distance on SO(3).
    
    Parameters
    ----------
    R1 : ndarray of shape (3,3)
        Proper rotation matrix in SO(3)
    geodesic_distance : float
        Desired geodesic distance (rotation angle) in radians
    
    Returns
    -------
    R2 : ndarray of shape (3,3)
        Rotation matrix at geodesic distance geodesic_distance from R1
    """
    
    np.random.seed(random_seed) 
    # Step 1: Sample a random rotation axis uniformly on the sphere
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    
    # Step 2: Construct rotation matrix about this axis by geodesic_distance
    x, y, z = axis
    c = np.cos(geodesic_distance)
    s = np.sin(geodesic_distance)
    C = 1 - c
    R_delta = np.array([
        [c + x*x*C, x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, c + y*y*C, y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, c + z*z*C]
    ])
    
    # Step 3: Multiply to get the new rotation
    R2 = R1 @ R_delta
    
    return R2


def pyxem_correlation_metric(image, py4DSTEM_bragg_vecto_list, k_max, pixel_numbers, intensity_gamma_correction = 0.5):
    
    qx = py4DSTEM_bragg_vecto_list.data['qx']
    qy = py4DSTEM_bragg_vecto_list.data['qy']
    intensity = py4DSTEM_bragg_vecto_list.data['intensity']
    
    if intensity_gamma_correction is not None:
        image = image ** intensity_gamma_correction;
    
    kinematic_diffraction_pattern_2D = generate_2D_IMG_array(
                                                qx,
                                                qy,
                                                intensity,
                                                k_max,
                                                pixel_numbers,
                                                )
    
    kinematic_diffraction_pattern_2D = kinematic_diffraction_pattern_2D / np.sqrt(np.sum(np.square(kinematic_diffraction_pattern_2D)))
    
    experimen_diffraction_pattern_2D = image / np.sqrt(np.sum(np.square(image)))
    
    correlation_score_Q = np.sum(np.multiply(kinematic_diffraction_pattern_2D,experimen_diffraction_pattern_2D))
    
    return correlation_score_Q

def generate_2D_IMG_array(qx,
                          qy,
                          intensity,
                          k_max,
                          pixel_numbers,
                          remove_direct_beam = True,
                          corner_centered = False,
                          position_tolerance = 1e-14,
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

    qx_refined = np.copy(qx)
    qy_refined = np.copy(qy)
    intensity_refined = np.copy(intensity)
    mask = None

    if remove_direct_beam:            
        mask = np.full((len(intensity)), True)
        radial_distances = np.linalg.norm(np.stack((qx,qy)).T, axis = 1)
        index_of_direct_beam = np.argmin(radial_distances)
        mask[index_of_direct_beam] = False
        qx_refined = qx_refined[mask]
        qy_refined = qy_refined[mask]
        intensity_refined = intensity_refined[mask]
        
    qx_refined[np.where(np.abs(qx_refined[:]) < (position_tolerance))[0]] = 0.0
    qy_refined[np.where(np.abs(qy_refined[:]) < (position_tolerance))[0]] = 0.0

    stack_qxqy = np.stack((qx_refined, qy_refined), axis = 1)
    width_of_diff2D_img = k_max * 2.0
    pixel_size = width_of_diff2D_img / pixel_numbers

    if pixel_numbers % 2 == 0:
        pixel_bins = np.arange(-k_max, k_max + pixel_size, pixel_size) # change
        pixel_bins[np.where(np.abs(pixel_bins[:]) < (position_tolerance))[0]] = 0.0 # change
        
        
        digitized_bin = np.digitize(stack_qxqy, pixel_bins) - 1
    else:
        pixel_bins = np.linspace(-k_max - pixel_size/2.0, 
                                 k_max + pixel_size/2.0, 
                                 pixel_numbers + 1, endpoint = True)
        digitized_bin = np.digitize(stack_qxqy, pixel_bins) - 1


    diff2D_img = np.zeros((pixel_numbers, pixel_numbers))
    diff2D_img[digitized_bin[:,0],digitized_bin[:,1]] = intensity_refined

    if corner_centered:
        diff2D_img = np.roll(diff2D_img, (-int(pixel_numbers / 2), -int(pixel_numbers / 2)), axis=(0, 1))

    return diff2D_img




def make_orientation_map_based_on_4D_rotation_matrices(
                                                        rotation_matrices_4D,
                                                        crystal,
                                                        match_ind = 0,
                                                        num_matches_return = 1,
                                                        correlation_thr = 100.,
                                                        ):
    """
    Generates an orientation map based on 4D rotation matrices.

    The function processes a 4D array of rotation matrices and constructs an 
    orientation map for a scan space. Each element in the scan space corresponds 
    to a 3x3 rotation matrix that describes the orientation of a crystal at that 
    pixel. The map is populated with orientation information, including matrix 
    values, correlation scores, and mirroring flags.

    Parameters:
    -----------
    rotation_matrices_4D : numpy.ndarray
        A 4D numpy array of shape (S_x, S_y, 3, 3) containing rotation matrices 
        for each pixel in the scan space. Each matrix is a 3x3 rotation matrix 
        associated with a specific pixel location in the scan space.
        
        - S_x: Number of pixels along the x-axis (scan space axis 0 dimension).
        - S_y: Number of pixels along the y-axis (scan space axis 1 dimension).
    
    crystal : Crystal
        A `Crystal` object that contains crystal symmetry operations and methods 
        for orientation reduction. The `crystal` object is used to apply symmetry 
        reductions to the calculated orientations, ensuring that the orientations 
        are consistent with the crystal's symmetry group.
        
        The `crystal` object must have a method `symmetry_reduce_directions(orientation, match_ind)` 
        which reduces the orientation matrix by applying the crystal's symmetries, 
        ensuring that the orientation is physically valid according to the crystal's 
        symmetry operations (e.g., rotation, reflection).
        
        NOTE! This code only works if all the rotations come from the same
        crystal systems. For different crystal systems, you have to introduce
        different py4DSTEM crystal object. This is easy to do. 
        Make another function that takes rotation matrices,
        py4DSTEM crystal 1, crystal 2, ..., crystal N,
        and 2D crystal system index array whose element is one of 0, ..., N-1.
        Then, for given crystal phase, you just have to apply crystal.symmetry_reduce_directions
        
    match_ind : int, optional, default=0
        Index indicating which match (orientation solution) to store in the 
        `Orientation` object. The default value is 0, which corresponds to 
        the first match.

    num_matches_return : int, optional, default=1
        The number of orientation matches to be returned and stored for each 
        pixel. Typically set to 1, but can be adjusted if multiple orientations 
        need to be tracked.

    Returns:
    --------
    new_orientation_map : OrientationMap
        An `OrientationMap` object containing the orientation data for the entire 
        scan space, including the 3x3 rotation matrices, correlation values, and 
        mirror flags for each pixel.
    """
    
    from py4DSTEM.process.diffraction.utils import OrientationMap, Orientation

    scan_space_dimension_x = rotation_matrices_4D.shape[0]
    scan_space_dimension_y = rotation_matrices_4D.shape[1]

    new_orientation_map = OrientationMap(
                                        num_x=scan_space_dimension_x,
                                        num_y=scan_space_dimension_y,
                                        num_matches=1,
    )

    for i in range(scan_space_dimension_x):
        for j in range(scan_space_dimension_y):
            if np.sum(np.abs(rotation_matrices_4D[i, j])) > 0.0:
                new_orientation = Orientation(num_matches=num_matches_return)
                new_orientation.matrix[match_ind] = rotation_matrices_4D[i, j]

                #######
                new_orientation.corr[match_ind] = correlation_thr # Set correlation value high such that crystal.plot_orientation_maps module can plot it.
                #######
    
                # if np.sum(new_orientation.matrix[match_ind][:,2]) < 0.0:
                #     new_orientation.mirror[match_ind] = True
                # else:
                #     new_orientation.mirror[match_ind] = False
                    
            else:
                new_orientation = Orientation(num_matches=num_matches_return)
                new_orientation.matrix[match_ind] = np.zeros((3,3))
                new_orientation.corr[match_ind] = 0.
                new_orientation.mirror[match_ind] = False
    
            new_orientation = crystal.symmetry_reduce_directions(
                        new_orientation,
                        match_ind=match_ind,
                    )
    
            # if np.sum(np.abs(new_orientation.matrix)) > 0.0:
            #     print("###############################################")
            #     print("xind, ying", i, j)
            #     print("new_orientation_map.matrix\n", new_orientation.matrix, "\n")
            #     print("new_orientation_map.family\n", new_orientation.family, "\n")
            #     print("new_orientation_map.mirror\n", new_orientation.mirror, "\n")
    
            new_orientation_map.set_orientation(new_orientation, i, j)

    return new_orientation_map

def process_pandas_tabular_data(
                                df,
                                num_bins_radialDistance,
                                num_bins_polarAngle,
                                num_bins_braggintensity,
                                max_sequence_length,
                                max_radial_distance,
                                max_braggIntensity,
                                min_braggIntensity = 0.001,
                                radial_distance_tolerance = 0.0001,
                                intensity_tolerance = 0.0001,
                                ):


    radial_bins = np.linspace(0.0, max_radial_distance + (radial_distance_tolerance), num_bins_radialDistance + 1)
    radial_bin_centers = (radial_bins[:-1] + radial_bins[1:]) / 2

    angle_bins = np.arange(-np.pi - np.pi/360., np.pi + np.pi/360., np.pi/180.)
    angle_bin_centers = (angle_bins[:-1] + angle_bins[1:]) / 2
    angle_bins[-1] = np.pi + np.pi/360 # further change the last element

    intensity_bins = np.linspace(min_braggIntensity, max_braggIntensity + (intensity_tolerance), num_bins_braggintensity + 1)
    intensity_bin_centers = (intensity_bins[:-1] + intensity_bins[1:]) / 2


        
    list_of_Bragg_disks_total = []

    # max_r = []
    # min_r = []
    for idx, diffractionPattern in df.items():
        np_diffractionPattern = np.array(diffractionPattern['input'])
        np_diffractionPattern[:, 0] = digitize_radial_distance(np_diffractionPattern[:,0], radial_bins)        
        np_diffractionPattern[:, 1] = digitize_polarAngle(np_diffractionPattern[:,1], angle_bins)
        np_diffractionPattern[:, 2] = digitize_braggIntensity(np_diffractionPattern[:,2], intensity_bins)
        np_diffractionPattern = np_diffractionPattern.astype(np.int32)   

            
        if idx == 0:
            if len(diffractionPattern['input']) < max_sequence_length:
                numbers_of_pad_tokens_to_add = max_sequence_length - len(diffractionPattern['input'])
                for recur in range(numbers_of_pad_tokens_to_add):
                    np_diffractionPattern = np.vstack((np_diffractionPattern, np.array([[0, 0, 0]])))

        # print("np_diffractionPattern after\n", np_diffractionPattern)
        
    
        list_of_Bragg_disks_total.append(torch.tensor(np_diffractionPattern))
    # max_r = np.array(max_r)
    # min_r = np.array(min_r)    
    
    radial_bins = torch.tensor(radial_bins, dtype = torch.float32)
    radial_bin_centers = torch.tensor(radial_bin_centers, dtype = torch.float32)
    
    angle_bins = torch.tensor(angle_bins, dtype = torch.float32)
    angle_bin_centers = torch.tensor(angle_bin_centers, dtype = torch.float32)
    
    intensity_bins = torch.tensor(intensity_bins, dtype = torch.float32)
    intensity_bin_centers = torch.tensor(intensity_bin_centers, dtype = torch.float32)
    
    return list_of_Bragg_disks_total, radial_bins, radial_bin_centers, angle_bins, angle_bin_centers, intensity_bins, intensity_bin_centers

def pre_process_experimental_BraggDisk(bragg_peaks, calibrated = True, remove_direct_beam = True):
    scan_x_dim, scan_y_dim = bragg_peaks.Rshape
    
    table_of_BraggDisk_qx_qy_intensity_for_eachScanIndex = {}
    dict_idx = 0
    for i in range(scan_x_dim):
        for j in range(scan_y_dim):
            # if len(bragg_peaks.cal[i,j].qx) > 2:
            number_of_Bragg_disks = 0
                
            if calibrated:

                qx = np.copy(bragg_peaks.cal[i,j].data["qx"])
                qy = np.copy(bragg_peaks.cal[i,j].data["qy"])
                intensity = np.copy(bragg_peaks.cal[i,j].data["intensity"])
                number_of_Bragg_disks = len(qx)
            
            else:
                # This is the case where we arbitrarily generate synthetic 4DSTEM
                qx = np.copy(bragg_peaks._v_uncal[i,j].data["qx"])
                qy = np.copy(bragg_peaks._v_uncal[i,j].data["qy"])
                intensity = np.copy(bragg_peaks._v_uncal[i,j].data["intensity"])
                number_of_Bragg_disks = len(qx)
            
            if remove_direct_beam:

                k_radial_distnaces_of_BPs = np.linalg.norm(np.stack((qx, qy)).T, axis = 1)
                index_of_direct_beam = np.argmin(k_radial_distnaces_of_BPs)
                qx = np.delete(qx, index_of_direct_beam)
                qy = np.delete(qy, index_of_direct_beam)
                intensity = np.delete(intensity, index_of_direct_beam)
                number_of_Bragg_disks = len(qx)
            
            if number_of_Bragg_disks > 1:
                            
                positions_of_Bragg_disks = np.stack((qx, qy)).T
                k_radial_distnaces_of_BPs = np.linalg.norm(positions_of_Bragg_disks, axis = 1)
                polar_angles = np.arctan2(positions_of_Bragg_disks[:,1], positions_of_Bragg_disks[:,0])
    
                table_of_BraggDisk_qx_qy_intensity_for_eachScanIndex[dict_idx] = {'input': np.stack((k_radial_distnaces_of_BPs, polar_angles, intensity/np.max(intensity))).T, 'scanIndices': [i,j]}
                
                dict_idx += 1
    return table_of_BraggDisk_qx_qy_intensity_for_eachScanIndex

def pre_process_synthetic_4DSTEM(bragg_peaks, ref_scan_indices, calibrated = True, remove_direct_beam = True):
    scan_x_dim, scan_y_dim = bragg_peaks.Rshape
    
    table_of_BraggDisk_qx_qy_intensity_for_eachScanIndex = {}
    dict_idx = 0
    for scan_index in ref_scan_indices:
        i = scan_index[0]
        j = scan_index[1]
        number_of_Bragg_disks = 0
            
        if calibrated:

            qx = np.copy(bragg_peaks.cal[i,j].data["qx"])
            qy = np.copy(bragg_peaks.cal[i,j].data["qy"])
            intensity = np.copy(bragg_peaks.cal[i,j].data["intensity"])
            number_of_Bragg_disks = len(qx)
        
        else:
            # This is the case where we arbitrarily generate synthetic 4DSTEM
            qx = np.copy(bragg_peaks._v_uncal[i,j].data["qx"])
            qy = np.copy(bragg_peaks._v_uncal[i,j].data["qy"])
            intensity = np.copy(bragg_peaks._v_uncal[i,j].data["intensity"])
            number_of_Bragg_disks = len(qx)
        
        if remove_direct_beam:

            k_radial_distnaces_of_BPs = np.linalg.norm(np.stack((qx, qy)).T, axis = 1)
            index_of_direct_beam = np.argmin(k_radial_distnaces_of_BPs)
            qx = np.delete(qx, index_of_direct_beam)
            qy = np.delete(qy, index_of_direct_beam)
            intensity = np.delete(intensity, index_of_direct_beam)
            number_of_Bragg_disks = len(qx)
        
        if number_of_Bragg_disks > 1:
                        
            positions_of_Bragg_disks = np.stack((qx, qy)).T
            k_radial_distnaces_of_BPs = np.linalg.norm(positions_of_Bragg_disks, axis = 1)
            polar_angles = np.arctan2(positions_of_Bragg_disks[:,1], positions_of_Bragg_disks[:,0])

            table_of_BraggDisk_qx_qy_intensity_for_eachScanIndex[dict_idx] = {'input': np.stack((k_radial_distnaces_of_BPs, polar_angles, intensity/np.max(intensity))).T, 'scanIndices': [i,j]}
            
            dict_idx += 1
        else:
            raise ValueError(f"number of Bragg disk in scan index: {i},{j} are less then 2. orientation mapping requires at least two Bragg disks in a diffraction pattern. Try to decrease intensity threshold for Bragg disk detection. Otherwise find another dataset.")
    
    
    assert dict_idx == ref_scan_indices.shape[0], "In one of Backgroun diffraction patterns, one or more Bragg disk is detected. This exlcudes direct beam. Please increase threshold for Bragg disk detection or try another probe kernel."
    return table_of_BraggDisk_qx_qy_intensity_for_eachScanIndex

def so3_correlation_map(R_field):
    """
    Compute 2D autocorrelation map for an SO(3) rotation field using cos(geodesic_distance).
    
    Parameters
    ----------
    R_field : ndarray, shape (H, W, 3, 3)
        Field of rotation matrices (each pixel is a 3x3 SO(3) rotation).
        
    Returns
    -------
    corr_map : ndarray, shape (2H-1, 2W-1)
        Correlation map, where corr_map[H-1+di, W-1+dj] is correlation at offset (di, dj).
    """
    H, W, _, _ = R_field.shape
    corr_map = np.zeros((2*H - 1, 2*W - 1), dtype=np.float64)
    
    # Compute traces for all pairwise dot products R_ij^T * R_i'j'
    R_flat = R_field.reshape(H*W, 3, 3)
    trace_mat = np.einsum('nki,mki->nm', R_flat, R_flat)
    trace_4d = trace_mat.reshape(H, W, H, W)

    # Vectorized offset correlation using strided windows
    for di in range(-H+1, H):
        for dj in range(-W+1, W):
            i1 = slice(max(0, -di), min(H, H - di))
            j1 = slice(max(0, -dj), min(W, W - dj))
            i2 = slice(max(0, di), min(H, H + di))
            j2 = slice(max(0, dj), min(W, W + dj))
            traces = trace_4d[i1, j1, i2, j2]
            corr_map[H-1+di, W-1+dj] = ((traces - 1)/2).mean()
    return corr_map


def calculate_correlation_function_btw_DPs_vectorized(
    fourD_STEM,
    real_space_grain_index,
    delta_max,
    ax0_Lbound,
    ax0_Ubound,
    ax1_Lbound,
    ax1_Ubound,
    index_of_crystal=1
):
    """
    Vectorized computation of the connected two-point correlation function between
    normalized diffraction patterns in a selected real-space region of 4D-STEM data.
    
    Parameters
    ----------
    fourD_STEM : ndarray, shape (H, W, k_H, k_W)
        2D diffraction pattern with shape (k_H, k_W) 
        at each scan point in space (H,W).
    real_space_grain_index : ndarray, shape (H, W)
        Integer labels for grains (or background) at each scan pixel.
    delta_max : int
        Maximum offset to compute the correlation function.
    ax*_bound : int
        Bounds of the real-space subregion to analyze.
    index_of_crystal : int
        Grain index for which to compute correlations.
    
    Returns
    -------
    corr_map : ndarray, shape (2*delta_max+1, 2*delta_max+1)
        Two-point correlation function as a function of offset (Δx, Δy).
    """

    # --- Step 1. Extract region of interest
    
    data_flattened = np.copy(fourD_STEM.reshape(fourD_STEM.shape[0], fourD_STEM.shape[1], fourD_STEM.shape[2] * fourD_STEM.shape[3]))
    data_flat = np.copy(data_flattened[ax0_Lbound:ax0_Ubound, ax1_Lbound:ax1_Ubound])
    grain_idx = np.copy(real_space_grain_index[ax0_Lbound:ax0_Ubound, ax1_Lbound:ax1_Ubound])
    mask = (grain_idx == index_of_crystal)

    H, W, P = data_flat.shape
    corr_map = np.zeros((2 * delta_max + 1, 2 * delta_max + 1), dtype=np.float64)

    # --- Step 2. Normalize each diffraction pattern to unit length
    norms = np.linalg.norm(data_flat, axis=-1, keepdims=True)
    data_norm = np.divide(data_flat, norms, out=np.zeros_like(data_flat), where=norms > 0)

    # --- Step 3. For each spatial offset, compute correlation in a vectorized way
    for dxi, dx in enumerate(range(-delta_max, delta_max + 1)):
        for dyi, dy in enumerate(range(-delta_max, delta_max + 1)):

            # Slices for overlapping region
            i1 = slice(max(0, -dx), min(H, H - dx))
            i2 = slice(max(0, dx), min(H, H + dx))
            j1 = slice(max(0, -dy), min(W, W - dy))
            j2 = slice(max(0, dy), min(W, W + dy))

            # Masks for valid grain pixels
            mask1 = mask[i1, j1]
            mask2 = mask[i2, j2]
            valid = mask1 & mask2
            if not np.any(valid):
                # corr_map[dxi, dyi] = np.nan
                corr_map[dxi, dyi] = 0.0
                continue

            # Select normalized diffraction patterns
            A = data_norm[i1, j1][valid]
            B = data_norm[i2, j2][valid]

            # --- Two-point correlation function
            # ⟨A·B⟩ - ⟨A⟩·⟨B⟩
            dot_mean = np.mean(np.sum(A * B, axis=1))
            meanA = np.mean(A, axis=0)
            meanB = np.mean(B, axis=0)
            corr_map[dxi, dyi] = dot_mean - np.dot(meanA, meanB)

    return corr_map

def so3_correlation_map_masked(R_field, real_space_grain_index, index_of_crystal=1):
    """
    Compute 2D autocorrelation map for an SO(3) rotation field using cos(geodesic_distance),
    but only for pairs of rotation matrices R_ij and R_i'j' where both pixels (i,j)
    and (i',j') belong to the specified crystal grain.

    Parameters
    ----------
    R_field : ndarray, shape (H, W, 3, 3)
        Field of rotation matrices (each pixel is a 3x3 SO(3) rotation).
    real_space_grain_index : ndarray, shape (H, W)
        Integer labels for grains (or background) at each scan pixel (e.g., 0 or 1).
    index_of_crystal : int, optional
        Grain index for which to compute correlations (default is 1).

    Returns
    -------
    corr_map : ndarray, shape (2H-1, 2W-1)
        Correlation map, where corr_map[H-1+di, W-1+dj] is correlation at offset (di, dj).
        The correlation is averaged only over valid (in-grain) pairs. Invalid offsets are 0.0.
    """
    H, W, _, _ = R_field.shape
    corr_map = np.zeros((2 * H - 1, 2 * W - 1), dtype=np.float64)
    
    # 1. Create a boolean mask for the target crystal grain
    mask = (real_space_grain_index == index_of_crystal)
    
    # 2. Compute traces for all pairwise dot products R_ij^T * R_i'j'
    # This step remains the same for efficiency, calculating ALL pairwise traces.
    R_flat = R_field.reshape(H * W, 3, 3)
    # R_flat[n] is R_ij, R_flat[m] is R_i'j'
    trace_mat = np.einsum('nki,mki->nm', R_flat, R_flat)
    trace_4d = trace_mat.reshape(H, W, H, W)

    # 3. Vectorized offset correlation with masking
    for di in range(-H + 1, H):
        for dj in range(-W + 1, W):
            
            # Slices for the overlapping region (spatial indices i, j)
            i1 = slice(max(0, -di), min(H, H - di)) # indices for R_ij (starting point)
            j1 = slice(max(0, -dj), min(W, W - dj))
            i2 = slice(max(0, di), min(H, H + di))  # indices for R_i'j' = R_{i+di, j+dj} (offset point)
            j2 = slice(max(0, dj), min(W, W + dj))

            # Apply mask to both sets of indices
            # mask1: True where R_ij is in the grain
            mask1 = mask[i1, j1]
            # mask2: True where R_{i+di, j+dj} is in the grain
            mask2 = mask[i2, j2]
            
            # Valid pairs: both R_ij and R_i'j' must be in the grain
            valid_pairs_mask = mask1 & mask2
            
            # Check if there are any valid pairs for this offset
            if not np.any(valid_pairs_mask):
                # If no valid pairs, the correlation remains 0.0 (from initialization)
                continue
            
            # Select only the traces corresponding to the valid in-grain pairs
            traces_overlap = trace_4d[i1, j1, i2, j2]
            valid_traces = traces_overlap[valid_pairs_mask]

            # Calculate the correlation for the valid pairs: cos(theta) = (Tr(R1^T R2) - 1) / 2
            correlations = (valid_traces - 1) / 2
            
            # Store the mean correlation for the current offset (di, dj)
            corr_map[H - 1 + di, W - 1 + dj] = correlations.mean()
            
    return corr_map


def read_4D(fname, trim_meta = True):

    '''
    Read the 4D dataset as a numpy array from .raw , . mat, .npy file.
    Input:

    fname: the file path

    Return: 

    dp       : numpy array
    dp_shape : the shape of the data

    '''

    fname_end = fname.split('.')[-1]

    if fname_end == 'raw':
        with open(fname, 'rb') as file:
            dp = np.fromfile(file, np.float32)

        columns = 128    
        rows = 130

        # print("dp.shape", dp.shape)
            
        sqpix = dp.size/columns/rows
        #Assuming square scan, i.e. same number of x and y scan points
        pix = int(sqpix**(0.5))
        
        dp = np.reshape(dp, (pix, pix, 130, 128), order = 'C')
        
        # Trim off the last two meta data rows if desired.  
        # The meta data is for EMPAD debugging, 
        # and generally doesn't need to be kept.
        if trim_meta:
            # dp = dp[:,:,0:128,:]
            # print("dp[:,:,128:,:].shape", dp[:,:,128:,:].shape, "\n")
            # print("dp[:,:,128:,:]\n", dp[:,:,128:,:], "\n")
            # print("dp[:,:,128,:]\n", dp[:,:,128,:], "\n")
            # print("dp[:,:,129,:]\n", dp[:,:,129,:], "\n")
            dp = dp[:,:,:128,:]

    ## Read 4D data from .mat file

    elif fname_end == 'mat':

        with h5py.File(fname, "r") as f:
            
            data_name = list(f.keys())[0]
            dp = np.array(list(f[data_name]))
    elif fname_end == 'npy':
        dp = np.load(fname)
    else:
        print('The Format is WRONG!! Only support .mat , .raw & .npy file !!') 


    sel = dp < 1
    dp[sel] = 1
    
    return dp

def align(cbed_data):

    '''

    Align the diffraction patterns through the Center of mass of the center beam
    '''

    x, y, kx, ky = np.shape(cbed_data)
    com_x, com_y = quickCOM(cbed_data) # need to add
    cbed_tran    = np.zeros((x, y, kx, ky))
    
    for i in range(x):
        for j in range(y):
            afine_tf = transform.AffineTransform(translation=(-kx//2+com_x[i,j], -ky//2+com_y[i,j]))
            cbed_tran[i,j,:,:] = transform.warp(cbed_data[i,j,:,:], inverse_map=afine_tf)
        sys.stdout.write('\r %d,%d' % (i, j) + ' '*10)
    com_x2, com_y2 = quickCOM(cbed_tran)
    std_com = (np.std(com_x2), np.std(com_y2))
    mean_com = (np.mean(com_x2), np.mean(com_y2))
    
    return cbed_tran, mean_com, std_com

def quickCOM(cbed_data):
    x, y, kx, ky = np.shape(cbed_data)
    center_x = kx//2 ; center_y = ky//2 
    disk = 5
    mask = spotmask(center_x,center_y, kx, disk)
    
    ap2_x, ap2_y = centroid2(cbed_data,x, y, kx, mask)
    
    return ap2_x, ap2_y

def spotmask(center_x,center_y, kx, disk):
    
    innerDisk   = disk

    mask = np.zeros((kx,kx))
    for i in range(kx):
        for j in range(kx):
            if (i - center_x) ** 2 + (j - center_y) ** 2 < innerDisk ** 2:
                mask[i][j] = 1

    return mask

def centroid2(fun, x, y, kx, mask):

    ap2_x = np.zeros((x,y)); ap2_y = np.zeros((x,y))
    rx, ry  = np.meshgrid(kx, kx)
    vx = np.arange(kx); vy = np.arange(kx)
    for i in range(x):
        for j in range(y):
            cbed = np.squeeze(fun[i,j, :, :] * mask)
            pnorm = np.sum(cbed)
            ap2_x[i,j] = np.sum(vx * np.sum(cbed, axis = 0))/pnorm
            ap2_y[i,j] = np.sum(vy * np.sum(cbed, axis = 1))/pnorm
            
    return ap2_x, ap2_y



def match_orientations_from_custom_bragg_peaks(
    custom_4D_Bragg_disk_pointList,
    crystal,
    num_matches_return: int = 1,
    min_angle_between_matches_deg=None,
    min_number_peaks: int = 3,
    inversion_symmetry: bool = True,
    multiple_corr_reset: bool = True,
    return_orientation: bool = True,
    progress_bar: bool = True,
    ):

    """
    Modified the function `py4DSTEM.process.diffraction.crystal_ACOM.match_orientations` 
    within the py4DSTEM library (v14.08).
    
    Parameters
    --------
    bragg_peaks_array: PointListArray
        PointListArray containing the Bragg peaks and intensities, with calibrations applied
    num_matches_return: int
        return these many matches as 3th dim of orient (matrix)
    min_angle_between_matches_deg: int
        Minimum angle between zone axis of multiple matches, in degrees.
        Note that I haven't thought how to handle in-plane rotations, since multiple matches are possible.
    min_number_peaks: int
        Minimum number of peaks required to perform ACOM matching
    inversion_symmetry: bool
        check for inversion symmetry in the matches
    multiple_corr_reset: bool
        keep original correlation score for multiple matches
    return_orientation: bool
        Return orientation map from function for inspection.
        The map is always stored in the Crystal object.
    progress_bar: bool
        Show or hide the progress bar

    """
    
    from py4DSTEM.process.diffraction.utils import OrientationMap

    orientation_map = OrientationMap(
                                    num_x = custom_4D_Bragg_disk_pointList.shape[0],
                                    num_y = custom_4D_Bragg_disk_pointList.shape[1],
                                    num_matches = num_matches_return,
    )

    for rx, ry in tqdmnd(
                        *custom_4D_Bragg_disk_pointList.shape,
                        desc="Matching Orientations",
                        unit=" PointList",
                        disable=not progress_bar,
    ):
            
        orientation  = crystal.match_single_pattern(
                                                    custom_4D_Bragg_disk_pointList._v_uncal[rx, ry],
                                                    num_matches_return=num_matches_return,
                                                    min_angle_between_matches_deg=min_angle_between_matches_deg,
                                                    min_number_peaks=min_number_peaks,
                                                    inversion_symmetry=inversion_symmetry,
                                                    multiple_corr_reset=multiple_corr_reset,
                                                    plot_corr=False,
                                                    verbose=False,
        )

        # visualize plot and double check that the orientation match looks reasonable.
        
        # bragg_peaks_fit = crystal.generate_diffraction_pattern(
        #     orientation_matrix = orientation.matrix[0],
        #     ind_orientation=0,
        #     sigma_excitation_error=sigma_compare)
        
        # # plot comparisons
        # py4DSTEM.process.diffraction.plot_diffraction_pattern(
        #     bragg_peaks_fit,
        #     bragg_peaks_compare=custom_4D_Bragg_disk_pointList._v_uncal[rx, ry],
        #     scale_markers=1000,
        #     scale_markers_compare=4e4,
        #     plot_range_kx_ky=range_plot,
        #     min_marker_size=1,
        #     figsize = (5,5),
        # )
        # plt.show()

        orientation_map.set_orientation(orientation, rx, ry)



    return orientation_map
    
def generate_custom_bragg_disks_pointList(
                                    # crystal,
                                    scan_space_dimension,
                                    diffraction_pattern_table_str,
                                    orientation_matrix_str,
                                    recip_x_dimension = 128,
                                    recip_y_dimension = 128,
                                    diffraction_space_pixel_size = 0.0328,
                                    np_random_seed = 64,
                                    cut_off_for_selecting_Bragg_disk_token = -100000.,
):
    np.random.seed(np_random_seed)
    from py4DSTEM import BraggVectors
    from itertools import product
    from py4DSTEM.data import QPoints
    

    bragg_disks_pointList = BraggVectors((scan_space_dimension, scan_space_dimension), (recip_x_dimension, recip_y_dimension))

    # Set calibration parameters
    bragg_disks_pointList.calibration.set_Q_pixel_size(diffraction_space_pixel_size)
    bragg_disks_pointList.calibration.set_Q_pixel_units('A^-1')
    bragg_disks_pointList.setcal()

    Bragg_disk_table_mmap = np.load(diffraction_pattern_table_str, mmap_mode = 'r')
    orientation_label_mmap = np.load(orientation_matrix_str, mmap_mode = 'r')
    
    number_of_a_Bragg_disk_list_in_entire_stack = len(orientation_label_mmap)
    number_of_a_Bragg_disk_list_to_sample = int(scan_space_dimension * scan_space_dimension)

    assert number_of_a_Bragg_disk_list_to_sample <= number_of_a_Bragg_disk_list_in_entire_stack, "number of scan space dimension should be equal to or lower than 500"

    permuted_indices = np.random.permutation(number_of_a_Bragg_disk_list_in_entire_stack)
    selected_indices = permuted_indices[:number_of_a_Bragg_disk_list_to_sample]

    orientation_matrices_4D = np.zeros((scan_space_dimension, scan_space_dimension,3,3), dtype = np.float32)

    for idx, (rx, ry) in enumerate(product(range(scan_space_dimension), repeat=2)):
        selected_index = selected_indices[idx]

        Bragg_disk_table = Bragg_disk_table_mmap[selected_index]

        orientation_matrices_4D[rx,ry] = orientation_label_mmap[selected_index]

        # print("Bragg_disk_table\n", Bragg_disk_table, "\n")

        indices_of_Bragg_disk_tokens = np.where(Bragg_disk_table[:,0] > cut_off_for_selecting_Bragg_disk_token)[0]

        # print("indices_of_Bragg_disk_tokens\n", indices_of_Bragg_disk_tokens)

        Bragg_disk_table_Bragg_disk_token_only = Bragg_disk_table[indices_of_Bragg_disk_tokens]

        # print("Bragg_disk_table_Bragg_disk_token_only\n", Bragg_disk_table_Bragg_disk_token_only, "\n")

        # Prepare the data structure for maxima with dtype (qx, qy, intensity)
        dtype = np.dtype([("x", float), ("y", float), ("intensity", float)])
        maxima = np.zeros(len(Bragg_disk_table_Bragg_disk_token_only), dtype=dtype)

        for i, bragg_vector in enumerate(Bragg_disk_table_Bragg_disk_token_only):
            maxima["x"][i] = bragg_vector[0]
            maxima["y"][i] = bragg_vector[1]
            maxima["intensity"][i] = bragg_vector[2]
        
        # Create QPoints object with already calibrated data
        maxima = QPoints(maxima)

        # No need to apply calibration transformation as data is already calibrated.
        # Directly assign the calibrated maxima data
        bragg_disks_pointList._v_uncal[rx, ry] = maxima

        

    return bragg_disks_pointList, orientation_matrices_4D

