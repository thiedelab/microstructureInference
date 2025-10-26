#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 08:20:54 2025

@author: kwang
"""
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from orientationMapping.dataModules import cubic_proper_point_group_operations
from orientationMapping.LossFunctions import pointGroup_map_rotation_prediction, pointGroup_map_rotation_and_phase_prediction, symmetric_orthogonalization

from scipy.spatial.distance import cdist
import scipy.sparse as sp
from scipy.optimize import linprog

def predict_rotation_sim_data_with_labels(model, dataloader, device, PAD = 0):
    point_group_op_matrices = cubic_proper_point_group_operations()
    point_group_op_matrices = point_group_op_matrices.to(device)
    
    predicted_rotation_matrices = []
    losses, count = [], 0
        

    with torch.no_grad():
        model.eval()
        for x, y in dataloader:
            features = x.to(device)
            labels_r  = y.to(device)
            pad_mask = (torch.sum(features, dim = 2) == PAD).view(features.size(0), 1, 1, features.size(1))
            
            
            
            pred = model(features, pad_mask)
            
            # print("pred", pred)
            loss = pointGroup_map_rotation_prediction(pred, labels_r, point_group_op_matrices)
            
            predicted_rotation_matrix = symmetric_orthogonalization(pred)
            predicted_rotation_matrices.append(predicted_rotation_matrix)
            
            losses.append(loss.item())
            count += 1
            
    return torch.vstack(predicted_rotation_matrices), np.mean(losses)


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
    
                if np.sum(new_orientation.matrix[match_ind][:,2]) < 0.0:
                    new_orientation.mirror[match_ind] = True
                else:
                    new_orientation.mirror[match_ind] = False
                    
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
    data_flattened,
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
    data_flattened : ndarray, shape (H, W, P)
        Flattened diffraction pattern at each scan point.
        P = 128*128 = number of pixels per diffraction pattern.
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