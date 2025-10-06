#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 13:40:45 2025

@author: kwang
"""
import numpy as np
from collections import Counter, defaultdict
from scipy.spatial.distance import pdist
from itertools import combinations
from scipy.spatial import KDTree

def rotation_wrt_zAxis(angle_in_rad):
    return np.array(
                    [
                        [np.cos(angle_in_rad), np.sin(angle_in_rad), 0],
                        [-np.sin(angle_in_rad), np.cos(angle_in_rad), 0],
                        [0, 0, 1],
                    ]
                    )

def make_proper_rotation_matrix(matrix):
    """Ensure the matrix is a proper orthogonal matrix with determinant +1."""
    U, _, Vt = np.linalg.svd(matrix)
    R = U @ Vt

    # Fix improper rotation (det(R) = -1)
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    return R

def rotation_wrt_xAxis(angle_in_rad):
    return np.array(
                    [
                        [1, 0, 0],
                        [0, np.cos(angle_in_rad), np.sin(angle_in_rad)],
                        [0, -np.sin(angle_in_rad), np.cos(angle_in_rad)],
                    ]
                    )


def rotMatrix_from_eulerAngles_ZXZ(
                                    angle_z1,
                                    angle_x1, 
                                    angle_z2,
                                    ):
    
    rotationMatrix =    rotation_wrt_zAxis(angle_z1) @ \
                        rotation_wrt_xAxis(angle_x1) @ \
                        rotation_wrt_zAxis(angle_z2)
        
    return rotationMatrix

def rotationMatrix_2D(angle_in_rad):
    return np.array(
                    [
                        [np.cos(angle_in_rad), np.sin(angle_in_rad)],
                        [np.sin(angle_in_rad), np.cos(angle_in_rad)],
                    ]
                    )
def calculate_rotation_matrix_for_zone_axis(zone_axis):
    elev = np.arctan2(
        np.hypot(zone_axis[0], zone_axis[1]),
        zone_axis[2],
    )
    
    azim = np.arctan2(zone_axis[0], zone_axis[1])

    new_rotation_matrix = rotation_wrt_zAxis(azim) @ rotation_wrt_xAxis(elev) @ rotation_wrt_zAxis(-azim)


    return new_rotation_matrix


def rotate_180(p, center):
    """
    Rotate point p 180 degrees around center.
    """
    p = np.asarray(p)
    c = np.asarray(center)
    return 2 * c - p


def has_2fold_rotational_symmetry(points, tolerance=1e-2):
    """
    Check for 180-degree rotational symmetry (2-fold) in a set of 2D points within tolerance.
    """
    if len(points) <= 1:
        return True

    point_array = np.array(points)
    kd_tree = KDTree(point_array)
    center = np.array([0.0, 0.0])
    # candidate_centers = get_candidate_centers(points)

    # for center in candidate_centers:
    rotated = np.array([rotate_180(p, center) for p in point_array])
    distances, _ = kd_tree.query(rotated, distance_upper_bound=tolerance)

    if np.all(distances != np.inf):
        # print(f"Approximate 2-fold rotational symmetry found around center {center} (tolerance = {tolerance})")
        return True
    else:
        return False


def normalize_direction(v):
    v = np.asarray(v)
    norm = np.linalg.norm(v)
    if norm < 1e-9:
        return None
    v = v / norm
    # Canonical form: always point to right or upward
    if v[0] < 0 or (abs(v[0]) < 1e-9 and v[1] < 0):
        v = -v
    return tuple(np.round(v, 8))

def reflect_point_over_origin_line(p, direction):
    p = np.asarray(p)
    direction = np.asarray(direction)
    direction = direction / np.linalg.norm(direction)
    projection = np.dot(p, direction) * direction
    reflected = 2 * projection - p
    return reflected

def get_candidate_mirror_directions(points):
    directions = set()
    points = list(points)

    for p1, p2 in combinations(points, 2):
        # print("")
        # print("p1", p1)
        # print("p2", p2)
        v = np.asarray(p2) - np.asarray(p1)
        if np.linalg.norm(v) < 1e-9:
            continue

        # --- Criteria 1: perpendicular to pair vector (between points)
        perp = (-v[1], v[0])
        dir1 = normalize_direction(perp)
        # print("dir1", dir1)
        if dir1:
            # print("keima")
            directions.add(dir1)

        # --- Criteria 2: line through both points also goes through origin
        # This is true if origin lies on the line defined by p1 and p2
        # Check if vector from p1 to origin is colinear with vector p1 -> p2
        origin = np.zeros(2)
        v1 = np.asarray(p1) - origin
        cross = v1[0] * v[1] - v1[1] * v[0]
        # print("v1", v1)
        # print("p2", p2)
        # print("cross", cross)
        # print("v",v)
        if abs(cross) < 5e-7:  # colinear with origin
            
            dir2 = normalize_direction(v)

            if dir2:
                directions.add(dir2)

    # Add some canonical directions to be robust
    directions.update([
        normalize_direction((1, 0)),
        normalize_direction((0, 1)),
        normalize_direction((1, 1)),
        normalize_direction((1, -1)),
    ])

    return directions

def is_MirrorSymmetric(points, tolerance=0.05):
    if len(points) <= 1:
        return True

    points = np.asarray(points)
    kd_tree = KDTree(points)
    directions = get_candidate_mirror_directions(points)

    for direction in directions:
        # print("direction", direction)
        reflected = np.array([
            reflect_point_over_origin_line(p, direction) for p in points
        ])
        
        distances, _ = kd_tree.query(reflected, distance_upper_bound=tolerance)
        # print("distances", distances)
        if np.all(distances != np.inf):
            return True

    return False


def signature(point_set, decimals=5):
    pts = np.array(point_set, dtype=np.float64)
    dists = pdist(pts, metric='euclidean')
    return np.round(np.sort(dists), decimals)

def match_signatures(sig_list1, sig_list2,  tol=1e-6):
    common = []
    unique_1 = []
    matched_2 = set()

    for i, sig1 in enumerate(sig_list1):
        found = False
        for j, sig2 in enumerate(sig_list2):
            if j in matched_2:
                continue
            if len(sig1) != len(sig2):
                continue
            diff = np.abs(sig1 - sig2)
            if np.all(diff < tol):
                common.append((i, j))  # match found
                matched_2.add(j)
                found = True
                break
        if not found:
            unique_1.append(i)

    # Items in dataset 2 that were never matched
    unique_2 = [j for j in range(len(sig_list2)) if j not in matched_2]

    return common, unique_1, unique_2

def compare_and_return_matched_and_unique_patterns(pattern_1, pattern_2, tolerance = 5e-3, decimals = 5):
    sig1 = [signature(s, decimals) for s in pattern_1]
    sig2 = [signature(s, decimals) for s in pattern_2]

    common, only_in_1, only_in_2 = match_signatures(sig1, sig2, tol = tolerance)

    return common, only_in_1, only_in_2


def get_closest_reference(dir_vec):
    
    reference_directions = {
                            (0, 0, 1): 1,  # [001]
                            (1, 1, 1): 2,  # [111]
                            (0, 1, 1): 3,  # [011]
    }
    # Normalize reference directions
    ref_dirs = {key: np.array(key) / np.linalg.norm(key) for key in reference_directions}
    
    best_priority = float('inf')
    best_angle = float('inf')
    for ref_vec, priority in reference_directions.items():
        ref = ref_dirs[ref_vec]
        dot = np.clip(np.dot(dir_vec, ref), -1.0, 1.0)
        angle = np.arccos(dot)  # in radians        
        if priority < best_priority or (priority == best_priority and angle < best_angle):
            best_priority = priority
            best_angle = angle
    return best_priority, best_angle

# Step 1: Union-Find implementation
class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        self.parent[self.find(x)] = self.find(y)


def remove_redundant_patterns_using_indices(common_indices_dictionary, 
                                            total_average_intensities,
                                            zone_axis_directions):



    # Step 2: Use Union-Find to group equivalent patterns
    uf = UnionFind()
    
    for (i, j), pairs in common_indices_dictionary.items():
        ## i and j are zone axis indices
        
        # print("")
        # print("(i,j)", i,j)
        # print("pairs\n", pairs)
        # if len(pairs) > 0:
        #     intensity_i = total_average_intensities[i]
        #     intensity_j = total_average_intensities[j]
            # print("pairs okey", pairs)
            # print("intensity_i\n", intensity_i)
            # print("intensity_j\n", intensity_j)
        for i_idx, j_idx in pairs:
            ## i_idx is thickness indices of zone axis i
            ## j_idx is thickness indices of zone axis j
            # print("i_idx", i_idx)
            # print("j_idx", j_idx)
            uf.union((i, i_idx), (j, j_idx))
    
    # Step 3: Build equivalence classes
    groups = defaultdict(list)
    for pattern in uf.parent:
        root = uf.find(pattern)
        groups[root].append(pattern)
    
    # Step 4: Decide which one to keep and which to remove
    # Let's always keep the one with the smallest (set_index, pattern_index)
    to_remove = defaultdict(set)
    to_keep = defaultdict(set)
    
    
    for group in groups.values():
        ranking = []
        # print("group\n", group)
        # print("#######################################################################")
        for set_idx, pat_idx in group:
            dir_vec = zone_axis_directions[set_idx]
            priority, angle = get_closest_reference(dir_vec)
            ranking.append((total_average_intensities[set_idx][pat_idx], set_idx, pat_idx))
            # print("ranking collecting\n", ranking, "\n")
    
        # Sort by priority, then angle
        ranking.sort(reverse=True)
        # print("ranking after sorting\n", ranking, "\n")
        
         # First one is to keep
        _, keep_set_idx, keep_pat_idx = ranking[0]
        to_keep[keep_set_idx].add(keep_pat_idx)
    
        # Others are to remove
        for _, set_idx, pat_idx in ranking[1:]:
            to_remove[set_idx].add(pat_idx)

    return to_keep, to_remove


def normalize_set_2D(point_list, decimals=5):
    def clean(val):
        # Fully cast to Python float before rounding
        val = float(val)
        val = round(val, decimals)
        return 0.0 if val == -0.0 else val

    # Return sorted tuple of clean (float, float, float)
    cleaned = [tuple(clean(v) for v in point) for point in point_list]
    return tuple(sorted(cleaned))

def normalize_set_3D(point_list, decimals=5):
    def clean(val):
        # Fully cast to Python float before rounding
        val = float(val)
        val = round(val, decimals)
        return 0.0 if val == -0.0 else val

    # Return sorted tuple of clean (float, float, float)
    cleaned = [tuple(clean(v) for v in point) for point in point_list]
    return tuple(sorted(cleaned))

def deduplicate_sets_with_index_2D(all_sets):
    unique_sets = {}
    
    for idx, point_set in enumerate(all_sets):
        norm = normalize_set_2D(point_set)
        if norm not in unique_sets:
            unique_sets[norm] = idx

    return unique_sets

def deduplicate_sets_with_index_3D(all_sets):
    unique_sets = {}
    
    for idx, point_set in enumerate(all_sets):
        norm = normalize_set_3D(point_set)
        if norm not in unique_sets:
            unique_sets[norm] = idx  # Save index of first occurrence

    return unique_sets



def action_01_collect_unique_thickness_for_each_zone_axis(
        crystal,
        unit_cell_length,
        lower_limit_unit_cell_num,
        upper_limit_unit_cell_num,
        k_max,
        zone_axes,
        excitation_error,
        intensity_threshold_for_each_Bragg_disk,
        intensity_threshold_for_disregarding_patterns_with_small_average_intensity = 1e-5,
        ):
    
    
    #### Sample thicknesses.
    thickness_lower_limit = unit_cell_length * lower_limit_unit_cell_num
    thickness_upper_limit = unit_cell_length * upper_limit_unit_cell_num
    thickness_num_for_sampling = upper_limit_unit_cell_num - lower_limit_unit_cell_num + 1
    thicknesses = np.linspace(thickness_lower_limit, thickness_upper_limit, thickness_num_for_sampling)
        
    
    total_unique_thicknesses = []
    total_average_intensities = []
    total_diffraction_patterns_uniques = []
    total_maximum_number_of_Bragg_disks = []
    
    for ZA_idx, ZA_of_interest in enumerate(zone_axes):
        print("##################################################")
        print("zone axis: ", ZA_of_interest)
        
        orientation_matrix = calculate_rotation_matrix_for_zone_axis(ZA_of_interest)
    
        beams = crystal.generate_diffraction_pattern(
                                    orientation_matrix = orientation_matrix,
                                    sigma_excitation_error = excitation_error,
                                    tol_intensity = 0.0,
                                    k_max = k_max * 6.,
        )
        
        dynamic_patterns = crystal.generate_dynamical_diffraction_pattern(
                            beams = beams,
                            orientation_matrix = orientation_matrix,
                            thickness=thicknesses,
                        )
        
        thickness_collection = []
        collections_DP = []
        collections_DP_and_intensity = []
        number_of_Bragg_disks = []
        for enIdx, pattern in enumerate(dynamic_patterns):
    
            qx = np.copy(pattern.data['qx'])
            qy = np.copy(pattern.data['qy'])
            intensity = np.copy(pattern.data['intensity'])
    
            initial_radial_distance = np.linalg.norm(np.stack((qx,qy)).T, axis = 1)    
            index_of_direct_beam = np.argmin(initial_radial_distance)
            qx = np.delete(qx, index_of_direct_beam)
            qy = np.delete(qy, index_of_direct_beam)
            intensity = np.delete(intensity, index_of_direct_beam)
            intensity_normalized = intensity / np.max(intensity)
    
    
            indices_where_intensity_below_threshold = np.where(intensity_normalized < intensity_threshold_for_each_Bragg_disk)[0]
            qx = np.delete(qx, indices_where_intensity_below_threshold)
            qy = np.delete(qy, indices_where_intensity_below_threshold)
            intensities_of_Bragg_disks = np.delete(intensity, indices_where_intensity_below_threshold)
                
            if len(qx) > 1:
                positions_of_Bragg_disks = np.stack((qx, qy)).T
                k_radial_distnaces_of_BPs = np.linalg.norm(positions_of_Bragg_disks, axis = 1)
    
                indices_where_cartesian_is_smaller_than_k_max_square = np.intersect1d(np.where(np.abs(positions_of_Bragg_disks[:,0]) < k_max)[0], np.where(np.abs(positions_of_Bragg_disks[:,1]) < k_max)[0])
                indices_where_radial_distance_smaller_than_k_max = np.where(k_radial_distnaces_of_BPs < k_max)[0]
                indices_where_radial_distance_smaller_than_k_max = np.intersect1d(indices_where_radial_distance_smaller_than_k_max, indices_where_cartesian_is_smaller_than_k_max_square)
    
                if len(indices_where_radial_distance_smaller_than_k_max) > 1:
    
                    final_qx = qx[indices_where_radial_distance_smaller_than_k_max]
                    final_qy = qy[indices_where_radial_distance_smaller_than_k_max]
                    final_intensities_of_Bragg_disks = intensities_of_Bragg_disks[indices_where_radial_distance_smaller_than_k_max]

                    BRAGG_DISKS_LIST_2D_CART_COORDS = np.stack((final_qx, final_qy)).T
                    BRAGG_DISKS_LIST_3D_CART_COORDS = np.stack((final_qx, final_qy, final_intensities_of_Bragg_disks)).T
                    
                    collections_DP.append(normalize_set_2D(BRAGG_DISKS_LIST_2D_CART_COORDS))
                    collections_DP_and_intensity.append(normalize_set_3D(BRAGG_DISKS_LIST_3D_CART_COORDS))
                    thickness_collection.append(thicknesses[enIdx])
                    
                    
        unique_sets_dict_2D = deduplicate_sets_with_index_2D(collections_DP)
        unique_sets_dict_3D = deduplicate_sets_with_index_3D(collections_DP_and_intensity)

        unique_thickness_given_pattern = []
        average_intensity_collection = []
        diffraction_patterns_uniques = []
        
        for key_2D, value_2D in unique_sets_dict_2D.items():

            average_intensity_for_given_unique_point_set = 0.0
            candidate_thickness_for_given_unique_point_set = thickness_collection[int(value_2D)]
            
            for key_3D, value_3D in unique_sets_dict_3D.items():
                key3D_xy = [(x, y) for (x, y, _) in key_3D]
                two_set_matches = key_2D == tuple(key3D_xy)

                if two_set_matches:
                    key3D_intensities = np.array([_ for (x, y, _) in key_3D])
                    average_key3D_intensities = np.mean(key3D_intensities)
                    if average_intensity_for_given_unique_point_set < average_key3D_intensities:
                        average_intensity_for_given_unique_point_set = average_key3D_intensities
                        candidate_thickness_for_given_unique_point_set = thickness_collection[int(value_3D)]
            
            
            if abs(average_intensity_for_given_unique_point_set) > intensity_threshold_for_disregarding_patterns_with_small_average_intensity:
                unique_thickness_given_pattern.append(candidate_thickness_for_given_unique_point_set)
                average_intensity_collection.append(average_intensity_for_given_unique_point_set)
                diffraction_patterns_uniques.append(np.array(key_2D))
                number_of_Bragg_disks.append(int(len(np.array(key_2D))))
            # assert average_intensity_for_given_unique_point_set > 0.0, "Average intensities of Bragg Disks is assinged as zero. Check point set comparison protocol"
        
        # print("np.where(np.array(average_intensity_collection) <1e-6)[0]\n",np.where(np.array(average_intensity_collection) <1e-6)[0])
        # print("np.min(np.array(average_intensity_collection))", np.min(np.array(average_intensity_collection)))            
        print("number of sampled thickness for given zone axis:", len(unique_thickness_given_pattern))
        print("maximum number of Bragg disks within a circle with radius ", k_max, " and above threshold", intensity_threshold_for_each_Bragg_disk, ": ", np.max(np.array(number_of_Bragg_disks)))
        print("minimum number of Bragg disks within a circle with radius ", k_max, " and above threshold", intensity_threshold_for_each_Bragg_disk, ": ", np.min(np.array(number_of_Bragg_disks)))
        
        total_maximum_number_of_Bragg_disks.append(np.max(np.array(number_of_Bragg_disks)))
        total_unique_thicknesses.append(unique_thickness_given_pattern)
        total_average_intensities.append(average_intensity_collection)
        total_diffraction_patterns_uniques.append(diffraction_patterns_uniques)
    
    print("")
    total_maximum_number_of_Bragg_disks = np.array(total_maximum_number_of_Bragg_disks)
    print("For ", len(zone_axes), " zone axes, maximum number of Bragg disks in a pattern within a circle with radius", k_max, " and above threshold", intensity_threshold_for_each_Bragg_disk, " is", np.max(total_maximum_number_of_Bragg_disks))
    print("")
        
        
            
    return total_unique_thicknesses, total_diffraction_patterns_uniques, total_average_intensities


def action_02_compare_DPs_from_different_zone_axes(
        total_diffraction_patterns_uniques,
        # total_average_intensities,
        zone_axes,
        tolerance_for_pattern_matching = 1e-2,
        decimals_for_setting_tuple = 5,
        ):        
        
    unique_indices_candiate_for_each_ZA = {}
    common_indices_for_each_ZA_pairs = {}
    
    for ZA_index_i in range(zone_axes.shape[0]):
        if ZA_index_i not in unique_indices_candiate_for_each_ZA:
            unique_indices_candiate_for_each_ZA[ZA_index_i] = []
        for ZA_index_j in range(zone_axes.shape[0]):
            if ZA_index_i < ZA_index_j:
                if ZA_index_j not in unique_indices_candiate_for_each_ZA:
                    unique_indices_candiate_for_each_ZA[ZA_index_j] = []
                
                patterns_i = total_diffraction_patterns_uniques[ZA_index_i]
                # intensities_of_patterns_i = total_average_intensities[ZA_index_i]
                patterns_j = total_diffraction_patterns_uniques[ZA_index_j]
                # intensities_of_patterns_j = total_average_intensities[ZA_index_j]
    
                common_pattern_index, thickness_indices_only_in_ZA_index_i, thickness_indices_only_in_ZA_index_j = compare_and_return_matched_and_unique_patterns(
                    patterns_i,
                    # intensities_of_patterns_i,                                                                                                                                                                  
                    patterns_j,
                    # intensities_of_patterns_j,
                    tolerance_for_pattern_matching,decimals_for_setting_tuple)
                
                unique_indices_candiate_for_each_ZA[ZA_index_i].extend(thickness_indices_only_in_ZA_index_i)
                unique_indices_candiate_for_each_ZA[ZA_index_j].extend(thickness_indices_only_in_ZA_index_j)
                common_indices_for_each_ZA_pairs[(ZA_index_i, ZA_index_j)] = common_pattern_index
    
    return common_indices_for_each_ZA_pairs, unique_indices_candiate_for_each_ZA


def action_03_remove_thickness_indices_of_common_patterns_for_lowSymmetry_zone_axes(
        zone_axes,
        total_average_intensities,
        common_indices_for_each_ZA_pairs,
        ):
    
    indices_selectively_assigned_to_each_ZA, indices_selectively_removed_from_each_ZA = remove_redundant_patterns_using_indices(common_indices_for_each_ZA_pairs, total_average_intensities, zone_axes)
        
    
    return indices_selectively_assigned_to_each_ZA, indices_selectively_removed_from_each_ZA

def action_04_finalize_unique_thickness_resulting_unique_pattern_for_each_ZA(
        number_of_zone_axes,
        unique_indices_candiate_for_each_ZA,
        indices_selectively_assigned_to_each_ZA,
        ):
    
    ZA_idx_unique_thickness_indices_final_save = {}
    
    for ZA_idx in range(len(unique_indices_candiate_for_each_ZA)):
        ZA_idx_unique_thickness_indices_final_save[ZA_idx] = []
        emergence_counts = Counter(unique_indices_candiate_for_each_ZA[ZA_idx])
        # print("########################################################################")
        # print("emergence_counts\n", emergence_counts, "\n")
        for thickness_key, count_of_emergence in emergence_counts.items():
            if count_of_emergence == int(number_of_zone_axes - 1):
                ZA_idx_unique_thickness_indices_final_save[ZA_idx].append(thickness_key)
        # print("ZA_idx_unique_thickness_indices_final_save[ZA_idx]\n",ZA_idx_unique_thickness_indices_final_save[ZA_idx], "\n\n")
    
    # print("ZA_idx_unique_thickness_indices_final_save\n", ZA_idx_unique_thickness_indices_final_save, "\n\n")
    
    for set_idx, idxs in indices_selectively_assigned_to_each_ZA.items():
        # print(f"zone axis {set_idx}: {sorted(idxs)}")
        ZA_idx_unique_thickness_indices_final_save[int(set_idx)].extend(idxs)
        ZA_idx_unique_thickness_indices_final_save[int(set_idx)] = sorted(ZA_idx_unique_thickness_indices_final_save[int(set_idx)])
    # print("")
    
    # print("ZA_idx_unique_thickness_indices_final_save\n", ZA_idx_unique_thickness_indices_final_save, "\n\n")
    
    for ZA_thickness_index_key, ZA_thickness_index_val in ZA_idx_unique_thickness_indices_final_save.items():
        # print("zone axis", ZA_thickness_index_key)
        # print("len(ZA_thickness_index_val)",len(ZA_thickness_index_val), "\n")
        count_number_of_thickness_indices = Counter(ZA_thickness_index_val)
        number_of_appearance = np.array(list(count_number_of_thickness_indices.values()))
        if len(np.where(number_of_appearance >1)[0]) > 0:
            raise ValueError("Thickness index of given zone axis appears more than once.")
    
    return ZA_idx_unique_thickness_indices_final_save
    

def action_05_sample_representative_thickness_for_each_ZA(
        total_number_for_sampling_thickness,
        ZA_idx_unique_thickness_indices_final_save,
        total_unique_thicknesses,
        total_diffraction_patterns_uniques,
        ):
    
    # category 1: sample thickness leading to diffractin patterns of small number of Bragg disks
    
    total_number_for_sampling_thickness_resulting_small_number_of_Bragg_disks = int(total_number_for_sampling_thickness / 2)

    # category 2: sample thickness leading to diffractin patterns of high number of Bragg disks
    total_number_for_sampling_thickness_resulting_large_number_of_Bragg_disks = total_number_for_sampling_thickness - total_number_for_sampling_thickness_resulting_small_number_of_Bragg_disks
    
    final_sampled_thickness_for_each_zone_axis_summary = {}

    for ZA_index, ZA_thickness_index_val in ZA_idx_unique_thickness_indices_final_save.items():
        thickness_indices_of_interest_for_current_zone_axis = np.array(ZA_thickness_index_val)
        unique_thickness_for_current_zone_axis = np.array(total_unique_thicknesses[ZA_index])[thickness_indices_of_interest_for_current_zone_axis]
    
        number_of_unique_thickness_for_current_zone_axis = unique_thickness_for_current_zone_axis.shape[0]
    
        final_thicknesses_for_current_zone_axis = []
    
        if number_of_unique_thickness_for_current_zone_axis < total_number_for_sampling_thickness:
    
            number_of_thickness_stack = int(total_number_for_sampling_thickness // number_of_unique_thickness_for_current_zone_axis) + 1
            for num_stack in range(number_of_thickness_stack):
                final_thicknesses_for_current_zone_axis.append(unique_thickness_for_current_zone_axis)
            final_thicknesses_for_current_zone_axis = np.hstack(final_thicknesses_for_current_zone_axis)
            final_thicknesses_for_current_zone_axis = final_thicknesses_for_current_zone_axis[:total_number_for_sampling_thickness]
        else:
    
            diffraction_patterns_for_current_zone_axis = total_diffraction_patterns_uniques[ZA_index]
    
            bragg_disk_numbers_and_corresponding_thickness_indices_tuple = []
    
            for i in thickness_indices_of_interest_for_current_zone_axis:
                number_of_Bragg_disks = diffraction_patterns_for_current_zone_axis[i].shape[0]
                bragg_disk_numbers_and_corresponding_thickness_indices_tuple.append((number_of_Bragg_disks, int(i)))
                # print(diffraction_patterns_for_current_zone_axis[i].shape[0])
    
            # print("bragg_disk_numbers_and_corresponding_thickness_indices_tuple before\n", bragg_disk_numbers_and_corresponding_thickness_indices_tuple, "\n")
            bragg_disk_numbers_and_corresponding_thickness_indices_tuple.sort()
            # print("bragg_disk_numbers_and_corresponding_thickness_indices_tuple after\n", bragg_disk_numbers_and_corresponding_thickness_indices_tuple, "\n")
    
            for j in range(total_number_for_sampling_thickness_resulting_small_number_of_Bragg_disks):
                disk_num, thickness_index = bragg_disk_numbers_and_corresponding_thickness_indices_tuple[j]
                final_thicknesses_for_current_zone_axis.append(total_unique_thicknesses[ZA_index][thickness_index])
                # print("disk_num, thickness_index", disk_num, thickness_index)
    
            for j in range(thickness_indices_of_interest_for_current_zone_axis.shape[0]-total_number_for_sampling_thickness_resulting_large_number_of_Bragg_disks,thickness_indices_of_interest_for_current_zone_axis.shape[0]):
                disk_num, thickness_index = bragg_disk_numbers_and_corresponding_thickness_indices_tuple[j]
                final_thicknesses_for_current_zone_axis.append(total_unique_thicknesses[ZA_index][thickness_index])
                # print("disk_num, thickness_index", disk_num, thickness_index)
            final_thicknesses_for_current_zone_axis = np.array(final_thicknesses_for_current_zone_axis)
    
        # print("final_thicknesses_for_current_zone_axis", final_thicknesses_for_current_zone_axis)
    
        final_sampled_thickness_for_each_zone_axis_summary[ZA_index] = final_thicknesses_for_current_zone_axis
    
        # print("final_thicknesses_for_current_zone_axis", final_thicknesses_for_current_zone_axis)
        # print("len(final_thicknesses_for_current_zone_axis)", len(final_thicknesses_for_current_zone_axis))
        
        assert len(final_thicknesses_for_current_zone_axis) == total_number_for_sampling_thickness, "ERROR in sampling representative  thicknesses"
    
    return final_sampled_thickness_for_each_zone_axis_summary

def action_06_check_symmetry_for_each_ZA_each_thickness(
        crystal,
        zone_axes,
        final_sampled_thickness_for_each_zone_axis_summary,
        k_max,
        excitation_error,
        intensity_threshold_for_each_Bragg_disk,
        intensity_threshold_for_disregarding_patterns_with_small_average_intensity = 1e-5,
        ):
    
    zone_axis_001 = np.array([0., 0., 1.0])
    zone_axis_011 = np.array([0., 1.0, 1.0])
    zone_axis_011 = zone_axis_011 / np.linalg.norm(zone_axis_011)
    zone_axis_111 = np.array([1.0, 1.0, 1.0])
    zone_axis_111 = zone_axis_111 / np.linalg.norm(zone_axis_111)
    zone_axis_check_dot_product_threshold = np.cos(0.1 * np.pi / 180.)
    
    per_zone_axis_per_thickness_checkMirrorSymmetry_and_upperBound_for_inPlangeRotation = {}
    
    for zone_axis_idx, zone_axis in enumerate(zone_axes):
        per_zone_axis_per_thickness_checkMirrorSymmetry_and_upperBound_for_inPlangeRotation[zone_axis_idx] = {}
        thicknesses_for_current_zone_axis = final_sampled_thickness_for_each_zone_axis_summary[zone_axis_idx]
            
        # print("thicknesses_for_current_zone_axis\n", thicknesses_for_current_zone_axis)
    
        orientation_matrix = calculate_rotation_matrix_for_zone_axis(zone_axis)
    
        beams = crystal.generate_diffraction_pattern(
                                    orientation_matrix = orientation_matrix,
                                    sigma_excitation_error = excitation_error,
                                    tol_intensity = 0.0,
                                    k_max = k_max * 6.,
        )
        
        dynamic_patterns = crystal.generate_dynamical_diffraction_pattern(
                            beams = beams,
                            orientation_matrix = orientation_matrix,
                            thickness=thicknesses_for_current_zone_axis,
                        )
        
        thickness_collection = []
        isMirror_collection = []
        UpperBound_for_angle_collection = []
    
        for enIdx, pattern in enumerate(dynamic_patterns):
    
            qx = np.copy(pattern.data['qx'])
            qy = np.copy(pattern.data['qy'])
            intensity = np.copy(pattern.data['intensity'])
    
            initial_radial_distance = np.linalg.norm(np.stack((qx,qy)).T, axis = 1)    
            index_of_direct_beam = np.argmin(initial_radial_distance)
            qx = np.delete(qx, index_of_direct_beam)
            qy = np.delete(qy, index_of_direct_beam)
            intensity = np.delete(intensity, index_of_direct_beam)
            intensity_normalized = intensity / np.max(intensity)
    
    
            indices_where_intensity_below_threshold = np.where(intensity_normalized < intensity_threshold_for_each_Bragg_disk)[0]
            qx = np.delete(qx, indices_where_intensity_below_threshold)
            qy = np.delete(qy, indices_where_intensity_below_threshold)
            # intensities_of_Bragg_disks = np.delete(intensity, indices_where_intensity_below_threshold)
                
            if len(qx) > 1:
                positions_of_Bragg_disks = np.stack((qx, qy)).T
                k_radial_distnaces_of_BPs = np.linalg.norm(positions_of_Bragg_disks, axis = 1)
    
                indices_where_cartesian_is_smaller_than_k_max_square = np.intersect1d(np.where(np.abs(positions_of_Bragg_disks[:,0]) < k_max)[0], np.where(np.abs(positions_of_Bragg_disks[:,1]) < k_max)[0])
                indices_where_radial_distance_smaller_than_k_max = np.where(k_radial_distnaces_of_BPs < k_max)[0]
                indices_where_radial_distance_smaller_than_k_max = np.intersect1d(indices_where_radial_distance_smaller_than_k_max, indices_where_cartesian_is_smaller_than_k_max_square)
    
                if len(indices_where_radial_distance_smaller_than_k_max) > 1:
    
                    
    
                    
    
                    final_qx = qx[indices_where_radial_distance_smaller_than_k_max]
                    final_qy = qy[indices_where_radial_distance_smaller_than_k_max]
                    # final_intensities_of_Bragg_disks = intensities_of_Bragg_disks[indices_where_radial_distance_smaller_than_k_max]
    
                    BRAGG_DISKS_LIST_2D_CART_COORDS = np.stack((final_qx, final_qy)).T
    
                    
    
                    ## For zone axis 001
                    if np.dot(zone_axis_001, zone_axis) > zone_axis_check_dot_product_threshold:
                        hasMirrorSymmetry = 1
                        hasRotationSymmetry = 1
                        upper_bound_for_in_plane_rotation_angle = int(90)
    
                    ## For zone axis 111
                    elif np.dot(zone_axis_111, zone_axis) > zone_axis_check_dot_product_threshold:
                        hasMirrorSymmetry = 1
    
                        hasRotationSymmetry = has_2fold_rotational_symmetry(BRAGG_DISKS_LIST_2D_CART_COORDS)
    
                        if hasRotationSymmetry:
                            # print("6 fold")
                            upper_bound_for_in_plane_rotation_angle = int(60)
    
                        else:
                            # print("3 fold")
                            upper_bound_for_in_plane_rotation_angle = int(120)
    
                    ## For zone axis 011
                    elif np.dot(zone_axis_011, zone_axis) > zone_axis_check_dot_product_threshold:
                        hasMirrorSymmetry = 1
                        hasRotationSymmetry = 1
                        upper_bound_for_in_plane_rotation_angle = int(180)
    
                    else:
                        hasMirrorSymmetry = is_MirrorSymmetric(BRAGG_DISKS_LIST_2D_CART_COORDS)
                        hasRotationSymmetry = has_2fold_rotational_symmetry(BRAGG_DISKS_LIST_2D_CART_COORDS)
                        
                        hasMirrorSymmetry = int(hasMirrorSymmetry)
    
                        if hasRotationSymmetry:
                            upper_bound_for_in_plane_rotation_angle = int(180)
                        else:
                            upper_bound_for_in_plane_rotation_angle = int(360)
    
                    for rep_count in range(int(360 / upper_bound_for_in_plane_rotation_angle)):
                        thickness_collection.append(thicknesses_for_current_zone_axis[enIdx])
                        isMirror_collection.append(hasMirrorSymmetry)
                        UpperBound_for_angle_collection.append(upper_bound_for_in_plane_rotation_angle)
    

        per_zone_axis_per_thickness_checkMirrorSymmetry_and_upperBound_for_inPlangeRotation[zone_axis_idx]['thickness'] = np.array(thickness_collection)
        per_zone_axis_per_thickness_checkMirrorSymmetry_and_upperBound_for_inPlangeRotation[zone_axis_idx]['isMirror'] = isMirror_collection
        per_zone_axis_per_thickness_checkMirrorSymmetry_and_upperBound_for_inPlangeRotation[zone_axis_idx]['InPlaneAngleUpperBound'] = UpperBound_for_angle_collection
    
    return per_zone_axis_per_thickness_checkMirrorSymmetry_and_upperBound_for_inPlangeRotation
    

def action_06_check_symmetry_for_each_ZA_each_thickness_monoclinic(
        crystal,
        zone_axes,
        final_sampled_thickness_for_each_zone_axis_summary,
        k_max,
        excitation_error,
        intensity_threshold_for_each_Bragg_disk,
        intensity_threshold_for_disregarding_patterns_with_small_average_intensity = 1e-5,
        ):
    
    
    per_zone_axis_per_thickness_checkMirrorSymmetry_and_upperBound_for_inPlangeRotation = {}
    
    for zone_axis_idx, zone_axis in enumerate(zone_axes):
        per_zone_axis_per_thickness_checkMirrorSymmetry_and_upperBound_for_inPlangeRotation[zone_axis_idx] = {}
        thicknesses_for_current_zone_axis = final_sampled_thickness_for_each_zone_axis_summary[zone_axis_idx]
            
        # print("thicknesses_for_current_zone_axis\n", thicknesses_for_current_zone_axis)
    
        orientation_matrix = calculate_rotation_matrix_for_zone_axis(zone_axis)
    
        beams = crystal.generate_diffraction_pattern(
                                    orientation_matrix = orientation_matrix,
                                    sigma_excitation_error = excitation_error,
                                    tol_intensity = 0.0,
                                    k_max = k_max * 6.,
        )
        
        dynamic_patterns = crystal.generate_dynamical_diffraction_pattern(
                            beams = beams,
                            orientation_matrix = orientation_matrix,
                            thickness=thicknesses_for_current_zone_axis,
                        )
        
        thickness_collection = []
        isMirror_collection = []
        UpperBound_for_angle_collection = []
        
    
        for enIdx, pattern in enumerate(dynamic_patterns):
    
            qx = np.copy(pattern.data['qx'])
            qy = np.copy(pattern.data['qy'])
            intensity = np.copy(pattern.data['intensity'])
    
            initial_radial_distance = np.linalg.norm(np.stack((qx,qy)).T, axis = 1)    
            index_of_direct_beam = np.argmin(initial_radial_distance)
            qx = np.delete(qx, index_of_direct_beam)
            qy = np.delete(qy, index_of_direct_beam)
            intensity = np.delete(intensity, index_of_direct_beam)
            intensity_normalized = intensity / np.max(intensity)
    
    
            indices_where_intensity_below_threshold = np.where(intensity_normalized < intensity_threshold_for_each_Bragg_disk)[0]
            qx = np.delete(qx, indices_where_intensity_below_threshold)
            qy = np.delete(qy, indices_where_intensity_below_threshold)
            # intensities_of_Bragg_disks = np.delete(intensity, indices_where_intensity_below_threshold)
                
            if len(qx) > 1:
                positions_of_Bragg_disks = np.stack((qx, qy)).T
                k_radial_distnaces_of_BPs = np.linalg.norm(positions_of_Bragg_disks, axis = 1)
    
                indices_where_cartesian_is_smaller_than_k_max_square = np.intersect1d(np.where(np.abs(positions_of_Bragg_disks[:,0]) < k_max)[0], np.where(np.abs(positions_of_Bragg_disks[:,1]) < k_max)[0])
                indices_where_radial_distance_smaller_than_k_max = np.where(k_radial_distnaces_of_BPs < k_max)[0]
                indices_where_radial_distance_smaller_than_k_max = np.intersect1d(indices_where_radial_distance_smaller_than_k_max, indices_where_cartesian_is_smaller_than_k_max_square)
    
                if len(indices_where_radial_distance_smaller_than_k_max) > 1:
    
                    
    
                    
    
                    final_qx = qx[indices_where_radial_distance_smaller_than_k_max]
                    final_qy = qy[indices_where_radial_distance_smaller_than_k_max]
                    # final_intensities_of_Bragg_disks = intensities_of_Bragg_disks[indices_where_radial_distance_smaller_than_k_max]
    
                    BRAGG_DISKS_LIST_2D_CART_COORDS = np.stack((final_qx, final_qy)).T
    
                    
                    hasMirrorSymmetry = is_MirrorSymmetric(BRAGG_DISKS_LIST_2D_CART_COORDS)
                    hasRotationSymmetry = has_2fold_rotational_symmetry(BRAGG_DISKS_LIST_2D_CART_COORDS)
                    
                    hasMirrorSymmetry = int(hasMirrorSymmetry)

                    if hasRotationSymmetry:
                        upper_bound_for_in_plane_rotation_angle = int(180)
                    else:
                        upper_bound_for_in_plane_rotation_angle = int(360)
    
                    for rep_count in range(int(360 / upper_bound_for_in_plane_rotation_angle)):
                        thickness_collection.append(thicknesses_for_current_zone_axis[enIdx])
                        isMirror_collection.append(hasMirrorSymmetry)
                        UpperBound_for_angle_collection.append(upper_bound_for_in_plane_rotation_angle)
    

        per_zone_axis_per_thickness_checkMirrorSymmetry_and_upperBound_for_inPlangeRotation[zone_axis_idx]['thickness'] = np.array(thickness_collection)
        per_zone_axis_per_thickness_checkMirrorSymmetry_and_upperBound_for_inPlangeRotation[zone_axis_idx]['isMirror'] = isMirror_collection
        per_zone_axis_per_thickness_checkMirrorSymmetry_and_upperBound_for_inPlangeRotation[zone_axis_idx]['InPlaneAngleUpperBound'] = UpperBound_for_angle_collection
    
    return per_zone_axis_per_thickness_checkMirrorSymmetry_and_upperBound_for_inPlangeRotation


def action_07_generate_dictionary_of_orientation_thickness_and_mirrorSymmOper(
        zone_axes,
        per_zone_axis_per_thickness_checkMirrorSymmetry_and_upperBound_for_inPlangeRotation,
        ):
    
    deploy_orientations_thickness_and_isMirrorSymmetry_for_data_generation = {}

    for zone_axis_idx, zone_axis in enumerate(zone_axes):
        deploy_orientations_thickness_and_isMirrorSymmetry_for_data_generation[zone_axis_idx] = {}
        
        per_thickness_checkMirrorSymmetry_and_upperBound_for_inPlangeRotation = per_zone_axis_per_thickness_checkMirrorSymmetry_and_upperBound_for_inPlangeRotation[zone_axis_idx]
    
        sampled_thickness = per_thickness_checkMirrorSymmetry_and_upperBound_for_inPlangeRotation['thickness']
        sampled_isMirrored = per_thickness_checkMirrorSymmetry_and_upperBound_for_inPlangeRotation['isMirror']
        sampled_inPlaneAngleUpperBound = per_thickness_checkMirrorSymmetry_and_upperBound_for_inPlangeRotation['InPlaneAngleUpperBound']
    
        assert len(sampled_thickness) == len(sampled_isMirrored), "ERROR in action 06; # of sampled thickness and # of sampled symmetryIndicator is not the same"
        assert len(sampled_thickness) == len(sampled_inPlaneAngleUpperBound), "ERROR in action 06; # of sampled thickness and # of sampled in-plane-angle upper bound is not the same"
    
        for global_in_plane_angle_in_degree in np.arange(0, 360, 1, dtype=np.int64):
            # print("global_in_plane_angle_in_degree", global_in_plane_angle_in_degree)
    
            deploy_orientations_thickness_and_isMirrorSymmetry_for_data_generation[zone_axis_idx][int(global_in_plane_angle_in_degree)] = {}
    
            thicknesses = []
            isMirrored = []
            
    
            for i in range(len(sampled_inPlaneAngleUpperBound)):
                if global_in_plane_angle_in_degree < sampled_inPlaneAngleUpperBound[i]:
                    # print("sampled_inPlaneAngleUpperBound[i]", sampled_inPlaneAngleUpperBound[i])
                    thicknesses.append(sampled_thickness[i])
                    isMirrored.append(sampled_isMirrored[i])
    
            deploy_orientations_thickness_and_isMirrorSymmetry_for_data_generation[zone_axis_idx][global_in_plane_angle_in_degree]['thickness'] = np.array(thicknesses)
            deploy_orientations_thickness_and_isMirrorSymmetry_for_data_generation[zone_axis_idx][global_in_plane_angle_in_degree]['isMirrored'] = isMirrored
    
    return deploy_orientations_thickness_and_isMirrorSymmetry_for_data_generation
    
