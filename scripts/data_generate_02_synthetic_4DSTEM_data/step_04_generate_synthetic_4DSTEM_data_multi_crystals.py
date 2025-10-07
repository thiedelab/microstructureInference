#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 13:40:45 2025

@author: kwang
"""
import numpy as np
from modules_generate_synthethic_4DSTEM import sample_diffraction_patterns_from_rotation_matrices, generate_synthetic_diffraction_pattern, process_background_diffraction_pattern
import helper_function as hf
import py4DSTEM

num_grains = 60
num_sampled_orientations = 60

assert num_grains <= num_sampled_orientations, "number of grains must be equal or smaller than number of sampled orientations"


syn_2D_scanSpace_map_side_dimension = 128

print("number of grains", num_grains)
print("scan space dimension (" + str(syn_2D_scanSpace_map_side_dimension) + "," + str(syn_2D_scanSpace_map_side_dimension) + ")")

BraggDisks_list_file_name_str = "m2_bragg_disks_corThForK80000_dog_sig1_2.00_sig2_8.00_cortThForTemp_5000"
rotMat_file_path = "./"

Cu_cif_path = "./"
k_max = 0.0328 * 64
accelerating_voltage = int(300e3)

list_of_crystal_names = ['Cu_fcc', 'Cu2O_cubic', 'CuO_monoclinic']
print("")
print("list_of_crystal_names", list_of_crystal_names, "\n")

saved_file_path = "./"
out_file_path = "./"
##############################################################################
##############################################################################
### READ backgrounds, correlation kernel, scan map, and rotations (START) ####
##############################################################################
##############################################################################

print("")
print("Action 1. READING backgrounds, correlation direct beam kernel, synthetic scan map, and rotation matrices  (START)")

backgrounds = np.load(saved_file_path + BraggDisks_list_file_name_str + "_backgrounds.npy")
correlation_kernel = np.load(saved_file_path + BraggDisks_list_file_name_str + "_direct_beam_kernel.npy")
syn_2D_scanSpace_map = np.load(saved_file_path + "syn_2D_scanSpace_map_side_dim%d_numGrains%d.npy"%(syn_2D_scanSpace_map_side_dimension, num_grains))
sampled_rotation_matrices = np.load(rotMat_file_path + "./randomly_sampled_%d_orientation_matrices_SO3.npy"%(num_sampled_orientations))

print("")
print("Action 1. READING backgrounds, correlation direct beam kernel, synthetic scan map, and rotation matrices  (END)")


##############################################################################
##############################################################################
#### READ backgrounds, correlation kernel, scan map, and rotations (END) #####
##############################################################################
##############################################################################


##############################################################################
##############################################################################
#### Perform electron diffraction simulation from sampled matrices (START) ###
##############################################################################
##############################################################################

print("")
print("Action 2. simulating electron diffraction (START)\n")

number_of_grains_and_rotations_for_each_crystal = int(num_grains / len(list_of_crystal_names))

print("number_of_grains_and_rotations_for_each_crystal\n", number_of_grains_and_rotations_for_each_crystal, "\n")

DP_images_for_each_crystal = []
rotation_labels_for_each_crystal = []
grain_indices_for_each_crystal = []

grain_indices_permuted = np.random.permutation(num_grains).astype(np.int64) + 1

for crystal_idx, crystal_name in enumerate(list_of_crystal_names):
    print("")
    print("Simulating diffration patterns of '" + crystal_name + "' from randomly sampled orientations\n")
    crystal = py4DSTEM.process.diffraction.Crystal.from_CIF(Cu_cif_path + crystal_name + ".cif")
    
    rotation_matrices_for_crystal = sampled_rotation_matrices[int(crystal_idx * number_of_grains_and_rotations_for_each_crystal):int((crystal_idx + 1) * number_of_grains_and_rotations_for_each_crystal)]
    sampled_grain_indices = grain_indices_permuted[int(crystal_idx * number_of_grains_and_rotations_for_each_crystal):int((crystal_idx + 1) * number_of_grains_and_rotations_for_each_crystal)]
    
    grain_indices_for_each_crystal.append(sampled_grain_indices)

    DPs_collection, labels_collection = sample_diffraction_patterns_from_rotation_matrices(
                                                            crystal, 
                                                            rotation_matrices_for_crystal,
                                                            accelerating_voltage,
                                                            k_max,
                                                            thickness_num_for_sampling = 1000
    )
    
    DP_images_for_each_crystal.append(DPs_collection)
    rotation_labels_for_each_crystal.append(labels_collection)



print("")
print("Action 2. simulating electron diffraction (END)")

##############################################################################
##############################################################################
##### Perform electron diffraction simulation from sampled matrices (END) ####
##############################################################################
##############################################################################


##############################################################################
##############################################################################
################# Generate synthetic 4DSTEM data (START) #####################
##############################################################################
##############################################################################

print("")
print("Action 3. Generating synthetic 4DSTEM data (START)\n")


total_synthetic_data_4D = []
total_labels_for_synthetic_data_4D = []
total_crystal_class_label = []
for i in range(syn_2D_scanSpace_map.shape[0]):
    row_synthetic_data_4D = []
    row_labels_for_synthetic_data_4D = []
    row_crystal_class_label = []
    for j in range(syn_2D_scanSpace_map.shape[1]):
        grain_index = int(syn_2D_scanSpace_map[i,j])
        if grain_index > 0:
            
            for crystal_idx, assigned_grain_indices_for_a_crystal in enumerate(grain_indices_for_each_crystal):
                
                index_where_grain_index_is_located = np.where(grain_index == assigned_grain_indices_for_a_crystal)[0]
                if len(index_where_grain_index_is_located) > 0:
                    
                    # print("index_where_grain_index_is_located", index_where_grain_index_is_located)
                    
                    DPs_collection = DP_images_for_each_crystal[crystal_idx]
                    labels_collection = rotation_labels_for_each_crystal[crystal_idx]

                    DP = DPs_collection[index_where_grain_index_is_located[0]]
                    orientation_matrix = labels_collection[index_where_grain_index_is_located[0]]
                    BG_idx = np.random.randint(0, high=backgrounds.shape[0], size=1, dtype=np.int64)
                    synthetic_diffraction_pattern = generate_synthetic_diffraction_pattern(DP, backgrounds[BG_idx[0]], correlation_kernel)
                    # print("synthetic_diffraction_pattern.shape", synthetic_diffraction_pattern.shape)
                    # print("DP.shape", DP.shape)
                    # print("orientation_matrix.shape", orientation_matrix.shape)
                    row_synthetic_data_4D.append(synthetic_diffraction_pattern)
                    row_labels_for_synthetic_data_4D.append(orientation_matrix)
                    row_crystal_class_label.append(crystal_idx)

                    
            
            # orientation_matrix = labels_collection[k-1]
            # DP = DPs_collection[k-1]
            # BG_idx = np.random.randint(0, high=backgrounds.shape[0], size=1, dtype=np.int64)
            # synthetic_diffraction_pattern = generate_synthetic_diffraction_pattern(DP, backgrounds[BG_idx[0]], correlation_kernel)
            # row_synthetic_data_4D.append(synthetic_diffraction_pattern)
            # row_labels_for_synthetic_data_4D.append(orientation_matrix)
        else:
            BG_idx = np.random.randint(0, high=backgrounds.shape[0], size=1, dtype=np.int64)
            synthetic_diffraction_pattern = process_background_diffraction_pattern(backgrounds[BG_idx[0]])            
            row_synthetic_data_4D.append(synthetic_diffraction_pattern)
            row_labels_for_synthetic_data_4D.append(np.zeros((3, 3), dtype=np.float32))
            row_crystal_class_label.append(len(list_of_crystal_names))
            # print("synthetic_diffraction_pattern.shape", synthetic_diffraction_pattern.shape)
    
    total_synthetic_data_4D.append(row_synthetic_data_4D)
    total_labels_for_synthetic_data_4D.append(row_labels_for_synthetic_data_4D)
    total_crystal_class_label.append(row_crystal_class_label)
    

total_synthetic_data_4D = np.array(total_synthetic_data_4D)
np.save("multi_crystal_synthetic_4DSTEM_data.npy", total_synthetic_data_4D)
print("Shape of synthetic 4D STEM data", total_synthetic_data_4D.shape, "\n")
del total_synthetic_data_4D

total_labels_for_synthetic_data_4D = np.array(total_labels_for_synthetic_data_4D)
np.save("multi_crystal_synthetic_4DSTEM_data_rotationMat_labels.npy", total_labels_for_synthetic_data_4D)
print("Shape of rotation matrix labels for synthetic 4DSTEM data", total_labels_for_synthetic_data_4D.shape, "\n")

total_crystal_class_label = np.array(total_crystal_class_label)
np.save("multi_crystal_synthetic_4DSTEM_data_crystalClass_labels.npy", total_crystal_class_label)
print("Shape of crystal class labels for synthetic 4DSTEM data", total_crystal_class_label.shape, "\n")

for crystal_idx, crystal_name in enumerate(list_of_crystal_names):
    print("crystal ", crystal_name, " class label: ", crystal_idx)

print("background class label:  ", len(list_of_crystal_names))

print("")
print("Action 3. Generating synthetic 4DSTEM data (END)")

##############################################################################
##############################################################################
################# Generate synthetic 4DSTEM data (END) #####################
##############################################################################
##############################################################################


