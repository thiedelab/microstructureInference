#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 13:40:45 2025

@author: kwang
"""
import numpy as np
from modules_generate_synthethic_4DSTEM import sample_diffraction_patterns_from_rotation_matrices, generate_synthetic_diffraction_pattern, process_background_diffraction_pattern
import py4DSTEM
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="information of orientations and crystal grains for generating synthetic data")
    parser.add_argument("--num_orientations", type = int, help="number of randomly sampled orientations", default = int(60))
    parser.add_argument("--num_grains", type = int, help="number of distinct crystal grains", default = int(60))
    parser.add_argument("--ScanDimension", type = int, help="scan space dimension", default = int(128))
 
    return parser.parse_args()


def main():
    
    args = parse_args()
    num_grains = args.num_grains
    syn_2D_scanSpace_map_side_dimension = args.ScanDimension
    num_sampled_orientations = args.num_orientations
    
    
    assert num_grains <= num_sampled_orientations, "number of grains must be equal or smaller than number of sampled orientations"
        
    
    
    BraggDisks_list_file_name_str = "m2_bragg_disks_corThForK80000_dog_sig1_2.00_sig2_8.00_cortThForTemp_5000"
    rotMat_file_path = "./"
    
    Cu_cif_path = "./"
    k_max = 0.0328 * 64
    accelerating_voltage = int(300e3)
    
    
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
    
    
    crystal = py4DSTEM.process.diffraction.Crystal.from_CIF(Cu_cif_path + "Cu_fcc.cif")
    
    DPs_collection, labels_collection, thickness_sampled = sample_diffraction_patterns_from_rotation_matrices(
                                                            crystal, 
                                                            sampled_rotation_matrices,
                                                            accelerating_voltage,
                                                            k_max,
                                                            thickness_num_for_sampling = 1000
    )
    
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
    total_thicknesses = []
    for i in range(syn_2D_scanSpace_map.shape[0]):
        row_synthetic_data_4D = []
        row_labels_for_synthetic_data_4D = []
        for j in range(syn_2D_scanSpace_map.shape[1]):
            k = int(syn_2D_scanSpace_map[i,j])
            if k > 0:
                orientation_matrix = labels_collection[k-1]
                DP = DPs_collection[k-1]
                total_thicknesses.append(thickness_sampled[k-1])
                BG_idx = np.random.randint(0, high=backgrounds.shape[0], size=1, dtype=np.int64)
                synthetic_diffraction_pattern = generate_synthetic_diffraction_pattern(DP, backgrounds[BG_idx[0]], correlation_kernel)
                row_synthetic_data_4D.append(synthetic_diffraction_pattern)
                row_labels_for_synthetic_data_4D.append(orientation_matrix)
            else:
                BG_idx = np.random.randint(0, high=backgrounds.shape[0], size=1, dtype=np.int64)
                synthetic_diffraction_pattern = process_background_diffraction_pattern(backgrounds[BG_idx[0]])            
                row_synthetic_data_4D.append(synthetic_diffraction_pattern)
                row_labels_for_synthetic_data_4D.append(np.zeros((3, 3), dtype=np.float32))
        total_synthetic_data_4D.append(row_synthetic_data_4D)
        total_labels_for_synthetic_data_4D.append(row_labels_for_synthetic_data_4D)
        
    
    total_synthetic_data_4D = np.array(total_synthetic_data_4D)
    np.save("single_Cu_fcc_crystal_synthetic_4DSTEM_data.npy", total_synthetic_data_4D)
    print("Shape of synthetic 4D STEM data", total_synthetic_data_4D.shape, "\n")
    del total_synthetic_data_4D
    
    total_labels_for_synthetic_data_4D = np.array(total_labels_for_synthetic_data_4D)
    np.save("single_Cu_fcc_crystal_synthetic_4DSTEM_data_rotationMat_labels.npy", total_labels_for_synthetic_data_4D)
    
    
    total_thicknesses = np.array(total_thicknesses)
    np.save("single_Cu_fcc_crystal_synthetic_4DSTEM_data_thicknesses.npy", total_thicknesses)
    print("Shape of rotation matrix labels for synthetic 4DSTEM data", total_labels_for_synthetic_data_4D.shape)
    
    print("")
    print("Action 3. Generating synthetic 4DSTEM data (END)")
    
    ##############################################################################
    ##############################################################################
    ################# Generate synthetic 4DSTEM data (END) #####################
    ##############################################################################
    ##############################################################################
    
    
if __name__ == "__main__":
    main()