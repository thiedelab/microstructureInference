#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 23 14:49:36 2025

@author: kwang
"""

import numpy as np
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="information of crystal, path of files, and other parameters for sampling thicknesses and orientations")
    parser.add_argument("--crystal", type = str, help="nomenclature of crystal", default = "Cu_fcc")
    parser.add_argument("--directoryPath", type = str, help="path of directory where unit cell cif file is located", default = "./")
    return parser.parse_args()

def main():
    
    seed = 42
    np.random.seed(seed)
    
    args = parse_args()
    crystal_name = args.crystal
    file_path = args.directoryPath    
    
    
    input_0 = crystal_name + "_list_of_Bragg_disks_total_0.npy"
    input_1 = crystal_name + "_list_of_Bragg_disks_total_1.npy"
    input_2 = crystal_name + "_list_of_Bragg_disks_total_2.npy"
    input_3 = crystal_name + "_list_of_Bragg_disks_total_3.npy"
    input_4 = crystal_name + "_list_of_Bragg_disks_total_4.npy"
    input_5 = crystal_name + "_list_of_Bragg_disks_total_5.npy"
    input_6 = crystal_name + "_list_of_Bragg_disks_total_6.npy"
    input_7 = crystal_name + "_list_of_Bragg_disks_total_7.npy"
    input_8 = crystal_name + "_list_of_Bragg_disks_total_8.npy"
    input_9 = crystal_name + "_list_of_Bragg_disks_total_9.npy"
    input_10 = crystal_name + "_list_of_Bragg_disks_total_10.npy"
    input_11 = crystal_name + "_list_of_Bragg_disks_total_11.npy"
    input_12 = crystal_name + "_list_of_Bragg_disks_total_12.npy"
    input_13 = crystal_name + "_list_of_Bragg_disks_total_13.npy"
    input_14 = crystal_name + "_list_of_Bragg_disks_total_14.npy"
    input_15 = crystal_name + "_list_of_Bragg_disks_total_15.npy"
    input_16 = crystal_name + "_list_of_Bragg_disks_total_16.npy"
    input_17 = crystal_name + "_list_of_Bragg_disks_total_17.npy"
    input_18 = crystal_name + "_list_of_Bragg_disks_total_18.npy"
    input_19 = crystal_name + "_list_of_Bragg_disks_total_19.npy"
    
    
    input_tensor_0 = np.load(file_path + input_0)
    input_tensor_1 = np.load(file_path + input_1)
    input_tensor_2 = np.load(file_path + input_2)
    input_tensor_3 = np.load(file_path + input_3)
    input_tensor_4 = np.load(file_path + input_4)
    input_tensor_5 = np.load(file_path + input_5)
    input_tensor_6 = np.load(file_path + input_6)
    input_tensor_7 = np.load(file_path + input_7)
    input_tensor_8 = np.load(file_path + input_8)
    input_tensor_9 = np.load(file_path + input_9)
    
    input_tensor_10 = np.load(file_path + input_10)
    input_tensor_11 = np.load(file_path + input_11)
    input_tensor_12 = np.load(file_path + input_12)
    input_tensor_13 = np.load(file_path + input_13)
    input_tensor_14 = np.load(file_path + input_14)
    input_tensor_15 = np.load(file_path + input_15)
    input_tensor_16 = np.load(file_path + input_16)
    input_tensor_17 = np.load(file_path + input_17)
    input_tensor_18 = np.load(file_path + input_18)
    input_tensor_19 = np.load(file_path + input_19)
    
    print("################################################\n\n")
    print("input_tensor_0.shape", input_tensor_0.shape)
    print("input_tensor_1.shape", input_tensor_1.shape)
    print("input_tensor_2.shape", input_tensor_2.shape)
    print("input_tensor_3.shape", input_tensor_3.shape)
    print("input_tensor_4.shape", input_tensor_4.shape)
    print("input_tensor_5.shape", input_tensor_5.shape)
    print("input_tensor_6.shape", input_tensor_6.shape)
    print("input_tensor_7.shape", input_tensor_7.shape)
    print("input_tensor_8.shape", input_tensor_8.shape)
    print("input_tensor_9.shape", input_tensor_9.shape)
    print("input_tensor_10.shape", input_tensor_10.shape)
    print("input_tensor_11.shape", input_tensor_11.shape)
    print("input_tensor_12.shape", input_tensor_12.shape)
    print("input_tensor_13.shape", input_tensor_13.shape)
    print("input_tensor_14.shape", input_tensor_14.shape)
    print("input_tensor_15.shape", input_tensor_15.shape)
    print("input_tensor_16.shape", input_tensor_16.shape)
    print("input_tensor_17.shape", input_tensor_17.shape)
    print("input_tensor_18.shape", input_tensor_18.shape)
    print("input_tensor_19.shape", input_tensor_19.shape)
    print("################################################\n\n")
    
    
    input_tensor_stacked = np.vstack([
                                        input_tensor_0,
                                        input_tensor_1,
                                        input_tensor_2, 
                                        input_tensor_3, 
                                        input_tensor_4, 
                                        input_tensor_5, 
                                        input_tensor_6,
                                        input_tensor_7,
                                        input_tensor_8,
                                        input_tensor_9,
                                        input_tensor_10,
                                        input_tensor_11,
                                        input_tensor_12, 
                                        input_tensor_13, 
                                        input_tensor_14, 
                                        input_tensor_15, 
                                        input_tensor_16,
                                        input_tensor_17,
                                        input_tensor_18,
                                        input_tensor_19
                                      ])
    
    print("input_tensor_stacked.shape before del", input_tensor_stacked.shape)
    
    del input_tensor_0
    del input_tensor_1
    del input_tensor_2
    del input_tensor_3
    del input_tensor_4
    del input_tensor_5
    del input_tensor_6
    del input_tensor_7
    del input_tensor_8
    del input_tensor_9
    del input_tensor_10
    del input_tensor_11
    del input_tensor_12
    del input_tensor_13
    del input_tensor_14
    del input_tensor_15
    del input_tensor_16
    del input_tensor_17
    del input_tensor_18
    del input_tensor_19
    
    print("input_tensor_stacked.shape after del", input_tensor_stacked.shape)
    
    permuted_indices_total = np.random.permutation(input_tensor_stacked.shape[0])
    
    input_tensor_stacked = input_tensor_stacked[permuted_indices_total]
    
    print("input_tensor_stacked.shape", input_tensor_stacked.shape)
    
    
    num_diffraction_patterns = input_tensor_stacked.shape[0]
    
    list_of_Bragg_disks_padded_train = input_tensor_stacked[:int(num_diffraction_patterns * 0.8)]
    list_of_Bragg_disks_padded_valid = input_tensor_stacked[int(num_diffraction_patterns * 0.8):]
    
    print("list_of_Bragg_disks_padded_train.shape", list_of_Bragg_disks_padded_train.shape)
    print("list_of_Bragg_disks_padded_valid.shape", list_of_Bragg_disks_padded_valid.shape)
    
    np.save(crystal_name + "_entire_Bradd_disks_padded_train.npy", list_of_Bragg_disks_padded_train)
    np.save(crystal_name + "_entire_Bradd_disks_padded_valid.npy", list_of_Bragg_disks_padded_valid)
    
    del list_of_Bragg_disks_padded_train
    del list_of_Bragg_disks_padded_valid
    del input_tensor_stacked
    
    
    
    orientation_canonical_0 = crystal_name + "_orientation_canonical_0.npy"
    orientation_canonical_1 = crystal_name + "_orientation_canonical_1.npy"
    orientation_canonical_2 = crystal_name + "_orientation_canonical_2.npy"
    orientation_canonical_3 = crystal_name + "_orientation_canonical_3.npy"
    orientation_canonical_4 = crystal_name + "_orientation_canonical_4.npy"
    orientation_canonical_5 = crystal_name + "_orientation_canonical_5.npy"
    orientation_canonical_6 = crystal_name + "_orientation_canonical_6.npy"
    orientation_canonical_7 = crystal_name + "_orientation_canonical_7.npy"
    orientation_canonical_8 = crystal_name + "_orientation_canonical_8.npy"
    orientation_canonical_9 = crystal_name + "_orientation_canonical_9.npy"
    orientation_canonical_10 = crystal_name + "_orientation_canonical_10.npy"
    orientation_canonical_11 = crystal_name + "_orientation_canonical_11.npy"
    orientation_canonical_12 = crystal_name + "_orientation_canonical_12.npy"
    orientation_canonical_13 = crystal_name + "_orientation_canonical_13.npy"
    orientation_canonical_14 = crystal_name + "_orientation_canonical_14.npy"
    orientation_canonical_15 = crystal_name + "_orientation_canonical_15.npy"
    orientation_canonical_16 = crystal_name + "_orientation_canonical_16.npy"
    orientation_canonical_17 = crystal_name + "_orientation_canonical_17.npy"
    orientation_canonical_18 = crystal_name + "_orientation_canonical_18.npy"
    orientation_canonical_19 = crystal_name + "_orientation_canonical_19.npy"
    
    
    orientation_canonical_label_0 = np.load(file_path + orientation_canonical_0)
    orientation_canonical_label_1 = np.load(file_path + orientation_canonical_1)
    orientation_canonical_label_2 = np.load(file_path + orientation_canonical_2)
    orientation_canonical_label_3 = np.load(file_path + orientation_canonical_3)
    orientation_canonical_label_4 = np.load(file_path + orientation_canonical_4)
    orientation_canonical_label_5 = np.load(file_path + orientation_canonical_5)
    orientation_canonical_label_6 = np.load(file_path + orientation_canonical_6)
    orientation_canonical_label_7 = np.load(file_path + orientation_canonical_7)
    orientation_canonical_label_8 = np.load(file_path + orientation_canonical_8)
    orientation_canonical_label_9 = np.load(file_path + orientation_canonical_9)
    orientation_canonical_label_10 = np.load(file_path + orientation_canonical_10)
    orientation_canonical_label_11 = np.load(file_path + orientation_canonical_11)
    orientation_canonical_label_12 = np.load(file_path + orientation_canonical_12)
    orientation_canonical_label_13 = np.load(file_path + orientation_canonical_13)
    orientation_canonical_label_14 = np.load(file_path + orientation_canonical_14)
    orientation_canonical_label_15 = np.load(file_path + orientation_canonical_15)
    orientation_canonical_label_16 = np.load(file_path + orientation_canonical_16)
    orientation_canonical_label_17 = np.load(file_path + orientation_canonical_17)
    orientation_canonical_label_18 = np.load(file_path + orientation_canonical_18)
    orientation_canonical_label_19 = np.load(file_path + orientation_canonical_19)
    
    
    print("################################################\n\n")
    print("orientation_canonical_label_0.shape", orientation_canonical_label_0.shape)
    print("orientation_canonical_label_1.shape", orientation_canonical_label_1.shape)
    print("orientation_canonical_label_2.shape", orientation_canonical_label_2.shape)
    print("orientation_canonical_label_3.shape", orientation_canonical_label_3.shape)
    print("orientation_canonical_label_4.shape", orientation_canonical_label_4.shape)
    print("orientation_canonical_label_5.shape", orientation_canonical_label_5.shape)
    print("orientation_canonical_label_6.shape", orientation_canonical_label_6.shape)
    print("orientation_canonical_label_7.shape", orientation_canonical_label_7.shape)
    print("orientation_canonical_label_8.shape", orientation_canonical_label_8.shape)
    print("orientation_canonical_label_9.shape", orientation_canonical_label_9.shape)
    print("orientation_canonical_label_10.shape", orientation_canonical_label_10.shape)
    print("orientation_canonical_label_11.shape", orientation_canonical_label_11.shape)
    print("orientation_canonical_label_12.shape", orientation_canonical_label_12.shape)
    print("orientation_canonical_label_13.shape", orientation_canonical_label_13.shape)
    print("orientation_canonical_label_14.shape", orientation_canonical_label_14.shape)
    print("orientation_canonical_label_15.shape", orientation_canonical_label_15.shape)
    print("orientation_canonical_label_16.shape", orientation_canonical_label_16.shape)
    print("orientation_canonical_label_17.shape", orientation_canonical_label_17.shape)
    print("orientation_canonical_label_18.shape", orientation_canonical_label_18.shape)
    print("orientation_canonical_label_19.shape", orientation_canonical_label_19.shape)
    print("################################################\n\n")
    
    orientation_canonical_label_stacked = np.vstack([
                                                    orientation_canonical_label_0, 
                                                    orientation_canonical_label_1, 
                                                    orientation_canonical_label_2, 
                                                    orientation_canonical_label_3, 
                                                    orientation_canonical_label_4, 
                                                    orientation_canonical_label_5, 
                                                    orientation_canonical_label_6,
                                                    orientation_canonical_label_7,
                                                    orientation_canonical_label_8,
                                                    orientation_canonical_label_9,
                                                    orientation_canonical_label_10, 
                                                    orientation_canonical_label_11, 
                                                    orientation_canonical_label_12, 
                                                    orientation_canonical_label_13, 
                                                    orientation_canonical_label_14, 
                                                    orientation_canonical_label_15, 
                                                    orientation_canonical_label_16,
                                                    orientation_canonical_label_17,
                                                    orientation_canonical_label_18,
                                                    orientation_canonical_label_19,
                                    ])
    
    orientation_canonical_label_stacked = orientation_canonical_label_stacked[permuted_indices_total]
    
    print("orientation_canonical_label_stacked.shape", orientation_canonical_label_stacked.shape)
    
    del orientation_canonical_label_0
    del orientation_canonical_label_1
    del orientation_canonical_label_2
    del orientation_canonical_label_3
    del orientation_canonical_label_4
    del orientation_canonical_label_5
    del orientation_canonical_label_6
    del orientation_canonical_label_7
    del orientation_canonical_label_8
    del orientation_canonical_label_9
    
    del orientation_canonical_label_10
    del orientation_canonical_label_11
    del orientation_canonical_label_12
    del orientation_canonical_label_13
    del orientation_canonical_label_14
    del orientation_canonical_label_15
    del orientation_canonical_label_16
    del orientation_canonical_label_17
    del orientation_canonical_label_18
    del orientation_canonical_label_19
    
    
    
    orientation_canonical_labels_train = orientation_canonical_label_stacked[:int(num_diffraction_patterns * 0.8)]
    orientation_canonical_labels_valid = orientation_canonical_label_stacked[int(num_diffraction_patterns * 0.8):]
    
    print("orientation_canonical_labels_train.shape", orientation_canonical_labels_train.shape)
    print("orientation_canonical_labels_valid.shape", orientation_canonical_labels_valid.shape)
    
    
    np.save(crystal_name + "_orientation_canonical_labels_train.npy", orientation_canonical_labels_train)
    np.save(crystal_name + "_orientation_canonical_labels_valid.npy", orientation_canonical_labels_valid)
    
    del orientation_canonical_label_stacked
    del orientation_canonical_labels_train
    del orientation_canonical_labels_valid
    
    orientation_original_0 = crystal_name + "_orientation_original_0.npy"
    orientation_original_1 = crystal_name + "_orientation_original_1.npy"
    orientation_original_2 = crystal_name + "_orientation_original_2.npy"
    orientation_original_3 = crystal_name + "_orientation_original_3.npy"
    orientation_original_4 = crystal_name + "_orientation_original_4.npy"
    orientation_original_5 = crystal_name + "_orientation_original_5.npy"
    orientation_original_6 = crystal_name + "_orientation_original_6.npy"
    orientation_original_7 = crystal_name + "_orientation_original_7.npy"
    orientation_original_8 = crystal_name + "_orientation_original_8.npy"
    orientation_original_9 = crystal_name + "_orientation_original_9.npy"
    orientation_original_10 = crystal_name + "_orientation_original_10.npy"
    orientation_original_11 = crystal_name + "_orientation_original_11.npy"
    orientation_original_12 = crystal_name + "_orientation_original_12.npy"
    orientation_original_13 = crystal_name + "_orientation_original_13.npy"
    orientation_original_14 = crystal_name + "_orientation_original_14.npy"
    orientation_original_15 = crystal_name + "_orientation_original_15.npy"
    orientation_original_16 = crystal_name + "_orientation_original_16.npy"
    orientation_original_17 = crystal_name + "_orientation_original_17.npy"
    orientation_original_18 = crystal_name + "_orientation_original_18.npy"
    orientation_original_19 = crystal_name + "_orientation_original_19.npy"
    
    
    orientation_original_label_0 = np.load(file_path + orientation_original_0)
    orientation_original_label_1 = np.load(file_path + orientation_original_1)
    orientation_original_label_2 = np.load(file_path + orientation_original_2)
    orientation_original_label_3 = np.load(file_path + orientation_original_3)
    orientation_original_label_4 = np.load(file_path + orientation_original_4)
    orientation_original_label_5 = np.load(file_path + orientation_original_5)
    orientation_original_label_6 = np.load(file_path + orientation_original_6)
    orientation_original_label_7 = np.load(file_path + orientation_original_7)
    orientation_original_label_8 = np.load(file_path + orientation_original_8)
    orientation_original_label_9 = np.load(file_path + orientation_original_9)
    orientation_original_label_10 = np.load(file_path + orientation_original_10)
    orientation_original_label_11 = np.load(file_path + orientation_original_11)
    orientation_original_label_12 = np.load(file_path + orientation_original_12)
    orientation_original_label_13 = np.load(file_path + orientation_original_13)
    orientation_original_label_14 = np.load(file_path + orientation_original_14)
    orientation_original_label_15 = np.load(file_path + orientation_original_15)
    orientation_original_label_16 = np.load(file_path + orientation_original_16)
    orientation_original_label_17 = np.load(file_path + orientation_original_17)
    orientation_original_label_18 = np.load(file_path + orientation_original_18)
    orientation_original_label_19 = np.load(file_path + orientation_original_19)
    
    
    print("################################################\n\n")
    print("orientation_original_label_0.shape", orientation_original_label_0.shape)
    print("orientation_original_label_1.shape", orientation_original_label_1.shape)
    print("orientation_original_label_2.shape", orientation_original_label_2.shape)
    print("orientation_original_label_3.shape", orientation_original_label_3.shape)
    print("orientation_original_label_4.shape", orientation_original_label_4.shape)
    print("orientation_original_label_5.shape", orientation_original_label_5.shape)
    print("orientation_original_label_6.shape", orientation_original_label_6.shape)
    print("orientation_original_label_7.shape", orientation_original_label_7.shape)
    print("orientation_original_label_8.shape", orientation_original_label_8.shape)
    print("orientation_original_label_9.shape", orientation_original_label_9.shape)
    print("orientation_original_label_10.shape", orientation_original_label_10.shape)
    print("orientation_original_label_11.shape", orientation_original_label_11.shape)
    print("orientation_original_label_12.shape", orientation_original_label_12.shape)
    print("orientation_original_label_13.shape", orientation_original_label_13.shape)
    print("orientation_original_label_14.shape", orientation_original_label_14.shape)
    print("orientation_original_label_15.shape", orientation_original_label_15.shape)
    print("orientation_original_label_16.shape", orientation_original_label_16.shape)
    print("orientation_original_label_17.shape", orientation_original_label_17.shape)
    print("orientation_original_label_18.shape", orientation_original_label_18.shape)
    print("orientation_original_label_19.shape", orientation_original_label_19.shape)
    print("################################################\n\n")
    
    orientation_original_label_stacked = np.vstack([
                                                    orientation_original_label_0, 
                                                    orientation_original_label_1, 
                                                    orientation_original_label_2, 
                                                    orientation_original_label_3, 
                                                    orientation_original_label_4, 
                                                    orientation_original_label_5, 
                                                    orientation_original_label_6,
                                                    orientation_original_label_7,
                                                    orientation_original_label_8,
                                                    orientation_original_label_9,
                                                    orientation_original_label_10, 
                                                    orientation_original_label_11, 
                                                    orientation_original_label_12, 
                                                    orientation_original_label_13, 
                                                    orientation_original_label_14, 
                                                    orientation_original_label_15, 
                                                    orientation_original_label_16,
                                                    orientation_original_label_17,
                                                    orientation_original_label_18,
                                                    orientation_original_label_19,
                                    ])
    
    orientation_original_label_stacked = orientation_original_label_stacked[permuted_indices_total]
    
    print("orientation_original_label_stacked.shape", orientation_original_label_stacked.shape)
    
    del orientation_original_label_0
    del orientation_original_label_1
    del orientation_original_label_2
    del orientation_original_label_3
    del orientation_original_label_4
    del orientation_original_label_5
    del orientation_original_label_6
    del orientation_original_label_7
    del orientation_original_label_8
    del orientation_original_label_9
    
    del orientation_original_label_10
    del orientation_original_label_11
    del orientation_original_label_12
    del orientation_original_label_13
    del orientation_original_label_14
    del orientation_original_label_15
    del orientation_original_label_16
    del orientation_original_label_17
    del orientation_original_label_18
    del orientation_original_label_19
    
    
    
    orientation_original_labels_train = orientation_original_label_stacked[:int(num_diffraction_patterns * 0.8)]
    orientation_original_labels_valid = orientation_original_label_stacked[int(num_diffraction_patterns * 0.8):]
    
    print("orientation_original_labels_train.shape", orientation_original_labels_train.shape)
    print("orientation_original_labels_valid.shape", orientation_original_labels_valid.shape)
    
    
    np.save(crystal_name + "_orientation_original_labels_train.npy", orientation_original_labels_train)
    np.save(crystal_name + "_orientation_original_labels_valid.npy", orientation_original_labels_valid)
    
    del orientation_original_label_stacked
    del orientation_original_labels_train
    del orientation_original_labels_valid
    
    
    
    mirror_0 = np.load(file_path + crystal_name + "_mirror_0.npy")
    mirror_1 = np.load(file_path + crystal_name + "_mirror_1.npy")
    mirror_2 = np.load(file_path + crystal_name + "_mirror_2.npy")
    mirror_3 = np.load(file_path + crystal_name + "_mirror_3.npy")
    mirror_4 = np.load(file_path + crystal_name + "_mirror_4.npy")
    mirror_5 = np.load(file_path + crystal_name + "_mirror_5.npy")
    mirror_6 = np.load(file_path + crystal_name + "_mirror_6.npy")
    mirror_7 = np.load(file_path + crystal_name + "_mirror_7.npy")
    mirror_8 = np.load(file_path + crystal_name + "_mirror_8.npy")
    mirror_9 = np.load(file_path + crystal_name + "_mirror_9.npy")
    
    mirror_10 = np.load(file_path + crystal_name + "_mirror_10.npy")
    mirror_11 = np.load(file_path + crystal_name + "_mirror_11.npy")
    mirror_12 = np.load(file_path + crystal_name + "_mirror_12.npy")
    mirror_13 = np.load(file_path + crystal_name + "_mirror_13.npy")
    mirror_14 = np.load(file_path + crystal_name + "_mirror_14.npy")
    mirror_15 = np.load(file_path + crystal_name + "_mirror_15.npy")
    mirror_16 = np.load(file_path + crystal_name + "_mirror_16.npy")
    mirror_17 = np.load(file_path + crystal_name + "_mirror_17.npy")
    mirror_18 = np.load(file_path + crystal_name + "_mirror_18.npy")
    mirror_19 = np.load(file_path + crystal_name + "_mirror_19.npy")
    
    
    
    print("################################################\n\n")
    print("mirror_0.shape", mirror_0.shape)
    print("mirror_1.shape", mirror_1.shape)
    print("mirror_2.shape", mirror_2.shape)
    print("mirror_3.shape", mirror_3.shape)
    print("mirror_4.shape", mirror_4.shape)
    print("mirror_5.shape", mirror_5.shape)
    print("mirror_6.shape", mirror_6.shape)
    print("mirror_7.shape", mirror_7.shape)
    print("mirror_8.shape", mirror_8.shape)
    print("mirror_9.shape", mirror_9.shape)
    
    print("mirror_10.shape", mirror_10.shape)
    print("mirror_11.shape", mirror_11.shape)
    print("mirror_12.shape", mirror_12.shape)
    print("mirror_13.shape", mirror_13.shape)
    print("mirror_14.shape", mirror_14.shape)
    print("mirror_15.shape", mirror_15.shape)
    print("mirror_16.shape", mirror_16.shape)
    print("mirror_17.shape", mirror_17.shape)
    print("mirror_18.shape", mirror_18.shape)
    print("mirror_19.shape", mirror_19.shape)
    
    print("################################################\n\n")
    
    
    
    
    mirror_stacked = np.vstack([
                                mirror_0,
                                mirror_1,
                                mirror_2, 
                                mirror_3, 
                                mirror_4, 
                                mirror_5, 
                                mirror_6,
                                mirror_7,
                                mirror_8,
                                mirror_9,                            
                                mirror_10,
                                mirror_11,
                                mirror_12, 
                                mirror_13, 
                                mirror_14, 
                                mirror_15, 
                                mirror_16,
                                mirror_17,
                                mirror_18,
                                mirror_19
                                ])
    
    print("mirror_stacked\n", mirror_stacked)
    print("mirror_stacked.shape", mirror_stacked.shape)
    
    mirror_stacked = mirror_stacked[permuted_indices_total]
    mirror_train = mirror_stacked[:int(num_diffraction_patterns * 0.8)]
    mirror_valid = mirror_stacked[int(num_diffraction_patterns * 0.8):]
    
    
    
    np.save(crystal_name + "_entire_mirror_train.npy", mirror_train)
    np.save(crystal_name + "_entire_mirror_valid.npy", mirror_valid)
    
    
    
    thickness_0 = np.load(file_path + crystal_name + "_thickness_0.npy")
    thickness_1 = np.load(file_path + crystal_name + "_thickness_1.npy")
    thickness_2 = np.load(file_path + crystal_name + "_thickness_2.npy")
    thickness_3 = np.load(file_path + crystal_name + "_thickness_3.npy")
    thickness_4 = np.load(file_path + crystal_name + "_thickness_4.npy")
    thickness_5 = np.load(file_path + crystal_name + "_thickness_5.npy")
    thickness_6 = np.load(file_path + crystal_name + "_thickness_6.npy")
    thickness_7 = np.load(file_path + crystal_name + "_thickness_7.npy")
    thickness_8 = np.load(file_path + crystal_name + "_thickness_8.npy")
    thickness_9 = np.load(file_path + crystal_name + "_thickness_9.npy")
    
    thickness_10 = np.load(file_path + crystal_name + "_thickness_10.npy")
    thickness_11 = np.load(file_path + crystal_name + "_thickness_11.npy")
    thickness_12 = np.load(file_path + crystal_name + "_thickness_12.npy")
    thickness_13 = np.load(file_path + crystal_name + "_thickness_13.npy")
    thickness_14 = np.load(file_path + crystal_name + "_thickness_14.npy")
    thickness_15 = np.load(file_path + crystal_name + "_thickness_15.npy")
    thickness_16 = np.load(file_path + crystal_name + "_thickness_16.npy")
    thickness_17 = np.load(file_path + crystal_name + "_thickness_17.npy")
    thickness_18 = np.load(file_path + crystal_name + "_thickness_18.npy")
    thickness_19 = np.load(file_path + crystal_name + "_thickness_19.npy")
    
    
    
    print("################################################\n\n")
    print("thickness_0.shape", thickness_0.shape)
    print("thickness_1.shape", thickness_1.shape)
    print("thickness_2.shape", thickness_2.shape)
    print("thickness_3.shape", thickness_3.shape)
    print("thickness_4.shape", thickness_4.shape)
    print("thickness_5.shape", thickness_5.shape)
    print("thickness_6.shape", thickness_6.shape)
    print("thickness_7.shape", thickness_7.shape)
    print("thickness_8.shape", thickness_8.shape)
    print("thickness_9.shape", thickness_9.shape)
    
    print("thickness_10.shape", thickness_10.shape)
    print("thickness_11.shape", thickness_11.shape)
    print("thickness_12.shape", thickness_12.shape)
    print("thickness_13.shape", thickness_13.shape)
    print("thickness_14.shape", thickness_14.shape)
    print("thickness_15.shape", thickness_15.shape)
    print("thickness_16.shape", thickness_16.shape)
    print("thickness_17.shape", thickness_17.shape)
    print("thickness_18.shape", thickness_18.shape)
    print("thickness_19.shape", thickness_19.shape)
    
    print("################################################\n\n")
    
    
    
    
    thickness_stacked = np.vstack([
                                thickness_0,
                                thickness_1,
                                thickness_2, 
                                thickness_3, 
                                thickness_4, 
                                thickness_5, 
                                thickness_6,
                                thickness_7,
                                thickness_8,
                                thickness_9,                            
                                thickness_10,
                                thickness_11,
                                thickness_12, 
                                thickness_13, 
                                thickness_14, 
                                thickness_15, 
                                thickness_16,
                                thickness_17,
                                thickness_18,
                                thickness_19
                                ])
    
    print("thickness_stacked\n", thickness_stacked)
    print("thickness_stacked.shape", thickness_stacked.shape)
    
    thickness_stacked = thickness_stacked[permuted_indices_total]
    thickness_train = thickness_stacked[:int(num_diffraction_patterns * 0.8)]
    thickness_valid = thickness_stacked[int(num_diffraction_patterns * 0.8):]
    
    
    
    np.save(crystal_name + "_entire_thickness_train.npy", thickness_train)
    np.save(crystal_name + "_entire_thickness_valid.npy", thickness_valid)
    
    
    np.save(crystal_name + "_entire_permuted_indices.npy", permuted_indices_total)

if __name__ == "__main__":
    main()