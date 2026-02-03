#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 23 14:49:36 2025

@author: kwang
"""

import numpy as np
import torch
import os
import argparse

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# os.environ['TORCH_USE_CUDA_DSA'] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

PAD = 0


num_bins_radialDistance = int(256)
num_bins_polarAngle = int(360)
num_bins_braggintensity = int(64)

max_sequence_length = 76

# file_path = os.getcwd() + "/"
file_path = "/media/kwang/LaCie/Kwang/Storage/py4DSTEM_Cu_simulated_data/complete_new_approach_0.040_4e3_Cu_fcc/"


parser = argparse.ArgumentParser(description="A simple script demonstrating argparse.")
parser.add_argument("--index", type=int, help="index_of_file")
args = parser.parse_args()

file_index = int(args.index)
file_index_string = str(file_index)

file_name = "py4DSTEM_Cu_ori_table_normalize_by_maxInt_4e3_sg0.040_merged_" + file_index_string


BD_input = np.load(file_path + file_name + '.npy')

orientation_canonical = np.load(file_path + "rot_canonical_" + file_name + '.npy')
orientation_original = np.load(file_path + "rot_original_" + file_name + '.npy')

orientation_canonical = torch.tensor(orientation_canonical)
orientation_original = torch.tensor(orientation_original)

thickness_total = np.load(file_path +"thickness_" + file_name + ".npy")
thickness_total = torch.tensor(thickness_total)
thickness_total = thickness_total.reshape((thickness_total.shape[0], 1))


mirror_total = np.load(file_path +"mirror_" + file_name + ".npy")
mirror_total = torch.tensor(mirror_total)
mirror_total = mirror_total.reshape((mirror_total.shape[0], 1))



print("BD_input.shape", BD_input.shape)
print("thickness_total.shape", thickness_total.shape)
print("mirror_total.shape", mirror_total.shape)
print("orientation_canonical.shape", orientation_canonical.shape)
print("orientation_original.shape", orientation_original.shape)


num_diffraction_patterns = len(BD_input)
print("number of diffraction patterns:", num_diffraction_patterns)

def digitize_radial_distance(radial_distances, radial_bins):
    return np.digitize(radial_distances, radial_bins) - 1

def digitize_polarAngle(polar_angles, angle_bins):
    return np.digitize(polar_angles, angle_bins) - 1

def digitize_braggIntensity(braggDisk_intensities, intensity_bins):
    return np.digitize(braggDisk_intensities, intensity_bins) - 1

def process_pandas_tabular_data(
                                BD_input, 
                                num_bins_radialDistance, 
                                num_bins_polarAngle, 
                                num_bins_braggintensity, 
                                max_sequence_length,
                                max_braggIntensity = 1.0,
                                min_braggIntensity = 0.001,
                                max_radial_distance = 2.99000,
                                radial_distance_tolerance = 0.0001,
                                intensity_tolerance = 0.0001,
                                ):
    
    

    radial_distance = []
    intensity_val = []
    
    number_of_tokens_in_sequences = []
    
    for idx, diffractionPattern in enumerate(BD_input):        
        indices_where_radial_distance_is_nonzero = np.where(diffractionPattern[:,0] > 0)[0]
    
        number_of_tokens_in_sequences.append(len(diffractionPattern[indices_where_radial_distance_is_nonzero]))
        radial_distance.append(diffractionPattern[:,0][indices_where_radial_distance_is_nonzero])
        intensity_val.append(diffractionPattern[:,2][indices_where_radial_distance_is_nonzero])
    
    number_of_tokens_in_sequences = np.array(number_of_tokens_in_sequences)
    
    intensity_val_extended = np.hstack(intensity_val)
    
    radial_bins = np.linspace(0.0, max_radial_distance + (radial_distance_tolerance), num_bins_radialDistance + 1)
    radial_bin_centers = (radial_bins[:-1] + radial_bins[1:]) / 2

    angle_bins = np.arange(-np.pi - np.pi/360., np.pi + np.pi/360., np.pi/180.)
    angle_bin_centers = (angle_bins[:-1] + angle_bins[1:]) / 2
    angle_bins[-1] = np.pi + np.pi/360 # further change the last element
    
    intensity_bins = np.linspace(min_braggIntensity, max_braggIntensity + (intensity_tolerance), num_bins_braggintensity + 1)
    intensity_bin_centers = (intensity_bins[:-1] + intensity_bins[1:]) / 2
    
        
    list_of_Bragg_disks_total = []
    for idx, diffractionPattern in enumerate(BD_input):
        np_diffractionPattern = np.zeros_like(diffractionPattern)
        np_diffractionPattern[:, 0] = digitize_radial_distance(diffractionPattern[:,0], radial_bins)
        np_diffractionPattern[:, 1] = digitize_polarAngle(diffractionPattern[:,1], angle_bins)
        np_diffractionPattern[:, 2] = digitize_braggIntensity(diffractionPattern[:,2], intensity_bins)   
        np_diffractionPattern = np_diffractionPattern.astype(np.int32)
        
        common_elements = np.intersect1d(np.where(np_diffractionPattern[:, 2]==-1)[0], np.intersect1d(np.where(np_diffractionPattern[:, 0]==0)[0], np.where(np_diffractionPattern[:, 1]==0)[0]))
        np_diffractionPattern[common_elements, 2] = int(0)
        if np.max(np_diffractionPattern[:, 2]) != 63:
            print("diffractionPattern\n", diffractionPattern, "\n")
    
        list_of_Bragg_disks_total.append(torch.tensor(np_diffractionPattern))
    
    radial_bins = torch.tensor(radial_bins, dtype = torch.float32)
    radial_bin_centers = torch.tensor(radial_bin_centers, dtype = torch.float32)
    
    angle_bins = torch.tensor(angle_bins, dtype = torch.float32)
    angle_bin_centers = torch.tensor(angle_bin_centers, dtype = torch.float32)
    
    intensity_bins = torch.tensor(intensity_bins, dtype = torch.float32)
    intensity_bin_centers = torch.tensor(intensity_bin_centers, dtype = torch.float32)
    
    return list_of_Bragg_disks_total,  radial_bins, radial_bin_centers, angle_bins, angle_bin_centers, intensity_bins, intensity_bin_centers


(list_of_Bragg_disks_total,  \
 radial_bins, radial_bin_centers, \
 angle_bins, angle_bin_centers, \
 intensity_bins, intensity_bin_centers) = process_pandas_tabular_data(
                                                    BD_input, 
                                                    num_bins_radialDistance, 
                                                    num_bins_polarAngle, 
                                                    num_bins_braggintensity, 
                                                    max_sequence_length)



del BD_input

###############################################################################
######## STEP 1. ADD [PAD] tokens and SHUFFLE processed data

list_of_Bragg_disks_total = torch.nn.utils.rnn.pad_sequence(
                                                    list_of_Bragg_disks_total, 
                                                    batch_first=True, 
                                                    padding_value = 0)


permuted_indices_total = torch.randperm(orientation_canonical.size(0))
print("list_of_Bragg_disks_total.shape", list_of_Bragg_disks_total.shape)
print("list_of_Bragg_disks_total[0].shape", list_of_Bragg_disks_total[0].shape)
print("list_of_Bragg_disks_total[0]\n", list_of_Bragg_disks_total[0], "\n")

list_of_Bragg_disks_total = list_of_Bragg_disks_total[permuted_indices_total]
orientation_canonical = orientation_canonical[permuted_indices_total]
orientation_original = orientation_original[permuted_indices_total]
thickness_total = thickness_total[permuted_indices_total]
mirror_total = mirror_total[permuted_indices_total]

print("list_of_Bragg_disks_total.shape", list_of_Bragg_disks_total.shape)
print("orientation_canonical.shape", orientation_canonical.shape)
print("orientation_original.shape", orientation_original.shape)
print("mirror_total.shape", mirror_total.shape)
print("thickness_total.shape", thickness_total.shape)

print("mirror_total\n", mirror_total, "\n")
print("thickness_total\n", thickness_total, "\n")

output_file_path = "/media/kwang/LaCie/Kwang/Storage/py4DSTEM_Cu_simulated_data/complete_new_approach_0.040_4e3_Cu_fcc/merge_076_intBins_64/"

np.save(output_file_path + "list_of_Bragg_disks_total_" + file_index_string + ".npy", list_of_Bragg_disks_total.detach().cpu().numpy())
np.save(output_file_path + "thickness_" + file_index_string + ".npy", thickness_total.detach().cpu().numpy())
np.save(output_file_path + "mirror_" + file_index_string + ".npy", mirror_total.detach().cpu().numpy())

np.save(output_file_path + "orientation_canonical_" + file_index_string + ".npy", orientation_canonical.detach().cpu().numpy())
np.save(output_file_path + "orientation_original_" + file_index_string + ".npy", orientation_original.detach().cpu().numpy())

np.save(output_file_path + "permuted_indices_" + file_index_string + ".npy", permuted_indices_total.detach().cpu().numpy())



print("JOB DONE.")
