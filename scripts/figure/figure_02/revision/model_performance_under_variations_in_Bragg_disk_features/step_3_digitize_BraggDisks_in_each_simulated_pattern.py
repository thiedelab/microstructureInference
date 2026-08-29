#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 23 14:49:36 2025

@author: kwang
"""

import numpy as np
import torch



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
file_path = "/home/kwang/Desktop/Storage/project/p03_orientation_mapping/figure/figure_02/revision_sensitivity_analyses/output/"
output_file_path = file_path


file_name = "postProc2_input_array"

BD_input = np.load(file_path + file_name + '.npy')

print("BD_input.shape", BD_input.shape)


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
    # radial_distance = np.array(radial_distance)
    
    intensity_val_extended = np.hstack(intensity_val)
    
    print("np.min(intensity_val_extended)", np.min(intensity_val_extended))
    
    # max_radial_distance = np.max(radial_distance_extended)
    # min_radial_distance = np.min(radial_distance_extended)
    
    # max_braggIntensity = np.max(intensity_val_extended)
    # min_braggIntensity = np.min(intensity_val_extended)
    
    print("")
    
    print("number_of_tokens_in_sequences", number_of_tokens_in_sequences)
    
    print("max_braggIntensity", max_braggIntensity)
    print("min_braggIntensity", min_braggIntensity)
    print("intensity_tolerance", intensity_tolerance, "\n")
    
    
    print("max_radial_distance", max_radial_distance)
    print("radial_distance_tolerance", radial_distance_tolerance,"\n")
    
    
    radial_bins = np.linspace(0.0, max_radial_distance + (radial_distance_tolerance), num_bins_radialDistance + 1)
    radial_bin_centers = (radial_bins[:-1] + radial_bins[1:]) / 2

    angle_bins = np.arange(-np.pi - np.pi/360., np.pi + np.pi/360., np.pi/180.)
    angle_bin_centers = (angle_bins[:-1] + angle_bins[1:]) / 2
    angle_bins[-1] = np.pi + np.pi/360 # further change the last element
    
    intensity_bins = np.linspace(min_braggIntensity, max_braggIntensity + (intensity_tolerance), num_bins_braggintensity + 1)
    intensity_bin_centers = (intensity_bins[:-1] + intensity_bins[1:]) / 2
    
        
    list_of_Bragg_disks_total = []
    # labels_total = []
    for idx, diffractionPattern in enumerate(BD_input):
    # for idx, diffractionPattern in df.items():
        # print("\n")
        # print("------------------------------------------------------------------------")
        # print("diffractionPattern\n", diffractionPattern,"\n")
        np_diffractionPattern = np.zeros_like(diffractionPattern)
        np_diffractionPattern[:, 0] = digitize_radial_distance(diffractionPattern[:,0], radial_bins)
        np_diffractionPattern[:, 1] = digitize_polarAngle(diffractionPattern[:,1], angle_bins)
        # print("digitize_polarAngle(diffractionPattern[:,1], angle_bins)\n", digitize_polarAngle(diffractionPattern[:,1], angle_bins), "\n")
        np_diffractionPattern[:, 2] = digitize_braggIntensity(diffractionPattern[:,2], intensity_bins)   
        np_diffractionPattern = np_diffractionPattern.astype(np.int32)
        # print("np_diffractionPattern\n", np_diffractionPattern, "\n")
        # print("np.where(np_diffractionPattern[:, 0]==0)", np.where(np_diffractionPattern[:, 0]==0)[0])
        # print("np.where(np_diffractionPattern[:, 1]==0)", np.where(np_diffractionPattern[:, 1]==0)[0])
        # print("np.where(np_diffractionPattern[:, 2]==-1)", np.where(np_diffractionPattern[:, 2]==-1)[0])
        
        common_elements = np.intersect1d(np.where(np_diffractionPattern[:, 2]==-1)[0], np.intersect1d(np.where(np_diffractionPattern[:, 0]==0)[0], np.where(np_diffractionPattern[:, 1]==0)[0]))
        # print("common_elements\n", common_elements, "\n")
        # np_diffractionPattern = np_diffractionPattern.astype(np.int32)
        np_diffractionPattern[common_elements, 2] = int(0)
        # print("np_diffractionPattern\n", np_diffractionPattern, "\n")
        # np.zeros()
        # print("np.max(np_diffractionPattern[:, 2])", np.max(np_diffractionPattern[:, 2]))
        if np.max(np_diffractionPattern[:, 2]) != 63:
            print("diffractionPattern\n", diffractionPattern, "\n")
        # print("np_diffractionPattern.shape", np_diffractionPattern.shape)
        # if idx == 0:
        #     if len(diffractionPattern['input']) < max_sequence_length:
        #         numbers_of_pad_tokens_to_add = max_sequence_length - len(diffractionPattern['input'])
        #         for recur in range(numbers_of_pad_tokens_to_add):
        #             np_diffractionPattern = np.vstack((np_diffractionPattern, np.array([[0, 0, 0]])))
    
        list_of_Bragg_disks_total.append(torch.tensor(np_diffractionPattern))
        # labels_total.append(diffractionPattern['rotationMatrix'])
        # labels_total.append(diffractionPattern['label1'])
    
    
    # labels_total = torch.tensor(labels_total)
    
    # print("list_of_Bragg_disks_total", list_of_Bragg_disks_total)
    
    # list_of_Bragg_disks_total = torch.tensor(list_of_Bragg_disks_total)
    
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

print("list_of_Bragg_disks_total.shape", list_of_Bragg_disks_total.shape)

output_file_name = "postProc3_input_array"

np.save(output_file_path + output_file_name + ".npy", list_of_Bragg_disks_total.detach().cpu().numpy())


print("JOB DONE.")
