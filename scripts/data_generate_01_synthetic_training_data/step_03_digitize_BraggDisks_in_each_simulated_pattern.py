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
                                max_radial_distance = 2.99000,
                                min_radial_distance = 0.45844,
                                min_braggIntensity = 0.004999,
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
    
    # max_radial_distance = np.max(radial_distance_extended)
    # min_radial_distance = np.min(radial_distance_extended)
    
    max_braggIntensity = np.max(intensity_val_extended)
    # min_braggIntensity = np.min(intensity_val_extended)
    
    print("number_of_tokens_in_sequences", number_of_tokens_in_sequences)
    
    print("max_braggIntensity", max_braggIntensity)
    print("min_braggIntensity", min_braggIntensity)
    
    print("max_radial_distance", max_radial_distance)
    print("min_radial_distance", min_radial_distance)
    
    
    radial_bins = np.linspace(0.0, max_radial_distance + (min_radial_distance*0.05), num_bins_radialDistance + 1)
    radial_bin_centers = (radial_bins[:-1] + radial_bins[1:]) / 2

    angle_bins = np.arange(-np.pi - np.pi/360., np.pi + np.pi/360., np.pi/180.)
    angle_bin_centers = (angle_bins[:-1] + angle_bins[1:]) / 2
    angle_bins[-1] = np.pi + np.pi/360 # further change the last element
    
    intensity_bins = np.linspace(0.0, max_braggIntensity + (min_braggIntensity*0.05), num_bins_braggintensity + 1)
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
        # print("np.max(np_diffractionPattern[:, 2])", np.max(np_diffractionPattern[:, 2]))
        if np.max(np_diffractionPattern[:, 2]) != 255:
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

def parse_args():
    parser = argparse.ArgumentParser(description="information of crystal, path of files, and other parameters for sampling thicknesses and orientations")
    parser.add_argument("--index", type=int, help="index_of_file", default = int(0))
    parser.add_argument("--crystal", type = str, help="nomenclature of crystal", default = "Cu_fcc")
    parser.add_argument("--directoryPath", type = str, help="path of directory where unit cell cif file is located", default = "./")
    parser.add_argument("--excitError", type=float, help="excitation error used for simulations", default = float(0.045))
    parser.add_argument("--intensThreshold", type=float, help="This threshold value is used to delete Bragg disks with a relative intensity smaller than it.", default = float(5e-3))
    return parser.parse_args()

def main():

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("torch device is: ", device)
    
    PAD = 0
    
    num_bins_radialDistance = int(256)
    num_bins_polarAngle = int(360)
    num_bins_braggintensity = int(256)
    
    max_sequence_length = 76
    
    file_path = os.getcwd() + "/"
    
    args = parse_args()
    crystal_name = args.crystal
    file_path = args.directoryPath
    excitation_error = float(args.excitError)
    intensThreshold = float(args.intensThreshold)
    
    
    file_index = int(args.index)
    file_index_string = str(file_index)
    
    file_name = crystal_name +"_ori_table_normalize_by_excitErr%4.3f_relIntThresh%4.3f_%d"%(excitation_error, intensThreshold, file_index_string)
    
    
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
                                                        padding_value = PAD)
    
    
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
    
    np.save(file_path + crystal_name + "_list_of_Bragg_disks_total_" + file_index_string + ".npy", list_of_Bragg_disks_total.detach().cpu().numpy())
    np.save(file_path + crystal_name + "_thickness_" + file_index_string + ".npy", thickness_total.detach().cpu().numpy())
    np.save(file_path + crystal_name + "_mirror_" + file_index_string + ".npy", mirror_total.detach().cpu().numpy())
    
    np.save(file_path + crystal_name + "_orientation_canonical_" + file_index_string + ".npy", orientation_canonical.detach().cpu().numpy())
    np.save(file_path + crystal_name + "_orientation_original_" + file_index_string + ".npy", orientation_original.detach().cpu().numpy())
    
    np.save(file_path + crystal_name + "_permuted_indices_" + file_index_string + ".npy", permuted_indices_total.detach().cpu().numpy())
    
    
    print("JOB DONE.")


if __name__ == "__main__":
    main()