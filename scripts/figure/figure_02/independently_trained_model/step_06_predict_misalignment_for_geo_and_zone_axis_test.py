#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 10:17:15 2025

@author: kwang
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import argparse
import time
import py4DSTEM
from microstructure_inference.dataModules import DataSetPointGroup_rotation, digitized_bin_centers, cubic_proper_point_group_operations
from microstructure_inference.dataProcessing import predict_rotation_sim_data_with_labels
from microstructure_inference.transformerModel import ModelConfig, make_model
from microstructure_inference.dataProcessing import return_predicted_rotation_matrices_in_canonical_form


def parse_args():
    parser = argparse.ArgumentParser(description="information of scan space dimension and number of crystalline grains")
    parser.add_argument("--embed_dim", type = int, help="embedded dimension", default = int(384))
    parser.add_argument("--accelerating_voltage", type = int, help="accelerating voltage of electron microscopy simulation", default = int(300e3))
    parser.add_argument("--diff_pixel_size", type = float, help="size of pixel of diffraction pattern", default = float(0.0328))
    parser.add_argument("--diff_pixel_numbers", type = int, help="number of pixels in a dimension of diffraction pattern", default = int(128))

    parser.add_argument("--max_sequence_length", type = int, help="maximum number of allowed tokens", default = int(76))
    parser.add_argument("--min_radial_distance", type = float, help="minimum raidus", default = float(0.45844))    
    parser.add_argument("--max_radial_distance", type = float, help="maximum raidus", default = float(2.99000))
    parser.add_argument("--min_braggIntensity", type = float, help="minimum intensity", default = float(0.001))    
    parser.add_argument("--max_braggIntensity", type = float, help="maximum intenisty", default = float(1.0))
    parser.add_argument("--num_bins_radialDistance", type = int, help="number of discretized bins for radius dimension", default = int(256))
    parser.add_argument("--num_bins_polarAngle", type = int, help="number of discretized bins for polar angle dimension", default = int(360))
    parser.add_argument("--num_bins_braggintensity", type = int, help="number of discretized bins for intensity dimension", default = int(64))
    parser.add_argument("--isMultitask", type = int, help="integer_indicating_multi_predictions", default = int(0))
    parser.add_argument("--seed", type = int, help="random number seed for numpy and torch", default = int(22))
    parser.add_argument("--PAD", type = int, help="integer indicating PAD token", default = int(0))
    parser.add_argument("--printArg", type = bool, help="boolean variable indicating whether to print all the arguments", default = bool(True))
    parser.add_argument("--printModelInfo", type = bool, help="boolean variable indicating whether to print model architecture", default = bool(True))
    parser.add_argument("--datasetString", type = str, help="boolean variable indicating whether to print model architecture", default = str("test"))
    return parser.parse_args()



def main():

    args = parse_args()

    data_set_str = str(args.datasetString)


    num_bins_radialDistance = args.num_bins_radialDistance
    num_bins_polarAngle = args.num_bins_polarAngle
    num_bins_braggintensity = args.num_bins_braggintensity

    accelerating_voltage = int(args.accelerating_voltage)
    pixel_size = args.diff_pixel_size
    pixel_numbers = int(args.diff_pixel_numbers)
    k_max = pixel_size * pixel_numbers / 2.
    
    embed_dim = args.embed_dim
    max_sequence_length = args.max_sequence_length
    
    max_radial_distance = args.max_radial_distance
    
    max_braggIntensity = args.max_braggIntensity

    isMultitask = int(args.isMultitask)
    
    seed = args.seed
    
    
    torch.manual_seed(seed)
    np.random.seed(seed)

    file_path = os.getcwd() + "/../"
    model_path = os.getcwd() + "/"
    Cu_cif = "Cu_fcc.cif"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("torch device", device, "\n")

    if args.printArg:

        print("Arguments passed:")
        for arg, value in vars(args).items():
            print(f"  {arg}: {value}")

    print("")
    
    sub_panel_directory_path = "panel_d/"



    radial_bins, radial_bin_centers, angle_bins, angle_bin_centers, intensity_bin_centers = digitized_bin_centers(
                                                    num_bins_radialDistance,
                                                    max_radial_distance,
                                                    num_bins_polarAngle,
                                                    num_bins_braggintensity,
                                                    max_braggIntensity,
    )
    
    data_set_i = DataSetPointGroup_rotation(file_path + "entire_Bradd_disks_padded_" + data_set_str +".npy", file_path + "orientation_original_labels_"+ data_set_str + ".npy", num_bins_polarAngle, transform = None)
    
    data_loader_i = DataLoader(
                            data_set_i,
                            batch_size = 2048,
                            shuffle = False,
                            num_workers = 16,
                            pin_memory=torch.cuda.is_available(),
                             )
    
    config = ModelConfig(
                         d_embed = embed_dim,
                         d_ff = 2 * embed_dim,
                         angle_bin_centers = angle_bin_centers,
                         intensity_bin_centers = intensity_bin_centers,
                         num_bins_radialDistance = num_bins_radialDistance,
                         device = device,
                         num_feature = 9,
                         h = 8,
                         N_encoder = 3,
                         max_seq_len = max_sequence_length,
                         dropout = 0.001,
                         multiTask = isMultitask,
                         )

    
    model = make_model(config)
    
    # checkpoint = torch.load('best_model.pth', map_location=torch.device('cpu')) # ie, model_best.pth.tar
    checkpoint = torch.load(model_path + 'best_model.pth') # ie, model_best.pth.tar
    model.load_state_dict(checkpoint['model_state_dict'])
    
    start_perf = time.perf_counter()

    rotation_matrices, geodesic_distance_stack, average_geodesic_error = predict_rotation_sim_data_with_labels(model, data_loader_i, device)
    rotation_matrices_np = rotation_matrices.detach().cpu().numpy()
    geodesic_distance_stack_np = geodesic_distance_stack.detach().cpu().numpy()


    crystal = py4DSTEM.process.diffraction.Crystal.from_CIF(file_path + Cu_cif)
    crystal.setup_diffraction(accelerating_voltage)
    crystal.calculate_structure_factors(k_max)
    
    crystal.orientation_plan(
        angle_step_zone_axis = 2,
        angle_step_in_plane = 2,
        accel_voltage = accelerating_voltage,
        # intensity_power = 0.5,
        # corr_kernel_size= 0.08, # was 0.08 before 0.12 not bad
        zone_axis_range='auto',
    )

    cubic_proper = cubic_proper_point_group_operations().detach().clone().cpu().numpy()

    canonical = return_predicted_rotation_matrices_in_canonical_form(crystal, rotation_matrices_np, cubic_proper)
    
    
    np.save(model_path + sub_panel_directory_path + "predicted_rotation_matrices_trained_" + data_set_str + ".npy", rotation_matrices_np)
    np.save(model_path + sub_panel_directory_path + "predicted_rotation_matrices_trained_" + data_set_str + "_canonical.npy", canonical)
    np.save(model_path + sub_panel_directory_path + "geodesic_distances_trained_" + data_set_str +".npy", geodesic_distance_stack_np)
    
    print("average_geodesic_error", average_geodesic_error)
    print("geodesic_distance_stack\n", geodesic_distance_stack, "\n")
    print("geodesic_distance_stack.shape", geodesic_distance_stack.shape)
    
    end_perf = time.perf_counter()
    elapsed_perf = end_perf - start_perf
    
    print(f"High-resolution execution time: {elapsed_perf:.6f} seconds")
    
    del rotation_matrices
    del rotation_matrices_np
    del geodesic_distance_stack
    del geodesic_distance_stack_np
    
    # untrained_model = make_model(config)
    # untrained_model.load_state_dict(torch.load(model_path + "untrained_model.pth"))
    
    
    # rotation_matrices, geodesic_distance_stack, average_geodesic_error = predict_rotation_sim_data_with_labels(untrained_model, data_loader_i, device)
    # rotation_matrices_np = rotation_matrices.detach().cpu().numpy()
    # geodesic_distance_stack_np = geodesic_distance_stack.detach().cpu().numpy()
    
    # np.save(model_path + sub_panel_directory_path + "predicted_rotation_matrices_untrained_" + data_set_str + ".npy", rotation_matrices_np)
    # np.save(model_path + sub_panel_directory_path + "geodesic_distances_untrained_" + data_set_str + ".npy", geodesic_distance_stack_np)
    
    # print("")    
    # print("untrained average_geodesic_error", average_geodesic_error)
    # print("untrained geodesic_distance_stack\n", geodesic_distance_stack, "\n")
    # print("untrained geodesic_distance_stack.shape", geodesic_distance_stack.shape)
    
    


if __name__ == "__main__":
    main()
