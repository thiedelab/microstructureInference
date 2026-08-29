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
from microstructure_inference.dataModules import DataSetPointGroup_rotation, digitized_bin_centers
from microstructure_inference.dataProcessing import predict_rotation_sim_data_with_labels
from microstructure_inference.transformerModel import ModelConfig, make_model

def parse_args():
    parser = argparse.ArgumentParser(description="information of scan space dimension and number of crystalline grains")
    parser.add_argument("--embed_dim", type = int, help="embedded dimension", default = int(384))
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
    parser.add_argument("--initial_run", type = bool, help="boolean variable indicating whether this training is the first training run", default = bool(True))
    
    parser.add_argument("--num_warmup_epochs", type = int, help="number of epochs for linear warm up learning rate scheduler", default = int(15))
    parser.add_argument("--cos_decay_epoch", type = int, help="number of epochs for cosine decay learning rate scheduler", default = int(250))

    parser.add_argument("--eta_intial", type = float, help="initial learning rate", default = float(0.00007))
    parser.add_argument("--eta_min", type = float, help="minimum learning rate in the last epoch", default = float(5e-7))
    parser.add_argument("--printArg", type = bool, help="boolean variable indicating whether to print all the arguments", default = bool(True))
    parser.add_argument("--printModelInfo", type = bool, help="boolean variable indicating whether to print model architecture", default = bool(True))
    parser.add_argument("--datasetString", type = str, help="boolean variable indicating whether to print model architecture", default = str("valid"))
    parser.add_argument("--input_dir", type = str, help="directory for input data", default = str("/home/kwang/Desktop/Storage/project/p03_orientation_mapping/figure/figure_02/revision_sensitivity_analyses/output/"))
    parser.add_argument("--output_dir", type = str, help="directory for saving digitized output tables", default = str("/home/kwang/Desktop/Storage/project/p03_orientation_mapping/figure/figure_02/revision_sensitivity_analyses/output/predict_using_trained_model/"))
    
    return parser.parse_args()



def main():

    args = parse_args()

    data_set_str = str(args.datasetString)


    num_bins_radialDistance = args.num_bins_radialDistance
    num_bins_polarAngle = args.num_bins_polarAngle
    num_bins_braggintensity = args.num_bins_braggintensity
    
    embed_dim = args.embed_dim
    max_sequence_length = args.max_sequence_length
    
    max_radial_distance = args.max_radial_distance
    
    max_braggIntensity = args.max_braggIntensity

    isMultitask = int(args.isMultitask)
    
    seed = args.seed
    
    
    torch.manual_seed(seed)
    np.random.seed(seed)

    file_path = str(args.input_dir)
    model_path =  "/home/kwang/Desktop/Storage/project/p03_orientation_mapping/figure/figure_02/independently_trained_model_2/"

    prediction_output_dir = str(args.output_dir)

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
    
    data_set_i = DataSetPointGroup_rotation(file_path + "postProc3_input_array.npy", file_path + "postProc2_output_label_orignial.npy", num_bins_polarAngle, transform = None)
    
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

    
    np.save(prediction_output_dir + "predicted_rotation_matrices_trained_" + data_set_str + ".npy", rotation_matrices_np)
    np.save(prediction_output_dir + "geodesic_distances_trained_" + data_set_str +".npy", geodesic_distance_stack_np)
    
    print("average_geodesic_error", average_geodesic_error)
    print("geodesic_distance_stack\n", geodesic_distance_stack, "\n")
    print("geodesic_distance_stack.shape", geodesic_distance_stack.shape)
    
    end_perf = time.perf_counter()
    elapsed_perf = end_perf - start_perf
    
    print(f"High-resolution execution time: {elapsed_perf:.6f} seconds")
    
    # del rotation_matrices
    # del rotation_matrices_np
    # del geodesic_distance_stack
    # del geodesic_distance_stack_np
    
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
