#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 23 14:49:36 2025

@author: kwang
"""

import py4DSTEM
import numpy as np
import time
import pickle
from microstructure_inference.transformerModel import ModelConfig, make_model
import torch
from microstructure_inference.dataModules import ExpDataset
from torch.utils.data import DataLoader
import pandas as pd
import argparse
from microstructure_inference.dataProcessing import pre_process_experimental_BraggDisk, process_pandas_tabular_data, predict_rotation_experimental_data, make_orientation_map_based_on_4D_rotation_matrices
print("py4DSTEM version: ", py4DSTEM.__version__)

def parse_args():
    parser = argparse.ArgumentParser(description="information of scan space dimension and digitization")
    parser.add_argument("--embed_dim", type = int, help="embedded dimension", default = int(384))
    parser.add_argument("--max_sequence_length", type = int, help="maximum number of allowed tokens", default = int(76))
    parser.add_argument("--min_radial_distance", type = float, help="minimum raidus", default = float(0.45844))
    parser.add_argument("--max_radial_distance", type = float, help="maximum raidus", default = float(2.99000))
    parser.add_argument("--min_braggIntensity", type = float, help="minimum intensity", default = float(0.001))
    parser.add_argument("--max_braggIntensity", type = float, help="maximum intenisty", default = float(1.0))
    parser.add_argument("--num_bins_radialDistance", type = int, help="number of discretized bins for radius dimension", default = int(256))
    parser.add_argument("--num_bins_polarAngle", type = int, help="number of discretized bins for polar angle dimension", default = int(360))
    parser.add_argument("--num_bins_braggintensity", type = int, help="number of discretized bins for intensity dimension", default = int(64))
    parser.add_argument("--seed", type = int, help="random number seed for numpy and torch", default = int(42))
    parser.add_argument("--PAD", type = int, help="integer indicating PAD token", default = int(0))
    parser.add_argument("--scan_dim", type=int,help="Scan dimension parameter", default = int(8))
    parser.add_argument("--isMultitask", type = int, help="integer_indicating_multi_predictions", default = int(0))
    
    parser.add_argument(
        "--input_path", 
        type=str, 
        required=True,
        help="Path to the input data directory or file"
    )

    parser.add_argument(
        "--output_path", 
        type=str, 
        required=True,
        help="Path to the directory for saving results"
    )

    parser.add_argument(
        "--run_id", 
        type=int, 
        required=True,
        help="slurm run id"
    )
    
    
    
    return parser.parse_args()

def main():
    
    ###########################################################################
    ###########################################################################
    ####################### setting variables and objects #####################
    ###########################################################################
    ###########################################################################
    
    start_perf = time.perf_counter()
    
    args = parse_args()
    num_bins_radialDistance = args.num_bins_radialDistance
    num_bins_polarAngle = args.num_bins_polarAngle
    num_bins_braggintensity = args.num_bins_braggintensity
    
    scan_dim = int(args.scan_dim)
    input_path = str(args.input_path)
    output_path = str(args.output_path)
    run_id = int(args.run_id)
    PAD = int(args.PAD)
    
    isMultitask = args.isMultitask
    
    embed_dim = args.embed_dim
    max_sequence_length = args.max_sequence_length
    
    max_radial_distance = args.max_radial_distance
    
    max_braggIntensity = args.max_braggIntensity
    min_braggIntensity = args.min_braggIntensity
    
    seed = args.seed
    
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    number_of_pixels_in_one_dimension = 128
    k_max = 0.0328 * number_of_pixels_in_one_dimension / 2
    accelerating_voltage = int(300e3)
    crystal = py4DSTEM.process.diffraction.Crystal.from_CIF(input_path + "Cu_fcc.cif")
    crystal.setup_diffraction(accelerating_voltage)
    crystal.calculate_structure_factors(k_max)
    
    crystal.orientation_plan(
                            angle_step_zone_axis = 2,
                            accel_voltage = accelerating_voltage,
                            zone_axis_range = 'auto',
                            )
    device = torch.device("cpu")
    print(f"Using device: {device}")  # print device
    
    end_perf = time.perf_counter()
    elapsed_perf = end_perf - start_perf
    
    print(f"setting parameters and initialzing crystal object: {elapsed_perf:.6f} seconds\n")
    
    ###########################################################################
    ###########################################################################
    ############################ Loading Bragg disks ##########################
    ###########################################################################
    ###########################################################################
    

    start_perf = time.perf_counter()

    
    file_name = "synthetic_bragg_vector_point_list_scan_space_dimension_%d_uncal"%(scan_dim)
    bragg_disks = py4DSTEM.read(input_path + file_name + ".h5")
    
    end_perf = time.perf_counter()
    elapsed_perf = end_perf - start_perf
    
    
    print(f"loading Bragg disk list: {elapsed_perf:.6f} seconds\n")
    
    ###########################################################################
    ###########################################################################
    # 4D Bragg disk array in cartesian coord. to torch tensor in polar coord. #
    ###########################################################################
    ###########################################################################
    
    start_perf = time.perf_counter()
    
    table_of_BraggDisk_qx_qy_intensity_for_eachScanIndex = pre_process_experimental_BraggDisk(bragg_disks, False, False)
    del bragg_disks
    
    
    df = pd.DataFrame(table_of_BraggDisk_qx_qy_intensity_for_eachScanIndex)
    (list_of_Bragg_disks_total, \
     radial_bins, radial_bin_centers, \
     angle_bins, angle_bin_centers, \
     intensity_bins, intensity_bin_centers) = process_pandas_tabular_data(
                                                     df, 
                                                     num_bins_radialDistance, 
                                                     num_bins_polarAngle, 
                                                     num_bins_braggintensity, 
                                                     max_sequence_length,
                                                     max_radial_distance,
                                                     max_braggIntensity)
    
    del table_of_BraggDisk_qx_qy_intensity_for_eachScanIndex
    
    
    ###############################################################################
    ######## STEP 1. ADD [PAD] tokens and SHUFFLE processed data
    
    list_of_Bragg_disks_total = torch.nn.utils.rnn.pad_sequence(
                                                        list_of_Bragg_disks_total, 
                                                        batch_first=True, 
                                                        padding_value = PAD)
    
    del df
    
    end_perf = time.perf_counter()
    elapsed_perf = end_perf - start_perf
    
    print(f"change coordinate and make torch tensor: {elapsed_perf:.6f} seconds\n")
    
    ###########################################################################
    ###########################################################################
    ############################# Model prediction ############################
    ###########################################################################
    ###########################################################################
    
    start_perf = time.perf_counter()
    experimental_dataset = ExpDataset(list_of_Bragg_disks_total, transform = None)
    
    exp_loader = DataLoader(
                            experimental_dataset,
                            batch_size = 2048,
                            shuffle = False,
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
    
    # checkpoint = torch.load('best_model_with_transform.pth') # ie, model_best.pth.tar
    checkpoint = torch.load(input_path + 'best_model.pth', map_location=device) # ie, model_best.pth.tar
    model.load_state_dict(checkpoint['model_state_dict'])
    
    end_perf = time.perf_counter()
    elapsed_perf = end_perf - start_perf
    
    print(f"Time for loading models and setting dataloader: {elapsed_perf:.6f} seconds\n")
    
    start_perf = time.perf_counter()
    
    with torch.no_grad():
        rotation_matrices_predicted = predict_rotation_experimental_data(model, exp_loader, device)
    
    
    end_perf = time.perf_counter()
    elapsed_perf = end_perf - start_perf
    print(f"Time for predicting orientations time: {elapsed_perf:.6f} seconds\n")
    
    ###########################################################################
    ###########################################################################
    ############################ Orientation Mapping ##########################
    ###########################################################################
    ###########################################################################
    
    start_perf = time.perf_counter()    
    rotation_matrices_predicted_np = rotation_matrices_predicted.detach().cpu().numpy()
    del rotation_matrices_predicted
    
    # print("rotation_matrices_predicted_np.shape", rotation_matrices_predicted_np.shape)
    
    
    
    rotation_matrices_predicted_np = rotation_matrices_predicted_np.reshape(scan_dim,scan_dim,3,3)
    
    orientation_Map = make_orientation_map_based_on_4D_rotation_matrices(rotation_matrices_predicted_np, crystal)
    
    pickle_filename = output_path + file_name +  '_ID%d_orientation_map_frompy4DSTEM_trans_pred_CPU.pkl'%(run_id)

    # Save the object
    with open(pickle_filename, 'wb') as f:
        pickle.dump(orientation_Map, f)
        
    del model, exp_loader, rotation_matrices_predicted_np, orientation_Map

    
    end_perf = time.perf_counter()
    elapsed_perf = end_perf - start_perf
    print(f"making orientation map: {elapsed_perf:.6f} seconds\n")


if __name__ == "__main__":
    main()
