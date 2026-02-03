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
from microstructure_inference.dataModules import DataSetPointGroup_rotation, digitized_bin_centers
from microstructure_inference.dataProcessing import predict_rotation_sim_data_with_labels
from microstructure_inference.transformerModel import ModelConfig, make_model
from microstructure_inference.LossFunctions import pointGroup_map_rotation_prediction, symmetric_orthogonalization
from microstructure_inference.dataModules import cubic_proper_point_group_operations

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
    parser.add_argument("--num_warmup_epochs", type = int, help="number of epochs for linear warm up learning rate scheduler", default = int(15))
    parser.add_argument("--cos_decay_epoch", type = int, help="number of epochs for cosine decay learning rate scheduler", default = int(250))
    parser.add_argument("--eta_intial", type = float, help="initial learning rate", default = float(0.00007))
    parser.add_argument("--eta_min", type = float, help="minimum learning rate in the last epoch", default = float(5e-7))
    parser.add_argument("--initial_run",action="store_true",help="whether this training is the first training run")
    parser.add_argument("--printArg",action="store_true",help="print all arguments")
    parser.add_argument("--printModelInfo",action="store_true",help="print model architecture")
    return parser.parse_args()


def main():

    args = parse_args()


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


    file_path = os.getcwd() + "/../"
    model_path = os.getcwd() + "/"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("torch device", device, "\n")

    if args.printArg:

        print("Arguments passed:")
        for arg, value in vars(args).items():
            print(f"  {arg}: {value}")

    print("")



    radial_bins, radial_bin_centers, angle_bins, angle_bin_centers, intensity_bin_centers = digitized_bin_centers(
                                                    num_bins_radialDistance,
                                                    max_radial_distance,
                                                    num_bins_polarAngle,
                                                    num_bins_braggintensity,
                                                    max_braggIntensity,
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
    
    torch.save(model.state_dict(), model_path + "untrained_model.pth")
    
    


if __name__ == "__main__":
    main()
