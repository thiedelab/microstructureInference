#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 23 14:49:36 2025

@author: kwang
"""

import numpy as np
import torch
import argparse
import os
from microstructure_inference.dataModules import DataSetPointGroup_rotation, digitized_bin_centers
from microstructure_inference.dataProcessing import predict_rotation_sim_data_with_labels
from microstructure_inference.transformerModel import ModelConfig, make_model

from torch.utils.data import DataLoader

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
    
    # print("np.min(intensity_val_extended)", np.min(intensity_val_extended))
    
    # max_radial_distance = np.max(radial_distance_extended)
    # min_radial_distance = np.min(radial_distance_extended)
    
    # max_braggIntensity = np.max(intensity_val_extended)
    # min_braggIntensity = np.min(intensity_val_extended)
    
    # print("")
    
    # print("number_of_tokens_in_sequences", number_of_tokens_in_sequences)
    
    # print("max_braggIntensity", max_braggIntensity)
    # print("min_braggIntensity", min_braggIntensity)
    # print("intensity_tolerance", intensity_tolerance, "\n")
    
    
    # print("max_radial_distance", max_radial_distance)
    # print("radial_distance_tolerance", radial_distance_tolerance,"\n")
    
    
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



def parse_args():
    parser = argparse.ArgumentParser(description="information for digitizing the Bragg disk features into discrete bins")
    parser.add_argument("--TrialVal", type = int, help="displacement trial index", default = int(1))
    parser.add_argument("--cartScale", type = float, help="scaling factor for cartesian displacements.", default = float(0.200))
    parser.add_argument("--intenScale", type = float, help="scaling factor for intensity displacements", default = float(0.040))
    parser.add_argument("--input_dir", type = str, help="directory for input data", default = str("/home/kwang/Desktop/Storage/project/p03_orientation_mapping/figure/figure_02/revision_sensitivity_analyses/output/noised/"))
    parser.add_argument("--output_dir", type = str, help="directory for saving digitized output tables", default = str("/home/kwang/Desktop/Storage/project/p03_orientation_mapping/figure/figure_02/revision_sensitivity_analyses/output/noised_digitized/"))
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
    parser.add_argument("--printArg", type = bool, help="boolean variable indicating whether to print all the arguments", default = bool(False))
    parser.add_argument("--printModelInfo", type = bool, help="boolean variable indicating whether to print model architecture", default = bool(True))
    parser.add_argument("--datasetString", type = str, help="boolean variable indicating whether to print model architecture", default = str("valid"))
    return parser.parse_args()




def main():

    args = parse_args()

    TrialValue = int(args.TrialVal)
    cartScale = float(args.cartScale)
    intenScale = float(args.intenScale)

    input_dir = str(args.input_dir)
    output_dir = str(args.output_dir)



    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # os.environ['TORCH_USE_CUDA_DSA'] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)

    PAD = 0


    num_bins_radialDistance = int(256)
    num_bins_polarAngle = int(360)
    num_bins_braggintensity = int(64)

    max_sequence_length = 76

    # file_path = os.getcwd() + "/"
    label_path = "/home/kwang/Desktop/Storage/project/p03_orientation_mapping/figure/figure_02/revision_sensitivity_analyses/output/"


    file_name = "postProc5_input_array_noised_cartesianOnly_cartScale%3.2f_seed%d"%(cartScale, TrialValue)

    BD_input = np.load(input_dir + file_name + '.npy')

    # print("BD_input.shape", BD_input.shape)


    num_diffraction_patterns = len(BD_input)
    # print("number of diffraction patterns:", num_diffraction_patterns)


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

    # print("list_of_Bragg_disks_total.shape", list_of_Bragg_disks_total.shape)

    output_file_name = file_name + "_digitized"


    np.save(output_dir + output_file_name + ".npy", list_of_Bragg_disks_total.detach().cpu().numpy())

    #############################################################################################################
    #############################################################################################################
    #############################################################################################################
    #############################################################################################################
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

    model_path =  "/home/kwang/Desktop/Storage/project/p03_orientation_mapping/figure/figure_02/independently_trained_model_2/"

    prediction_output_dir = str(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("torch device", device, "\n")

    if args.printArg:

        print("Arguments passed:")
        for arg, value in vars(args).items():
            print(f"  {arg}: {value}")

    # print("")
    
    


    radial_bins, radial_bin_centers, angle_bins, angle_bin_centers, intensity_bin_centers = digitized_bin_centers(
                                                    num_bins_radialDistance,
                                                    max_radial_distance,
                                                    num_bins_polarAngle,
                                                    num_bins_braggintensity,
                                                    max_braggIntensity,
    )
    
    data_set_i = DataSetPointGroup_rotation(output_dir + output_file_name + ".npy", label_path + "postProc2_output_label_orignial.npy", num_bins_polarAngle, transform = None)
    
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
    

    rotation_matrices, geodesic_distance_stack, average_geodesic_error = predict_rotation_sim_data_with_labels(model, data_loader_i, device)
    rotation_matrices_np = rotation_matrices.detach().cpu().numpy()
    geodesic_distance_stack_np = geodesic_distance_stack.detach().cpu().numpy()

    np.save(output_dir + output_file_name + "_geodesic_distances.npy", geodesic_distance_stack_np)
    # np.save(prediction_output_dir + "predicted_rotation_matrices_trained_" + data_set_str + ".npy", rotation_matrices_np)
    # np.save(prediction_output_dir + "geodesic_distances_trained_" + data_set_str +".npy", geodesic_distance_stack_np)
    
    # print("average_geodesic_error", average_geodesic_error)
    # print("geodesic_distance_stack\n", geodesic_distance_stack, "\n")
    # print("geodesic_distance_stack.shape", geodesic_distance_stack.shape)
    
    

    print("JOB DONE.\n\n")

if __name__ == "__main__":
    main()
