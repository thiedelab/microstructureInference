#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 21 14:50:43 2025

@author: kwang
"""
import numpy as np
import torch
from orientationMapping.dataAugmentation import  custom_transforms_for_Data_Aug
from torch.utils.data import DataLoader
import os
import argparse
#import torchinfo
from torch.utils.data.sampler import SubsetRandomSampler
from orientationMapping.dataModules import MyDatasetMultiTask
# from orientationMapping.transformerModel import ModelConfig, make_model
from orientationMapping.deepSetModel import ModelConfig, make_model
from orientationMapping.trainer import train, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="information of scan space dimension and number of crystalline grains")
    parser.add_argument("--embed_dim", type = int, help="embedded dimension", default = int(384))
    parser.add_argument("--max_sequence_length", type = int, help="maximum number of allowed tokens", default = int(76))
    parser.add_argument("--min_radial_distance", type = float, help="minimum raidus", default = float(0.45844))    
    parser.add_argument("--max_radial_distance", type = float, help="maximum raidus", default = float(2.99000))
    parser.add_argument("--min_braggIntensity", type = float, help="minimum intensity", default = float(0.000999))    
    parser.add_argument("--max_braggIntensity", type = float, help="maximum intenisty", default = float(1.0))
    parser.add_argument("--num_bins_radialDistance", type = int, help="number of discretized bins for radius dimension", default = int(256))
    parser.add_argument("--num_bins_polarAngle", type = int, help="number of discretized bins for polar angle dimension", default = int(360))
    parser.add_argument("--num_bins_braggintensity", type = int, help="number of discretized bins for intensity dimension", default = int(256))
    parser.add_argument("--seed", type = int, help="random number seed for numpy and torch", default = int(22))
    parser.add_argument("--PAD", type = int, help="integer indicating PAD token", default = int(0))
    parser.add_argument("--initial_run", type = bool, help="boolean variable indicating whether this training is the first training run", default = bool(True))    
    parser.add_argument("--num_warmup_epochs", type = int, help="number of epochs for linear warm up learning rate scheduler", default = int(15))
    parser.add_argument("--cos_decay_epoch", type = int, help="number of epochs for cosine decay learning rate scheduler", default = int(300))
    parser.add_argument("--eta_intial", type = float, help="initial learning rate", default = float(0.00008))
    parser.add_argument("--eta_min", type = float, help="minimum learning rate in the last epoch", default = float(5e-7))
    parser.add_argument("--printArg", type = bool, help="boolean variable indicating whether to print all the arguments", default = bool(True))
    parser.add_argument("--printModelInfo", type = bool, help="boolean variable indicating whether to print model architecture", default = bool(True))
    return parser.parse_args()



def main():

    args = parse_args()
    

    num_bins_radialDistance = args.num_bins_radialDistance
    num_bins_polarAngle = args.num_bins_polarAngle
    num_bins_braggintensity = args.num_bins_braggintensity
    
    embed_dim = args.embed_dim
    max_sequence_length = args.max_sequence_length
    
    min_radial_distance = args.min_radial_distance
    max_radial_distance = args.max_radial_distance
    
    min_braggIntensity = args.min_braggIntensity
    max_braggIntensity = args.max_braggIntensity
    
    seed = args.seed
    PAD = args.PAD
    initial_run = args.initial_run
    
    num_warmup_epochs = args.num_warmup_epochs
    cos_decay_epoch = args.cos_decay_epoch
    
    
    eta_intial = args.eta_intial
    eta_min = args.eta_min
    
    
    initial_run = args.initial_run
    num_epochs = int(num_warmup_epochs + cos_decay_epoch + 1)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    file_path = os.getcwd() + "/"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("torch device", device, "\n")
    
    if args.printArg:
        
        print("Arguments passed:")
        for arg, value in vars(args).items():
            print(f"  {arg}: {value}")
    
    print("")
    
    
    
    
    radial_bins = np.linspace(0.0, max_radial_distance + (min_radial_distance*0.05), num_bins_radialDistance + 1)
    radial_bin_centers = (radial_bins[:-1] + radial_bins[1:]) / 2
    
    angle_bins = np.arange(-np.pi - np.pi/360., np.pi + np.pi/360., np.pi/180.)
    angle_bin_centers = (angle_bins[:-1] + angle_bins[1:]) / 2
    angle_bins[-1] = np.pi + np.pi/360 # further change the last element
    
    intensity_bins = np.linspace(0.0, max_braggIntensity + (min_braggIntensity*0.05), num_bins_braggintensity + 1)
    intensity_bin_centers = (intensity_bins[:-1] + intensity_bins[1:]) / 2
    
    
    radial_bins = torch.tensor(radial_bins, dtype = torch.float32)
    radial_bin_centers = torch.tensor(radial_bin_centers, dtype = torch.float32)
    
    angle_bins = torch.tensor(angle_bins, dtype = torch.float32)
    angle_bin_centers = torch.tensor(angle_bin_centers, dtype = torch.float32)
    
    intensity_bins = torch.tensor(intensity_bins, dtype = torch.float32)
    intensity_bin_centers = torch.tensor(intensity_bin_centers, dtype = torch.float32)
    
    
    transforms = custom_transforms_for_Data_Aug(
                                                        num_bins_radialDistance, 
                                                        num_bins_polarAngle, 
                                                        num_bins_braggintensity
                                                        )
    
    
    train_transforms = transforms
    val_transforms = transforms
    
    
    
    
    
    
    ####################################################################################
    ############ Set dataset and dataloader that can be used as input of torch Dataset class.
    
    
    kwang_dataset_train = MyDatasetMultiTask(file_path + "entire_Bradd_disks_padded_train.npy", file_path + "orientation_canonical_labels_train.npy", file_path + "entire_mirror_train.npy", num_bins_polarAngle,  transform = train_transforms)
    kwang_dataset_val = MyDatasetMultiTask(file_path + "entire_Bradd_disks_padded_valid.npy", file_path + "orientation_canonical_labels_valid.npy", file_path + "entire_mirror_valid.npy", num_bins_polarAngle, transform = val_transforms)
    
    
    train_indices = np.arange(len(kwang_dataset_train))
    np.random.shuffle(train_indices)
    train_sampler = SubsetRandomSampler(train_indices)
    
    
    train_loader = DataLoader(
                            kwang_dataset_train,
                            batch_size = 1024,
                            sampler = train_sampler,
                            # shuffle = True,
                            num_workers = 16,
                            pin_memory=torch.cuda.is_available(),
    
                             )
    
    val_loader = DataLoader(
                            kwang_dataset_val,
                            batch_size = 1024,
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
                         N_encoder = 1,
                         max_seq_len = max_sequence_length,
                         dropout = 0.005
                         )
    
    model = make_model(config)
    
    if args.printModelInfo:
        from torchinfo import summary
        summary(model);
    
    print("")
        
    optimizer = torch.optim.AdamW(model.parameters(), lr = eta_intial)
    
    
    
    linear_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1 / num_warmup_epochs, end_factor=1.0, total_iters = num_warmup_epochs - 1, last_epoch=-1)
    cos_decay     = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max= cos_decay_epoch, eta_min=eta_min)
    
    if initial_run == True:
        start_epoch = 0
    else:    
        checkpoint_path = file_path + 'best_model.pth'
        model, optimizer, cos_decay, start_epoch = load_checkpoint(model, optimizer, cos_decay, checkpoint_path, device)

    
    file_path_o = file_path + "output/"
    
    train_error, valid_error = train(model, train_loader, val_loader, num_epochs, optimizer, linear_warmup, cos_decay, num_warmup_epochs, cos_decay_epoch,  device, file_path_o, PAD = PAD, start_epoch = start_epoch)
    #train_error, valid_error = train(model, train_loader, val_loader, num_epochs, optimizer, linear_warmup, cos_decay, num_warmup_epochs, cos_decay_epoch,  device, file_path, PAD = PAD, start_epoch = 0)
    
    train_error = np.save(file_path_o + "train_error.npy", np.array(train_error))
    valid_error = np.save(file_path_o + "valid_error.npy", np.array(valid_error))
    
    print("Hello world!")
    
    PATH = file_path_o + "supervisedReggresion_last.pt"
    torch.save(model.state_dict(), PATH)

if __name__ == "__main__":
    main()
