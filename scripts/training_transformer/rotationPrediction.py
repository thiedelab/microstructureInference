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
#import torchinfo
from torch.utils.data.sampler import SubsetRandomSampler
from orientationMapping.dataModules import MyDatasetMultiTask
from orientationMapping.transformerModel import ModelConfig, make_model
from orientationMapping.trainer import train
###############################################################################
######## STEP 0. SET PARAMETERS

seed = 22
torch.manual_seed(seed)
np.random.seed(seed)

def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Start from the next epoch
    start_epoch = checkpoint['epoch'] + 1

    # Resume scheduler with last_epoch as the last epoch in the checkpoint
    scheduler.last_epoch = checkpoint['epoch']

    return model, optimizer, scheduler, start_epoch

file_path = os.getcwd() + "/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

PAD = 0
num_feature = 9

num_bins_radialDistance = int(256)
num_bins_polarAngle = int(360)
num_bins_braggintensity = int(256)

embed_dim = int(384)
max_sequence_length = 76

min_radial_distance = 0.45844
max_radial_distance = 2.99000

min_braggIntensity = 0.004999
max_braggIntensity = 1.0


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

# list_of_Bragg_disks_padded_train = torch.load(file_path + "Bragg_disks_padded_train.pt")
# labels_train = torch.load(file_path + "labels_train.pt")
# labels_train = labels_train.float()

# labels_train_m = torch.zeros((list_of_Bragg_disks_padded_train.shape[0], 1))


# list_of_Bragg_disks_padded_val = torch.load(file_path + "Bragg_disks_padded_valid.pt")
# labels_val =  torch.load(file_path + "labels_valid.pt")
# labels_val = labels_val.float()

# labels_val_m = torch.zeros((list_of_Bragg_disks_padded_val.shape[0], 1))

# print("list_of_Bragg_disks_padded_train.shape", list_of_Bragg_disks_padded_train.shape)
# print("list_of_Bragg_disks_padded_val.shape", list_of_Bragg_disks_padded_val.shape)
# print("")
# print("labels_train.shape", labels_train.shape)
# print("labels_val.shape", labels_val.shape)
# print("")

# print("list_of_Bragg_disks_padded_train.requires_grad", list_of_Bragg_disks_padded_train.requires_grad)

###############################################################################
############ Define transforms, torch Dataset, and torch Dataloader.

# train_transforms = custom_transforms_for_Data_Aug(
#                                                     num_bins_radialDistance, 
#                                                     num_bins_polarAngle, 
#                                                     num_bins_braggintensity
#                                                     )
# val_transforms = train_transforms

transforms = custom_transforms_for_Data_Aug(
                                                    num_bins_radialDistance, 
                                                    num_bins_polarAngle, 
                                                    num_bins_braggintensity
                                                    )


train_transforms = transforms
val_transforms = transforms






####################################################################################
############ Set dataset and dataloader that can be used as input of torch Dataset class.


kwang_dataset_train = pPd.MyDatasetMultiTask("./entire_Bradd_disks_padded_train.npy", "./orientation_canonical_labels_train.npy", "./entire_mirror_train.npy", num_bins_polarAngle,  transform = train_transforms)
kwang_dataset_val = pPd.MyDatasetMultiTask("./entire_Bradd_disks_padded_valid.npy", "./orientation_canonical_labels_valid.npy", "./entire_mirror_valid.npy", num_bins_polarAngle, transform = val_transforms)

#kwang_dataset_train = pPd.MyDataset(list_of_Bragg_disks_padded_train, labels_train, transform = None)
#kwang_dataset_val = pPd.MyDataset(list_of_Bragg_disks_padded_val, labels_val, transform = None)

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
                     num_feature = num_feature,
                     h = 8,
                     N_encoder = 3,
                     max_seq_len = max_sequence_length,
                     dropout = 0.005
                     )

model = make_model(config)
#print(torchinfo.summary(model))
optimizer = torch.optim.AdamW(model.parameters(), lr = 0.00008)

num_epochs = 316
num_warmup_epochs = int(15)

eta_min = 5e-7

cos_decay_epoch = 300

linear_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1 / num_warmup_epochs, end_factor=1.0, total_iters = num_warmup_epochs - 1, last_epoch=-1)
cos_decay     = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max= cos_decay_epoch, eta_min=eta_min)

checkpoint_path = './best_model_r1.pth'
model, optimizer, scheduler, start_epoch = load_checkpoint(model, optimizer, cos_decay, checkpoint_path, device)
# Initialize best validation loss

file_path_o = "/tmp/kwang/diM01/"

train_error, valid_error = train(model, train_loader, val_loader, num_epochs, optimizer, linear_warmup, cos_decay, num_warmup_epochs, cos_decay_epoch,  device, file_path_o, PAD = PAD, start_epoch = start_epoch)
#train_error, valid_error = train(model, train_loader, val_loader, num_epochs, optimizer, linear_warmup, cos_decay, num_warmup_epochs, cos_decay_epoch,  device, file_path, PAD = PAD, start_epoch = 0)

train_error = np.save(file_path_o + "train_error.npy", np.array(train_error))
valid_error = np.save(file_path_o + "valid_error.npy", np.array(valid_error))

print("Hello world!")

PATH = file_path_o + "supervisedReggresion_last.pt"
torch.save(model.state_dict(), PATH)
