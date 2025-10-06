#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 08:20:54 2025

@author: kwang
"""
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import random
from dataAugmentation import applyMirrorOperation, applyMirrorLabels

def digitize_radial_distance(radial_distances, radial_bins):
    return np.digitize(radial_distances, radial_bins) - 1

def digitize_polarAngle(polar_angles, angle_bins):
    return np.digitize(polar_angles, angle_bins) - 1

def digitize_braggIntensity(braggDisk_intensities, intensity_bins):
    return np.digitize(braggDisk_intensities, intensity_bins) - 1

class MyExpDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data.long()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]

        if self.transform:
            x = self.transform(x)

        return x

    def __len__(self):
        return len(self.data)

class MyDatasetMultiTask(Dataset):
    def __init__(self, input_file, rot_file, mir_file, angle_bin_center_num, transform=None):
        self.data = np.load(input_file, mmap_mode='r')
        self.target_rot = np.load(rot_file, mmap_mode='r')
        self.target_mir = np.load(mir_file, mmap_mode='r')
        self.transform = transform
        # self.applyMirrorOperation = applyMirrorOperation(angle_bin_center_num)
        # self.applyMirrorLabels = applyMirrorLabels()

    def __getitem__(self, index):
        x = self.data[index]
        y_rot = self.target_rot[index]
        y_mir = self.target_mir[index]

        # Convert numpy arrays to PyTorch tensors
        x = torch.tensor(x, dtype=torch.int64)
        y_rot = torch.tensor(y_rot, dtype=torch.float32)
        y_mir = torch.tensor(y_mir, dtype=torch.float32)


        if self.transform:
            x = self.transform(x)

        return x, y_rot, y_mir

    def __len__(self):
        return len(self.data)


# backup    
# class MyDatasetMultiTask(Dataset):
#     def __init__(self, data, target_rot, target_mir, angle_bin_center_num, transform=None):
#         self.data = data.long()
#         self.target_rot = target_rot
#         self.target_mir = target_mir
#         self.transform = transform
#         self.applyMirrorOperation = applyMirrorOperation(angle_bin_center_num)
#         self.applyMirrorLabels = applyMirrorLabels()

#     def __getitem__(self, index):
#         x = self.data[index]
#         y_rot = self.target_rot[index]
#         y_mir = self.target_mir[index]

#         if random.random() > 0.5:
#             # print("x\n", x)
#             # print("y", y)

#             x = self.applyMirrorOperation.apply_mirror(x)
#             y_mir = self.applyMirrorLabels.mirrorLabel(y_mir)

#             # print("x\n", x)
#             # print("y", y)

#         if self.transform:
#             x = self.transform(x)

#         return x, y_rot, y_mir

#     def __len__(self):
#         return len(self.data)

class MyDatasetMirPredTask(Dataset):
    def __init__(self, data, target, angle_bin_center_num, transform=None):
        self.data = data.long()
        self.target = target
        self.transform = transform
        self.applyMirrorOperation = applyMirrorOperation(angle_bin_center_num)
        self.applyMirrorLabels = applyMirrorLabels()

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        if random.random() > 0.5:
            # print("x\n", x)
            # print("y", y)

            x = self.applyMirrorOperation.apply_mirror(x)
            y = self.applyMirrorLabels.mirrorLabel(y)

            # print("x\n", x)
            # print("y", y)

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)


class MyDatasetRotPredTask(Dataset):
    def __init__(self, data, target1, transform=None):
        self.data = data.long()
        self.target1 = target1
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y1 = self.target1[index]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y1
    
    def __len__(self):
        return len(self.data)




def sign_consistency_loss(output, labels):
    label_signs = torch.sign(labels[:,:,2])
    label_sign_tot = torch.unsqueeze((torch.sign(torch.sum(label_signs, dim = 1)) +1)/2, dim = 1)
    return F.binary_cross_entropy_with_logits(torch.unsqueeze(output[:,0], dim = 1), label_sign_tot)

def reshapeOutputToRotationMatrix(output):
    """
    For each input data, the neural network model outputs
    6-dimensional vector (6D). This function use the 6D vector to
    create a rotation matrix with shape (3,3). The rotation matrix
    is a desired label (continuous value) for the input data.
    
    Args:
        output:
                    output of a neural network model.
                    torch.Tensor with shape (batch_size, 6)
    returns:
        rotation_matrices:
                    output further processed into 
                    rotation matrices.
                    torch.Tensor with shape (batch_size, 3, 3)        
    
    """
    isMirrorSymm = torch.unsqueeze(((output[:,0] >= 0.0).float() * 2.) - 1., dim = 1)
    
    reshaped_output = torch.reshape(
                                    output[:,1:],
                                    (output.shape[0], 2, 3)
                                    )
    reshaped_output_vectorSoftPlused = torch.stack(((F.celu(reshaped_output[:,0]) + 1.) * isMirrorSymm, reshaped_output[:,1]), dim = 1)
    normalized_reshaped_output = F.normalize(
                                    reshaped_output_vectorSoftPlused,
                                    p = 2, 
                                    dim = 2
                                    )
    R3 = F.normalize(
                    torch.cross(
                                normalized_reshaped_output[:,0], 
                                normalized_reshaped_output[:,1], 
                                dim = 1
                                ), 
                    dim = 1
                    )
    
    ## R2
    R2 = torch.cross(R3, normalized_reshaped_output[:,0], dim = 1)
    # print("R2[1]\n", R2[1], "\n")
    
    stacked = torch.stack((R2, R3, normalized_reshaped_output[:,0]), dim = 1)
    rotation_matrices = torch.transpose(stacked, 1, 2)
    # print("rotation_matrices[1]\n", rotation_matrices[1], "\n")

    return rotation_matrices


def symmetric_orthogonalization(x):
    """Maps 9D input vectors onto SO(3) via symmetric orthogonalization.
        x: should have size [batch_size, 9]

        Output has size [batch_size, 3, 3], where each inner 3x3 matrix is in SO(3).
    """
    m = x.view(-1, 3, 3)
    u, s, v = torch.svd(m)
    vt = torch.transpose(v, 1, 2)


    det = torch.det(torch.matmul(u, vt))
    det = det.view(-1, 1, 1)
    vt = torch.cat((vt[:, :2, :], vt[:, -1:, :] * det), 1)
    r = torch.matmul(u, vt)
    return r
