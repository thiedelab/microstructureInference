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
from orientationMapping.dataAugmentation import applyMirrorOperation, applyMirrorLabels

def cubic_proper_point_group_operations():

    cubic_point_group_operations = []
    
    
    # 1
    cubic_point_group_operations.append(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]    
        ]
    )
    
    # 2
    cubic_point_group_operations.append(
        [
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]    
        ]
    )
    
    # 3
    cubic_point_group_operations.append(
        [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]    
        ]
    )
    
    # 4
    cubic_point_group_operations.append(
        [
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]    
        ]
    )
    
    # 5
    cubic_point_group_operations.append(
        [
            [0, 0, -1],
            [-1, 0, 0],
            [0, -1, 0]    
        ]
    )
    
    # 6
    cubic_point_group_operations.append(
        [
            [0, -1, 0],
            [0, 0, -1],
            [-1, 0, 0]    
        ]
    )
    
    # 7
    cubic_point_group_operations.append(
        [
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]    
        ]
    )
    
    # 8
    cubic_point_group_operations.append(
        [
            [0, 0, -1],
            [-1, 0, 0],
            [0, 1, 0]    
        ]
    )
    
    # 9
    cubic_point_group_operations.append(
        [
            [0, -1, 0],
            [0, 0, 1],
            [-1, 0, 0]    
        ]
    )
    
    # 10
    cubic_point_group_operations.append(
        [
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, -1]    
        ]
    )
    
    # 11
    cubic_point_group_operations.append(
        [
            [0, -1, 0],
            [-1, 0, 0],
            [0, 0, -1]    
        ]
    )
    
    # 12
    cubic_point_group_operations.append(
        [
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]    
        ]
    )
    
    # 13
    cubic_point_group_operations.append(
        [
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1]    
        ]
    )
    
    # 14
    cubic_point_group_operations.append(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, -1]    
        ]
    )
    
    # 15
    cubic_point_group_operations.append(
        [
            [0, 0, 1],
            [1, 0, 0],
            [0, -1, 0]    
        ]
    )
    
    # 16
    cubic_point_group_operations.append(
        [
            [0, 1, 0],
            [0, 0, -1],
            [1, 0, 0]    
        ]
    )
    
    # 17
    cubic_point_group_operations.append(
        [
            [0, -1, 0],
            [-1, 0, 0],
            [0, 0, 1]    
        ]
    )
    
    # 18
    cubic_point_group_operations.append(
        [
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1]    
        ]
    )
    
    # 19
    cubic_point_group_operations.append(
        [
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, -1]    
        ]
    )
    
    # 20
    cubic_point_group_operations.append(
        [
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, -1]    
        ]
    )
    
    # 21
    cubic_point_group_operations.append(
        [
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, -1]    
        ]
    )
    
    # 22
    cubic_point_group_operations.append(
        [
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0]    
        ]
    )
    
    # 23
    cubic_point_group_operations.append(
        [
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0]    
        ]
    )
    
    # 24
    cubic_point_group_operations.append(
        [
            [0, 0, 1],
            [0, -1, 0],
            [1, 0, 0]    
        ]
    )
    
    # 25
    cubic_point_group_operations.append(
        [
            [0, 0, -1],
            [0, -1, 0],
            [-1, 0, 0]    
        ]
    )
    
    # 26
    cubic_point_group_operations.append(
        [
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0]    
        ]
    )
    
    # 27
    cubic_point_group_operations.append(
        [
            [0, 0, -1],
            [0, 1, 0],
            [1, 0, 0]    
        ]
    )
    
    # 28
    cubic_point_group_operations.append(
        [
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]    
        ]
    )
    
    # 29
    cubic_point_group_operations.append(
        [
            [0, 0, -1],
            [1, 0, 0],
            [0, 1, 0]    
        ]
    )
    
    # 30
    cubic_point_group_operations.append(
        [
            [0, 1, 0],
            [0, 0, 1],
            [-1, 0, 0]    
        ]
    )
    
    # 31
    cubic_point_group_operations.append(
        [
            [0, 0, -1],
            [0, 1, 0],
            [-1, 0, 0]    
        ]
    )
    
    # 32
    cubic_point_group_operations.append(
        [
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0]    
        ]
    )
    
    # 33
    cubic_point_group_operations.append(
        [
            [0, 0, -1],
            [0, -1, 0],
            [1, 0, 0]    
        ]
    )
    
    # 34
    cubic_point_group_operations.append(
        [
            [0, 0, 1],
            [0, -1, 0],
            [-1, 0, 0]    
        ]
    )
    
    # 35
    cubic_point_group_operations.append(
        [
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]    
        ]
    )
    
    # 36
    cubic_point_group_operations.append(
        [
            [0, 0, -1],
            [1, 0, 0],
            [0, -1, 0]    
        ]
    )
    
    # 37
    cubic_point_group_operations.append(
        [
            [0, 1, 0],
            [0, 0, -1],
            [-1, 0, 0]    
        ]
    )
    
    # 38
    cubic_point_group_operations.append(
        [
            [-1, 0, 0],
            [0, 0, 1],
            [0, 1, 0]    
        ]
    )
    
    # 39
    cubic_point_group_operations.append(
        [
            [-1, 0, 0],
            [0, 0, -1],
            [0, -1, 0]    
        ]
    )
    
    # 40
    cubic_point_group_operations.append(
        [
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]    
        ]
    )
    
    # 41
    cubic_point_group_operations.append(
        [
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]    
        ]
    )
    
    # 42
    cubic_point_group_operations.append(
        [
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]    
        ]
    )
    
    # 43
    cubic_point_group_operations.append(
        [
            [0, 0, 1],
            [-1, 0, 0],
            [0, 1, 0]    
        ]
    )
    
    # 44
    cubic_point_group_operations.append(
        [
            [0, -1, 0],
            [0, 0, 1],
            [1, 0, 0]    
        ]
    )
    
    # 45
    cubic_point_group_operations.append(
        [
            [1, 0, 0],
            [0, 0, -1],
            [0, -1, 0]    
        ]
    )
    
    # 46
    cubic_point_group_operations.append(
        [
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0]    
        ]
    )
    
    # 47
    cubic_point_group_operations.append(
        [
            [-1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]    
        ]
    )
    
    # 48
    cubic_point_group_operations.append(
        [
            [-1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]    
        ]
    )
    
    cubic_point_group_operations = np.array(cubic_point_group_operations)
   
    proper_cubic_point_group_operations = []

    for idx,mat in enumerate(cubic_point_group_operations):
        if np.linalg.det(mat) > 0.0:
            proper_cubic_point_group_operations.append(mat)
    
    proper_cubic_point_group_operations = np.array(proper_cubic_point_group_operations)
    proper_cubic_point_group_operations_pt = torch.from_numpy(proper_cubic_point_group_operations).float()    
    
    return proper_cubic_point_group_operations_pt
    

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

class DataSetPointGroup_rotation(Dataset):
    def __init__(self, input_file, rot_file, angle_bin_center_num, transform=None):
        self.data = np.load(input_file, mmap_mode='r')
        self.target_rot = np.load(rot_file, mmap_mode='r')
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y_rot = self.target_rot[index]

        # Convert numpy arrays to PyTorch tensors
        x = torch.tensor(x, dtype=torch.int64)
        y_rot = torch.tensor(y_rot, dtype=torch.float32)


        if self.transform:
            x = self.transform(x)

        return x, y_rot

    def __len__(self):
        return len(self.data)


class DataSetPointGroup_rotation_and_phase(Dataset):
    def __init__(self, input_file, rot_file, phase_file, angle_bin_center_num, transform=None):
        self.data = np.load(input_file, mmap_mode='r')
        self.target_rot = np.load(rot_file, mmap_mode='r')
        self.target_phase = np.load(phase_file, mmap_mode='r')
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y_rot = self.target_rot[index]
        y_phase = self.target_phase[index]

        # Convert numpy arrays to PyTorch tensors
        x = torch.tensor(x, dtype=torch.int64)
        y_rot = torch.tensor(y_rot, dtype=torch.float32)
        y_phase = torch.tensor(y_phase, dtype=torch.float32)


        if self.transform:
            x = self.transform(x)

        return x, y_rot, y_phase

    def __len__(self):
        return len(self.data)


def digitized_bin_centers(num_bins_radialDistance,
                          max_radial_distance,
                          num_bins_polarAngle,
                          num_bins_braggintensity,
                          max_braggIntensity,
                          radial_distance_tolerance = 0.0001,
                          intensity_tolerance = 0.0001,
                          ):
    
    radial_bins = np.linspace(0.0, max_radial_distance + (radial_distance_tolerance), num_bins_radialDistance + 1)
    radial_bin_centers = (radial_bins[:-1] + radial_bins[1:]) / 2
    
    angle_bins = np.arange(-np.pi - np.pi/360., np.pi + np.pi/360., np.pi/180.)
    angle_bin_centers = (angle_bins[:-1] + angle_bins[1:]) / 2
    angle_bins[-1] = np.pi + np.pi/360 # further change the last element
    
    intensity_bins = np.linspace(0.0, max_braggIntensity + (intensity_tolerance), num_bins_braggintensity + 1)
    intensity_bin_centers = (intensity_bins[:-1] + intensity_bins[1:]) / 2
    
    
    radial_bins = torch.tensor(radial_bins, dtype = torch.float32)
    radial_bin_centers = torch.tensor(radial_bin_centers, dtype = torch.float32)
    
    angle_bins = torch.tensor(angle_bins, dtype = torch.float32)
    angle_bin_centers = torch.tensor(angle_bin_centers, dtype = torch.float32)
    
    intensity_bins = torch.tensor(intensity_bins, dtype = torch.float32)
    intensity_bin_centers = torch.tensor(intensity_bin_centers, dtype = torch.float32)
    
    return radial_bins, radial_bin_centers, angle_bins, angle_bin_centers, intensity_bin_centers
