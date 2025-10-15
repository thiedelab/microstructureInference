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


def digitized_bin_centers(num_bins_radialDistance,
                          min_radial_distance,
                          max_radial_distance,
                          num_bins_polarAngle,
                          num_bins_braggintensity,
                          min_braggIntensity,
                          max_braggIntensity
                          ):
    
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
    
    return intensity_bin_centers, angle_bin_centers
