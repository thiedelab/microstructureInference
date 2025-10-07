#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 15:19:35 2025

@author: kwang
"""

import numpy as np
from modules_detect_BraggDisks import detect_Bragg_Disks_in_4DSTEM_data
import read_and_align_4DSTEM   as rA
import argparse
import py4DSTEM

def parse_args():
    parser = argparse.ArgumentParser(description="information of scan space dimension and number of crystalline grains")
    parser.add_argument("--correlationThresholdTemplateMatch", type = int, help="scan space dimension", default = int(12000))   
    return parser.parse_args()

def main():
    
    args = parse_args()
    correlation_thresh_for_templ_match = args.correlationThresholdTemplateMatch
    
    correlation_thresh_for_direct_beam = 8e4
    diffGauss_sigma1 = 2 
    diffGauss_sigma2 = 8
    
    file_path = "./"
    # file_name_braggdisks = file_path + 'm2_bragg_disks_corThForK%d_dog_sig1_%3.2f_sig2_%3.2f_cortThForTemp_%d.h5'%(correlation_thresh_for_direct_beam, diffGauss_sigma1, diffGauss_sigma2, correlation_thresh_for_templ_match)
    BraggDisks_list_file_name_str = 'm2_bragg_disks_corThForK%d_dog_sig1_%3.2f_sig2_%3.2f_cortThForTemp_%d'%(correlation_thresh_for_direct_beam, diffGauss_sigma1, diffGauss_sigma2, correlation_thresh_for_templ_match)
    BraggDisks_list_file_name = BraggDisks_list_file_name_str +".h5"
    BraggDisks_list_filepath = file_path + BraggDisks_list_file_name
    

    dirpath = "/home/kwang/Desktop/Storage/project/p03_orientation_mapping/experimental_data/4D-STEM_T-control_Echem_copper/PAD03_40kx_-3V_-40C_256pixels_8ms_good/"
    filepath_data = dirpath + 'scan_x256_y256.raw'
    
    data = rA.read_4D(filepath_data)
    nan_pos_data = np.isnan(data)
    data[nan_pos_data] = 1
    aligned_data = rA.alignment(data)
    aligned_data = aligned_data[0]
    aligned_data = aligned_data[:,:150]
    
    bragg_vectors = detect_Bragg_Disks_in_4DSTEM_data(
                        aligned_data, 
                        file_path,
                        correlation_thresh_for_templ_match = correlation_thresh_for_templ_match,
                        correlation_thresh_for_direct_beam = correlation_thresh_for_direct_beam,
                        diffGauss_sigma1 = diffGauss_sigma1, 
                        diffGauss_sigma2 = diffGauss_sigma2,
                        )
    
    py4DSTEM.save(
        BraggDisks_list_filepath,
        bragg_vectors,
        mode='o',
    )
    
    print("saved detected py4DSTEM Bragg Vectors to h5 file\n")
    print("file name:", BraggDisks_list_file_name, "\n\n")
    

    del bragg_vectors

if __name__ == "__main__":
    main()