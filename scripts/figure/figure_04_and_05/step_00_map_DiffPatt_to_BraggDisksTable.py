#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 15:19:35 2025

@author: kwang
"""

import numpy as np
import argparse
import py4DSTEM
from microstructure_inference.dataProcessing import read_4D, alignment, detect_Bragg_Disks_in_4DSTEM_data_normalize_intensity

def parse_args():
    parser = argparse.ArgumentParser(description="information of scan space dimension and number of crystalline grains")
    parser.add_argument("--correlationThresholdTemplateMatch", type = int, help="scan space dimension", default = int(10000))   
    return parser.parse_args()

def main():
    
    args = parse_args()
    correlation_thresh_for_templ_match = args.correlationThresholdTemplateMatch
    
    correlation_thresh_for_direct_beam = 8e4
    diffGauss_sigma1 = 2 
    diffGauss_sigma2 = 6
    
    file_path = "./"
    BraggDisks_list_file_name_str = 'bragg_disks_corThForK%d_dog_sig1_%3.2f_sig2_%3.2f_cortThForTemp_%d'%(correlation_thresh_for_direct_beam, diffGauss_sigma1, diffGauss_sigma2, correlation_thresh_for_templ_match)
    BraggDisks_list_file_name = BraggDisks_list_file_name_str +".h5"
    BraggDisks_list_filepath = file_path + BraggDisks_list_file_name
    

    dirpath = "/home/kwang/Desktop/Storage/project/p03_orientation_mapping/figure/experimental_raw_data/"
    filepath_data = dirpath + 'scan_x256_y256.raw'
    
    data = read_4D(filepath_data)
    nan_pos_data = np.isnan(data)
    data[nan_pos_data] = 1
    aligned_data = alignment(data)
    aligned_data = aligned_data[0]
    aligned_data = aligned_data[:,:150]
    
    bragg_vectors = detect_Bragg_Disks_in_4DSTEM_data_normalize_intensity(
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