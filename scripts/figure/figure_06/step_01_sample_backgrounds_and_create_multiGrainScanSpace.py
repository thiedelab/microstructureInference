#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 13:40:45 2025

@author: kwang
"""
import numpy as np
from modules_generate_synthethic_4DSTEM import generate_grains
import py4DSTEM
import argparse
from microstructure_inference.dataProcessing import read_4D, alignment, detect_Bragg_Disks_in_4DSTEM_data_normalize_intensity

def parse_args():
    parser = argparse.ArgumentParser(description="information of scan space dimension and number of crystalline grains")
    parser.add_argument("--num_grains", type = int, help="number of distinct crystal grains", default = int(60))
    parser.add_argument("--ScanDimension", type = int, help="scan space dimension", default = int(128))
 
    
    return parser.parse_args()


def main():
    args = parse_args()
    num_grains = args.num_grains
    syn_2D_scanSpace_map_side_dimension = args.ScanDimension


    correlation_thresh_for_templ_match = 4000
    correlation_thresh_for_direct_beam = 8e4
    diffGauss_sigma1 = 2 
    diffGauss_sigma2 = 6
    
    
    experimental_4DSTEM_dirpath = "/home/kwang/Desktop/Storage/project/p03_orientation_mapping/figure/experimental_raw_data/"
    experimental_4DSTEM_filepath = experimental_4DSTEM_dirpath + 'scan_x256_y256.raw'
    
    file_path = "./"
    # file_name_braggdisks = file_path + 'm2_bragg_disks_corThForK%d_dog_sig1_%3.2f_sig2_%3.2f_cortThForTemp_%d.h5'%(correlation_thresh_for_direct_beam, diffGauss_sigma1, diffGauss_sigma2, correlation_thresh_for_templ_match)
    BraggDisks_list_file_name_str = 'bragg_disks_corThForK%d_dog_sig1_%3.2f_sig2_%3.2f_cortThForTemp_%d'%(correlation_thresh_for_direct_beam, diffGauss_sigma1, diffGauss_sigma2, correlation_thresh_for_templ_match)
    BraggDisks_list_file_name = BraggDisks_list_file_name_str +".h5"
    BraggDisks_list_filepath = file_path + BraggDisks_list_file_name
    
    ##############################################################################
    ##############################################################################
    ####################### READ exp 4DSTEM data (START) #########################
    ##############################################################################
    ##############################################################################
    
    print("")
    print("Action 1. READING experimental 4DSTEM data (START)\n")
    data = read_4D(experimental_4DSTEM_filepath)
    nan_pos_data = np.isnan(data)
    data[nan_pos_data] = 1
    aligned_data = alignment(data)
    aligned_data = aligned_data[0]
    aligned_data = aligned_data[:,:150]
    
    print("\n")
    print("Action 1. READING experimental 4DSTEM data (END)\n\n")
    
    ##############################################################################
    ##############################################################################
    ######################## READ exp 4DSTEM data (END) ##########################
    ##############################################################################
    ##############################################################################
    
    
    ##############################################################################
    ##############################################################################
    ### SAVE detected bragg disks for each diffraction pattern in data (START) ###
    ##############################################################################
    ##############################################################################
    
    print("Action 2. Detecting Bragg disks using py4DSTEM module and saving the detected outcome (START)\n\n")
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
    
    
    print("Action 2. Detecting Bragg disks using py4DSTEM module and saving the detected outcome (END)\n")
    
    ##############################################################################
    ##############################################################################
    ### SAVE detected bragg disks for each diffraction pattern in data (END) ###
    ##############################################################################
    ##############################################################################
    
    ##############################################################################
    ##############################################################################
    #### collect correlation kernel & background diffraction patterns (START) ####
    ##############################################################################
    ##############################################################################
    
    print("Action 3. SAVING direction beam as correlation kernel and saving background diffraction patterns (START)\n")
    
    
    centers = []
    diffraction_pattern_side_dim = aligned_data[0,0].shape[0]
    DPImgcenter = int(diffraction_pattern_side_dim / 2)
    collection_pixel_nums = 8
    cut_off = 8.5e4
    
    
    index_of_backgrounds = []
    backgrounds = []
    
    for i in range(aligned_data.shape[0]):
        for j in range(aligned_data.shape[1]):
            if len(bragg_vectors.cal[i,j].qx) <= 1:
                index_of_backgrounds.append([i,j])
                current_diffPatt = aligned_data[i,j]
                current_center = current_diffPatt[DPImgcenter - collection_pixel_nums:DPImgcenter + collection_pixel_nums, DPImgcenter - collection_pixel_nums:DPImgcenter + collection_pixel_nums]
                centers.append(current_center)
                backgrounds.append(aligned_data[i,j])
    
    centers = np.array(centers)
    centers_av = np.average(centers, axis = 0)
    
    correlation_kernel = np.zeros((diffraction_pattern_side_dim, diffraction_pattern_side_dim))
    correlation_kernel[DPImgcenter - collection_pixel_nums:DPImgcenter + collection_pixel_nums, DPImgcenter - collection_pixel_nums:DPImgcenter + collection_pixel_nums] = centers_av 
    correlation_kernel = np.where(correlation_kernel > cut_off, correlation_kernel, 0)
    
    np.save(BraggDisks_list_file_name_str + "_direct_beam_kernel.npy", correlation_kernel)
    
    backgrounds = np.array(backgrounds)
    np.save(BraggDisks_list_file_name_str +"_backgrounds.npy", backgrounds)
    
    del aligned_data
    del backgrounds
    del bragg_vectors
    
    print("Action 3. SAVING direction beam as correlation kernel and saving background diffraction patterns (END)\n")
    
    ##############################################################################
    ##############################################################################
    #### collect correlation kernel & background diffraction patterns (END) ####
    ##############################################################################
    ##############################################################################
    
    
    ##############################################################################
    ##############################################################################
    ######### Generate synthetic 2D scan map of crystal grains (START) ###########
    ##############################################################################
    ##############################################################################
    
    print("Action 4. generate synthetic 2D scan map (START)\n")
    
    # Generate and visualize
    syn_2D_scanSpace_map = generate_grains(size = syn_2D_scanSpace_map_side_dimension, num_grains = num_grains, boundary_thickness=1)
    
    np.save("./syn_2D_scanSpace_map_side_dim%d_numGrains%d"%(syn_2D_scanSpace_map_side_dimension, num_grains), syn_2D_scanSpace_map)
    
    print("Action 4. generate synthetic 2D scan map (END)\n")
    
    ##############################################################################
    ##############################################################################
    ########## Generate synthetic 2D scan map of crystal grains (END) ############
    ##############################################################################
    ##############################################################################
    
if __name__ == "__main__":
    main()