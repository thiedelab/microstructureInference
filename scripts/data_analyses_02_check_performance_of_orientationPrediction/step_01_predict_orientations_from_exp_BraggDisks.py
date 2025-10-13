import py4DSTEM
import numpy as np
import pandas as pd
import torch
import os
import time
import pickle
from orientationMapping.dataModules import MyExpDataset
from orientationMapping.transformerModel import ModelConfig, make_model
from modules_for_mapping_exper_Bragg_disks_to_orientations import process_pandas_tabular_data,  pre_process_experimental_BraggDisk, map_exp_BraggDisk_to_orienation_via_Transformer
from torch.utils.data import DataLoader
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="information of scan space dimension and number of crystalline grains")
    parser.add_argument("--correlationThresholdTemplateMatch", type = int, help="scan space dimension", default = int(12000))   
    parser.add_argument("--embed_dim", type = int, help="embedded dimension", default = int(384))
    parser.add_argument("--max_sequence_length", type = int, help="maximum number of allowed tokens", default = int(76))
    parser.add_argument("--min_radial_distance", type = float, help="minimum raidus", default = float(0.45844))    
    parser.add_argument("--max_radial_distance", type = float, help="maximum raidus", default = float(2.99000))
    parser.add_argument("--min_braggIntensity", type = float, help="minimum intensity", default = float(0.000999))    
    parser.add_argument("--max_braggIntensity", type = float, help="maximum intenisty", default = float(1.0))
    parser.add_argument("--num_bins_radialDistance", type = int, help="number of discretized bins for radius dimension", default = int(256))
    parser.add_argument("--num_bins_polarAngle", type = int, help="number of discretized bins for polar angle dimension", default = int(360))
    parser.add_argument("--num_bins_braggintensity", type = int, help="number of discretized bins for intensity dimension", default = int(256))
    parser.add_argument("--seed", type = int, help="random number seed for numpy and torch", default = int(42))
    parser.add_argument("--PAD", type = int, help="integer indicating PAD token", default = int(0))
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    correlationThresholdTemplateMatch = args.correlationThresholdTemplateMatch

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
    
    start_perf = time.perf_counter()
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("")
    print("torch device:", device, "\n")
    
    file_path = os.getcwd() + "/"
    Bragg_vector_file_name = "m2_bragg_disks_corThForK80000_dog_sig1_2.00_sig2_8.00_cortThForTemp_%d"%(correlationThresholdTemplateMatch)
    filepath_braggdisks_cal = file_path + Bragg_vector_file_name + ".h5"
    bragg_disks = py4DSTEM.read(filepath_braggdisks_cal)
    
    
    end_perf = time.perf_counter()
    elapsed_perf = end_perf - start_perf
    print("\n")
    print(f"Loading Bragg disk list: {elapsed_perf:.6f} seconds\n\n")
    
    
    start_perf = time.perf_counter()
    #### NORMALIZE ENTIRE STACK
    
    table_of_BraggDisk_qx_qy_intensity_for_eachScanIdx = pre_process_experimental_BraggDisk(bragg_disks)
    
    with open(file_path + Bragg_vector_file_name + '_preProcessed_dictionray.pkl', 'wb') as f:
        pickle.dump(table_of_BraggDisk_qx_qy_intensity_for_eachScanIdx, f)
    
    
    df = pd.DataFrame(table_of_BraggDisk_qx_qy_intensity_for_eachScanIdx)
    df.to_json(file_path + Bragg_vector_file_name + '_preProcessed_df.json', index=True)
    
    ###############################################################################
    
    (list_of_Bragg_disks_total, \
     radial_bins, radial_bin_centers, \
     angle_bins, angle_bin_centers, \
     intensity_bins, intensity_bin_centers) = process_pandas_tabular_data(
                                                        df, 
                                                        num_bins_radialDistance, 
                                                        num_bins_polarAngle, 
                                                        num_bins_braggintensity, 
                                                        max_sequence_length,
                                                        min_radial_distance,
                                                        max_radial_distance,
                                                        min_braggIntensity,
                                                        max_braggIntensity)
    
    
    
    ###############################################################################
    ######## STEP 1. ADD [PAD] tokens and SHUFFLE processed data
    
    list_of_Bragg_disks_total = torch.nn.utils.rnn.pad_sequence(
                                                        list_of_Bragg_disks_total, 
                                                        batch_first=True, 
                                                        padding_value = 0)
    
    #print("list_of_Bragg_disks_total.shape", list_of_Bragg_disks_total.shape)
    #print("list_of_Bragg_disks_total\n", list_of_Bragg_disks_total, "\n")
    
    torch.save(list_of_Bragg_disks_total, file_path + Bragg_vector_file_name + "_table.pt")
    
    
    end_perf = time.perf_counter()
    elapsed_perf = end_perf - start_perf
    
    print(f"Mapping Bragg disk list to torch table tensor inputs: {elapsed_perf:.6f} seconds\n\n")
    
    
    
    start_perf = time.perf_counter()
    experimental_dataset = MyExpDataset(list_of_Bragg_disks_total, transform = None)
    
    exp_loader = DataLoader(
                            experimental_dataset,
                            batch_size = 2048,
                            shuffle = False,
                            num_workers = 10,
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
                         dropout = 0.0
                         )
    
    
    model = make_model(config)
    
    
    # checkpoint = torch.load('best_model_with_transform.pth') # ie, model_best.pth.tar
    checkpoint = torch.load('best_model.pth', map_location=torch.device('cpu')) # ie, model_best.pth.tar
    model.load_state_dict(checkpoint['model_state_dict'])
    
    end_perf = time.perf_counter()
    elapsed_perf = end_perf - start_perf
    
    print(f"Time for loading models and setting dataloader: {elapsed_perf:.6f} seconds\n\n")
    
    start_perf = time.perf_counter()
    
    rotation_matrices_predicted = map_exp_BraggDisk_to_orienation_via_Transformer(model, exp_loader, device)
    rotation_matrices_predicted_np = rotation_matrices_predicted.detach().cpu().numpy()
    
    np.save(file_path + Bragg_vector_file_name + "_rotation_matrices.npy", rotation_matrices_predicted_np)
    
    end_perf = time.perf_counter()
    elapsed_perf = end_perf - start_perf
    print(f"Time for predicting orientations time: {elapsed_perf:.6f} seconds\n\n")
    
    print("JOB DONE\n\n")

if __name__ == "__main__":
    main()