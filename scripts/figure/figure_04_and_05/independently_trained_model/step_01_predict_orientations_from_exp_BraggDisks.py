import py4DSTEM
import numpy as np
import pandas as pd
import torch
import os
import time
import pickle
from microstructure_inference.dataModules import ExpDataset, cubic_proper_point_group_operations
from microstructure_inference.dataProcessing import pre_process_experimental_BraggDisk, predict_rotation_experimental_data, return_predicted_rotation_matrices_in_canonical_form, process_pandas_tabular_data
from microstructure_inference.transformerModel import ModelConfig, make_model
from torch.utils.data import DataLoader
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="information of scan space dimension and number of crystalline grains")
    parser.add_argument("--embed_dim", type = int, help="embedded dimension", default = int(384))
    parser.add_argument("--accelerating_voltage", type = int, help="accelerating voltage of electron microscopy simulation", default = int(300e3))
    parser.add_argument("--diff_pixel_size", type = float, help="size of pixel of diffraction pattern", default = float(0.0328))
    parser.add_argument("--diff_pixel_numbers", type = int, help="number of pixels in a dimension of diffraction pattern", default = int(128))


    parser.add_argument("--correlationThresholdTemplateMatch", type = int, help="scan space dimension", default = int(15000))
    parser.add_argument("--max_sequence_length", type = int, help="maximum number of allowed tokens", default = int(76))
    parser.add_argument("--max_radial_distance", type = float, help="maximum raidus", default = float(2.99000))
    parser.add_argument("--max_braggIntensity", type = float, help="maximum intenisty", default = float(1.0))
    parser.add_argument("--num_bins_radialDistance", type = int, help="number of discretized bins for radius dimension", default = int(256))
    parser.add_argument("--num_bins_polarAngle", type = int, help="number of discretized bins for polar angle dimension", default = int(360))
    parser.add_argument("--num_bins_braggintensity", type = int, help="number of discretized bins for intensity dimension", default = int(64))
    parser.add_argument("--isMultitask", type = int, help="integer_indicating_multi_predictions", default = int(0))
    parser.add_argument("--seed", type = int, help="random number seed for numpy and torch", default = int(22))
    parser.add_argument("--PAD", type = int, help="integer indicating PAD token", default = int(0))
    parser.add_argument("--printArg", action="store_true", help="print all arguments")
    parser.add_argument("--printModelInfo", action="store_true", help="print all arguments")
    parser.add_argument("--trainedTrial", type = str, help="index that indicate independetly trained model", default = str("sing_05"))
    return parser.parse_args()


def main():
    args = parse_args()
    
    correlationThresholdTemplateMatch = args.correlationThresholdTemplateMatch

    num_bins_radialDistance = args.num_bins_radialDistance
    num_bins_polarAngle = args.num_bins_polarAngle
    num_bins_braggintensity = args.num_bins_braggintensity
    
    embed_dim = args.embed_dim
    max_sequence_length = args.max_sequence_length
    
    isMultitask = int(args.isMultitask)
    max_radial_distance = args.max_radial_distance
    
    max_braggIntensity = args.max_braggIntensity

    trained_model_indicator_index = args.trainedTrial

    accelerating_voltage = int(args.accelerating_voltage)

    pixel_size = args.diff_pixel_size
    pixel_numbers = args.diff_pixel_numbers

    k_max = pixel_size * pixel_numbers / 2.
    
    seed = args.seed
    
    start_perf = time.perf_counter()
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("")
    print("torch device:", device, "\n")
    

    model_path = os.getcwd() + "/"
    procssed_data_saving_path = model_path + "processed_data/"
    file_path = model_path + "../"
    Cu_cif = "Cu_fcc.cif"
    Bragg_vector_file_name = "bragg_disks_corThForK80000_dog_sig1_2.00_sig2_6.00_cortThForTemp_%d"%(correlationThresholdTemplateMatch)
    filepath_braggdisks_cal = file_path + Bragg_vector_file_name + ".h5"
    bragg_disks = py4DSTEM.read(filepath_braggdisks_cal)
    
    
    end_perf = time.perf_counter()
    elapsed_perf = end_perf - start_perf
    print("\n")
    print(f"Loading Bragg disk list: {elapsed_perf:.6f} seconds\n\n")
    
    
    start_perf = time.perf_counter()
    #### NORMALIZE ENTIRE STACK
    
    table_of_BraggDisk_qx_qy_intensity_for_eachScanIdx = pre_process_experimental_BraggDisk(bragg_disks)
    
    with open(procssed_data_saving_path + trained_model_indicator_index + "_" + Bragg_vector_file_name + '_preProcessed_dictionray.pkl', 'wb') as f:
        pickle.dump(table_of_BraggDisk_qx_qy_intensity_for_eachScanIdx, f)
    
    
    df = pd.DataFrame(table_of_BraggDisk_qx_qy_intensity_for_eachScanIdx)
    df.to_json(procssed_data_saving_path + trained_model_indicator_index + "_" + Bragg_vector_file_name + '_preProcessed_df.json', index=True)
    
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
                                                        max_radial_distance,
                                                        max_braggIntensity)
    
    
    
    ###############################################################################
    ######## STEP 1. ADD [PAD] tokens and SHUFFLE processed data
    
    list_of_Bragg_disks_total = torch.nn.utils.rnn.pad_sequence(
                                                        list_of_Bragg_disks_total, 
                                                        batch_first=True, 
                                                        padding_value = 0)
    
    #print("list_of_Bragg_disks_total.shape", list_of_Bragg_disks_total.shape)
    #print("list_of_Bragg_disks_total\n", list_of_Bragg_disks_total, "\n")
    
    torch.save(list_of_Bragg_disks_total, procssed_data_saving_path + trained_model_indicator_index + "_" + Bragg_vector_file_name + "_table.pt")
    
    
    end_perf = time.perf_counter()
    elapsed_perf = end_perf - start_perf
    
    print(f"Mapping Bragg disk list to torch table tensor inputs: {elapsed_perf:.6f} seconds\n\n")
    
    
    
    start_perf = time.perf_counter()
    experimental_dataset = ExpDataset(list_of_Bragg_disks_total, transform = None)
    
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
                         dropout = 0.001,
                         multiTask = isMultitask,
                         )

    
    
    model = make_model(config)
    
    
    # checkpoint = torch.load('best_model_with_transform.pth') # ie, model_best.pth.tar
    checkpoint = torch.load(model_path + 'best_model.pth', map_location=torch.device('cpu')) # ie, model_best.pth.tar
    model.load_state_dict(checkpoint['model_state_dict'])
    
    end_perf = time.perf_counter()
    elapsed_perf = end_perf - start_perf
    
    print(f"Time for loading models and setting dataloader: {elapsed_perf:.6f} seconds\n\n")
    
    start_perf = time.perf_counter()
    
    rotation_matrices_predicted = predict_rotation_experimental_data(model, exp_loader, device)
    rotation_matrices_predicted_np = rotation_matrices_predicted.detach().cpu().numpy()
    
    np.save(procssed_data_saving_path + trained_model_indicator_index + "_" + Bragg_vector_file_name + "_rotation_matrices.npy", rotation_matrices_predicted_np)


    


    crystal = py4DSTEM.process.diffraction.Crystal.from_CIF(file_path + Cu_cif)
    crystal.setup_diffraction(accelerating_voltage)
    crystal.calculate_structure_factors(k_max)
    
    crystal.orientation_plan(
        angle_step_zone_axis = 2,
        angle_step_in_plane = 2,
        accel_voltage = accelerating_voltage,
        # intensity_power = 0.5,
        # corr_kernel_size= 0.08, # was 0.08 before 0.12 not bad
        zone_axis_range='auto',
    )


    cubic_proper = cubic_proper_point_group_operations().detach().clone().cpu().numpy()
                        
    
    canonical = return_predicted_rotation_matrices_in_canonical_form(crystal, rotation_matrices_predicted_np, cubic_proper)
    
    np.save(procssed_data_saving_path + trained_model_indicator_index + "_" +  Bragg_vector_file_name + "_rotation_matrices_canonical.npy", canonical)
    
    end_perf = time.perf_counter()
    elapsed_perf = end_perf - start_perf
    print(f"Time for predicting orientations time: {elapsed_perf:.6f} seconds\n\n")
    
    print("JOB DONE\n\n")

if __name__ == "__main__":
    main()
