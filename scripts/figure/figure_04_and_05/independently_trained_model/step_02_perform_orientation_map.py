
import py4DSTEM
import numpy as np
import torch
import os
import time
import matplotlib.pyplot as plt
from microstructure_inference.dataProcessing import  make_orientation_map_based_on_4D_rotation_matrices
import argparse
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description="information of scan space dimension and number of crystalline grains")
    parser.add_argument("--accelerating_voltage", type = int, help="accelerating voltage of electron microscopy simulation", default = int(300e3))
    parser.add_argument("--diff_pixel_size", type = float, help="size of pixel of diffraction pattern", default = float(0.0328))
    parser.add_argument("--diff_pixel_numbers", type = int, help="number of pixels in a dimension of diffraction pattern", default = int(128))
    parser.add_argument("--correlationThresholdTemplateMatch", type = int, help="scan space dimension", default = int(14000))
    parser.add_argument("--seed", type = int, help="random number seed for numpy and torch", default = int(22))
    parser.add_argument("--printArg", action="store_true", help="print all arguments")
    parser.add_argument("--trainedTrial", type = str, help="index that indicate independetly trained model", default = str("sing_02"))
    parser.add_argument("--perform_py4DSTEM_orientation_map", action="store_true",help="perform py4DSTEM orientation mapping of given Bragg PointList using ACOM")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    correlationThresholdTemplateMatch = args.correlationThresholdTemplateMatch
    accelerating_voltage = int(args.accelerating_voltage)

    pixel_size = args.diff_pixel_size
    pixel_numbers = args.diff_pixel_numbers
    
    trained_model_indicator_index = args.trainedTrial

    k_max = pixel_size * pixel_numbers / 2.
    
    seed = args.seed
    
    start_perf = time.perf_counter()
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("")
    print("torch device:", device, "\n")
    

    model_path = os.getcwd() + "/"
    processed_data_saving_path = model_path + "processed_data/"
    image_output_path = model_path + "figure_04_panel_a_c/"
    file_path = model_path + "../"
    Cu_cif = "Cu_fcc.cif"
    Bragg_vector_file_name = "bragg_disks_corThForK80000_dog_sig1_2.00_sig2_6.00_cortThForTemp_%d"%(correlationThresholdTemplateMatch)
    # torch_input = torch.load(procssed_data_saving_path + Bragg_vector_file_name + "_table.pt")
    prediced_rotation_matrices = np.load(processed_data_saving_path + trained_model_indicator_index + "_" + Bragg_vector_file_name + "_rotation_matrices_canonical.npy")
    filepath_braggdisks_cal = file_path + Bragg_vector_file_name + ".h5"
    bragg_disks = py4DSTEM.read(filepath_braggdisks_cal)
    
    
    end_perf = time.perf_counter()
    elapsed_perf = end_perf - start_perf
    print("\n")
    print(f"Loading Bragg disk list: {elapsed_perf:.6f} seconds\n\n")
    
    
    
        
    
    table_BPs_scanIndex = {}
    dict_idx = 0
    for i in range(bragg_disks.shape[0]):
        for j in range(bragg_disks.shape[1]):
            if len(bragg_disks.cal[i,j].qx) > 2:
    
                qx = np.copy(bragg_disks.cal[i,j].data["qx"])
                qy = np.copy(bragg_disks.cal[i,j].data["qy"])
                intensity = np.copy(bragg_disks.cal[i,j].data["intensity"] / np.max(bragg_disks.cal[i,j].data["intensity"]))
    
                k_radial_distnaces_of_BPs = np.linalg.norm(np.stack((qx, qy)).T, axis = 1)
                index_of_direct_beam = np.argmin(k_radial_distnaces_of_BPs)
    
                qx = np.delete(qx, index_of_direct_beam)
                qy = np.delete(qy, index_of_direct_beam)
                intensity = np.delete(intensity, index_of_direct_beam)
    
                
    
                positions_of_Bragg_disks = np.stack((qx, qy)).T
                k_radial_distnaces_of_BPs = np.linalg.norm(positions_of_Bragg_disks, axis = 1)
                polar_angles = np.arctan2(positions_of_Bragg_disks[:,1], positions_of_Bragg_disks[:,0])
    
    
                table_BPs_scanIndex[dict_idx] = {'input': np.stack((k_radial_distnaces_of_BPs, polar_angles, intensity)).T, 'scanIndices': [i,j]}
    
                dict_idx += 1
                


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
    
    rotation_map_4D = np.zeros((256,150,3,3))

    for key, value in table_BPs_scanIndex.items():
        
    
        i, j = value['scanIndices'][0], value['scanIndices'][1]
        
        rotation_map_4D[i,j] = prediced_rotation_matrices[key]
    
    orientation_map = make_orientation_map_based_on_4D_rotation_matrices(rotation_map_4D, crystal)
    
    images_orientation, fig, ax  = crystal.plot_orientation_maps(
        orientation_map,
        # orientation_ind=0,
        # symmetry_order = 6,
        corr_range = [0.9, 1.0],
        figsize = (9,9),
        returnfig = True,
    )

    # fig.savefig(image_output_path + trained_model_indicator_index + "_orientation_map_from_transformer_model_cortThForTemp_%d.pdf"%(correlationThresholdTemplateMatch), bbox_inches='tight')
    # fig.savefig(image_output_path + trained_model_indicator_index + "_orientation_map_from_transformer_model_cortThForTemp_%d.png"%(correlationThresholdTemplateMatch), bbox_inches='tight')
    plt.close(fig)
    # plt.show()
    
    if args.perform_py4DSTEM_orientation_map:
    
        py4DSTEM_orientation_map = crystal.match_orientations(
                                                bragg_disks,
                                                )

        py4DSTEM_orientationMapping_pickle_filename = processed_data_saving_path + "orientation_map_from_py4DSTEM_cortThForTemp_%d.pkl"%(correlationThresholdTemplateMatch)

        # Save the py4DSTEM orientation map object
        with open(py4DSTEM_orientationMapping_pickle_filename, 'wb') as f:
            pickle.dump(py4DSTEM_orientation_map, f)
        
        images_orientation, fig, ax  = crystal.plot_orientation_maps(
            orientation_map,
            # orientation_ind=0,
            # symmetry_order = 6,
            corr_range = [0.9, 1.0],
            figsize = (9,9),
            returnfig = True,
        )
        
        images_orientation, fig, ax  = crystal.plot_orientation_maps(
            py4DSTEM_orientation_map,
            # orientation_ind=0,
            # symmetry_order = 6,
            corr_range = [0.9, 1.0],
            figsize = (9,9),
            returnfig = True,
        )
    
        fig.savefig(image_output_path + "orientation_map_from_py4DSTEM_cortThForTemp_%d.pdf"%(correlationThresholdTemplateMatch), bbox_inches='tight')
        fig.savefig(image_output_path + "orientation_map_from_py4DSTEM_cortThForTemp_%d.png"%(correlationThresholdTemplateMatch), bbox_inches='tight')
        plt.close(fig)
        # plt.show()

    
    print("JOB DONE\n\n")

if __name__ == "__main__":
    main()
