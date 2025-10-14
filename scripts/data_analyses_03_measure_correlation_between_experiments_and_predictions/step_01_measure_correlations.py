import py4DSTEM
import numpy as np
import read_and_align_4DSTEM as r4D
import os
import time
import pickle
import argparse
from correlation_metric import assignment_cost_pairwise_distance, pyxem_correlation_metric

def parse_args():
    parser = argparse.ArgumentParser(description="information of scan space dimension and number of crystalline grains")
    parser.add_argument("--correlationThresholdTemplateMatch", type = int, help="scan space dimension", default = int(12000))   
    return parser.parse_args()


def main():
    args = parse_args()
    
    correlationThresholdTemplateMatch = args.correlationThresholdTemplateMatch
    
    
    start_perf = time.perf_counter()
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("")
    # print("torch device:", device, "\n")
    
    
    
    file_path = os.getcwd() + "/"
    raw_data = "/home/kwang/Desktop/Storage/project/p03_orientation_mapping/experimental_data/4D-STEM_T-control_Echem_copper/PAD03_40kx_-3V_-40C_256pixels_8ms_good/scan_x256_y256.raw"

    data = r4D.read_4D(raw_data)
    nan_pos_data = np.isnan(data)
    data[nan_pos_data] = 1
    aligned_data = r4D.alignment(data)
    aligned_data = aligned_data[0]
    aligned_data = aligned_data[:,:150]
    # datacube_aligned_data = py4DSTEM.DataCube(aligned_data)
    del data
    # del aligned_data
    print("\n")
    
    Bragg_vector_file_name = "m2_bragg_disks_corThForK80000_dog_sig1_2.00_sig2_8.00_cortThForTemp_%d"%(correlationThresholdTemplateMatch)
    filepath_braggdisks_cal = file_path + Bragg_vector_file_name + ".h5"
    bragg_disks = py4DSTEM.read(filepath_braggdisks_cal)
    prediced_rotation_matrices = np.load(file_path + Bragg_vector_file_name + "_rotation_matrices.npy")
    with open(file_path + Bragg_vector_file_name + '_preProcessed_dictionray.pkl', 'rb') as f:
        table_BPs_scanIndex = pickle.load(f)
    
    
    end_perf = time.perf_counter()
    elapsed_perf = end_perf - start_perf
    print("\n")
    print(f"Loading Bragg disk list and predicted rotation matrices: {elapsed_perf:.6f} seconds\n\n")
    
    
    start_perf = time.perf_counter()
    
    pixel_size = 0.0328
    sigma_compare = 0.02
    pixel_numbers = 128
    
    k_max = pixel_size * pixel_numbers / 2.
    accelerating_voltage = int(300e3)
    crystal = py4DSTEM.process.diffraction.Crystal.from_CIF(file_path + "Cu_fcc.cif")
    crystal.setup_diffraction(accelerating_voltage)
    crystal.calculate_structure_factors(
        k_max,
    )
    crystal.orientation_plan(
        angle_step_zone_axis = 4,
        angle_step_in_plane = 4,
        # corr_kernel_size= 0.08, # was 0.08 before 0.12 not bad
        zone_axis_range='auto',
    )
    
        
    
    end_perf = time.perf_counter()
    elapsed_perf = end_perf - start_perf
    
    print("")    
    print(f"Initialized py4DSTEM crystal object and created template library using py4DSTEM ACOM orientation_plan: {elapsed_perf:.6f} seconds\n\n")
    
    
    
    start_perf = time.perf_counter()
    
    linearAssign_cost_dontconsiderInten_py4DSTEM = []
    linearAssign_cost_dontconsiderInten_transformer = []
    
    py4DSTEM_predicted_rotation_matrices = []
    
    py4DSTEMCorrScore_considerInten_py4DSTEM = []
    py4DSTEMCorrScore_considerInten_transformer = []
    
    pyxemCorrScore_considerInten_py4DSTEM = []
    pyxemCorrScore_considerInten_transformer = []
    for key, value in table_BPs_scanIndex.items():

        i, j = value['scanIndices'][0], value['scanIndices'][1]
    


        
    
        orientation = crystal.match_single_pattern(
            bragg_disks.cal[i,j],
        #     plot_corr = True,
            # num_matches_return = 406,
        #     plot_polar = False,
            verbose = False,
        )


        crystal_for_cor = py4DSTEM.process.diffraction.Crystal.from_CIF(file_path + "Cu_fcc.cif")
        crystal_for_cor.setup_diffraction(accelerating_voltage)
        crystal_for_cor.calculate_structure_factors(k_max)
        crystal_for_cor.orientation_plan_customized(orientation.matrix[0],
                                                   accel_voltage = accelerating_voltage,
                                                    angle_step_in_plane = 1,
                                                    angle_step_zone_axis = 1.0,
                                                    zone_axis_range='auto',
                                                    progress_bar = False)
        py4DSTEM_predicted_rotation_matrices.append(orientation.matrix[0])

        orientation_py4D, corr_full, corr_full_inv  = crystal_for_cor.match_single_pattern_customized(
                                                                                bragg_disks.cal[i,j],
                                                                                verbose = False,
                                                                                                )

        # print("py4DSTEM correlation score:", orientation_py4D.corr[0])
        del crystal_for_cor

        
        bragg_disks_fit_py = crystal.generate_diffraction_pattern(
                            orientation_matrix = orientation.matrix[0],
                            ind_orientation=0,
                            sigma_excitation_error=sigma_compare)

        exp_stacks = np.stack((bragg_disks.cal[i,j].data['qx'], bragg_disks.cal[i,j].data['qy'])).T
        py4DSTEM_prediction_stack = np.stack((bragg_disks_fit_py.data['qx'], bragg_disks_fit_py.data['qy'])).T
        lin_assign_cost = assignment_cost_pairwise_distance(exp_stacks,  py4DSTEM_prediction_stack)
        # print("py4DSTEM linear assignment cost", lin_assign_cost)
        py4DSTEM_pyxem_correlation_score = pyxem_correlation_metric(aligned_data[i, j],bragg_disks_fit_py, k_max, pixel_numbers)
        linearAssign_cost_dontconsiderInten_py4DSTEM.append(lin_assign_cost)
        pyxemCorrScore_considerInten_py4DSTEM.append(py4DSTEM_pyxem_correlation_score)

        # print("######################################################## py4DSTEM orientation prediction\n", orientation.matrix[0])
    


        rotation_matrix = prediced_rotation_matrices[key]
    
    
        
        # print("")
        # print("######################################################## transformer orientation prediction\n", rotation_matrix)

        crystal_for_cor = py4DSTEM.process.diffraction.Crystal.from_CIF(file_path + "Cu_fcc.cif")
        crystal_for_cor.setup_diffraction(accelerating_voltage)
        crystal_for_cor.calculate_structure_factors(k_max)
        crystal_for_cor.orientation_plan_customized(rotation_matrix,
                                                   accel_voltage = accelerating_voltage,
                                                    corr_kernel_size= pixel_size * 3., # was 0.08 before 0.12 not bad
                                                    angle_step_in_plane = 1,
                                                    angle_step_zone_axis = 1.0,
                                                    zone_axis_range='auto',
                                                    progress_bar = False)


        orientation_trans, corr_full, corr_full_inv  = crystal_for_cor.match_single_pattern_customized(
                                                                                bragg_disks.cal[i,j],
                                                                                verbose = False,
                                                                                                )

        
    
        bragg_disks_fit = crystal.generate_diffraction_pattern(
                            orientation_matrix = rotation_matrix,
                            ind_orientation=0,
                            sigma_excitation_error=sigma_compare)


        exp_stacks = np.stack((bragg_disks.cal[i,j].data['qx'], bragg_disks.cal[i,j].data['qy'])).T
        trans_prediction_stack = np.stack((bragg_disks_fit.data['qx'], bragg_disks_fit.data['qy'])).T
        lin_assign_cost = assignment_cost_pairwise_distance(exp_stacks,  trans_prediction_stack)
        transformer_pyxem_correlation_score = pyxem_correlation_metric(aligned_data[i, j],bragg_disks_fit, k_max, pixel_numbers)
        linearAssign_cost_dontconsiderInten_transformer.append(lin_assign_cost)            
        
        pyxemCorrScore_considerInten_transformer.append(transformer_pyxem_correlation_score)


        py4DSTEMCorrScore_considerInten_py4DSTEM.append(orientation_py4D.corr[0])
        py4DSTEMCorrScore_considerInten_transformer.append(orientation_trans.corr[0])

    py4DSTEM_predicted_rotation_matrices = np.array(py4DSTEM_predicted_rotation_matrices)
    linearAssign_cost_dontconsiderInten_transformer = np.array(linearAssign_cost_dontconsiderInten_transformer)
    linearAssign_cost_dontconsiderInten_py4DSTEM = np.array(linearAssign_cost_dontconsiderInten_py4DSTEM)
    
    py4DSTEMCorrScore_considerInten_py4DSTEM = np.array(py4DSTEMCorrScore_considerInten_py4DSTEM)
    py4DSTEMCorrScore_considerInten_transformer = np.array(py4DSTEMCorrScore_considerInten_transformer)
    
    pyxemCorrScore_considerInten_py4DSTEM = np.array(pyxemCorrScore_considerInten_py4DSTEM)
    pyxemCorrScore_considerInten_transformer = np.array(pyxemCorrScore_considerInten_transformer)

    np.save('py4DSTEM_predicted_rotation_matrices_%d.npy'%(correlationThresholdTemplateMatch), py4DSTEM_predicted_rotation_matrices)
    np.save('linearAssign_cost_dontconsiderInten_transformer_%d.npy'%(correlationThresholdTemplateMatch), linearAssign_cost_dontconsiderInten_transformer)
    np.save('linearAssign_cost_dontconsiderInten_py4DSTEM_%d.npy'%(correlationThresholdTemplateMatch), linearAssign_cost_dontconsiderInten_py4DSTEM)
    np.save('py4DSTEMCorrScore_considerInten_py4DSTEM_%d.npy'%(correlationThresholdTemplateMatch), py4DSTEMCorrScore_considerInten_py4DSTEM)
    np.save('py4DSTEMCorrScore_considerInten_transformer_%d.npy'%(correlationThresholdTemplateMatch), py4DSTEMCorrScore_considerInten_transformer)
    np.save('pyxemCorrScore_considerInten_py4DSTEM_%d.npy'%(correlationThresholdTemplateMatch), pyxemCorrScore_considerInten_py4DSTEM)
    np.save('pyxemCorrScore_considerInten_transformer_%d.npy'%(correlationThresholdTemplateMatch), pyxemCorrScore_considerInten_transformer)




if __name__ == "__main__":
    main()