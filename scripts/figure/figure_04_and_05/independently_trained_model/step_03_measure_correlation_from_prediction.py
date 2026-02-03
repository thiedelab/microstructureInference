import py4DSTEM
import numpy as np
import matplotlib.pyplot as plt

# from chamferdist import ChamferDistance
from microstructure_inference.analysis.measure_correlation_score import measure_sparseCorr_of_single_pattern, orientation_plan_init_for_single_pattern, Q_calculation
from py4DSTEM import BraggVectors

import py4DSTEM
import numpy as np
import os
import time
import pickle
import argparse
from microstructure_inference.dataProcessing import read_4D, align

def parse_args():
    parser = argparse.ArgumentParser(description="information of scan space dimension and number of crystalline grains")
    parser.add_argument("--correlationThresholdTemplateMatch", type = int, help="scan space dimension", default = int(14000))   
    parser.add_argument("--trainedTrial", type = str, help="index that indicate independetly trained model", default = str("sing_06"))
    return parser.parse_args()


def main():
    args = parse_args()
    
    correlationThresholdTemplateMatch = args.correlationThresholdTemplateMatch
    trained_model_indicator_index = args.trainedTrial
    
    
    start_perf = time.perf_counter()
    
    trained_model_indicator_index = args.trainedTrial
    
    print("")
        
    script_path = os.getcwd() + "/"
    processed_data_saving_path = script_path + "processed_data/"
    image_output_path = script_path + "figure_04_panel_a_c/"
    file_path = script_path + "../"

    
    raw_file_path = "/home/kwang/Desktop/Storage/project/p03_orientation_mapping/figure/experimental_raw_data/"
    raw_data = raw_file_path + "scan_x256_y256.raw"

    data = read_4D(raw_data)
    nan_pos_data = np.isnan(data)
    data[nan_pos_data] = 1
    aligned_data = align(data)
    aligned_data = aligned_data[0]
    aligned_data = aligned_data[:,:150]
    del data
    print("\n")
    
    Bragg_vector_file_name = "bragg_disks_corThForK80000_dog_sig1_2.00_sig2_6.00_cortThForTemp_%d"%(correlationThresholdTemplateMatch)
    filepath_braggdisks_cal = file_path + Bragg_vector_file_name + ".h5"
    bragg_disks = py4DSTEM.read(filepath_braggdisks_cal)
    
    processed_Bragg_vector_file_name = trained_model_indicator_index + "_" + Bragg_vector_file_name
    prediced_rotation_matrices = np.load(processed_data_saving_path +processed_Bragg_vector_file_name + "_rotation_matrices_canonical.npy")
    
    with open(processed_data_saving_path + processed_Bragg_vector_file_name + '_preProcessed_dictionray.pkl', 'rb') as f:
        table_BPs_scanIndex = pickle.load(f)
    
    
    end_perf = time.perf_counter()
    elapsed_perf = end_perf - start_perf
    print("\n")
    print(f"Loading raw data, Bragg disk Pointlist, and predicted rotation matrices: {elapsed_perf:.6f} seconds\n\n")
    
    
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
        angle_step_zone_axis = 2,
        angle_step_in_plane = 2,
        zone_axis_range='auto',
    )
    
        
    
    end_perf = time.perf_counter()
    elapsed_perf = end_perf - start_perf
    
    print("")    
    print(f"Initialized py4DSTEM crystal object and created template library using py4DSTEM ACOM orientation_plan: {elapsed_perf:.6f} seconds\n\n")
    
    
    
    start_perf = time.perf_counter()
        
    py4DSTEM_predicted_rotation_matrices = []
    
    sparseCorrValue_from_py4DSTEM_prediction = []
    sparseCorrValue_from_transformer_prediction = []
    
    QCorrValue_from_py4DSTEM_prediction = []
    QCorrValue_from_transformer_prediction = []
    for key, value in table_BPs_scanIndex.items():


        i, j = value['scanIndices'][0], value['scanIndices'][1]            

        # print("i, j", i, j)
    
        orientation = crystal.match_single_pattern(
            bragg_disks.cal[i,j],
            verbose = False,
        )


        crystal_for_cor = py4DSTEM.process.diffraction.Crystal.from_CIF(file_path + "Cu_fcc.cif")
        crystal_for_cor.setup_diffraction(accelerating_voltage)
        crystal_for_cor.calculate_structure_factors(k_max)
        
        idx_of_zone_axis_for_replacement, is_input_rotation_matrix_inverted_wrt_canonical = orientation_plan_init_for_single_pattern(
            crystal_for_cor,
            orientation.matrix[0],
            angle_step_zone_axis = 2,
            angle_step_in_plane = 2,
            accel_voltage = 300e3,
            zone_axis_range='auto',
            progress_bar = False,
        )
        
        predicted_orientation_object, py4DSTEM_corr_value = measure_sparseCorr_of_single_pattern(
                            crystal = crystal_for_cor,
                            bragg_peaks = bragg_disks.cal[i,j],
                            inversion_symmetry = is_input_rotation_matrix_inverted_wrt_canonical,
                               )
        
        py4DSTEM_predicted_rotation_matrices.append(orientation.matrix[0])

        
        bragg_disks_fit_py = crystal.generate_diffraction_pattern(
                            orientation_matrix = orientation.matrix[0],
                            ind_orientation=0,
                            sigma_excitation_error=sigma_compare)
        
        py4DSTEM_Q_correlation_score = Q_calculation(aligned_data[i, j],bragg_disks_fit_py, k_max, pixel_numbers)
        QCorrValue_from_py4DSTEM_prediction.append(py4DSTEM_Q_correlation_score)
        sparseCorrValue_from_py4DSTEM_prediction.append(py4DSTEM_corr_value)

        del crystal_for_cor, idx_of_zone_axis_for_replacement, is_input_rotation_matrix_inverted_wrt_canonical, py4DSTEM_corr_value, predicted_orientation_object
        
        
        #######################################################################
        #######################################################################
        #######################################################################
        #######################################################################
        #######################################################################
        #######################################################################
    


        rotation_matrix_trans = prediced_rotation_matrices[key]        
        
        crystal_for_cor = py4DSTEM.process.diffraction.Crystal.from_CIF(file_path + "Cu_fcc.cif")
        crystal_for_cor.setup_diffraction(accelerating_voltage)
        crystal_for_cor.calculate_structure_factors(k_max)
        
        idx_of_zone_axis_for_replacement, is_input_rotation_matrix_inverted_wrt_canonical = orientation_plan_init_for_single_pattern(
            crystal_for_cor,
            rotation_matrix_trans,
            angle_step_zone_axis = 2,
            angle_step_in_plane = 2,
            accel_voltage = 300e3,
            zone_axis_range='auto',
            progress_bar = False,
        )
        
        predicted_orientation_object, transformer_corr_value = measure_sparseCorr_of_single_pattern(
                            crystal = crystal_for_cor,
                            bragg_peaks = bragg_disks.cal[i,j],
                            inversion_symmetry = is_input_rotation_matrix_inverted_wrt_canonical,
                               )    
    
    
        bragg_disks_fit_tr = crystal.generate_diffraction_pattern(
                            orientation_matrix = rotation_matrix_trans,
                            ind_orientation=0,
                            sigma_excitation_error=sigma_compare)

        transformer_Q_correlation_score = Q_calculation(aligned_data[i, j], bragg_disks_fit_tr, k_max, pixel_numbers)
        QCorrValue_from_transformer_prediction.append(transformer_Q_correlation_score)
        sparseCorrValue_from_transformer_prediction.append(transformer_corr_value)
        
        del crystal_for_cor, idx_of_zone_axis_for_replacement, is_input_rotation_matrix_inverted_wrt_canonical, transformer_corr_value, predicted_orientation_object

    py4DSTEM_predicted_rotation_matrices = np.array(py4DSTEM_predicted_rotation_matrices)    
    sparseCorrValue_from_py4DSTEM_prediction = np.array(sparseCorrValue_from_py4DSTEM_prediction)
    sparseCorrValue_from_transformer_prediction = np.array(sparseCorrValue_from_transformer_prediction)
    QCorrValue_from_py4DSTEM_prediction = np.array(QCorrValue_from_py4DSTEM_prediction)
    QCorrValue_from_transformer_prediction = np.array(QCorrValue_from_transformer_prediction)

    np.save(processed_data_saving_path + trained_model_indicator_index + "_" + 'py4DSTEM_predicted_rotation_matrices_%d.npy'%(correlationThresholdTemplateMatch), py4DSTEM_predicted_rotation_matrices)
    np.save(processed_data_saving_path + trained_model_indicator_index + "_" + 'sparseCorrValue_from_py4DSTEM_prediction_%d.npy'%(correlationThresholdTemplateMatch), sparseCorrValue_from_py4DSTEM_prediction)
    np.save(processed_data_saving_path + trained_model_indicator_index + "_" + 'sparseCorrValue_from_transformer_prediction_%d.npy'%(correlationThresholdTemplateMatch), sparseCorrValue_from_transformer_prediction)
    np.save(processed_data_saving_path + trained_model_indicator_index + "_" + 'QCorrValue_from_py4DSTEM_prediction_%d.npy'%(correlationThresholdTemplateMatch), QCorrValue_from_py4DSTEM_prediction)
    np.save(processed_data_saving_path + trained_model_indicator_index + "_" + 'QCorrValue_from_transformer_prediction_%d.npy'%(correlationThresholdTemplateMatch), QCorrValue_from_transformer_prediction)




if __name__ == "__main__":
    main()
