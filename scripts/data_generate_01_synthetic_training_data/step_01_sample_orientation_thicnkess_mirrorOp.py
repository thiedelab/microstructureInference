#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 13:40:45 2025

@author: kwang
"""
import py4DSTEM
import numpy as np
import pickle
from modules_step_01_sample_orientation_thickness import action_01_collect_unique_thickness_for_each_zone_axis
from modules_step_01_sample_orientation_thickness import action_02_compare_DPs_from_different_zone_axes
from modules_step_01_sample_orientation_thickness import action_03_remove_thickness_indices_of_common_patterns_for_lowSymmetry_zone_axes
from modules_step_01_sample_orientation_thickness import action_04_finalize_unique_thickness_resulting_unique_pattern_for_each_ZA
from modules_step_01_sample_orientation_thickness import action_05_sample_representative_thickness_for_each_ZA
from modules_step_01_sample_orientation_thickness import action_06_check_symmetry_for_each_ZA_each_thickness, action_06_check_symmetry_for_each_ZA_each_thickness_monoclinic
from modules_step_01_sample_orientation_thickness import action_07_generate_dictionary_of_orientation_thickness_and_mirrorSymmOper
import time
import argparse



def parse_args():
    parser = argparse.ArgumentParser(description="information of crystal, path of files, and other parameters for sampling thicknesses and orientations")
    parser.add_argument("--crystal", type = str, help="nomenclature of crystal", default = "Cu_fcc")
    parser.add_argument("--directoryPath", type = str, help="path of directory where unit cell cif file is located", default = "./")
    parser.add_argument("--outOfPlaneAngleDisp", type=float, help="out of plane angle displacement used in py4DSTEM orientation plan", default = float(2))
    parser.add_argument("--excitError", type=float, help="excitation error used for simulations", default = float(0.045))
    parser.add_argument("--intensThreshold", type=float, help="This threshold value is used to delete Bragg disks with a relative intensity smaller than it.", default = float(5e-3))
    parser.add_argument("--saveIntermediatePklFiles", type=int, help="whether to save intermediate pikcle files; can be helpful for debug.", default = 0)
    
    return parser.parse_args()

def main():
    
    args = parse_args()
    crystal_name = args.crystal
    crystal_cif = crystal_name + ".cif"
    unit_cell_path = args.directoryPath
    excitation_error = float(args.excitError)
    intensThreshold = float(args.intensThreshold)
    outOfPlaneAngleDisp = args.outOfPlaneAngleDisp
    saveIntermediatePklFiles = args.saveIntermediatePklFiles
    
    
    if crystal_name not in ("Cu_fcc", "Cu2O_cubic", "CuO_monoclinic"):
        raise AssertionError(" '--crystal' argument must be one one of 'Cu_fcc', 'Cu2O_cubic', or 'CuO_monoclinic' .")
        
    
    
    
    # print("")
    # print("Sampling orientations and thickness for crystal ", crystal_name)
    # print("excitation error", excitation_error)
    # print("Threshold value for removing Bragg disks with small relative intensity", intensThreshold)
    # print("out of plane angle displacement", outOfPlaneAngleDisp)
    # print("")
    
    print("")
    print(f"{'Sampling orientations and thickness for crystal':80} {crystal_name}")
    print(f"{'excitation error':80} {excitation_error}")
    print(f"{'Threshold value for removing Bragg disks with small relative intensity':80} {intensThreshold}")
    print(f"{'out of plane angle displacement':80} {outOfPlaneAngleDisp}")
    print("\n\n")
        
    ###############################################################################
    ###############################################################################
    ########################### Set global parameters #############################
    ###############################################################################
    ###############################################################################
    
    start_perf = time.perf_counter()
    print("Setting and initializing parameters, variables, and py4DSTEM crystal object\n\n")
        
    pixel_size = 0.0328
    pixel_numbers = 128
    total_number_for_sampling_thickness = 80
    accelerating_voltage = int(300e3)
    upper_limit_unit_cell_num = 1000
    lower_limit_unit_cell_num = 2
    
    k_max = pixel_size * pixel_numbers / 2.    
    crystal = py4DSTEM.process.diffraction.Crystal.from_CIF(unit_cell_path + crystal_cif)
    unit_cell_length = crystal.lat_real[0][0]
    
    crystal.setup_diffraction(accelerating_voltage)
    crystal.calculate_structure_factors(
        k_max,
    )
    
    crystal.calculate_structure_factors(k_max * 4.)
        
    # Convert the V_g to relativistic-corrected U_g and store in a datastructure optimized
    # for access by the Bloch code
    crystal.calculate_dynamical_structure_factors(
        accelerating_voltage, "WK-CP", k_max=k_max * 4., thermal_sigma=0.08, tol_structure_factor=-1.0
    )
    
    
    # Create an orientation plan for Cu fcc
    crystal.orientation_plan(
        angle_step_zone_axis = outOfPlaneAngleDisp,
        angle_step_in_plane = 1,
        accel_voltage = accelerating_voltage,
        zone_axis_range='auto',
    )
    
    # Save Zone axes
    zone_axes = crystal.orientation_vecs
    
    if crystal_name == "CuO_monoclinic":
        use_functions_for_monoclinic = 1
        zone_axes = zone_axes[:int(-1)]
        print("")
        print("For monoclinic crystal, zone axis [-100] is not considered\n")
    else:
        use_functions_for_monoclinic = 0
    
    print("")    
    print("Number of sampled zone axes", zone_axes.shape[0], "\n\n")
    
    np.save(crystal_name + "_zone_axes_out_of_plane_displacement_%2.1f"%(outOfPlaneAngleDisp) + "_degree.npy", zone_axes)
    
    
    
    
    end_perf = time.perf_counter()
    elapsed_perf = end_perf - start_perf
    print("\n\n")
    print(f"                                                                           execution time: {elapsed_perf:.6f} seconds\n\n")
    
    ###############################################################################
    ###############################################################################
    ############## ACTION 1. For each zone axis, collect thickness ################
    ##############           resulting unique diffraction pattern  ################
    ###############################################################################
    ###############################################################################
    
    start_perf = time.perf_counter()
    print("Action 1. for each zone axis, collect thickness resulting unique diffraction patterns (START)\n\n")
    
    
    total_unique_thicknesses, total_diffraction_patterns_uniques, total_average_intensities = action_01_collect_unique_thickness_for_each_zone_axis(
                                                                            crystal,
                                                                            unit_cell_length,
                                                                            lower_limit_unit_cell_num,
                                                                            upper_limit_unit_cell_num,
                                                                            k_max,
                                                                            zone_axes,
                                                                            excitation_error,
                                                                            intensThreshold,
                                                                            
    )
    
   
    
    print("")
    print("Action 1. for each zone axis collect thickness resulting unique diffraction patterns (END)\n\n")
    
    end_perf = time.perf_counter()
    elapsed_perf = end_perf - start_perf
    print(f"                                                                           execution time: {elapsed_perf:.6f} seconds\n\n")
    
    ###############################################################################
    ###############################################################################
    ######### ACTION 2. collect thickness indices resulting the same and      #####
    #########           unique diffracion patterns unique diffraction pattern #####
    ###############################################################################
    ###############################################################################
    
    start_perf = time.perf_counter()
    print("Action 2. for each zone axis, collect thickness indices resulting overlapping and unique diffraction patterns (START)\n\n")
    
    common_indices_for_each_ZA_pairs, unique_indices_candiate_for_each_ZA = action_02_compare_DPs_from_different_zone_axes(
                                                                                total_diffraction_patterns_uniques,
                                                                                # total_average_intensities,
                                                                                zone_axes,
                                                                                tolerance_for_pattern_matching = 1e-2,
                                                                                decimals_for_setting_tuple = 6,
    )
    
    
    
    print("")
    print("Action 2. for each zone axis, collect thickness indices resulting overlapping and unique diffraction patterns (END)\n\n")
    
    end_perf = time.perf_counter()
    elapsed_perf = end_perf - start_perf
    print(f"                                                                           execution time: {elapsed_perf:.6f} seconds\n\n")
    
    ###############################################################################
    ###############################################################################
    ##### ACTION 3. Remove thickness indices of diffraction patterns ############## 
    #####           common in multiple zone axis                     ##############
    ###############################################################################
    ###############################################################################
    
    start_perf = time.perf_counter()
    print("Action 3. for each zone axis, remove thickness indices resulting overlapping diffraction patterns (START)\n\n")
    
    indices_selectively_assigned_to_each_ZA, indices_selectively_removed_from_each_ZA = action_03_remove_thickness_indices_of_common_patterns_for_lowSymmetry_zone_axes(
        zone_axes,
        total_average_intensities,
        common_indices_for_each_ZA_pairs,
    )
    
    
            
    print("")
    print("Action 3. for each zone axis, remove thickness indices resulting overlapping diffraction patterns (END)\n\n")
    
    end_perf = time.perf_counter()
    elapsed_perf = end_perf - start_perf
    print(f"                                                                           execution time: {elapsed_perf:.6f} seconds\n\n")
    
    ###############################################################################
    ###############################################################################
    ##### ACTION 4. For each zone axis key, assign thickness indices that #########
    #####           could be used for simulations.                        #########
    ###############################################################################
    ###############################################################################
    
    start_perf = time.perf_counter()
    print("Action 4. for each zone axis, assign thickness indices that may be used for simulations (START)\n\n")
    
    ZA_idx_unique_thickness_indices_final_save = action_04_finalize_unique_thickness_resulting_unique_pattern_for_each_ZA(
            zone_axes.shape[0],
            unique_indices_candiate_for_each_ZA,
            indices_selectively_assigned_to_each_ZA,
            )
    
    
    print("")
    print("Action 4. for each zone axis, assign thickness indices that may be used for simulations (END)\n\n")
    
    end_perf = time.perf_counter()
    elapsed_perf = end_perf - start_perf
    print(f"                                                                           execution time: {elapsed_perf:.6f} seconds\n\n")
    
    ###############################################################################
    ###############################################################################
    ##### ACTION 5. For each zone axis key, sample and assign       ###############
    #####           thickness that could be used for simulations.   ###############
    ###############################################################################
    ###############################################################################
    
    start_perf = time.perf_counter()
    print("Action 5. for each zone axis, sample thickness that may be used for simulations (START)\n\n")
    
    final_sampled_thickness_for_each_zone_axis_summary = action_05_sample_representative_thickness_for_each_ZA(
                                                                                                                total_number_for_sampling_thickness,
                                                                                                                ZA_idx_unique_thickness_indices_final_save,
                                                                                                                total_unique_thicknesses,
                                                                                                                total_diffraction_patterns_uniques,
                                                                                                                )
    
    
    
                
    
    print("")
    print("Action 5. for each zone axis, sample thickness that may be used for simulations (END)\n\n")
    
    end_perf = time.perf_counter()
    elapsed_perf = end_perf - start_perf
    print(f"                                                                           execution time: {elapsed_perf:.6f} seconds\n\n")
    
    ###############################################################################
    ###############################################################################
    ##### ACTION 6. For each zone axis, for each thickness check whether ##########
    #####           diffraction pattern feature mirror symmetry and      ##########
    #####           rotation symmetry                                    ##########
    ###############################################################################
    ###############################################################################
    
    start_perf = time.perf_counter()
    print("Action 6. for each zone axis, for each thickness, check mirror symmetry and rotation symmetry, and then further asisgn upper bound for possible in-plane angles (START)\n\n")
    
    
    if use_functions_for_monoclinic:
        per_zone_axis_per_thickness_checkMirrorSymmetry_and_upperBound_for_inPlangeRotation = action_06_check_symmetry_for_each_ZA_each_thickness_monoclinic(
                                                                                                                                        crystal,
                                                                                                                                        zone_axes,
                                                                                                                                        final_sampled_thickness_for_each_zone_axis_summary,
                                                                                                                                        k_max,
                                                                                                                                        excitation_error,
                                                                                                                                        intensThreshold,
        )
    
    else:
        per_zone_axis_per_thickness_checkMirrorSymmetry_and_upperBound_for_inPlangeRotation = action_06_check_symmetry_for_each_ZA_each_thickness(
                                                                                                                                        crystal,
                                                                                                                                        zone_axes,
                                                                                                                                        final_sampled_thickness_for_each_zone_axis_summary,
                                                                                                                                        k_max,
                                                                                                                                        excitation_error,
                                                                                                                                        intensThreshold,
        )
    
    
    
    
    
    print("")
    print("Action 6. for each zone axis, for each thickness, check mirror symmetry and rotation symmetry, and then further asisgn upper bound for possible in-plane angles (END)\n\n")
    
    end_perf = time.perf_counter()
    elapsed_perf = end_perf - start_perf
    print(f"                                                                           execution time: {elapsed_perf:.6f} seconds\n\n")
    
    ###############################################################################
    ###############################################################################
    ##### ACTION 7. For each zone axis, for each in-plane angle, assign  ##########
    #####           thickness and presence of mirror operation           ##########
    ###############################################################################
    ###############################################################################
    
    start_perf = time.perf_counter()
    print("Action 7. for each orientation (zone axis and in-plane angle), assign thickness and presence of mirror operation index (START)\n\n")
    
    deploy_orientations_thickness_and_isMirrorSymmetry_for_data_generation = action_07_generate_dictionary_of_orientation_thickness_and_mirrorSymmOper(
        zone_axes,
        per_zone_axis_per_thickness_checkMirrorSymmetry_and_upperBound_for_inPlangeRotation,
    )
    
    if saveIntermediatePklFiles:
    
        with open(crystal_name + '_total_unique_thicknesses' + '_excitErr%4.3f'%(excitation_error) + '_relIntThresh%4.3f'%(intensThreshold)  + '.pkl', 'wb') as f:
            pickle.dump(total_unique_thicknesses, f)
        
        with open(crystal_name + '_total_diffraction_patterns_uniques' + '_excitErr%4.3f'%(excitation_error) + '_relIntThresh%4.3f'%(intensThreshold)  + '.pkl', 'wb') as f:
            pickle.dump(total_diffraction_patterns_uniques, f)
        
        with open(crystal_name + '_total_average_intensities' + '_excitErr%4.3f'%(excitation_error) + '_relIntThresh%4.3f'%(intensThreshold)  + '.pkl', 'wb') as f:
            pickle.dump(total_average_intensities, f)
        
        with open(crystal_name + '_common_indices_for_each_ZA_pairs' + '_excitErr%4.3f'%(excitation_error) + '_relIntThresh%4.3f'%(intensThreshold)  + '.pkl', 'wb') as f:
            pickle.dump(common_indices_for_each_ZA_pairs, f)
            
        with open(crystal_name + '_unique_indices_candiate_for_each_ZA' + '_excitErr%4.3f'%(excitation_error) + '_relIntThresh%4.3f'%(intensThreshold)  + '.pkl', 'wb') as f:
            pickle.dump(unique_indices_candiate_for_each_ZA, f)
        
        with open(crystal_name + '_indices_selectively_assigned_to_each_ZA' + '_excitErr%4.3f'%(excitation_error) + '_relIntThresh%4.3f'%(intensThreshold)  + '.pkl', 'wb') as f:
            pickle.dump(indices_selectively_assigned_to_each_ZA, f)
        
        with open(crystal_name + '_ZA_idx_unique_thickness_indices_final_save' + '_excitErr%4.3f'%(excitation_error) + '_relIntThresh%4.3f'%(intensThreshold)  + '.pkl', 'wb') as f:
            pickle.dump(ZA_idx_unique_thickness_indices_final_save, f)
        
        with open(crystal_name + '_ZA_idx_unique_thickness_indices_final_save' + '_excitErr%4.3f'%(excitation_error) + '_relIntThresh%4.3f'%(intensThreshold)  + '.pkl', 'wb') as f:
            pickle.dump(ZA_idx_unique_thickness_indices_final_save, f)
        
        with open(crystal_name + '_final_sampled_thickness_for_each_zone_axis_summary' + '_excitErr%4.3f'%(excitation_error) + '_relIntThresh%4.3f'%(intensThreshold)  + '.pkl', 'wb') as f:
            pickle.dump(final_sampled_thickness_for_each_zone_axis_summary, f)
        
        with open(crystal_name + '_per_zone_axis_per_thickness_checkMirrorSymmetry_and_upperBound_for_inPlangeRotation' + '_excitErr%4.3f'%(excitation_error) + '_relIntThresh%4.3f'%(intensThreshold)  + '.pkl', 'wb') as f:
            pickle.dump(per_zone_axis_per_thickness_checkMirrorSymmetry_and_upperBound_for_inPlangeRotation, f)        
    
    with open(crystal_name + '_orientations_thickness_and_isMirrorSymmetry_for_data_generation' + '_excitErr%4.3f'%(excitation_error) + '_relIntThresh%4.3f'%(intensThreshold)  + '.pkl', 'wb') as f:
        pickle.dump(deploy_orientations_thickness_and_isMirrorSymmetry_for_data_generation, f)
    
    print("")
    print("Action 7. for each orientation (zone axis and in-plane angle), assign thickness and presence of mirror operation index (END)\n\n")     
    
    end_perf = time.perf_counter()
    elapsed_perf = end_perf - start_perf
    print(f"                                                                           execution time: {elapsed_perf:.6f} seconds\n\n")
    
    
    print("")
    print("You need following two files for dynamic Bloch method simulations and synthetic training data generation:\n\n")
    print("file 1")
    print(crystal_name + '_orientations_thickness_and_isMirrorSymmetry_for_data_generation' + '_excitErr%4.3f'%(excitation_error) + '_relIntThresh%4.3f'%(intensThreshold)  + '.pkl\n\n')
    print("file 2")
    print(crystal_name + "_zone_axes_out_of_plane_displacement_%2.1f"%(outOfPlaneAngleDisp) + "_degree.npy\n\n")
    print("orientation and thickness sampling for crystal " + crystal_name + " complete.")
    print("")
    print("----------------------------------------------------------------------------------------------------------------------")


if __name__ == "__main__":
    main()
