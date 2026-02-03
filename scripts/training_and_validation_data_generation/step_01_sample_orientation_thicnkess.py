#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 13:40:45 2025

@author: kwang
"""
import py4DSTEM
import numpy as np
import pickle
from modules_select_orientation_thicnkess_mirrorOp import action_01_collect_unique_thickness_for_each_zone_axis
from modules_select_orientation_thicnkess_mirrorOp import action_02_compare_DPs_from_different_zone_axes
from modules_select_orientation_thicnkess_mirrorOp import action_03_remove_thickness_indices_of_common_patterns_for_lowSymmetry_zone_axes
from modules_select_orientation_thicnkess_mirrorOp import action_04_finalize_unique_thickness_resulting_unique_pattern_for_each_ZA
from modules_select_orientation_thicnkess_mirrorOp import action_05_sample_representative_thickness_for_each_ZA
from modules_select_orientation_thicnkess_mirrorOp import action_06_check_symmetry_for_each_ZA_each_thickness
from modules_select_orientation_thicnkess_mirrorOp import action_07_generate_dictionary_of_orientation_thickness_and_mirrorSymmOper
import time
###############################################################################
###############################################################################
########################### Set global parameters #############################
###############################################################################
###############################################################################

start_perf = time.perf_counter()
print("Setting and initializing parameters, variables, and py4DSTEM crystal object\n\n")

unit_cell_path = "./"
Cu_cif = "Cu_fcc.cif"

pixel_size = 0.0328
sigma_compare = 0.02
pixel_numbers = 128


k_max = pixel_size * pixel_numbers / 2.
accelerating_voltage = int(300e3)
crystal = py4DSTEM.process.diffraction.Crystal.from_CIF(unit_cell_path + Cu_cif)

upper_limit_unit_cell_num = 1000
lower_limit_unit_cell_num = 2
unit_cell_length = crystal.lat_real[0][0]

crystal.setup_diffraction(accelerating_voltage)
crystal.calculate_structure_factors(
    k_max,
)

#crystal.calculate_structure_factors(k_max * 2.)
    
# Convert the V_g to relativistic-corrected U_g and store in a datastructure optimized
# for access by the Bloch code
crystal.calculate_dynamical_structure_factors(
    300e3, "WK-CP", k_max=k_max * 2., thermal_sigma=0.08, tol_structure_factor=-1.0
)


# Create an orientation plan for Cu fcc
crystal.orientation_plan(
    angle_step_zone_axis = 2,
    angle_step_in_plane = 1,
    accel_voltage = 300e3,
    corr_kernel_size= 0.08, # was 0.08 before 0.12 not bad
    zone_axis_range='auto',
)

# Save Zone axes
zone_axes = crystal.orientation_vecs
np.save("zone_axes_out_of_plane_displacement_2_degree.npy", zone_axes)

total_number_for_sampling_thickness = 80


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
                                                                        k_max
)

with open('total_unique_thicknesses_Cu_fcc_4e3_ee0.04.pkl', 'wb') as f:
    pickle.dump(total_unique_thicknesses, f)

with open('total_diffraction_patterns_uniques_Cu_fcc_4e3_ee0.04.pkl', 'wb') as f:
    pickle.dump(total_diffraction_patterns_uniques, f)

with open('total_average_intensities_Cu_fcc_4e3_ee0.04.pkl', 'wb') as f:
    pickle.dump(total_average_intensities, f)

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

with open('common_indices_for_each_ZA_pairs_Cu_fcc_4e3_ee0.04.pkl', 'wb') as f:
    pickle.dump(common_indices_for_each_ZA_pairs, f)
with open('unique_indices_candiate_for_each_ZA_Cu_fcc_4e3_ee0.04.pkl', 'wb') as f:
    pickle.dump(unique_indices_candiate_for_each_ZA, f)

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

with open('indices_selectively_assigned_to_each_ZA_Cu_fcc_4e3_ee0.04.pkl', 'wb') as f:
    pickle.dump(indices_selectively_assigned_to_each_ZA, f)
        
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

with open('ZA_idx_unique_thickness_indices_final_save_Cu_fcc_4e3_ee0.04.pkl', 'wb') as f:
    pickle.dump(ZA_idx_unique_thickness_indices_final_save, f)

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



            
with open('final_sampled_thickness_for_each_zone_axis_summary_Cu_fcc_4e3_ee0.04.pkl', 'wb') as f:
    pickle.dump(final_sampled_thickness_for_each_zone_axis_summary, f)

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

per_zone_axis_per_thickness_checkMirrorSymmetry_and_upperBound_for_inPlangeRotation = action_06_check_symmetry_for_each_ZA_each_thickness(
                                                                                                                                crystal,
                                                                                                                                zone_axes,
                                                                                                                                final_sampled_thickness_for_each_zone_axis_summary,
                                                                                                                                k_max
)


with open('per_zone_axis_per_thickness_checkMirrorSymmetry_and_upperBound_for_inPlangeRotation_Cu_fcc_4e3_ee0.04.pkl', 'wb') as f:
    pickle.dump(per_zone_axis_per_thickness_checkMirrorSymmetry_and_upperBound_for_inPlangeRotation, f)    


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

                
with open('orientations_thickness_and_isMirrorSymmetry_for_data_generation_Cu_fcc_4e3_ee0.04.pkl', 'wb') as f:
    pickle.dump(deploy_orientations_thickness_and_isMirrorSymmetry_for_data_generation, f)

print("")
print("Action 7. for each orientation (zone axis and in-plane angle), assign thickness and presence of mirror operation index (END)\n\n")     

end_perf = time.perf_counter()
elapsed_perf = end_perf - start_perf
print(f"                                                                           execution time: {elapsed_perf:.6f} seconds\n\n")


print("")
print("JONE DONE")
print("")
