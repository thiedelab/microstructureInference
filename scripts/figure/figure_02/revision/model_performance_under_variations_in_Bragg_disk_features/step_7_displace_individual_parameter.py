import os
import numpy as np
import time

def noise_apply_gaussian_noise_to_disk_positions(
    input_polar_table,
    cartesian_disp_std,
    seed = 1,
    cartesian_disp_mean = 0.0,
    DP_2D_pattern_dimension = 128,
    diffraction_space_pixel_size = 0.0328,
    threshold = 1e-8,
    ):

    # Fix random number generator
    rng = np.random.default_rng(seed)

    input_polar_table_noised = np.copy(input_polar_table)


    q_dimension_max = diffraction_space_pixel_size * (DP_2D_pattern_dimension/2.)
    buffer = diffraction_space_pixel_size / 2.0
    q_dimension_lower_bound = -q_dimension_max + buffer
    q_dimension_upper_bound = q_dimension_max - buffer
    
    for count, polar_table in enumerate(input_polar_table):
        radial_distances = np.copy(polar_table[:,0])
        polar_angles = np.copy(polar_table[:,1])
        intensities = np.copy(polar_table[:,2])
        # print("polar_table\n", polar_table)
        
        BD_indices = np.intersect1d(np.where(radial_distances > threshold)[0], np.where(intensities > threshold)[0])

        numberOfBraggDisks_to_displace = len(BD_indices)

        radial_distances_BDs = radial_distances[BD_indices]
        polar_angles_BDs = polar_angles[BD_indices]
        intensities_BDs = intensities[BD_indices]

        # Use np.cos, np.sin, and np.random.normal
        displaced_qx = radial_distances_BDs * np.cos(polar_angles_BDs) + rng.normal(cartesian_disp_mean, cartesian_disp_std, numberOfBraggDisks_to_displace)
        displaced_qy = radial_distances_BDs * np.sin(polar_angles_BDs) + rng.normal(cartesian_disp_mean, cartesian_disp_std, numberOfBraggDisks_to_displace)

        displaced_qx = np.clip(displaced_qx, a_min = q_dimension_lower_bound, a_max = q_dimension_upper_bound)
        displaced_qy = np.clip(displaced_qy, a_min = q_dimension_lower_bound, a_max = q_dimension_upper_bound)

        # np.hypot calculates sqrt(qx^2 + qy^2) element-wise
        displaced_r = np.hypot(displaced_qx, displaced_qy)
        
        # Use np.arctan2
        displaced_angle = np.arctan2(displaced_qy, displaced_qx)

        new_BD_features = np.stack((displaced_r,displaced_angle,intensities_BDs)).T

        input_polar_table_noised[count, BD_indices] = new_BD_features

        # print("BD_indices", BD_indices)
        # print("input_polar_table_noised[count]\n",input_polar_table_noised[count])

    return input_polar_table_noised



def noise_apply_gaussian_noise_to_disk_intensities(
    input_polar_table,
    intensities_disp_std,
    seed = 1,
    intensities_disp_mean = 0.0,
    relative_intensity_dimension_lower_bound = 4e-3,
    relative_intensity_dimension_upper_bound = 1.0,
    threshold = 1e-8,
    ):

    # Fix random number generator
    rng = np.random.default_rng(seed)

    input_polar_table_noised = np.copy(input_polar_table)    
    
    for count, polar_table in enumerate(input_polar_table):
        radial_distances = np.copy(polar_table[:,0])
        polar_angles = np.copy(polar_table[:,1])
        intensities = np.copy(polar_table[:,2])
        # print("polar_table\n", polar_table)
        
        BD_indices = np.intersect1d(np.where(radial_distances > threshold)[0], np.where(intensities > threshold)[0])

        numberOfBraggDisks_to_displace = len(BD_indices)

        radial_distances_BDs = radial_distances[BD_indices]
        polar_angles_BDs = polar_angles[BD_indices]
        intensities_BDs = intensities[BD_indices]

        displaced_intensities = np.copy(intensities_BDs) + rng.normal(intensities_disp_mean, intensities_disp_std, numberOfBraggDisks_to_displace)
        displaced_intensities = np.clip(displaced_intensities, a_min = relative_intensity_dimension_lower_bound, a_max = relative_intensity_dimension_upper_bound)
        displaced_intensities = displaced_intensities / np.max(displaced_intensities)

        new_BD_features = np.stack((radial_distances_BDs,polar_angles_BDs,displaced_intensities)).T

        input_polar_table_noised[count, BD_indices] = new_BD_features

        # print("BD_indices", BD_indices)
        # print("input_polar_table_noised[count]\n",input_polar_table_noised[count])

    return input_polar_table_noised



def noise_apply_gaussian_noise_to_both_disk_positions_intensities(
    input_polar_table,
    cartesian_disp_std,
    intensities_disp_std,
    seed = 1,
    cartesian_disp_mean = 0.0,
    intensities_disp_mean = 0.0,
    DP_2D_pattern_dimension = 128,
    diffraction_space_pixel_size = 0.0328,
    relative_intensity_dimension_lower_bound = 4e-3,
    relative_intensity_dimension_upper_bound = 1.0,
    threshold = 1e-8,    
    ):

    # Fix random number generator
    rng = np.random.default_rng(seed)

    input_polar_table_noised = np.copy(input_polar_table)    

    q_dimension_max = diffraction_space_pixel_size * (DP_2D_pattern_dimension/2.)
    buffer = diffraction_space_pixel_size / 2.0
    q_dimension_lower_bound = -q_dimension_max + buffer
    q_dimension_upper_bound = q_dimension_max - buffer
    
    for count, polar_table in enumerate(input_polar_table):
        radial_distances = np.copy(polar_table[:,0])
        polar_angles = np.copy(polar_table[:,1])
        intensities = np.copy(polar_table[:,2])
        # print("polar_table\n", polar_table)
        
        BD_indices = np.intersect1d(np.where(radial_distances > threshold)[0], np.where(intensities > threshold)[0])

        numberOfBraggDisks_to_displace = len(BD_indices)

        radial_distances_BDs = radial_distances[BD_indices]
        polar_angles_BDs = polar_angles[BD_indices]
        intensities_BDs = intensities[BD_indices]

        # Use np.cos, np.sin, and np.random.normal
        displaced_qx = radial_distances_BDs * np.cos(polar_angles_BDs) + rng.normal(cartesian_disp_mean, cartesian_disp_std, numberOfBraggDisks_to_displace)
        displaced_qy = radial_distances_BDs * np.sin(polar_angles_BDs) + rng.normal(cartesian_disp_mean, cartesian_disp_std, numberOfBraggDisks_to_displace)

        displaced_qx = np.clip(displaced_qx, a_min = q_dimension_lower_bound, a_max = q_dimension_upper_bound)
        displaced_qy = np.clip(displaced_qy, a_min = q_dimension_lower_bound, a_max = q_dimension_upper_bound)

        # np.hypot calculates sqrt(qx^2 + qy^2) element-wise
        displaced_r = np.hypot(displaced_qx, displaced_qy)
        
        # Use np.arctan2
        displaced_angle = np.arctan2(displaced_qy, displaced_qx)

        displaced_intensities = np.copy(intensities_BDs) + rng.normal(intensities_disp_mean, intensities_disp_std, numberOfBraggDisks_to_displace)
        displaced_intensities = np.clip(displaced_intensities, a_min = relative_intensity_dimension_lower_bound, a_max = relative_intensity_dimension_upper_bound)
        displaced_intensities = displaced_intensities / np.max(displaced_intensities)

        new_BD_features = np.stack((displaced_r,displaced_angle,displaced_intensities)).T

        input_polar_table_noised[count, BD_indices] = new_BD_features

        # print("BD_indices", BD_indices)
        # print("input_polar_table_noised[count]\n",input_polar_table_noised[count])

    return input_polar_table_noised


file_dir = "/home/kwang/Desktop/Storage/project/p03_orientation_mapping/figure/figure_02/revision_sensitivity_analyses/output/"

input_array = np.load(file_dir + "postProc2_input_array.npy")

output_dir = file_dir + "noised/"

diffraction_space_pixel_size = 0.0328
intensityDisp_std_reference = 1.0
cartesianDisp_std_reference = diffraction_space_pixel_size

cartesian_displacement_scaling_factor_unit = 0.20
intensity_displacement_scaling_factor_unit = 0.05

cartesian_displacement_scaling_factors = np.arange(cartesian_displacement_scaling_factor_unit, cartesian_displacement_scaling_factor_unit * 19 ,cartesian_displacement_scaling_factor_unit)
intensity_displacement_scaling_factors = np.arange(intensity_displacement_scaling_factor_unit, intensity_displacement_scaling_factor_unit * 18 ,intensity_displacement_scaling_factor_unit)

print("Starting sensitivity analysis generation...")
start_time = time.perf_counter()


max_displacement_trials_begin = 0
max_displacement_trials_end = 20

# Displace each parameter individually.
for seedValue in range(max_displacement_trials_begin + 1, max_displacement_trials_end + 1, 1):

    print("Trial:", seedValue)

    # displace positions only:
    for cartesian_displacement_scaling_factor in cartesian_displacement_scaling_factors:

        cartesianDisp_std = diffraction_space_pixel_size * cartesian_displacement_scaling_factor

        input_polar_table_positions_noised = noise_apply_gaussian_noise_to_disk_positions(input_array, cartesianDisp_std, seedValue)

        np.save(output_dir + "postProc5_input_array_noised_cartesianOnly_cartScale%3.2f_seed%d"%(cartesian_displacement_scaling_factor, seedValue), input_polar_table_positions_noised)

    # displace intensities only:
    for intensity_displacement_scaling_factor in intensity_displacement_scaling_factors:

        intensityDisp_std = intensityDisp_std_reference * intensity_displacement_scaling_factor

        input_polar_table_intensities_noised = noise_apply_gaussian_noise_to_disk_intensities(input_array, intensityDisp_std, seedValue)

        np.save(output_dir + "postProc5_input_array_noised_intensityOnly_intenScale%3.2f_seed%d"%(intensity_displacement_scaling_factor, seedValue), input_polar_table_intensities_noised)

end_time = time.perf_counter()  # <-- STOP TIMER
elapsed_time = end_time - start_time

# Print out the total time taken in seconds (formatted to 2 decimal places)
print(f"Finished individual displacements successfully in {elapsed_time:.2f} seconds!")

# start_time = time.perf_counter()

# # # jointly displace two parameter individually.

# for cartesian_displacement_scaling_factor in cartesian_displacement_scaling_factors:
#     cartesianDisp_std = diffraction_space_pixel_size * cartesian_displacement_scaling_factor
#     for intensity_displacement_scaling_factor in intensity_displacement_scaling_factors:
#         intensityDisp_std = intensityDisp_std_reference * intensity_displacement_scaling_factor
#         for seedValue in range(1, max_displacement_trials + 1, 1):
#             input_polar_table_both_noised = noise_apply_gaussian_noise_to_both_disk_positions_intensities(input_array, cartesianDisp_std, intensityDisp_std, seedValue)

#             np.save(output_dir + "postProc5_input_array_noised_Both_cartScale%3.2f_intenScale%3.2f_seed%d"%(cartesian_displacement_scaling_factor,intensity_displacement_scaling_factor, seedValue), input_polar_table_both_noised)
# end_time = time.perf_counter()  # <-- STOP TIMER
# elapsed_time = end_time - start_time

# Print out the total time taken in seconds (formatted to 2 decimal places)
# print(f"Finished joint displacements successfully in {elapsed_time:.2f} seconds!")
