import py4DSTEM
import numpy as np

########################################################################################
########################################################################################
##############################    parameter definitions    #############################
########################################################################################
########################################################################################

N = 1000
input_dir = "/home/kwang/Desktop/Storage/project/p03_orientation_mapping/figure/figure_02/revision_sensitivity_analyses/"
output_dir = "/home/kwang/Desktop/Storage/project/p03_orientation_mapping/figure/figure_02/revision_sensitivity_analyses/output/"
k_max_radial = 2.98
max_sequence_length = 76
thickness_num_for_sampling = 1000
upper_limit_unit_cell_num = 10000
lower_limit_unit_cell_num = 2


# 1. Fix the seed for reproducibility (e.g., seed=42)
rng = np.random.default_rng(42)

# 1. Load the file using memory-mapping to save RAM and time
# 'r' means read-only
orientations = np.load(input_dir + 'orientation_original_labels_train.npy', mmap_mode='r')

# 2. Get the total number of elements along axis 0
total_elements = orientations.shape[0] # 20000000


# 3. Generate N random, unique indices
# replace=False ensures you don't pick the same index twice
random_indices = rng.choice(total_elements, size = N, replace = False)
# 4. Extract the random elements
# Because 'data' is memory-mapped, this is the only step that actually loads data into RAM
randomly_sampled_orientations = orientations[random_indices]

del orientations


k_max = 0.0328 * 64
accelerating_voltage = int(300e3)
crystal = py4DSTEM.process.diffraction.Crystal.from_CIF(input_dir + "Cu_fcc.cif")
crystal.setup_diffraction(accelerating_voltage)
crystal.calculate_structure_factors(k_max)

# Convert the V_g to relativistic-corrected U_g and store in a datastructure optimized
# for access by the Bloch code
crystal.calculate_dynamical_structure_factors(
    300e3, "WK-CP", k_max=k_max * 2., thermal_sigma=0.08, tol_structure_factor=-1.0
)




unit_cell_length = crystal.lat_real[0][0]

#### Sample thicknesses.
thickness_lower_limit = unit_cell_length * lower_limit_unit_cell_num
thickness_upper_limit = unit_cell_length * upper_limit_unit_cell_num

sampled_thicknesses = np.linspace(thickness_lower_limit, thickness_upper_limit, thickness_num_for_sampling)

input_array = []
output_label_orignial = []
total_BD_number_count_for_each_orientation_matrix = []


for count, orientation_matrix in enumerate(randomly_sampled_orientations):

    print("count", count)

    beams = crystal.generate_diffraction_pattern(
        orientation_matrix = orientation_matrix,
        sigma_excitation_error = 0.04,
        tol_intensity = 0.0,
        k_max = k_max,
    )
    
    
    dynamic_patterns = crystal.generate_dynamical_diffraction_pattern(
                    beams = beams,
                    orientation_matrix = orientation_matrix,
                    thickness = sampled_thicknesses,
                )
    
    
    collection_together = []

    current_orientation_matrix_BD_number_count = []
    
    for enIdx, pattern in enumerate(dynamic_patterns):
        # print("thickness", thicknesses[enIdx])
        
        qx = np.copy(pattern.data['qx'])
        qy = np.copy(pattern.data['qy'])
        intensity = np.copy(pattern.data['intensity'])
        initial_radial_distance = np.linalg.norm(np.stack((qx,qy)).T, axis = 1)
        
        index_of_direct_beam = np.argmin(initial_radial_distance)
        qx = np.delete(qx, index_of_direct_beam)
        qy = np.delete(qy, index_of_direct_beam)
        intensity = np.delete(intensity, index_of_direct_beam)
        intensity = intensity / np.max(intensity)
        
        
        indices_where_intensity_below_threshold = np.where(intensity < 4e-3)[0]
        qx = np.delete(qx, indices_where_intensity_below_threshold)
        qy = np.delete(qy, indices_where_intensity_below_threshold)
        intensities_of_Bragg_disks = np.delete(intensity, indices_where_intensity_below_threshold)
        
        # collection_together.append(np.stack((qx, qy, intensity)).T)
        if len(qx) > 0:
        
            positions_of_Bragg_disks = np.stack((qx, qy)).T
            k_radial_distnaces_of_BPs = np.linalg.norm(positions_of_Bragg_disks, axis = 1)
            polar_angles = np.arctan2(positions_of_Bragg_disks[:,1], positions_of_Bragg_disks[:,0])
                
            indices_where_cartesian_is_smaller_than_k_max_square = np.intersect1d(np.where(np.abs(positions_of_Bragg_disks[:,0]) < k_max)[0], np.where(np.abs(positions_of_Bragg_disks[:,1]) < k_max)[0])
            indices_where_radial_distance_smaller_than_k_max = np.where(k_radial_distnaces_of_BPs < k_max_radial)[0]
            indices_where_radial_distance_smaller_than_k_max = np.intersect1d(indices_where_radial_distance_smaller_than_k_max, indices_where_cartesian_is_smaller_than_k_max_square)
        
            if len(indices_where_radial_distance_smaller_than_k_max) > 1:

                
                input_for_network = np.stack((k_radial_distnaces_of_BPs[indices_where_radial_distance_smaller_than_k_max], polar_angles[indices_where_radial_distance_smaller_than_k_max], intensities_of_Bragg_disks[indices_where_radial_distance_smaller_than_k_max] / np.max(intensities_of_Bragg_disks[indices_where_radial_distance_smaller_than_k_max]))).T

                number_of_disks = input_for_network.shape[0]

                if number_of_disks not in current_orientation_matrix_BD_number_count:
                    current_orientation_matrix_BD_number_count.append(number_of_disks)
                    total_BD_number_count_for_each_orientation_matrix.append(number_of_disks)

    
    
                    numbers_to_pad = max_sequence_length - input_for_network.shape[0]
                    for numStack in range(numbers_to_pad):
                        input_for_network = np.vstack((input_for_network, np.array([[0.0, -np.pi + 0.00001, 0.0]])))
                    input_array.append(input_for_network)
                    output_label_orignial.append(orientation_matrix)

                    
     
        
# 1. Convert to float32 using the dtype argument
total_BD_number_count_for_each_orientation_matrix = np.array(total_BD_number_count_for_each_orientation_matrix, dtype=np.float32)
input_array = np.array(input_array, dtype=np.float32)
output_label_orignial = np.array(output_label_orignial, dtype=np.float32)

# 2. Save each array as a separate .npy file
np.save(output_dir + "postProc1_input_array.npy", input_array)
np.save(output_dir + "postProc1_total_BD_number_count_for_each_orientation_matrix.npy", total_BD_number_count_for_each_orientation_matrix)
np.save(output_dir + "postProc1_output_label_orignial.npy", output_label_orignial)
