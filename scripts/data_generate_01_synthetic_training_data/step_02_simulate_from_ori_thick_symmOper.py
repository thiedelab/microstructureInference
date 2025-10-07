import py4DSTEM
import numpy as np
import time
import argparse
import pickle
# import matplotlib.pyplot as plt
print(py4DSTEM.__version__)

def rotation_wrt_zAxis(angle_in_rad):
    return np.array([
                        [np.cos(angle_in_rad), np.sin(angle_in_rad), 0],
                        [-np.sin(angle_in_rad), np.cos(angle_in_rad), 0],
                        [0, 0, 1]
                    ]
                    )


def rotation_wrt_xAxis(angle_in_rad):
    return np.array(
                    [
                        [1, 0, 0],
                        [0, np.cos(angle_in_rad), np.sin(angle_in_rad)],
                        [0, -np.sin(angle_in_rad), np.cos(angle_in_rad)],
                    ]
                    )


def rotMatrix_from_eulerAngles_ZXZ(
                                    angle_z1,
                                    angle_x1, 
                                    angle_z2,
                                    ):
    
    rotationMatrix =    rotation_wrt_zAxis(angle_z1) @ \
                        rotation_wrt_xAxis(angle_x1) @ \
                        rotation_wrt_zAxis(angle_z2)
        
    return rotationMatrix

def calculate_rotation_matrix_for_zone_axis(zone_axis):
    elev = np.arctan2(
        np.hypot(zone_axis[0], zone_axis[1]),
        zone_axis[2],
    )
    
    azim = np.arctan2(zone_axis[0], zone_axis[1])

    new_rotation_matrix = rotation_wrt_zAxis(azim) @ rotation_wrt_xAxis(elev) @ rotation_wrt_zAxis(-azim)


    return new_rotation_matrix

def get_bounds(save_string_integer, number_of_zone_axis_per_run = 21):
    bound_range = []
    for i in range(20):
        bound_range.append([int(i * number_of_zone_axis_per_run), int((i + 1) * number_of_zone_axis_per_run)])
    return int(bound_range[save_string_integer][0]), int(bound_range[save_string_integer][1])


parser = argparse.ArgumentParser(description="A simple script demonstrating argparse.")
parser.add_argument("--index", type=int, help="index_of_file")
args = parser.parse_args()
save_string = int(args.index)

print("file_index", save_string)
left_bound, right_bound = get_bounds(save_string)
if save_string == 19:
    right_bound = 406
print("left_bound, right_bound", left_bound, right_bound)

file_path = "/projects/bdts/jekw/projects/py4DSTEM_table/"
Cu_cif_path = "/projects/bdts/jekw/projects/py4DSTEM_table/"
# rotation_matrices = np.load(rot_matrices_path + "rotation_matrices_merged.npy")

with open(file_path + 'Cu_fcc_orientations_thickness_and_isMirrorSymmetry_for_data_generation_excitErr0.045_relIntThresh0.005.pkl', 'rb') as f:
    deploy_orientations_thickness_and_isMirrorSymmetry_for_data_generation = pickle.load(f)

zone_axes = np.load(file_path + "Cu_fcc_zone_axes_out_of_plane_displacement_2.0_degree.npy")

k_max = 0.0328 * 64
accelerating_voltage = int(300e3)
crystal = py4DSTEM.process.diffraction.Crystal.from_CIF(Cu_cif_path + "Cu_fcc.cif")
crystal.setup_diffraction(accelerating_voltage)
crystal.calculate_structure_factors(k_max * 4.)

# Convert the V_g to relativistic-corrected U_g and store in a datastructure optimized
# for access by the Bloch code
crystal.calculate_dynamical_structure_factors(
    accelerating_voltage, "WK-CP", k_max=k_max * 4., thermal_sigma=0.08, tol_structure_factor=-1.0
)

range_plot = np.array([k_max+0.1,k_max+0.1])



k_max_radial = 2.98

start_time = time.perf_counter()

max_sequence_length = 76

max_number_of_Bragg_disks = []

input_array = []
output_label_canonical = []
output_label_orignial = []
output_mirror = []
numbers_for_each_sg = []
thickness_save = []
num_BD = []
max_distances = []
min_distances = []


for zone_axis_index, zone_axis_dictionary in deploy_orientations_thickness_and_isMirrorSymmetry_for_data_generation.items():
    if left_bound <= zone_axis_index and zone_axis_index < right_bound:
        print("##################################")    
        current_zone_axis = zone_axes[zone_axis_index]
        print("zone_axis_index", zone_axis_index)
        print("current_zone_axis", current_zone_axis)

        ref_orientation_matrix = calculate_rotation_matrix_for_zone_axis(current_zone_axis)
        
        for in_plane_angle_degree, thickness_and_symmetry_values in zone_axis_dictionary.items():

        
            current_thicknesses = thickness_and_symmetry_values['thickness']
            current_hasMirrorSymmetry = thickness_and_symmetry_values['isMirrored']
    
            if len(current_thicknesses) > 0:
                in_plane_rotation_matrix = rotation_wrt_zAxis(float(in_plane_angle_degree) * np.pi / 180.)
                orientation_matrix = ref_orientation_matrix @ in_plane_rotation_matrix


                beams = crystal.generate_diffraction_pattern(
                        orientation_matrix = orientation_matrix, 
                        sigma_excitation_error = 0.045,
                        tol_intensity = 0.0, 
                        k_max = k_max * 6.,
                    )
        
            
                dynamic_patterns = crystal.generate_dynamical_diffraction_pattern(
                                    beams = beams, 
                                    orientation_matrix = orientation_matrix,
                                    thickness = current_thicknesses, 
                                )
                
            
                collection_together = []
                
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
            
                    
                    indices_where_intensity_below_threshold = np.where(intensity < 5e-3)[0]
                    qx = np.delete(qx, indices_where_intensity_below_threshold)
                    qy = np.delete(qy, indices_where_intensity_below_threshold)
                    intensities_of_Bragg_disks = np.delete(intensity, indices_where_intensity_below_threshold)
                    num_BD.append(len(qx))
            
                    # collection_together.append(np.stack((qx, qy, intensity)).T)
                    if len(qx) > 0:
                        
                        positions_of_Bragg_disks = np.stack((qx, qy)).T
                        k_radial_distnaces_of_BPs = np.linalg.norm(positions_of_Bragg_disks, axis = 1)
                        polar_angles = np.arctan2(positions_of_Bragg_disks[:,1], positions_of_Bragg_disks[:,0])
                        
                        if current_hasMirrorSymmetry[enIdx] == 0:
                            positions_of_Bragg_disks_mirrored = np.stack((-qx, qy)).T
                            k_radial_distnaces_of_BPs_mirrored = np.linalg.norm(positions_of_Bragg_disks_mirrored, axis = 1)
                            polar_angles_mirrored = np.arctan2(positions_of_Bragg_disks_mirrored[:,1], positions_of_Bragg_disks_mirrored[:,0])
                        
            
                        indices_where_cartesian_is_smaller_than_k_max_square = np.intersect1d(np.where(np.abs(positions_of_Bragg_disks[:,0]) < k_max)[0], np.where(np.abs(positions_of_Bragg_disks[:,1]) < k_max)[0])
                        indices_where_radial_distance_smaller_than_k_max = np.where(k_radial_distnaces_of_BPs < k_max_radial)[0]     
                        indices_where_radial_distance_smaller_than_k_max = np.intersect1d(indices_where_radial_distance_smaller_than_k_max, indices_where_cartesian_is_smaller_than_k_max_square)
                        
                        if len(indices_where_radial_distance_smaller_than_k_max) > 1:
                            input_for_network = np.stack((k_radial_distnaces_of_BPs[indices_where_radial_distance_smaller_than_k_max], polar_angles[indices_where_radial_distance_smaller_than_k_max], intensities_of_Bragg_disks[indices_where_radial_distance_smaller_than_k_max] / np.max(intensities_of_Bragg_disks[indices_where_radial_distance_smaller_than_k_max]))).T

            
            
                            numbers_to_pad = max_sequence_length - input_for_network.shape[0]
                            for numStack in range(numbers_to_pad):
                                input_for_network = np.vstack((input_for_network, np.array([[0.0, -np.pi + 0.00001, 0.0]])))
                            input_array.append(input_for_network)

                            thickness_save.append(current_thicknesses[enIdx])
                            max_number_of_Bragg_disks.append(len(k_radial_distnaces_of_BPs[indices_where_radial_distance_smaller_than_k_max]))
                            output_mirror.append(0)
                            output_label_canonical.append(orientation_matrix)
                            output_label_orignial.append(orientation_matrix)
                            

                            if current_hasMirrorSymmetry[enIdx] == 0:
                                input_for_network_mirrored = np.stack((k_radial_distnaces_of_BPs[indices_where_radial_distance_smaller_than_k_max], polar_angles_mirrored[indices_where_radial_distance_smaller_than_k_max], intensities_of_Bragg_disks[indices_where_radial_distance_smaller_than_k_max] / np.max(intensities_of_Bragg_disks[indices_where_radial_distance_smaller_than_k_max]))).T
                                for numStack in range(numbers_to_pad):
                                    input_for_network_mirrored = np.vstack((input_for_network_mirrored, np.array([[0.0, -np.pi + 0.00001, 0.0]])))
                                input_array.append(input_for_network_mirrored)
                                
                                thickness_save.append(current_thicknesses[enIdx])
                                max_number_of_Bragg_disks.append(len(k_radial_distnaces_of_BPs[indices_where_radial_distance_smaller_than_k_max]))
                                output_mirror.append(1)
                                output_label_canonical.append(orientation_matrix)
                                orientation_matrix_c = np.copy(orientation_matrix)
                                orientation_matrix_c[:,1:] *= (-1.0)
                                output_label_orignial.append(orientation_matrix_c)
                    
    
input_array = np.array(input_array)
output_label_canonical = np.array(output_label_canonical)
output_label_orignial = np.array(output_label_orignial)
output_mirror = np.array(output_mirror)
thickness_save = np.array(thickness_save)

filename = "py4DSTEM_Cu_ori_table_normalize_by_maxInt_5e3_sg0.045_merged_%d"%(save_string)



print("input_array.shape", input_array.shape)
np.save(filename + ".npy", input_array)

print("output_label_canonical.shape", output_label_canonical.shape)
print("output_label_orignial.shape", output_label_orignial.shape)

np.save("rot_canonical_" + filename + ".npy", output_label_canonical)
np.save("rot_original_" + filename + ".npy", output_label_orignial)

print("len(thickness_save)", len(thickness_save))
np.save("thickness_" + filename + ".npy", thickness_save)

print("len(output_mirror)", len(output_mirror))
np.save("mirror_" + filename + ".npy", output_mirror)


max_number_of_Bragg_disks = np.array(max_number_of_Bragg_disks)
print("np.max(max_number_of_Bragg_disks)", np.max(max_number_of_Bragg_disks))
print("")
