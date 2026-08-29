import numpy as np

########################################################################################
########################################################################################
##############################    parameter definitions    #############################
########################################################################################
########################################################################################





rng = np.random.default_rng(78)

max_number_of_Bragg_disks_cutoff = 200

simulation_output_dir = "/home/kwang/Desktop/Storage/project/p03_orientation_mapping/figure/figure_02/revision_sensitivity_analyses/output/"

input_array = np.load(simulation_output_dir + "postProc1_input_array.npy")
total_BD_number_count_for_each_orientation_matrix = np.load(simulation_output_dir + "postProc1_total_BD_number_count_for_each_orientation_matrix.npy")
total_BD_number_count_for_each_orientation_matrix = np.array(total_BD_number_count_for_each_orientation_matrix, dtype=np.int32)
output_label_orignial = np.load(simulation_output_dir + "postProc1_output_label_orignial.npy")

# 1. Get the total number of elements along axis 0
num_samples = input_array.shape[0]

# 3. Generate a single array of shuffled indices from 0 to num_samples - 1
shuffled_indices = rng.permutation(num_samples)

# 4. Apply the exact same shuffled indices to all three arrays
input_array_shuffled = input_array[shuffled_indices]
BD_count_shuffled = total_BD_number_count_for_each_orientation_matrix[shuffled_indices]
output_label_shuffled = output_label_orignial[shuffled_indices]

post_processed_number = []
post_processed_orientation_label = []
post_processed_polar_table = []

for N in range(2,37,1):

    indices_corresponding_to_BD_N = np.where(BD_count_shuffled == N)[0]
    indices_corresponding_to_BD_N = rng.choice(indices_corresponding_to_BD_N, size=max_number_of_Bragg_disks_cutoff, replace=False)

    for index in indices_corresponding_to_BD_N:
        post_processed_number.append(N)
        post_processed_orientation_label.append(output_label_shuffled[index])
        post_processed_polar_table.append(input_array_shuffled[index])

post_processed_number = np.array(post_processed_number, dtype=np.int32)
post_processed_orientation_label = np.array(post_processed_orientation_label, dtype=np.float32)
post_processed_polar_table = np.array(post_processed_polar_table, dtype=np.float32)

np.save(simulation_output_dir + "postProc2_input_array.npy", post_processed_polar_table)
np.save(simulation_output_dir + "postProc2_total_BD_number_count_for_each_orientation_matrix.npy", post_processed_number)
np.save(simulation_output_dir + "postProc2_output_label_orignial.npy", post_processed_orientation_label)

print("total number of unique orientations: ", np.unique(post_processed_orientation_label, axis = 0).shape[0])


# rng = np.random.default_rng(3)

# max_number_of_Bragg_disks_cutoff = 100

# simulation_output_dir = "/home/kwang/Desktop/Storage/project/p03_orientation_mapping/figure/figure_02/revision_sensitivity_analyses/output/"

# input_array = np.load(simulation_output_dir + "postProc1_input_array.npy")
# total_BD_number_count_for_each_orientation_matrix = np.load(simulation_output_dir + "postProc1_total_BD_number_count_for_each_orientation_matrix.npy")
# total_BD_number_count_for_each_orientation_matrix = np.array(total_BD_number_count_for_each_orientation_matrix, dtype=np.int32)
# output_label_orignial = np.load(simulation_output_dir + "postProc1_output_label_orignial.npy")

# # 1. Get the total number of elements along axis 0
# num_samples = input_array.shape[0]

# # 3. Generate a single array of shuffled indices from 0 to num_samples - 1
# shuffled_indices = rng.permutation(num_samples)

# # 4. Apply the exact same shuffled indices to all three arrays
# input_array_shuffled = input_array[shuffled_indices]
# BD_count_shuffled = total_BD_number_count_for_each_orientation_matrix[shuffled_indices]
# output_label_shuffled = output_label_orignial[shuffled_indices]

# post_processed_number = []
# post_processed_orientation_label = []
# post_processed_polar_table = []

# for N in range(2,36,1):

#     indices_corresponding_to_BD_N = np.where(total_BD_number_count_for_each_orientation_matrix == N)[0]
#     indices_corresponding_to_BD_N = rng.choice(indices_corresponding_to_BD_N, size=max_number_of_Bragg_disks_cutoff, replace=False)

#     for index in indices_corresponding_to_BD_N:
#         post_processed_number.append(N)
#         post_processed_orientation_label.append(output_label_orignial[index])
#         post_processed_polar_table.append(input_array[index])

# post_processed_number = np.array(post_processed_number, dtype=np.int32)
# post_processed_orientation_label = np.array(post_processed_orientation_label, dtype=np.float32)
# post_processed_polar_table = np.array(post_processed_polar_table, dtype=np.float32)

# np.save(simulation_output_dir + "postProc2_input_array.npy", post_processed_polar_table)
# np.save(simulation_output_dir + "postProc2_total_BD_number_count_for_each_orientation_matrix.npy", post_processed_number)
# np.save(simulation_output_dir + "postProc2_output_label_orignial.npy", post_processed_orientation_label)

# print("total number of unique orientations: ", np.unique(post_processed_orientation_label, axis = 0).shape[0])