from py4DSTEM import BraggVectors
from py4DSTEM.data import QPoints
import py4DSTEM
from emdfile import tqdmnd, PointList, PointListArray
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Optional
import torch
from microstructure_inference.dataModules import cubic_proper_point_group_operations
from microstructure_inference.LossFunctions import pointGroup_map_rotation_prediction_return_geodesic_distance_stack
from microstructure_inference.analysis import soft_disk_loss


def compute_symmetry_aware_geodesic_loss_between_two_rotation_matrices(rotation_matrix_1, rotation_matrix_2, point_group_op_matrices, device = "cpu"):

    rotation_tensor_1 = torch.from_numpy(rotation_matrix_1).to(torch.float32)
    rotation_tensor_2 = torch.from_numpy(rotation_matrix_2).to(torch.float32)
    
    rotation_tensor_1 = rotation_tensor_1.unsqueeze(0)
    rotation_tensor_2 = rotation_tensor_2.unsqueeze(0)
    
    loss, geodesic_distances = pointGroup_map_rotation_prediction_return_geodesic_distance_stack(rotation_tensor_1, rotation_tensor_2, point_group_op_matrices)

    return float(geodesic_distances[0])
    
def plot_labelINPUT_labelBD_diffraction_pattern(
    bragg_peaks: PointList,
    sim_BD_marker = "p",
    sim_BD_color = "#007544",
    sim_BD_linewidth = 0.9,
    observed_BD_color = "#F2FFF7",    
    observed_pattern_linewidth = 1.1,
    bragg_peaks_compare: PointList = None,
    scale_markers: float = 500,
    scale_markers_compare: Optional[float] = None,
    power_markers: float = 0.8,
    plot_range_kx_ky: Optional[Union[list, tuple, np.ndarray]] = None,
    add_labels: bool = True,
    shift_labels: float = 0.08,
    shift_marker: float = 0.005,
    min_marker_size: float = 1e-6,
    max_marker_size: float = 1000,
    figsize: Union[list, tuple, np.ndarray] = (12, 6),
    returnfig: bool = False,
    input_fig_handle=None,
    showTicks: bool = False,
):
    """
    2D scatter plot of the Bragg peaks

    Args:
        bragg_peaks (PointList):        numpy array containing ('qx', 'qy', 'intensity', 'h', 'k', 'l')
        bragg_peaks_compare(PointList): numpy array containing ('qx', 'qy', 'intensity')
        scale_markers (float):          size scaling for markers
        scale_markers_compare (float):  size scaling for markers of comparison
        power_markers (float):          power law scaling for marks (default is 1, i.e. amplitude)
        plot_range_kx_ky (float):       2 element numpy vector giving the plot range
        add_labels (bool):              flag to add hkl labels to peaks
        min_marker_size (float):        minimum marker size for the comparison peaks
        max_marker_size (float):        maximum marker size for the comparison peaks
        figsize (2 element float):      size scaling of figure axes
        returnfig (bool):               set to True to return figure and axes handles
        input_fig_handle (fig,ax)       Tuple containing a figure / axes handle for the plot.
    """

    # 2D plotting
    if input_fig_handle is None:
        # fig = plt.figure(figsize=figsize)
        # ax = fig.add_subplot()
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = input_fig_handle[0]
        ax_parent = input_fig_handle[1]
        ax = ax_parent[0]

    if power_markers == 2:
        marker_size = scale_markers * bragg_peaks.data["intensity"]
    else:
        marker_size = scale_markers * (
            bragg_peaks.data["intensity"] ** (power_markers / 2)
        )

    # Apply marker size limits to primary plot
    marker_size = np.clip(marker_size, min_marker_size, max_marker_size)

    if bragg_peaks_compare is None:
        ax.scatter(
            bragg_peaks.data["qy"], bragg_peaks.data["qx"], s=marker_size, facecolor=observed_pattern_custom_color, linewidths=observed_pattern_linewidth, edgecolors = "k",
        )
    else:
        if scale_markers_compare is None:
            scale_markers_compare = scale_markers

        if power_markers == 2:
            marker_size_compare = np.clip(
                scale_markers_compare * bragg_peaks_compare.data["intensity"],
                min_marker_size,
                max_marker_size,
            )
        else:
            marker_size_compare = np.clip(
                scale_markers_compare
                * (bragg_peaks_compare.data["intensity"] ** (power_markers / 2)),
                min_marker_size,
                max_marker_size,
            )

        ax.scatter(
            bragg_peaks_compare.data["qy"],
            bragg_peaks_compare.data["qx"],
            s=marker_size_compare,
            marker="o",
            facecolor=observed_BD_color, linewidths=observed_pattern_linewidth, edgecolors = "k"
        )
        ax.scatter(
            bragg_peaks.data["qy"],
            bragg_peaks.data["qx"],
            s=marker_size,
            marker=sim_BD_marker,
            facecolor=sim_BD_color,
            linewidths = sim_BD_linewidth,
            
            # marker="o",
            # facecolor="none",
            # edgecolors="#007544",
            # linewidths=2.0,
        )

    # ax.set_xlabel("$q_y$ [Å$^{-1}$]")
    # ax.set_ylabel("$q_x$ [Å$^{-1}$]")

    if plot_range_kx_ky is not None:
        plot_range_kx_ky = np.array(plot_range_kx_ky)
        if plot_range_kx_ky.ndim == 0:
            plot_range_kx_ky = np.array((plot_range_kx_ky, plot_range_kx_ky))
        ax.set_xlim((-plot_range_kx_ky[0], plot_range_kx_ky[0]))
        ax.set_ylim((-plot_range_kx_ky[1], plot_range_kx_ky[1]))
    else:
        k_range = 1.05 * np.sqrt(
            np.max(bragg_peaks.data["qx"] ** 2 + bragg_peaks.data["qy"] ** 2)
        )
        ax.set_xlim((-k_range, k_range))
        ax.set_ylim((-k_range, k_range))

    ax.invert_yaxis()
    ax.set_box_aspect(1)
    ax.xaxis.tick_top()

    if showTicks:

        ax.set_xticks([-1.5, 0, 1.5])
        ax.set_yticks([-1.5, 0, 1.5])

    else:

        ax.set_xticks([])
        ax.set_yticks([])

    # Labels for all peaks
    if add_labels is True:
        text_params = {
            "ha": "center",
            "va": "center",
            "family": "sans-serif",
            "fontweight": "normal",
            "color": "r",
            "size": 10,
        }

        def overline(x):
            return str(x) if x >= 0 else (r"\overline{" + str(np.abs(x)) + "}")

        for a0 in range(bragg_peaks.data.shape[0]):
            h = bragg_peaks.data["h"][a0]
            k = bragg_peaks.data["k"][a0]
            l = bragg_peaks.data["l"][a0]

            ax.text(
                bragg_peaks.data["qy"][a0],
                bragg_peaks.data["qx"][a0]
                - shift_labels
                - shift_marker * np.sqrt(marker_size[a0]),
                "$" + overline(h) + overline(k) + overline(l) + "$",
                **text_params,
            )

    # Force plot to have 1:1 aspect ratio
    ax.set_aspect("equal")

    ax.tick_params(
        axis="both",      # x and y
        which="major",   # major ticks
        length=11,         # tick length
        width=0.8,        # tick thickness
        labelsize=26      # tick-label font size
    )

    # if input_fig_handle is None:
    #     plt.show()

    if returnfig:
        return fig, ax

def return_BD_positions_and_intensities(input_polar_table, threshold = 1e-8):
    
    radial_distances = np.copy(input_polar_table[:,0])
    polar_angles = np.copy(input_polar_table[:,1])
    intensities = np.copy(input_polar_table[:,2])
    
    BD_indices = np.intersect1d(np.where(radial_distances > threshold)[0], np.where(intensities > threshold)[0])
    
    numberOfBraggDisks_to_displace = len(BD_indices)
    
    radial_distances_BDs = radial_distances[BD_indices]
    polar_angles_BDs = polar_angles[BD_indices]
    intensities_BDs = intensities[BD_indices]
    
    qx_BDs = radial_distances_BDs * np.cos(polar_angles_BDs)
    qy_BDs = radial_distances_BDs * np.sin(polar_angles_BDs)

    return np.stack((qx_BDs, qy_BDs, intensities_BDs)).T


def return_py4DSTEM_point_list(bragg_vectors_np):
    
    # Prepare the data structure for maxima with dtype (qx, qy, intensity)
    dtype = np.dtype([("x", float), ("y", float), ("intensity", float)])
    maxima = np.zeros(len(bragg_vectors_np), dtype=dtype)

    for i, bragg_vector in enumerate(bragg_vectors_np):
        maxima["x"][i] = bragg_vector[0]
        maxima["y"][i] = bragg_vector[1]
        maxima["intensity"][i] = bragg_vector[2]
    
    # Create QPoints object with already calibrated data
    maxima = QPoints(maxima)

    return maxima



def remove_direct_beam_and_normalize(py4DSTEM_bragg_disk_object):

    qx = np.copy(py4DSTEM_bragg_disk_object.data["qx"])
    qy = np.copy(py4DSTEM_bragg_disk_object.data["qy"])
    intensity = np.copy(py4DSTEM_bragg_disk_object.data["intensity"])

    k_radial_distnaces_of_BPs = np.linalg.norm(np.stack((qx, qy)).T, axis = 1)
    index_of_direct_beam = np.argmin(k_radial_distnaces_of_BPs)

    qx = np.delete(qx, index_of_direct_beam)
    qy = np.delete(qy, index_of_direct_beam)
    intensity = np.delete(intensity, index_of_direct_beam)
    intensity = intensity / np.max(intensity)

    Bragg_disk_list = np.stack((qx, qy, intensity)).T

    return Bragg_disk_list



simulation_output_dir = "/home/kwang/Desktop/Storage/project/p03_orientation_mapping/figure/figure_02/revision_sensitivity_analyses/output/"
prediction_output_dir = "/home/kwang/Desktop/Storage/project/p03_orientation_mapping/figure/figure_02/revision_sensitivity_analyses/output/predict_using_trained_model/"

label_rotation_matrices = np.load(simulation_output_dir + "postProc2_output_label_orignial.npy")

pred_rotation_matrices = np.load(prediction_output_dir + "predicted_rotation_matrices_trained_valid.npy")
loss_for_predictions = np.load(prediction_output_dir + "geodesic_distances_trained_valid.npy")

number_of_Braqgg_disks = np.load(simulation_output_dir + "postProc2_total_BD_number_count_for_each_orientation_matrix.npy")

input_arrays = np.load(simulation_output_dir + "postProc2_input_array.npy")


device = "cpu"
point_group_op_matrices = cubic_proper_point_group_operations()
point_group_op_matrices = point_group_op_matrices.to(device)

Cu_cif_path = "../"

k_max = 0.0328 * 64
accelerating_voltage = int(300e3)
crystal = py4DSTEM.process.diffraction.Crystal.from_CIF(Cu_cif_path + "Cu_fcc.cif")
crystal.setup_diffraction(accelerating_voltage)
crystal.calculate_structure_factors(k_max)


# Create an orientation plan for [0001] WS2
crystal.orientation_plan(
    angle_step_zone_axis = 2,
    angle_step_in_plane = 2,
    accel_voltage = accelerating_voltage,
    corr_kernel_size=0.08,
    zone_axis_range='auto',
)

index_for_000_reciprocal_space = np.where(crystal.g_vec_leng<1e-16)[0]
intensity_of_direct_beam = crystal.struct_factors_int[index_for_000_reciprocal_space][0]
direct_beam = np.array([0.0, 0.0, intensity_of_direct_beam], dtype = np.float32)

range_plot = [k_max, k_max]
scale_markers = 1800
scale_markers_compare = 4e3
excitation_error_threshold = 0.03

total_symmetry_aware_geodesic_loss_transformer = []
total_symmetry_aware_geodesic_loss_py4DSTEM = []

overall_loss_for_label = []
overall_loss_for_transformer = []
overall_loss_for_py4DSTEM = []

indices = np.where(number_of_Braqgg_disks == 2)[0]

for index in indices:

    if index < 2e3:

        ### Retreive orientation label and prediction from tranformer.
        label = label_rotation_matrices[index]
        prediction = pred_rotation_matrices[index]

        polar_table = input_arrays[index]

        input_BDs_np = return_BD_positions_and_intensities(polar_table)
        input_BDs_np_BraggVector = return_py4DSTEM_point_list(input_BDs_np)

        input_BDs_np_for_py4DSTEM = np.vstack((input_BDs_np, direct_beam))
        input_BDs_np_for_py4DSTEM_BraggVector = return_py4DSTEM_point_list(input_BDs_np_for_py4DSTEM)


        

        py4DSTEM_ACOM = crystal.match_single_pattern(
                                input_BDs_np_for_py4DSTEM_BraggVector,
                                verbose = False,
        )

        py4DSTEM_ACOM_predicted_orientation = py4DSTEM_ACOM.matrix[0]

        symmetry_aware_geodesic_loss_transformer = loss_for_predictions[index]


        computed_loss_transformer = compute_symmetry_aware_geodesic_loss_between_two_rotation_matrices(prediction, label, point_group_op_matrices)
        
        # Assert that the difference is LESS THAN OR EQUAL to 1e-4
        assert abs(symmetry_aware_geodesic_loss_transformer - computed_loss_transformer) <= 1e-4, \
            f"Loss mismatch! Expected {symmetry_aware_geodesic_loss_transformer:.6f}, got {computed_loss_transformer:.6f}"
        
        computed_loss_py4DSTEM = compute_symmetry_aware_geodesic_loss_between_two_rotation_matrices(py4DSTEM_ACOM_predicted_orientation, label, point_group_op_matrices)
        
        total_symmetry_aware_geodesic_loss_transformer.append(computed_loss_transformer)
        total_symmetry_aware_geodesic_loss_py4DSTEM.append(computed_loss_py4DSTEM)


        # print("orientation label\n", label)
        
    
        bragg_peaks_fit = crystal.generate_diffraction_pattern(
                                        orientation_matrix = label,
                                        ind_orientation=0,
                                        sigma_excitation_error=excitation_error_threshold) 

    
        bragg_peaks_fit_transformer = crystal.generate_diffraction_pattern(
                                        orientation_matrix = prediction,
                                        ind_orientation=0,
                                        sigma_excitation_error=excitation_error_threshold)

    
        bragg_peaks_fit_py4DSTEM = crystal.generate_diffraction_pattern(
                                        orientation_matrix = py4DSTEM_ACOM_predicted_orientation,
                                        ind_orientation=0,
                                        sigma_excitation_error=excitation_error_threshold)
    

        observation_normalized = remove_direct_beam_and_normalize(input_BDs_np_for_py4DSTEM_BraggVector)
        kinematic_label_normalized = remove_direct_beam_and_normalize(bragg_peaks_fit)
        kinematic_transformer_normalized = remove_direct_beam_and_normalize(bragg_peaks_fit_transformer)
        kinematic_py4DSTEM_normalized = remove_direct_beam_and_normalize(bragg_peaks_fit_py4DSTEM)

        loss_for_label = soft_disk_loss.soft_disk_loss(
                                observation_normalized,
                                kinematic_label_normalized,
        )

        overall_loss_for_label.append(loss_for_label)
        

        loss_for_transformer = soft_disk_loss.soft_disk_loss(
                                observation_normalized,
                                kinematic_transformer_normalized,
        )

        overall_loss_for_transformer.append(loss_for_transformer)

        loss_for_py4DSTEM = soft_disk_loss.soft_disk_loss(
                                observation_normalized,
                                kinematic_py4DSTEM_normalized,
        )

        overall_loss_for_py4DSTEM.append(loss_for_py4DSTEM)


overall_loss_for_label = np.array(overall_loss_for_label)
overall_loss_for_transformer = np.array(overall_loss_for_transformer)
overall_loss_for_py4DSTEM = np.array(overall_loss_for_py4DSTEM)

delta_loss_for_transformer = overall_loss_for_transformer - overall_loss_for_label
delta_loss_for_py4DSTEM = overall_loss_for_py4DSTEM - overall_loss_for_label


nbins = 50

# 1. Create a figure with 2 rows, 1 column. 
# Adjusted the height to 6 to stack them comfortably.
# sharex=True perfectly aligns their x-axes so you can compare the distributions.
# sharey=True ensures the vertical scale (fraction heights) is also identical.
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True, sharey=True)

x_ticks = np.array([0.0, 0.5, 1.0])

# ==========================================
# TOP PANEL (ax1): Transformer
# ==========================================
weights_transformer = np.ones_like(total_symmetry_aware_geodesic_loss_transformer) / len(total_symmetry_aware_geodesic_loss_transformer)

ax1.hist(
    total_symmetry_aware_geodesic_loss_transformer, 
    bins=nbins, 
    weights=weights_transformer,
    # alpha=0.9, 
    color="#F55B27", 
    edgecolor="black", 
    linewidth=0.5,
    label="Transformer"
)

ax1.set_ylabel("Fraction", fontsize=24)
ax1.set_xlim(-0.01, 1.07)
ax1.tick_params(axis='both', which='major', labelsize=29, length=13, width=0.9)

# Optional: Add a legend or title to the top panel
ax1.legend(fontsize=24, frameon=False, loc="upper right")


# ==========================================
# BOTTOM PANEL (ax2): py4DSTEM
# ==========================================
weights_py4DSTEM = np.ones_like(total_symmetry_aware_geodesic_loss_py4DSTEM) / len(total_symmetry_aware_geodesic_loss_py4DSTEM)

ax2.hist(
    total_symmetry_aware_geodesic_loss_py4DSTEM, 
    bins=nbins, 
    weights=weights_py4DSTEM,
    # alpha=0.9, 
    color="#2164EB", 
    edgecolor="black", 
    linewidth=0.5,
    label="py4DSTEM"
)

# Only the bottom panel needs the x-axis label when sharex=True
ax2.set_xlabel("Geodesic Loss", fontsize=29)
ax2.set_ylabel("Fraction", fontsize=27)

ax2.set_xticks(x_ticks)
ax2.tick_params(axis='both', which='major', labelsize=29, length=13, width=0.9)

ax2.legend(fontsize=24, frameon=False, loc="upper right")

# Apply tight layout to prevent overlapping text between the top and bottom panels
plt.tight_layout()

fig.savefig(
    prediction_output_dir + 'geodesicLoss_tranfsormer_py4DSTEM.pdf',
    format='pdf',
    bbox_inches='tight',
    dpi=400,
)

fig.savefig(
    prediction_output_dir + 'geodesicLoss_tranfsormer_py4DSTEM.png',
    bbox_inches='tight',
    dpi=400,
)
plt.close(fig)


fig, ax = plt.subplots(figsize=(7, 6.05))

weights_transformer = np.ones_like(delta_loss_for_transformer) / len(delta_loss_for_transformer)

ax.hist(
    delta_loss_for_transformer, 
    bins=nbins, 
    weights=weights_transformer,
    # alpha=0.9, 
    color="#F55B27", 
    edgecolor="black", 
    linewidth=0.5,
    label="transformer"
)

# # 2. Plot the vertical lines for the median and quantile
# ax.axvline(
#     median_val, 
#     color='black', 
#     linestyle='dashed', 
#     linewidth=2, 
#     label=f'Median: {median_val:.4f}'
# )

ax.set_xlabel(r'$L_{\mathrm{soft, transformer}} - L_{\mathrm{soft, label}}$', fontsize=33)
ax.set_ylabel("Fraction of Predictions", fontsize=28)
# ax.set_title("Distribution of Orientation Prediction Errors", fontsize=16, pad=15)



ax.tick_params(
    axis='both',
    which='major',
    labelsize=29,
    length=13,
    width=0.9,
)

plt.tight_layout()

fig.savefig(
    prediction_output_dir + 'softloss_difference_transformer_sub_label.pdf',
    format='pdf',
    bbox_inches='tight',
    dpi=400,
)

fig.savefig(
    prediction_output_dir + 'softloss_difference_transformer_sub_label.png',
    bbox_inches='tight',
    dpi=400,
)



plt.close(fig)



######## plot the case where loss appear high but in fact the predictions are not bad



for index in indices:



    ### Retreive orientation label and prediction from tranformer.
    label = label_rotation_matrices[index]
    prediction = pred_rotation_matrices[index]

    polar_table = input_arrays[index]

    input_BDs_np = return_BD_positions_and_intensities(polar_table)
    input_BDs_np_BraggVector = return_py4DSTEM_point_list(input_BDs_np)

    input_BDs_np_for_py4DSTEM = np.vstack((input_BDs_np, direct_beam))
    input_BDs_np_for_py4DSTEM_BraggVector = return_py4DSTEM_point_list(input_BDs_np_for_py4DSTEM)


    

    py4DSTEM_ACOM = crystal.match_single_pattern(
                            input_BDs_np_for_py4DSTEM_BraggVector,
                            verbose = False,
    )

    py4DSTEM_ACOM_predicted_orientation = py4DSTEM_ACOM.matrix[0]

    symmetry_aware_geodesic_loss_transformer = loss_for_predictions[index]



    # print("")
    # print("------------------------------------------------------------------------------------------------------------\n\n")
    # print("symmetry_aware_geodesic_loss_transformer", symmetry_aware_geodesic_loss_transformer, "\n")
    computed_loss_transformer = compute_symmetry_aware_geodesic_loss_between_two_rotation_matrices(prediction, label, point_group_op_matrices)
    
    # Assert that the difference is LESS THAN OR EQUAL to 1e-4
    assert abs(symmetry_aware_geodesic_loss_transformer - computed_loss_transformer) <= 1e-4, \
        f"Loss mismatch! Expected {symmetry_aware_geodesic_loss_transformer:.6f}, got {computed_loss_transformer:.6f}"
    
    computed_loss_py4DSTEM = compute_symmetry_aware_geodesic_loss_between_two_rotation_matrices(py4DSTEM_ACOM_predicted_orientation, label, point_group_op_matrices)
    


    # print("orientation label\n", label)
    

    bragg_peaks_fit = crystal.generate_diffraction_pattern(
                                    orientation_matrix = label,
                                    ind_orientation=0,
                                    sigma_excitation_error=excitation_error_threshold) 



    # print("orientation prediction\n", prediction)

    bragg_peaks_fit_transformer = crystal.generate_diffraction_pattern(
                                    orientation_matrix = prediction,
                                    ind_orientation=0,
                                    sigma_excitation_error=excitation_error_threshold)

    


    # print("py4DSTEM orientation prediction\n", py4DSTEM_ACOM_predicted_orientation)

    

    bragg_peaks_fit_py4DSTEM = crystal.generate_diffraction_pattern(
                                    orientation_matrix = py4DSTEM_ACOM_predicted_orientation,
                                    ind_orientation=0,
                                    sigma_excitation_error=excitation_error_threshold)


    observation_normalized = remove_direct_beam_and_normalize(input_BDs_np_for_py4DSTEM_BraggVector)
    kinematic_label_normalized = remove_direct_beam_and_normalize(bragg_peaks_fit)
    kinematic_transformer_normalized = remove_direct_beam_and_normalize(bragg_peaks_fit_transformer)
    kinematic_py4DSTEM_normalized = remove_direct_beam_and_normalize(bragg_peaks_fit_py4DSTEM)

    loss_for_label = soft_disk_loss.soft_disk_loss(
                            observation_normalized,
                            kinematic_label_normalized,
    )

    

    loss_for_transformer = soft_disk_loss.soft_disk_loss(
                            observation_normalized,
                            kinematic_transformer_normalized,
    )


    loss_for_py4DSTEM = soft_disk_loss.soft_disk_loss(
                            observation_normalized,
                            kinematic_py4DSTEM_normalized,
    )


    #####################################################################################################################################
    #####################################################################################################################################
    ##########################################################     Plot     ##########################################################
    #####################################################################################################################################
    #####################################################################################################################################


    if index in [0, 35, 37, 45, 58, 61, 85, 96]:
        print("index", index)
        print("symmetry_aware_geodesic_loss transformer:", symmetry_aware_geodesic_loss_transformer)
        print("computed_loss_transformer", computed_loss_transformer)
        print("computed_loss_py4DSTEM", computed_loss_py4DSTEM)
        print("soft loss transformer: ", loss_for_transformer)
        print("soft loss py4DSTEM: ", loss_for_py4DSTEM)
        

        fig, ax = plot_labelINPUT_labelBD_diffraction_pattern(
                    bragg_peaks_fit,
                    sim_BD_marker = "p",
                    sim_BD_color = "#007544",
                    bragg_peaks_compare=input_BDs_np_BraggVector,
                    scale_markers=scale_markers*0.71,
                    scale_markers_compare=scale_markers_compare,
                    plot_range_kx_ky=range_plot,
                    min_marker_size=3,
                    figsize = (5,5),
                    shift_labels = 0.2,
                    add_labels = False,
                    returnfig = True,
        )
        fig.savefig(prediction_output_dir + 'index_%d_kimSim_input_and_BD_from_label.pdf'%(index), bbox_inches='tight', dpi=300)
        plt.close(fig)



        fig, ax = plot_labelINPUT_labelBD_diffraction_pattern(
                    bragg_peaks_fit_transformer,
                    sim_BD_marker = "1",
                    sim_BD_color = "#F55B27",
                    sim_BD_linewidth = 2,
                    bragg_peaks_compare=input_BDs_np_BraggVector,
                    scale_markers=scale_markers * 1.2,
                    scale_markers_compare=scale_markers_compare,
                    plot_range_kx_ky=range_plot,
                    min_marker_size=3,
                    figsize = (5,5),
                    shift_labels = 0.2,
                    add_labels = False,
                    returnfig = True,
        )
        fig.savefig(prediction_output_dir + 'index_%d_kimSim_input_and_BD_from_transformer.pdf'%(index), bbox_inches='tight', dpi=300)
        plt.close(fig)

        fig, ax = plot_labelINPUT_labelBD_diffraction_pattern(
                    bragg_peaks_fit_py4DSTEM,
                    sim_BD_marker = "+",
                    sim_BD_color = "#2164EB",
                    sim_BD_linewidth = 2,
                    bragg_peaks_compare=input_BDs_np_BraggVector,
                    scale_markers=scale_markers,
                    scale_markers_compare=scale_markers_compare,
                    plot_range_kx_ky=range_plot,
                    min_marker_size=3,
                    figsize = (5,5),
                    shift_labels = 0.2,
                    add_labels = False,
                    returnfig = True,
        )
        fig.savefig(prediction_output_dir + 'index_%d_kimSim_input_and_BD_from_py4DSTEM.pdf'%(index), bbox_inches='tight', dpi=300)
        plt.close(fig)
    