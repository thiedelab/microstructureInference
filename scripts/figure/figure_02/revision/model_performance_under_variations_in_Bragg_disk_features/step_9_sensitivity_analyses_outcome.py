import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker

def return_stdError_and_populationSTD(std_geodesic_error, N):


    sigma_pop = std_geodesic_error * np.sqrt((N - 1) / N)
    standard_error = sigma_pop / np.sqrt(N)
    return sigma_pop, standard_error


def return_average_and_std_error(geodesic_losses, numbers):

    total_average = []
    total_standard_error = []
    
    for N in range(2,37,1):
        indices_where_number_Bragg_disks_is_N = np.where(numbers == N)[0]
    
        geodesic_loss_for_the_Bragg_disks_is_N = geodesic_losses[indices_where_number_Bragg_disks_is_N]
    
        average_geodesic_loss = np.average(geodesic_loss_for_the_Bragg_disks_is_N)
    
        std_geodesic_loss = geodesic_loss_for_the_Bragg_disks_is_N.std(ddof=1).item()
    
        population_error, standard_error = return_stdError_and_populationSTD(std_geodesic_loss, len(indices_where_number_Bragg_disks_is_N))
    
        total_average.append(average_geodesic_loss)
        total_standard_error.append(standard_error)

    total_average = np.array(total_average)
    total_standard_error = np.array(total_standard_error)


    return total_average, total_standard_error

def return_fig_ax_for_Lgeo_vs_numbers(total_average, total_standard_error, x_disks = np.arange(2, 37, 1)):
    
    fig, ax = plt.subplots(figsize=(8.5, 7))
    
    # Plot the average with standard error bars
    # fmt='-o' creates a line with circular markers for the data points
    # capsize adds the little horizontal caps to the top and bottom of the error bars
    ax.errorbar(
        x_disks, 
        total_average, 
        yerr=total_standard_error, 
        fmt='-o', 
        color='black', 
        ecolor='gray',       # Make error bars slightly lighter than the main points
        capsize=4,           # Width of the caps on the error bars
        elinewidth=1.5,      # Thickness of the error bar lines
        markersize=6, 
        alpha=0.9
    )
    
    # Format the axes
    ax.set_xlabel('Number of Bragg disks', fontsize=33)
    ax.set_ylabel('Average Geodesic Loss', fontsize=33)
    
    # Set the x-limits slightly wider than the data so points don't get cut off
    ax.set_xlim(0, 37) 
    
    # Set appropriate x-ticks
    x_ticks = np.array([5, 15, 25, 35])
    ax.set_xticks(x_ticks)
    
    # Add a grid to make it easier to read
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Make tick labels large and readable
    ax.tick_params(
        axis='both',
        which='major',
        labelsize=30,
        length=11,
        width=0.9,
    )

    plt.tight_layout()

    return fig, ax

def plot_sensitivity_heatmap_mark_boundary_for_Lgeo_theshold(averaged_data, scaling_factors, x_disks, title, ylabel, vmin, vmax, cbar_label = r'$L_{\mathrm{geo, noised}} - L_{\mathrm{geo}}$', threshold=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 1. ENFORCE GLOBAL COLOR SCALE using vmin and vmax
    im = ax.imshow(averaged_data, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    
    # 2. MARK THE RECTANGULAR BOUNDARY IF A THRESHOLD IS PROVIDED
    if threshold is not None:
        # Create a boolean mask where the value is less than the threshold
        mask = averaged_data < threshold
        rows, cols = mask.shape
        
        # Iterate over the grid to draw borders only on the outer edges of the masked region
        for i in range(rows):
            for j in range(cols):
                if mask[i, j]:
                    # In imshow with origin='lower', pixel (i,j) spans x from j-0.5 to j+0.5, 
                    # and y from i-0.5 to i+0.5
                    
                    # Top edge (check if top neighbor is outside mask or grid)
                    if i == rows - 1 or not mask[i+1, j]:
                        ax.plot([j-0.5, j+0.5], [i+0.5, i+0.5], color='red', linewidth=3)
                    
                    # Bottom edge (check if bottom neighbor is outside mask or grid)
                    if i == 0 or not mask[i-1, j]:
                        ax.plot([j-0.5, j+0.5], [i-0.5, i-0.5], color='red', linewidth=3)
                        
                    # Right edge (check if right neighbor is outside mask or grid)
                    if j == cols - 1 or not mask[i, j+1]:
                        ax.plot([j+0.5, j+0.5], [i-0.5, i+0.5], color='red', linewidth=3)
                        
                    # Left edge (check if left neighbor is outside mask or grid)
                    if j == 0 or not mask[i, j-1]:
                        ax.plot([j-0.5, j-0.5], [i-0.5, i+0.5], color='red', linewidth=3)
    
    # 3. LOCK THE PLOT DIMENSIONS so the colorbar doesn't steal space unevenly
    divider = make_axes_locatable(ax)
    cax_cbar = divider.append_axes("right", size="5%", pad=0.15)
    
    # 4. Apply colorbar to the locked axis
    cbar = fig.colorbar(im, cax=cax_cbar)
    cbar.set_label(cbar_label, fontsize=34)
    cbar.ax.tick_params(labelsize=33, length=12, width=0.7)
    
    # --- Format X-axis (Bragg Disks) ---
    ax.set_xlabel('Number of Bragg disks', fontsize=34)
    x_tick_indices = np.arange(0, len(x_disks), 6) 
    ax.set_xticks(x_tick_indices)
    ax.set_xticklabels(x_disks[x_tick_indices])
    
    # --- Format Y-axis (Scaling Factors) ---
    ax.set_ylabel(ylabel, fontsize=40)
    y_tick_indices = np.arange(0, len(scaling_factors), 4)
    ax.set_yticks(y_tick_indices)
    ax.set_yticklabels([f"{val:.2f}" for val in scaling_factors[y_tick_indices]])
    
    # Set tick parameters
    ax.tick_params(axis='both', which='major', labelsize=36, length=12, width=0.7)
    
    ax.set_title(title, fontsize=33, pad=15, x=0.5)
    plt.tight_layout()
    
    return fig, ax

def plot_sensitivity_heatmap(averaged_data, scaling_factors, x_disks, title, ylabel, vmin, vmax, cbar_label = r'$L_{\mathrm{geo, noised}} - L_{\mathrm{geo}}$'):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 1. ENFORCE GLOBAL COLOR SCALE using vmin and vmax
    im = ax.imshow(averaged_data, aspect='auto', origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)
    
    # 2. LOCK THE PLOT DIMENSIONS so the colorbar doesn't steal space unevenly
    divider = make_axes_locatable(ax)
    cax_cbar = divider.append_axes("right", size="5%", pad=0.15)
    
    # 3. Apply colorbar to the locked axis
    cbar = fig.colorbar(im, cax=cax_cbar)
    cbar.set_label(cbar_label, fontsize=34)
    cbar.ax.tick_params(labelsize=33, length=12, width=0.7)
    
    # cbar.formatter.set_scientific(True)
    # cbar.formatter.set_powerlimits((0, 0))
    # cbar.formatter.set_useMathText(True)
    # cbar.update_ticks()
    # cbar.ax.yaxis.get_offset_text().set_fontsize(26)
    
    # --- Format X-axis (Bragg Disks) ---
    ax.set_xlabel('Number of Bragg disks', fontsize=34)
    x_tick_indices = np.arange(0, len(x_disks), 6) 
    ax.set_xticks(x_tick_indices)
    ax.set_xticklabels(x_disks[x_tick_indices])
    
    # --- Format Y-axis (Scaling Factors) ---
    ax.set_ylabel(ylabel, fontsize=40)
    y_tick_indices = np.arange(0, len(scaling_factors), 4)
    ax.set_yticks(y_tick_indices)
    ax.set_yticklabels([f"{val:.2f}" for val in scaling_factors[y_tick_indices]])
    
    # Set tick parameters
    ax.tick_params(axis='both', which='major', labelsize=36, length=12, width=0.7)
    
    ax.set_title(title, fontsize=33, pad=15, x=0.5)
    plt.tight_layout()
    
    return fig, ax

geodesic_distances_dir = "/home/kwang/Desktop/Storage/project/p03_orientation_mapping/figure/figure_02/revision_sensitivity_analyses/output/noised_digitized/"
number_dir = "/home/kwang/Desktop/Storage/project/p03_orientation_mapping/figure/figure_02/revision_sensitivity_analyses/output/"
numbers = np.load(number_dir + "postProc2_total_BD_number_count_for_each_orientation_matrix.npy")

reference_geodesic_loss_dir = "/home/kwang/Desktop/Storage/project/p03_orientation_mapping/figure/figure_02/revision_sensitivity_analyses/output/predict_using_trained_model/"
reference_geodesic_losses_no_noise = np.load(reference_geodesic_loss_dir + "geodesic_distances_trained_valid.npy")
reference_Lgeo_average, reference_Lgeo_stdErr = return_average_and_std_error(reference_geodesic_losses_no_noise, numbers)

cartesian_displacement_scaling_factor_unit = 0.20
intensity_displacement_scaling_factor_unit = 0.05

cartesian_displacement_scaling_factors = np.arange(cartesian_displacement_scaling_factor_unit, cartesian_displacement_scaling_factor_unit * 19 ,cartesian_displacement_scaling_factor_unit)
intensity_displacement_scaling_factors = np.arange(intensity_displacement_scaling_factor_unit, intensity_displacement_scaling_factor_unit * 18 ,intensity_displacement_scaling_factor_unit)


diffraction_space_pixel_size = 0.0328
intensityDisp_std_reference = 1.0


max_displacement_trials_begin = 0
max_displacement_trials_end = 20

tables_for_geodesic_loss_cartesian_position_displaced = np.zeros((cartesian_displacement_scaling_factors.shape[0], max_displacement_trials_end, np.unique(numbers).shape[0]), dtype = np.float32)
tables_for_geodesic_loss_intensity_displaced = np.zeros((intensity_displacement_scaling_factors.shape[0], max_displacement_trials_end, np.unique(numbers).shape[0]), dtype = np.float32)



# Displace each parameter individually.


    # displace positions only:
for scaling_factor_index, cartesian_displacement_scaling_factor in enumerate(cartesian_displacement_scaling_factors):

    cartesianDisp_std = diffraction_space_pixel_size * cartesian_displacement_scaling_factor

    for seedValue in range(max_displacement_trials_begin + 1, max_displacement_trials_end + 1, 1):

        # print("Trial:", seedValue)

        geod_dist_for_position_noised = np.load(geodesic_distances_dir + "postProc5_input_array_noised_cartesianOnly_cartScale%3.2f_seed%d_digitized_geodesic_distances.npy"%(cartesian_displacement_scaling_factor, seedValue))

        cartDisp_Lgeo_average, cartDisp_Lgeo_stdErr = return_average_and_std_error(geod_dist_for_position_noised, numbers)

        tables_for_geodesic_loss_cartesian_position_displaced[scaling_factor_index, seedValue - 1, :] = cartDisp_Lgeo_average

        # fig, ax = return_fig_ax_for_Lgeo_vs_numbers(cartDisp_Lgeo_average, cartDisp_Lgeo_stdErr)

        # Save the figure in both vector and raster formats
        # fig.savefig(geodesic_loss_dir + 'average_loss_vs_disks_with_error.pdf', format='pdf', bbox_inches='tight')
        # fig.savefig(geodesic_loss_dir + 'average_loss_vs_disks_with_error.png', bbox_inches='tight', dpi=400)
        
        # plt.show()

    # displace intensities only:
for scaling_factor_index, intensity_displacement_scaling_factor in enumerate(intensity_displacement_scaling_factors):

    intensityDisp_std = intensityDisp_std_reference * intensity_displacement_scaling_factor

    for seedValue in range(max_displacement_trials_begin + 1, max_displacement_trials_end + 1, 1):

        # print("Trial:", seedValue)

        geod_dist_for_intensity_noised = np.load(geodesic_distances_dir + "postProc5_input_array_noised_intensityOnly_intenScale%3.2f_seed%d_digitized_geodesic_distances.npy"%(intensity_displacement_scaling_factor, seedValue))

        intensityDisp_Lgeo_average, intensityDisp_Lgeo_stdErr = return_average_and_std_error(geod_dist_for_intensity_noised, numbers)

        tables_for_geodesic_loss_intensity_displaced[scaling_factor_index, seedValue - 1, :] = intensityDisp_Lgeo_average

        # fig, ax = return_fig_ax_for_Lgeo_vs_numbers(intensityDisp_Lgeo_average, intensityDisp_Lgeo_stdErr)

        # Save the figure in both vector and raster formats
        # fig.savefig(geodesic_loss_dir + 'average_loss_vs_disks_with_error.pdf', format='pdf', bbox_inches='tight')
        # fig.savefig(geodesic_loss_dir + 'average_loss_vs_disks_with_error.png', bbox_inches='tight', dpi=400)
        
        # plt.show()

# Average across the 20 trials (axis=1) to get shape (19, 35)
cartesian_averaged_loss_across_trials = np.mean(tables_for_geodesic_loss_cartesian_position_displaced, axis=1)

# Average across the 20 trials (axis=1) to get shape (17, 35)
intensity_averaged_loss_across_trials = np.mean(tables_for_geodesic_loss_intensity_displaced, axis=1)

cartesian_loss_difference = cartesian_averaged_loss_across_trials - reference_Lgeo_average
intensity_loss_difference = intensity_averaged_loss_across_trials - reference_Lgeo_average

# X-axis values for labeling
x_disks = np.arange(2, 37, 1)



# Plot Cartesian Heat Map (passing the difference array)
# 1. Find the absolute minimum and maximum across BOTH datasets
global_vmin = min(np.min(cartesian_averaged_loss_across_trials), np.min(intensity_averaged_loss_across_trials))
global_vmax = max(np.max(cartesian_averaged_loss_across_trials), np.max(intensity_averaged_loss_across_trials))

# Define the absolute error bound threshold (approx 2 degrees)
absolute_error_bound = 3.5 * np.pi / 180.

# 2. Plot Cartesian Heat Map (Absolute Loss)
fig_cart, ax_cart = plot_sensitivity_heatmap_mark_boundary_for_Lgeo_theshold(
    averaged_data=cartesian_averaged_loss_across_trials, 
    scaling_factors=cartesian_displacement_scaling_factors, 
    x_disks=x_disks, 
    title='sensitivity to cartesian position noise', 
    ylabel= r'$\sigma_{\mathrm{position}}$ / $d_{\mathrm{pixel}}$',
    vmin=global_vmin,  
    vmax=global_vmax,  
    cbar_label = r'$L_{\mathrm{geo, noised}}$',
    threshold=absolute_error_bound # <-- ADDED THRESHOLD HERE
)
fig_cart.savefig(number_dir + 'sensitivity_to_cartesian_positions_noise_Lgeo.pdf', format='pdf', bbox_inches='tight')
fig_cart.savefig(number_dir + 'sensitivity_to_cartesian_positions_noise_Lgeo.png', bbox_inches='tight', dpi=400)
plt.close(fig_cart)

# 3. Plot Intensity Heat Map (Absolute Loss)
fig_inten, ax_inten = plot_sensitivity_heatmap_mark_boundary_for_Lgeo_theshold(
    averaged_data=intensity_averaged_loss_across_trials, 
    scaling_factors=intensity_displacement_scaling_factors, 
    x_disks=x_disks, 
    title='sensitivity to intensity noise', 
    ylabel= r'$\sigma_{\mathrm{intensity}}$ / $I_{\mathrm{max}}$',
    vmin=global_vmin,  
    vmax=global_vmax,  
    cbar_label = r'$L_{\mathrm{geo, noised}}$',
    threshold=absolute_error_bound # <-- ADDED THRESHOLD HERE
)
fig_inten.savefig(number_dir + 'sensitivity_to_intensity_noise_Lgeo.pdf', format='pdf', bbox_inches='tight')
fig_inten.savefig(number_dir + 'sensitivity_to_intensity_noise_Lgeo.png', bbox_inches='tight', dpi=400)
plt.close(fig_inten)

global_vmin = min(np.min(cartesian_loss_difference), np.min(intensity_loss_difference))
global_vmax = max(np.max(cartesian_loss_difference), np.max(intensity_loss_difference))
    
# 2. Plot Cartesian Heat Map
fig_cart, ax_cart = plot_sensitivity_heatmap(
    averaged_data=cartesian_loss_difference, 
    scaling_factors=cartesian_displacement_scaling_factors, 
    x_disks=x_disks, 
    title='sensitivity to cartesian position noise', 
    ylabel= r'$\sigma_{\mathrm{position}}$ / $d_{\mathrm{pixel}}$',
    vmin=global_vmin,  # <-- Pass global min
    vmax=global_vmax   # <-- Pass global max
)
fig_cart.savefig(number_dir + 'sensitivity_to_cartesian_positions_noise_DeltaLgeo.pdf', format='pdf', bbox_inches='tight')
fig_cart.savefig(number_dir + 'sensitivity_to_cartesian_positions_noise_DeltaLgeo.png', bbox_inches='tight', dpi=400)
plt.close(fig_cart)

# 3. Plot Intensity Heat Map
fig_inten, ax_inten = plot_sensitivity_heatmap(
    averaged_data=intensity_loss_difference, 
    scaling_factors=intensity_displacement_scaling_factors, 
    x_disks=x_disks, 
    title='sensitivity to intensity noise', 
    ylabel= r'$\sigma_{\mathrm{intensity}}$ / $I_{\mathrm{max}}$',
    vmin=global_vmin,  # <-- Pass global min
    vmax=global_vmax   # <-- Pass global max
)
fig_inten.savefig(number_dir + 'sensitivity_to_intensity_noise_DeltaLgeo.pdf', format='pdf', bbox_inches='tight')
fig_inten.savefig(number_dir + 'sensitivity_to_intensity_noise_DeltaLgeo.png', bbox_inches='tight', dpi=400)
plt.close(fig_inten)

print("Sensitivity analyses completed. Figures saved in the output directory.")


