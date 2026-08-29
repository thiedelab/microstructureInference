import matplotlib.pyplot as plt
import numpy as np

def return_stdError_and_populationSTD(sample_std, N):
    # 1. Convert sample standard deviation to population standard deviation
    sigma_pop = sample_std * np.sqrt((N - 1) / N)
    
    # 2. Calculate standard error using the SAMPLE standard deviation
    standard_error = sample_std / np.sqrt(N)
    
    return sigma_pop, standard_error

number_dir = "/home/kwang/Desktop/Storage/project/p03_orientation_mapping/figure/figure_02/revision_sensitivity_analyses/output/"
geodesic_loss_dir = "/home/kwang/Desktop/Storage/project/p03_orientation_mapping/figure/figure_02/revision_sensitivity_analyses/output/predict_using_trained_model/"


numbers = np.load(number_dir + "postProc2_total_BD_number_count_for_each_orientation_matrix.npy")
geodesic_losses = np.load(geodesic_loss_dir + "geodesic_distances_trained_valid.npy")

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



# Create the x-axis array from 2 to 35
x_disks = np.arange(2, 37, 1)

fig, ax = plt.subplots(figsize=(9, 6.6))

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
ax.set_xlabel('Number of Bragg disks', fontsize=34)
ax.set_ylabel('Geodesic Loss', fontsize=36)

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

# Save the figure in both vector and raster formats
fig.savefig(geodesic_loss_dir + 'average_loss_vs_disks_with_error.pdf', format='pdf', bbox_inches='tight')
fig.savefig(geodesic_loss_dir + 'average_loss_vs_disks_with_error.png', bbox_inches='tight', dpi=400)

plt.close(fig)
