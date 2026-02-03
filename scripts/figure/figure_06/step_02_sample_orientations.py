import py4DSTEM
import numpy as np
import math
import argparse
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

############ STEP 0. DEFINE functions and modules ############

class MyCustomError(Exception):
    """Custom exception for specific error conditions."""
    pass


def rotation_wrt_zAxis(angle_in_rad):
    return np.array(
                    [
                        [np.cos(angle_in_rad), np.sin(angle_in_rad), 0],
                        [-np.sin(angle_in_rad), np.cos(angle_in_rad), 0],
                        [0, 0, 1],
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

def sample_zone_axes_on_unit_sphere(number_of_zone_axes_to_sample):
    # Generate N uniform samples for u and theta
    u = np.random.uniform(-1, 1, number_of_zone_axes_to_sample)
    theta = np.random.uniform(0, 2 * np.pi, number_of_zone_axes_to_sample)
    
    # Compute the spherical coordinates for each sample
    x = np.sqrt(1 - u**2) * np.cos(theta)
    y = np.sqrt(1 - u**2) * np.sin(theta)
    z = u
    
    # Stack the coordinates into an Nx3 array
    points = np.vstack((x, y, z)).T
    
    return points

def sample_polarAngles_on_unit_circle(number_of_polar_angles_to_sample):
    # Sample N random angles uniformly between 0 and 2*pi
    theta = np.random.uniform(0, 2 * np.pi, number_of_polar_angles_to_sample)
    return theta

def parse_args():
    parser = argparse.ArgumentParser(description="number for sampling random orienations")
    parser.add_argument("--num_orientations", type = int, help="number of randomly sampled orientations", default = int(60))
    # parser.add_argument("--random_seed", type = int, help="random seed for numpy random number generator", default = int(77))
    parser.add_argument("--random_seed", type = int, help="random seed for numpy random number generator", default = int(7))
 
    return parser.parse_args()

def main():
    
    args = parse_args()
    number_of_orientations_to_sample = args.num_orientations
    
    random_seed = args.random_seed
    np.random.seed(random_seed) 
    
    print("")
    print("Action 1. Sampling random orientation (rotation) matrix (START)\n")
        
    randomly_sampled_zone_axes = sample_zone_axes_on_unit_sphere(number_of_orientations_to_sample)
    randomly_sampled_inPlane_angle = sample_polarAngles_on_unit_circle(number_of_orientations_to_sample)
    
    
    randomly_sampled_orientation_matrices = []
    
    for i in range(number_of_orientations_to_sample):
        zone_axes_rotation_matrix = calculate_rotation_matrix_for_zone_axis(randomly_sampled_zone_axes[i])
        
        in_plane_rotation_matrix = rotation_wrt_zAxis(randomly_sampled_inPlane_angle[i])
    
        rotation_matrix = zone_axes_rotation_matrix @ in_plane_rotation_matrix
        
        randomly_sampled_orientation_matrices.append(rotation_matrix)
        
        assert math.isclose(np.linalg.norm(randomly_sampled_zone_axes[i]), 1.0, abs_tol = 1e-5), "zone axis does not lie on unit sphere"
        assert math.isclose(np.linalg.det(rotation_matrix), 1.0, abs_tol = 1e-5), "randomly sampled rotation matrix is not SO3 proper rotation"
        
        
    
    randomly_sampled_orientation_matrices = np.array(randomly_sampled_orientation_matrices)
    np.save("./randomly_sampled_%d_orientation_matrices_SO3.npy"%(number_of_orientations_to_sample), randomly_sampled_orientation_matrices)
    
    print("sampled and saved ", randomly_sampled_orientation_matrices.shape[0], " orientations.\n")
    print("Action 1. Sampling random orientation (rotation) matrix (END)\n\n")
    
    print("JOB DONE.")
    

if __name__ == "__main__":
    main()
