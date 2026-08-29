#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  1 14:11:33 2026

@author: kwang
"""

import numpy as np

from typing import Union
from emdfile import tqdmnd, PointList
from py4DSTEM.process.diffraction.utils import Orientation
from py4DSTEM.process.utils import electron_wavelength_angstrom
import py4DSTEM
import matplotlib.pyplot as plt

# from chamferdist import ChamferDistance
from py4DSTEM.data import QPoints


try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    cp = None

orientation_ranges = {
    "1": ["fiber", [0, 0, 1], [180.0, 360.0]],
    "-1": ["full", None, None],
    "2": ["fiber", [0, 0, 1], [180.0, 180.0]],
    "m": ["full", None, None],
    "2/m": ["half", None, None],
    "222": ["fiber", [0, 0, 1], [90.0, 180.0]],
    "mm2": ["fiber", [0, 0, 1], [180.0, 90.0]],
    "mmm": [[[1, 0, 0], [0, 1, 0]], None, None],
    "4": ["fiber", [0, 0, 1], [90.0, 180.0]],
    "-4": ["half", None, None],
    "4/m": [[[1, 0, 0], [0, 1, 0]], None, None],
    "422": ["fiber", [0, 0, 1], [180.0, 45.0]],
    "4mm": ["fiber", [0, 0, 1], [180.0, 45.0]],
    "-42m": ["fiber", [0, 0, 1], [180.0, 45.0]],
    "4/mmm": [[[1, 0, 0], [1, 1, 0]], None, None],
    "3": ["fiber", [0, 0, 1], [180.0, 120.0]],
    "-3": ["fiber", [0, 0, 1], [180.0, 60.0]],
    "32": ["fiber", [0, 0, 1], [90.0, 60.0]],
    "3m": ["fiber", [0, 0, 1], [180.0, 60.0]],
    "-3m": ["fiber", [0, 0, 1], [90.0, 60.0]],
    "6": ["fiber", [0, 0, 1], [180.0, 60.0]],
    "-6": ["fiber", [0, 0, 1], [180.0, 60.0]],
    "6/m": [[[1, 0, 0], [0.5, 0.5 * np.sqrt(3), 0]], None, None],
    "622": ["fiber", [0, 0, 1], [180.0, 30.0]],
    "6mm": ["fiber", [0, 0, 1], [180.0, 30.0]],
    "-6m2": ["fiber", [0, 0, 1], [90.0, 60.0]],
    "6/mmm": [[[0.5 * np.sqrt(3), 0.5, 0.0], [1, 0, 0]], None, None],
    "23": [
        [[1, 0, 0], [1, 1, 1]],
        None,
        None,
    ],  # this is probably wrong, it is half the needed range
    "m-3": [[[1, 0, 0], [1, 1, 1]], None, None],
    "432": [[[1, 0, 0], [1, 1, 1]], None, None],
    "-43m": [[[1, -1, 1], [1, 1, 1]], None, None],
    "m-3m": [[[0, 1, 1], [1, 1, 1]], None, None],
}


def generate_2D_IMG_array(qx,
                          qy,
                          intensity,
                          k_max,
                          pixel_numbers,
                          remove_direct_beam = True,
                          corner_centered = False,
                          position_tolerance = 1e-14,
                             ):
    """
    Map py4DSTEM pointList object to 2D diffraction pattern array

    Args:            
        corner_centered: (boolean)
            boolean variable that determines whether to set the location of
            direct beam (qx, qy) = (0, 0) to corner of 2D numpy array of 
            diffraction image (diff2D_img) at index [0][0]. If True, 
            direct beam (qx, qy) = (0, 0) is located at the corner of 
            diff2D_img at index[0][0]. If False, direct beam (qx, qy) = (0, 0)
            is located at the center of 2D numpy array
            default: False

        remove_direct_beam: (boolean)
            boolean variable that directs whether to remove
            direct beam or not.
            Default is False

    Additional attributes:
        indices_of_qx_in_diff2d_img:
            indices of pixels of qx of Bragg peaks    
            
        indices_of_qy_in_diff2d_img:
            indices of pixels of qy of Bragg peaks

    returns:
          diff2D_img: 2D array of Bragg peaks 
          (shape: ($pixel_numbers$, $pixel_numbers$))
    """

    qx_refined = np.copy(qx)
    qy_refined = np.copy(qy)
    intensity_refined = np.copy(intensity)
    mask = None

    if remove_direct_beam:            
        mask = np.full((len(intensity)), True)
        radial_distances = np.linalg.norm(np.stack((qx,qy)).T, axis = 1)
        index_of_direct_beam = np.argmin(radial_distances)
        mask[index_of_direct_beam] = False
        qx_refined = qx_refined[mask]
        qy_refined = qy_refined[mask]
        intensity_refined = intensity_refined[mask]
        
    qx_refined[np.where(np.abs(qx_refined[:]) < (position_tolerance))[0]] = 0.0
    qy_refined[np.where(np.abs(qy_refined[:]) < (position_tolerance))[0]] = 0.0

    stack_qxqy = np.stack((qx_refined, qy_refined), axis = 1)
    width_of_diff2D_img = k_max * 2.0
    pixel_size = width_of_diff2D_img / pixel_numbers

    if pixel_numbers % 2 == 0:
        pixel_bins = np.arange(-k_max, k_max + pixel_size, pixel_size) # change
        pixel_bins[np.where(np.abs(pixel_bins[:]) < (position_tolerance))[0]] = 0.0 # change
        
        
        digitized_bin = np.digitize(stack_qxqy, pixel_bins) - 1
    else:
        pixel_bins = np.linspace(-k_max - pixel_size/2.0, 
                                 k_max + pixel_size/2.0, 
                                 pixel_numbers + 1, endpoint = True)
        digitized_bin = np.digitize(stack_qxqy, pixel_bins) - 1


    diff2D_img = np.zeros((pixel_numbers, pixel_numbers))
    diff2D_img[digitized_bin[:,0],digitized_bin[:,1]] = intensity_refined

    if corner_centered:
        diff2D_img = np.roll(diff2D_img, (-int(pixel_numbers / 2), -int(pixel_numbers / 2)), axis=(0, 1))

    return diff2D_img

def Q_calculation(image, py4DSTEM_bragg_vecto_list, k_max, pixel_numbers, intensity_gamma_correction = 0.5):
    
    qx = py4DSTEM_bragg_vecto_list.data['qx']
    qy = py4DSTEM_bragg_vecto_list.data['qy']
    intensity = py4DSTEM_bragg_vecto_list.data['intensity']
    
    if intensity_gamma_correction is not None:
        image = image ** intensity_gamma_correction;
    
    kinematic_diffraction_pattern_2D = generate_2D_IMG_array(
                                                qx,
                                                qy,
                                                intensity,
                                                k_max,
                                                pixel_numbers,
                                                )
    
    kinematic_diffraction_pattern_2D = kinematic_diffraction_pattern_2D / np.sqrt(np.sum(np.square(kinematic_diffraction_pattern_2D)))
    
    experimen_diffraction_pattern_2D = image / np.sqrt(np.sum(np.square(image)))
    
    correlation_score_Q = np.sum(np.multiply(kinematic_diffraction_pattern_2D,experimen_diffraction_pattern_2D))
    
    return correlation_score_Q

def geodesic_distance_SO3(R1, R2):
    """
    Geodesic distance between two 3x3 rotation matrices in SO(3).
    Returns the rotation angle in radians, in [0, pi].
    """
    R = R1.T @ R2
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # numerical safety
    return np.arccos(cos_theta)


def rotation_wrt_zAxis(angle_in_rad):
    return np.array(
                    [
                        [np.cos(angle_in_rad), np.sin(angle_in_rad), 0],
                        [-np.sin(angle_in_rad), np.cos(angle_in_rad), 0],
                        [0, 0, 1],
                    ]
                    )

def make_proper_rotation_matrix(matrix):
    """Ensure the matrix is a proper orthogonal matrix with determinant +1."""
    U, _, Vt = np.linalg.svd(matrix)
    R = U @ Vt

    # Fix improper rotation (det(R) = -1)
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    return R

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

def rotationMatrix_2D(angle_in_rad):
    return np.array(
                    [
                        [np.cos(angle_in_rad), np.sin(angle_in_rad)],
                        [np.sin(angle_in_rad), np.cos(angle_in_rad)],
                    ]
                    )
def calculate_rotation_matrix_for_zone_axis(zone_axis):
    elev = np.arctan2(
        np.hypot(zone_axis[0], zone_axis[1]),
        zone_axis[2],
    )
    
    azim = np.arctan2(zone_axis[0], zone_axis[1])

    new_rotation_matrix = rotation_wrt_zAxis(azim) @ rotation_wrt_xAxis(elev) @ rotation_wrt_zAxis(-azim)


    return new_rotation_matrix

def returnBPVEC(crystal_1, orientation_matrix, displace_sigma = 0.1, excitation_errors = 0.02, draw_inverse_as_well = False):

    orientation_matrix_canonical = np.copy(orientation_matrix)

    if draw_inverse_as_well:

        
        rng = np.random.default_rng()
        x = rng.random()
    
        if x < 0.5:
            orientation_matrix_canonical[:,1:] = orientation_matrix_canonical[:,1:] * (-1.0) 
        





    bragg_peaks_fit = crystal_1.generate_diffraction_pattern(
                                orientation_matrix = orientation_matrix_canonical,
                                ind_orientation=0,
                                sigma_excitation_error=excitation_errors)
            
    qx = np.copy(bragg_peaks_fit.data['qx'])
    qy = np.copy(bragg_peaks_fit.data['qy'])
    intensity = np.copy(bragg_peaks_fit.data['intensity'])
    
    radial_distance = np.linalg.norm(np.stack((qx,qy)).T, axis = 1)
    mask_of_BD_for_position_displacement = np.ones((qx.shape[0],), dtype=bool)
    index_of_direct_beam = np.argmin(radial_distance)
    mask_of_BD_for_position_displacement[index_of_direct_beam] = False
    
    qx_direct_beam = np.copy(qx[index_of_direct_beam])
    intensity_direct_beam = np.copy(intensity[index_of_direct_beam])
    qy_direct_beam = np.copy(qy[index_of_direct_beam])
    
    qx_BDs = np.copy(qx[mask_of_BD_for_position_displacement])
    qy_BDs = np.copy(qy[mask_of_BD_for_position_displacement])
    intensity_BDs = np.copy(intensity[mask_of_BD_for_position_displacement])
    
    
    
    qx_BDs_displaced = qx_BDs + np.random.normal(loc=0.0, scale = displace_sigma, size = qx_BDs.shape[0])
    qy_BDs_displaced = qy_BDs + np.random.normal(loc=0.0, scale = displace_sigma, size = qy_BDs.shape[0])
    
    # qx_BDs_displaced = qx_BDs
    # qy_BDs_displaced = qy_BDs + qy_BDs * np.random.normal(loc=0.0, scale = displace_sigma, size = qy_BDs.shape[0])
    
    new_qx_set = np.hstack((qx_BDs_displaced, qx_direct_beam))
    new_qy_set = np.hstack((qy_BDs_displaced, qy_direct_beam))
    new_intensity_set = np.hstack((intensity_BDs, intensity_direct_beam))
    
    
    # Extract (qx, qy, intensity) from the diffraction pattern
    bragg_vectors_np = np.stack((new_qx_set, new_qy_set, new_intensity_set)).T
    
    # Prepare the data structure for maxima with dtype (qx, qy, intensity)
    dtype = np.dtype([("x", float), ("y", float), ("intensity", float)])
    synthetic_experimental_DP = np.zeros(len(bragg_vectors_np), dtype=dtype)
    
    for i, bragg_vector in enumerate(bragg_vectors_np):
        synthetic_experimental_DP["x"][i] = bragg_vector[0]
        synthetic_experimental_DP["y"][i] = bragg_vector[1]
        synthetic_experimental_DP["intensity"][i] = bragg_vector[2]
    
    # Create QPoints object with already calibrated data
    synthetic_experimental_DP = QPoints(synthetic_experimental_DP)

    return synthetic_experimental_DP

def find_angle_from_sin_cos(sin_value, cos_value):
    # Clip sine and cosine values to be within the range [-1, 1]
    sin_value = np.clip(sin_value, -1.0, 1.0)
    cos_value = np.clip(cos_value, -1.0, 1.0)
    
    # Calculate the angle using arctan2 to handle all quadrants
    angle = np.arctan2(sin_value, cos_value)
    
    # Ensure the angle is within [0, 2pi)
    if angle < 0:
        angle += 2 * np.pi
    
    return angle




def orientation_plan_init_for_single_pattern(
    crystal,
    input_rotation_matrix = None,
    is_input_rotation_matrix_inverted_wrt_canonical = None,
    zone_axis_range: np.ndarray = np.array([[0, 1, 1], [1, 1, 1]]),
    angle_step_zone_axis: float = 2.0,
    angle_coarse_zone_axis: float = None,
    angle_refine_range: float = None,
    angle_step_in_plane: float = 2.0,
    accel_voltage: float = 300e3,
    corr_kernel_size: float = 0.08,
    radial_power: float = 1.0,
    intensity_power: float = 0.25,  # New default intensity power scaling
    calculate_correlation_array=True,
    tol_peak_delete=None,
    tol_distance: float = 0.01,
    fiber_axis=None,
    fiber_angles=None,
    figsize: Union[list, tuple, np.ndarray] = (6, 6),
    CUDA: bool = False,
    progress_bar: bool = True,
):
    
    """
    This function is adapted from the open-source library py4DSTEM
    (https://github.com/py4dstem/py4DSTEM), licensed under GPL-3.0.
    
    Calculate the rotation basis arrays for an SO(3) rotation correlogram.

    Args:
        zone_axis_range (float): Row vectors give the range for zone axis orientations.
                                 If user specifies 2 vectors (2x3 array), we start at [0,0,1]
                                    to make z-x-z rotation work.
                                 If user specifies 3 vectors (3x3 array), plan will span these vectors.
                                 Setting to 'full' as a string will use a hemispherical range.
                                 Setting to 'half' as a string will use a quarter sphere range.
                                 Setting to 'fiber' as a string will make a spherical cap around a given vector.
                                 Setting to 'auto' will use pymatgen to determine the point group symmetry
                                    of the structure and choose an appropriate zone_axis_range
        angle_step_zone_axis (float): Approximate angular step size for zone axis search [degrees]
        angle_coarse_zone_axis (float): Coarse step size for zone axis search [degrees]. Setting to
                                        None uses the same value as angle_step_zone_axis.
        angle_refine_range (float):   Range of angles to use for zone axis refinement. Setting to
                                      None uses same value as angle_coarse_zone_axis.

        angle_step_in_plane (float):  Approximate angular step size for in-plane rotation [degrees]
        accel_voltage (float):        Accelerating voltage for electrons [Volts]
        corr_kernel_size (float):        Correlation kernel size length in Angstroms
        radial_power (float):          Power for scaling the correlation intensity as a function of the peak radius
        intensity_power (float):       Power for scaling the correlation intensity as a function of the peak intensity
        calculate_correlation_array (bool):     Set to false to skip calculating the correlation array.
                                                This is useful when we only want the angular range / rotation matrices.
        tol_peak_delete (float):      Distance to delete peaks for multiple matches.
                                      Default is kernel_size * 0.5
        tol_distance (float):         Distance tolerance for radial shell assignment [1/Angstroms]
        fiber_axis (float):           (3,) vector specifying the fiber axis
        fiber_angles (float):         (2,) vector specifying angle range from fiber axis, and in-plane angular range [degrees]
        cartesian_directions (bool): When set to true, all zone axes and projection directions
                                     are specified in Cartesian directions.
        figsize (float):            (2,) vector giving the figure size
        CUDA (bool):             Use CUDA for the Fourier operations.
        progress_bar (bool):    If false no progress bar is displayed
    """

    if input_rotation_matrix is not None:
        input_rotation_matrix_canonical = np.copy(input_rotation_matrix)
        if np.sum(input_rotation_matrix[:,2]) < 0.0:
            input_rotation_matrix_canonical = np.copy(input_rotation_matrix)
            input_rotation_matrix_canonical[:,1:] = input_rotation_matrix[:,1:] * (-1.0)
            is_input_rotation_matrix_inverted_wrt_canonical = True
        else:
            is_input_rotation_matrix_inverted_wrt_canonical = False

    # print("input_rotation_matrix\n ", input_rotation_matrix, "\n")
    # print("input_rotation_matrix_canonical\n", input_rotation_matrix_canonical, "\n")
        

    # Store inputs
    crystal.accel_voltage = np.asarray(accel_voltage)
    crystal.orientation_kernel_size = np.asarray(corr_kernel_size)
    if tol_peak_delete is None:
        crystal.orientation_tol_peak_delete = crystal.orientation_kernel_size * 0.5
    else:
        crystal.orientation_tol_peak_delete = np.asarray(tol_peak_delete)
    if fiber_axis is None:
        crystal.orientation_fiber_axis = None
    else:
        crystal.orientation_fiber_axis = np.asarray(fiber_axis)
    if fiber_angles is None:
        crystal.orientation_fiber_angles = None
    else:
        crystal.orientation_fiber_angles = np.asarray(fiber_angles)
    crystal.CUDA = CUDA

    # Calculate wavelenth
    crystal.wavelength = electron_wavelength_angstrom(crystal.accel_voltage)

    # store the radial and intensity scaling to use later for generating test patterns
    crystal.orientation_radial_power = radial_power
    crystal.orientation_intensity_power = intensity_power

    # Calculate the ratio between coarse and fine refinement
    if angle_coarse_zone_axis is not None:
        crystal.orientation_refine = True
        crystal.orientation_refine_ratio = np.round(
            angle_coarse_zone_axis / angle_step_zone_axis
        ).astype("int")
        crystal.orientation_angle_coarse = angle_coarse_zone_axis
        if angle_refine_range is None:
            crystal.orientation_refine_range = angle_coarse_zone_axis
        else:
            crystal.orientation_refine_range = angle_refine_range
    else:
        crystal.orientation_refine_ratio = 1.0
        crystal.orientation_refine = False

    if crystal.pymatgen_available:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        from pymatgen.core.structure import Structure

        structure = Structure(
            crystal.lat_real, crystal.numbers, crystal.positions, coords_are_cartesian=False
        )
        crystal.pointgroup = SpacegroupAnalyzer(structure)

    # Handle the "auto" case first, since it works by overriding zone_axis_range,
    #   fiber_axis, and fiber_angles then using the regular parser:
    if isinstance(zone_axis_range, str) and zone_axis_range == "auto":
        assert (
            crystal.pointgroup.get_point_group_symbol() in orientation_ranges
        ), "Unrecognized pointgroup returned by pymatgen!"

        zone_axis_range, fiber_axis, fiber_angles = orientation_ranges[
            crystal.pointgroup.get_point_group_symbol()
        ]
        if isinstance(zone_axis_range, list):
            zone_axis_range = np.array(zone_axis_range)
        elif zone_axis_range == "fiber":
            crystal.orientation_fiber_axis = np.asarray(fiber_axis)
            crystal.orientation_fiber_angles = np.asarray(fiber_angles)

        # print(
        #     f"Automatically detected point group {crystal.pointgroup.get_point_group_symbol()},\n"
        #     f" using arguments: zone_axis_range = \n{zone_axis_range}, \n fiber_axis={fiber_axis}, fiber_angles={fiber_angles}."
        # )

    if isinstance(zone_axis_range, str):
        if (
            zone_axis_range == "fiber"
            and fiber_axis is not None
            and fiber_angles is not None
        ):
            # Determine vector ranges
            crystal.orientation_fiber_axis = np.array(
                crystal.orientation_fiber_axis, dtype="float"
            )
            # if crystal.cartesian_directions:
            crystal.orientation_fiber_axis = crystal.orientation_fiber_axis / np.linalg.norm(
                crystal.orientation_fiber_axis
            )

            # update fiber axis to be centered on the 1st unit cell vector
            v3 = np.cross(crystal.orientation_fiber_axis, crystal.lat_real[0, :])
            v2 = np.cross(
                v3,
                crystal.orientation_fiber_axis,
            )
            v2 = v2 / np.linalg.norm(v2)
            v3 = v3 / np.linalg.norm(v3)

            if crystal.orientation_fiber_angles[0] == 0:
                crystal.orientation_zone_axis_range = np.vstack(
                    (crystal.orientation_fiber_axis, v2, v3)
                ).astype("float")
            else:
                if crystal.orientation_fiber_angles[0] == 180:
                    theta = np.pi / 2.0
                else:
                    theta = crystal.orientation_fiber_angles[0] * np.pi / 180.0
                if (
                    crystal.orientation_fiber_angles[1] == 180
                    or crystal.orientation_fiber_angles[1] == 360
                ):
                    phi = np.pi / 2.0
                else:
                    phi = crystal.orientation_fiber_angles[1] * np.pi / 180.0

                # Generate zone axis range
                v2output = crystal.orientation_fiber_axis * np.cos(theta) + v2 * np.sin(
                    theta
                )
                v3output = (
                    crystal.orientation_fiber_axis * np.cos(theta)
                    + (v2 * np.sin(theta)) * np.cos(phi)
                    + (v3 * np.sin(theta)) * np.sin(phi)
                )
                v2output = (
                    crystal.orientation_fiber_axis * np.cos(theta)
                    + (v2 * np.sin(theta)) * np.cos(phi / 2)
                    - (v3 * np.sin(theta)) * np.sin(phi / 2)
                )
                v3output = (
                    crystal.orientation_fiber_axis * np.cos(theta)
                    + (v2 * np.sin(theta)) * np.cos(phi / 2)
                    + (v3 * np.sin(theta)) * np.sin(phi / 2)
                )

                crystal.orientation_zone_axis_range = np.vstack(
                    (crystal.orientation_fiber_axis, v2output, v3output)
                ).astype("float")

            crystal.orientation_full = False
            crystal.orientation_half = False
            crystal.orientation_fiber = True
        else:
            crystal.orientation_zone_axis_range = np.array(
                [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
            )
            if zone_axis_range == "full":
                crystal.orientation_full = True
                crystal.orientation_half = False
                crystal.orientation_fiber = False
            elif zone_axis_range == "half":
                crystal.orientation_full = False
                crystal.orientation_half = True
                crystal.orientation_fiber = False
            else:
                if zone_axis_range == "fiber" and fiber_axis is None:
                    raise ValueError(
                        "For fiber zone axes, you must specify the fiber axis and angular ranges"
                    )
                else:
                    raise ValueError(
                        "Zone axis range must be a 2x3 array, 3x3 array, or full, half or fiber"
                    )

    else:
        crystal.orientation_zone_axis_range = np.array(zone_axis_range, dtype="float")

        # Define 3 vectors which span zone axis orientation range, normalize
        if zone_axis_range.shape[0] == 3:
            crystal.orientation_zone_axis_range = np.array(
                crystal.orientation_zone_axis_range, dtype="float"
            )
            crystal.orientation_zone_axis_range[0, :] /= np.linalg.norm(
                crystal.orientation_zone_axis_range[0, :]
            )
            crystal.orientation_zone_axis_range[1, :] /= np.linalg.norm(
                crystal.orientation_zone_axis_range[1, :]
            )
            crystal.orientation_zone_axis_range[2, :] /= np.linalg.norm(
                crystal.orientation_zone_axis_range[2, :]
            )

        elif zone_axis_range.shape[0] == 2:
            crystal.orientation_zone_axis_range = np.vstack(
                (
                    np.array([0, 0, 1]),
                    np.array(crystal.orientation_zone_axis_range, dtype="float"),
                )
            ).astype("float")
            crystal.orientation_zone_axis_range[1, :] /= np.linalg.norm(
                crystal.orientation_zone_axis_range[1, :]
            )
            crystal.orientation_zone_axis_range[2, :] /= np.linalg.norm(
                crystal.orientation_zone_axis_range[2, :]
            )
        crystal.orientation_full = False
        crystal.orientation_half = False
        crystal.orientation_fiber = False

    # Solve for number of angular steps in zone axis (rads)
    angle_u_v = np.arccos(
        np.sum(
            crystal.orientation_zone_axis_range[0, :]
            * crystal.orientation_zone_axis_range[1, :]
        )
    )
    angle_u_w = np.arccos(
        np.sum(
            crystal.orientation_zone_axis_range[0, :]
            * crystal.orientation_zone_axis_range[2, :]
        )
    )
    step = np.maximum(
        (180 / np.pi) * angle_u_v / angle_step_zone_axis,
        (180 / np.pi) * angle_u_w / angle_step_zone_axis,
    )
    crystal.orientation_zone_axis_steps = (
        np.round(step / crystal.orientation_refine_ratio) * crystal.orientation_refine_ratio
    ).astype(np.int64)

    if crystal.orientation_fiber and crystal.orientation_fiber_angles[0] == 0:
        crystal.orientation_num_zones = int(1)
        crystal.orientation_vecs = np.zeros((1, 3))
        crystal.orientation_vecs[0, :] = crystal.orientation_zone_axis_range[0, :]
        crystal.orientation_inds = np.zeros((1, 3), dtype="int")

    else:
        # Generate points spanning the zone axis range
        # Calculate points along u and v using the SLERP formula
        # https://en.wikipedia.org/wiki/Slerp
        weights = np.linspace(0, 1, crystal.orientation_zone_axis_steps + 1)
        pv = crystal.orientation_zone_axis_range[0, :] * np.sin(
            (1 - weights[:, None]) * angle_u_v
        ) / np.sin(angle_u_v) + crystal.orientation_zone_axis_range[1, :] * np.sin(
            weights[:, None] * angle_u_v
        ) / np.sin(
            angle_u_v
        )

        # Calculate points along u and w using the SLERP formula
        pw = crystal.orientation_zone_axis_range[0, :] * np.sin(
            (1 - weights[:, None]) * angle_u_w
        ) / np.sin(angle_u_w) + crystal.orientation_zone_axis_range[2, :] * np.sin(
            weights[:, None] * angle_u_w
        ) / np.sin(
            angle_u_w
        )

        # Init array to hold all points
        crystal.orientation_num_zones = (
            (crystal.orientation_zone_axis_steps + 1)
            * (crystal.orientation_zone_axis_steps + 2)
            / 2
        ).astype(np.int64)
        crystal.orientation_vecs = np.zeros((crystal.orientation_num_zones, 3))
        crystal.orientation_vecs[0, :] = crystal.orientation_zone_axis_range[0, :]
        crystal.orientation_inds = np.zeros((crystal.orientation_num_zones, 3), dtype="int")

        # Calculate zone axis points on the unit sphere with another application of SLERP,
        # or circular arc SLERP for fiber texture
        for a0 in np.arange(1, crystal.orientation_zone_axis_steps + 1):
            inds = np.arange(a0 * (a0 + 1) / 2, a0 * (a0 + 1) / 2 + a0 + 1).astype(
                np.int64
            )

            p0 = pv[a0, :]
            p1 = pw[a0, :]

            weights = np.linspace(0, 1, a0 + 1)

            if crystal.orientation_fiber:
                # For fiber texture, place points on circular arc perpendicular to the fiber axis
                crystal.orientation_vecs[inds, :] = p0[None, :]

                p_proj = (
                    np.dot(p0, crystal.orientation_fiber_axis)
                    * crystal.orientation_fiber_axis
                )
                p0_sub = p0 - p_proj
                p1_sub = p1 - p_proj

                angle_p_sub = np.arccos(
                    np.sum(p0_sub * p1_sub)
                    / np.linalg.norm(p0_sub)
                    / np.linalg.norm(p1_sub)
                )

                crystal.orientation_vecs[inds, :] = (
                    p_proj
                    + p0_sub[None, :]
                    * np.sin((1 - weights[:, None]) * angle_p_sub)
                    / np.sin(angle_p_sub)
                    + p1_sub[None, :]
                    * np.sin(weights[:, None] * angle_p_sub)
                    / np.sin(angle_p_sub)
                )
            else:
                angle_p = np.arccos(np.sum(p0 * p1))

                crystal.orientation_vecs[inds, :] = p0[None, :] * np.sin(
                    (1 - weights[:, None]) * angle_p
                ) / np.sin(angle_p) + p1[None, :] * np.sin(
                    weights[:, None] * angle_p
                ) / np.sin(
                    angle_p
                )

            crystal.orientation_inds[inds, 0] = a0
            crystal.orientation_inds[inds, 1] = np.arange(a0 + 1)

    if crystal.orientation_fiber and crystal.orientation_fiber_angles[0] == 180:
        # Mirror about the equator of fiber_zone_axis
        m = np.identity(3) - 2 * (
            crystal.orientation_fiber_axis[:, None] @ crystal.orientation_fiber_axis[None, :]
        )

        vec_new = np.copy(crystal.orientation_vecs) @ m
        orientation_sector = np.zeros(vec_new.shape[0], dtype="int")

        keep = np.zeros(vec_new.shape[0], dtype="bool")
        for a0 in range(keep.size):
            if (
                np.sqrt(
                    np.min(
                        np.sum((crystal.orientation_vecs - vec_new[a0, :]) ** 2, axis=1)
                    )
                )
                > tol_distance
            ):
                keep[a0] = True

        crystal.orientation_vecs = np.vstack((crystal.orientation_vecs, vec_new[keep, :]))
        crystal.orientation_num_zones = crystal.orientation_vecs.shape[0]

        crystal.orientation_inds = np.vstack(
            (crystal.orientation_inds, crystal.orientation_inds[keep, :])
        ).astype("int")
        crystal.orientation_inds[:, 2] = np.hstack(
            (orientation_sector, np.ones(np.sum(keep), dtype="int"))
        )

    # Fiber texture angle 1 extend to 180 degree angular range if needed
    if (
        crystal.orientation_fiber
        and crystal.orientation_fiber_angles[0] != 0
        and (
            crystal.orientation_fiber_angles[1] == 180
            or crystal.orientation_fiber_angles[1] == 360
        )
    ):
        # Mirror about the axes 0 and 1
        n = np.cross(
            crystal.orientation_zone_axis_range[0, :],
            crystal.orientation_zone_axis_range[1, :],
        )
        n = n / np.linalg.norm(n)

        # n = crystal.orientation_zone_axis_range[2,:]
        m = np.identity(3) - 2 * (n[:, None] @ n[None, :])

        vec_new = np.copy(crystal.orientation_vecs) @ m
        orientation_sector = np.zeros(vec_new.shape[0], dtype="int")

        keep = np.zeros(vec_new.shape[0], dtype="bool")
        for a0 in range(keep.size):
            if (
                np.sqrt(
                    np.min(
                        np.sum((crystal.orientation_vecs - vec_new[a0, :]) ** 2, axis=1)
                    )
                )
                > tol_distance
            ):
                keep[a0] = True

        crystal.orientation_vecs = np.vstack((crystal.orientation_vecs, vec_new[keep, :]))
        crystal.orientation_num_zones = crystal.orientation_vecs.shape[0]

        crystal.orientation_inds = np.vstack(
            (crystal.orientation_inds, crystal.orientation_inds[keep, :])
        ).astype("int")
        crystal.orientation_inds[:, 2] = np.hstack(
            (orientation_sector, np.ones(np.sum(keep), dtype="int"))
        )
    # Fiber texture extend to 360 angular range if needed
    if (
        crystal.orientation_fiber
        and crystal.orientation_fiber_angles[0] != 0
        and crystal.orientation_fiber_angles[1] == 360
    ):
        # Mirror about the axes 0 and 2
        n = np.cross(
            crystal.orientation_zone_axis_range[0, :],
            crystal.orientation_zone_axis_range[2, :],
        )
        n = n / np.linalg.norm(n)

        # n = crystal.orientation_zone_axis_range[2,:]
        m = np.identity(3) - 2 * (n[:, None] @ n[None, :])

        vec_new = np.copy(crystal.orientation_vecs) @ m
        orientation_sector = np.zeros(vec_new.shape[0], dtype="int")

        keep = np.zeros(vec_new.shape[0], dtype="bool")
        for a0 in range(keep.size):
            if (
                np.sqrt(
                    np.min(
                        np.sum((crystal.orientation_vecs - vec_new[a0, :]) ** 2, axis=1)
                    )
                )
                > tol_distance
            ):
                keep[a0] = True

        crystal.orientation_vecs = np.vstack((crystal.orientation_vecs, vec_new[keep, :]))
        crystal.orientation_num_zones = crystal.orientation_vecs.shape[0]

        crystal.orientation_inds = np.vstack(
            (crystal.orientation_inds, crystal.orientation_inds[keep, :])
        ).astype("int")
        crystal.orientation_inds[:, 2] = np.hstack(
            (orientation_sector, np.ones(np.sum(keep), dtype="int"))
        )

    # expand to quarter sphere if needed
    if crystal.orientation_half or crystal.orientation_full:
        vec_new = np.copy(crystal.orientation_vecs) * np.array([-1, 1, 1])
        orientation_sector = np.zeros(vec_new.shape[0], dtype="int")

        keep = np.zeros(vec_new.shape[0], dtype="bool")
        for a0 in range(keep.size):
            if (
                np.sqrt(
                    np.min(
                        np.sum((crystal.orientation_vecs - vec_new[a0, :]) ** 2, axis=1)
                    )
                )
                > tol_distance
            ):
                keep[a0] = True

        crystal.orientation_vecs = np.vstack((crystal.orientation_vecs, vec_new[keep, :]))
        crystal.orientation_num_zones = crystal.orientation_vecs.shape[0]

        crystal.orientation_inds = np.vstack(
            (crystal.orientation_inds, crystal.orientation_inds[keep, :])
        ).astype("int")
        crystal.orientation_inds[:, 2] = np.hstack(
            (orientation_sector, np.ones(np.sum(keep), dtype="int"))
        )

    # expand to hemisphere if needed
    if crystal.orientation_full:
        vec_new = np.copy(crystal.orientation_vecs) * np.array([1, -1, 1])

        keep = np.zeros(vec_new.shape[0], dtype="bool")
        for a0 in range(keep.size):
            if (
                np.sqrt(
                    np.min(
                        np.sum((crystal.orientation_vecs - vec_new[a0, :]) ** 2, axis=1)
                    )
                )
                > tol_distance
            ):
                keep[a0] = True

        crystal.orientation_vecs = np.vstack((crystal.orientation_vecs, vec_new[keep, :]))
        crystal.orientation_num_zones = crystal.orientation_vecs.shape[0]

        orientation_sector = np.hstack(
            (crystal.orientation_inds[:, 2], crystal.orientation_inds[keep, 2] + 2)
        )
        crystal.orientation_inds = np.vstack(
            (crystal.orientation_inds, crystal.orientation_inds[keep, :])
        ).astype("int")
        crystal.orientation_inds[:, 2] = orientation_sector

    # If needed, create coarse orientation sieve
    if crystal.orientation_refine:
        crystal.orientation_sieve = np.logical_and(
            np.mod(crystal.orientation_inds[:, 0], crystal.orientation_refine_ratio) == 0,
            np.mod(crystal.orientation_inds[:, 1], crystal.orientation_refine_ratio) == 0,
        )
        if crystal.CUDA:
            crystal.orientation_sieve_CUDA = cp.asarray(crystal.orientation_sieve)

    if input_rotation_matrix is not None:
        zone_axis_vector_of_input_rotation_matrix = np.array([input_rotation_matrix_canonical[0,2], input_rotation_matrix_canonical[1,2], input_rotation_matrix_canonical[2,2]])
        # print("zone_axis_vector_of_input_rotation_matrix\n", zone_axis_vector_of_input_rotation_matrix, "\n")
        # print("crystal.orientation_vecs\n", crystal.orientation_vecs.shape)
        # print("input_rotation_matrix_canonical\n", input_rotation_matrix_canonical, "\n")
        idx_of_zone_axis_for_replacement = np.argmax(np.dot(crystal.orientation_vecs, zone_axis_vector_of_input_rotation_matrix))
        crystal.orientation_vecs[idx_of_zone_axis_for_replacement] = zone_axis_vector_of_input_rotation_matrix
        crystal.idx_of_zone_axis_for_replacement = idx_of_zone_axis_for_replacement
        

    # Convert to spherical coordinates
    elev = np.arctan2(
        np.hypot(crystal.orientation_vecs[:, 0], crystal.orientation_vecs[:, 1]),
        crystal.orientation_vecs[:, 2],
    )
    # azim = np.pi / 2 + np.arctan2(
    #     crystal.orientation_vecs[:, 1], crystal.orientation_vecs[:, 0]
    # )
    azim = np.arctan2(crystal.orientation_vecs[:, 0], crystal.orientation_vecs[:, 1])

    # Solve for number of angular steps along in-plane rotation direction
    crystal.orientation_in_plane_steps = np.round(360 / angle_step_in_plane).astype(
        np.int64
    )

    # Calculate -z angles (Euler angle 3)
    crystal.orientation_gamma = np.linspace(
        0, 2 * np.pi, crystal.orientation_in_plane_steps, endpoint=False
    )

    # Determine the radii of all spherical shells
    radii_test = np.round(crystal.g_vec_leng / tol_distance) * tol_distance
    radii = np.unique(radii_test)
    # Remove zero beam
    keep = np.abs(radii) > tol_distance
    crystal.orientation_shell_radii = radii[keep]

    # init
    crystal.orientation_shell_index = -1 * np.ones(crystal.g_vec_all.shape[1], dtype="int")
    crystal.orientation_shell_count = np.zeros(crystal.orientation_shell_radii.size)

    # Assign each structure factor point to a radial shell
    for a0 in range(crystal.orientation_shell_radii.size):
        sub = np.abs(crystal.orientation_shell_radii[a0] - radii_test) <= tol_distance / 2

        crystal.orientation_shell_index[sub] = a0
        crystal.orientation_shell_count[a0] = np.sum(sub)
        crystal.orientation_shell_radii[a0] = np.mean(crystal.g_vec_leng[sub])

    # init storage arrays
    crystal.orientation_rotation_angles = np.zeros((crystal.orientation_num_zones, 3))
    crystal.orientation_rotation_matrices = np.zeros((crystal.orientation_num_zones, 3, 3))

    # If possible,  Get symmetry operations for this spacegroup, store in matrix form
    if crystal.pymatgen_available:
        # get operators
        ops = crystal.pointgroup.get_point_group_operations()

        # Inverse of lattice
        zone_axis_range_inv = np.linalg.inv(crystal.orientation_zone_axis_range)

        # init
        num_sym = len(ops)
        crystal.symmetry_operators = np.zeros((num_sym, 3, 3))
        crystal.symmetry_reduction = np.zeros((num_sym, 3, 3))

        # calculate symmetry and reduction matrices
        for a0 in range(num_sym):
            crystal.symmetry_operators[a0] = (
                crystal.lat_inv.T @ ops[a0].rotation_matrix.T @ crystal.lat_real
            )
            crystal.symmetry_reduction[a0] = (
                zone_axis_range_inv.T @ crystal.symmetry_operators[a0]
            ).T

        # Remove duplicates
        keep = np.ones(num_sym, dtype="bool")
        for a0 in range(num_sym):
            if keep[a0]:
                diff = np.sum(
                    np.abs(crystal.symmetry_operators - crystal.symmetry_operators[a0]),
                    axis=(1, 2),
                )
                sub = diff < 1e-3
                sub[: a0 + 1] = False
                keep[sub] = False
        crystal.symmetry_operators = crystal.symmetry_operators[keep]
        crystal.symmetry_reduction = crystal.symmetry_reduction[keep]

        if (
            crystal.orientation_fiber_angles is not None
            and np.abs(crystal.orientation_fiber_angles[0] - 180.0) < 1e-3
        ):
            zone_axis_range_flip = crystal.orientation_zone_axis_range.copy()
            zone_axis_range_flip[0, :] = -1 * zone_axis_range_flip[0, :]
            zone_axis_range_inv = np.linalg.inv(zone_axis_range_flip)

            num_sym = crystal.symmetry_operators.shape[0]
            crystal.symmetry_operators = np.tile(crystal.symmetry_operators, [2, 1, 1])
            crystal.symmetry_reduction = np.tile(crystal.symmetry_reduction, [2, 1, 1])

            for a0 in range(num_sym):
                crystal.symmetry_reduction[a0 + num_sym] = (
                    zone_axis_range_inv.T @ crystal.symmetry_operators[a0 + num_sym]
                ).T

    # Calculate rotation matrices for zone axes

    # if input_rotation_matrix is not None:
        # for a0 in np.arange(crystal.orientation_num_zones):
        #     m1z = np.array(
        #         [
        #             [np.cos(azim[a0]), np.sin(azim[a0]), 0],
        #             [-np.sin(azim[a0]), np.cos(azim[a0]), 0],
        #             [0, 0, 1],
        #         ]
        #     )
        #     m2x = np.array(
        #         [
        #             [1, 0, 0],
        #             [0, np.cos(elev[a0]), np.sin(elev[a0])],
        #             [0, -np.sin(elev[a0]), np.cos(elev[a0])],
        #         ]
        #     )

        #     zone_axis_aligment_matrix = m1z @ m2x

        #     in_plane_rotation_matrix_with_repsect_to_z = zone_axis_aligment_matrix.T @ input_rotation_matrix

        #     in_plane_angle_rotation_matrix_with_repsect_to_z = np.arccos(np.clip(in_plane_rotation_matrix_with_repsect_to_z[0,0], -1., 1.))

        #     m3z = np.array(
        #         [
        #             [np.cos(in_plane_angle_rotation_matrix_with_repsect_to_z), -np.sin(in_plane_angle_rotation_matrix_with_repsect_to_z), 0],
        #             [np.sin(in_plane_angle_rotation_matrix_with_repsect_to_z), np.cos(in_plane_angle_rotation_matrix_with_repsect_to_z), 0],
        #             [0, 0, 1],
        #         ]
        #     )

        # crystal.orientation_rotation_matrices[a0, :, :] = input_rotation_matrix
        # crystal.orientation_rotation_angles[a0, :] = [azim[a0], elev[a0], in_plane_angle_rotation_matrix_with_repsect_to_z]
    # else:
    for a0 in np.arange(crystal.orientation_num_zones):
        m1z = np.array(
            [
                [np.cos(azim[a0]), np.sin(azim[a0]), 0],
                [-np.sin(azim[a0]), np.cos(azim[a0]), 0],
                [0, 0, 1],
            ]
        )
        m2x = np.array(
            [
                [1, 0, 0],
                [0, np.cos(elev[a0]), np.sin(elev[a0])],
                [0, -np.sin(elev[a0]), np.cos(elev[a0])],
            ]
        )
        m3z = np.array(
            [
                [np.cos(azim[a0]), -np.sin(azim[a0]), 0],
                [np.sin(azim[a0]), np.cos(azim[a0]), 0],
                [0, 0, 1],
            ]
        )
        crystal.orientation_rotation_matrices[a0, :, :] = m1z @ m2x @ m3z            
        crystal.orientation_rotation_angles[a0, :] = [azim[a0], elev[a0], -azim[a0]]

    if input_rotation_matrix is not None:
        # if not is_input_rotation_matrix_inverted_wrt_canonical:
        m1z = np.array(
            [
                [np.cos(azim[idx_of_zone_axis_for_replacement]), np.sin(azim[idx_of_zone_axis_for_replacement]), 0],
                [-np.sin(azim[idx_of_zone_axis_for_replacement]), np.cos(azim[idx_of_zone_axis_for_replacement]), 0],
                [0, 0, 1],
            ]
        )
        m2x = np.array(
            [
                [1, 0, 0],
                [0, np.cos(elev[idx_of_zone_axis_for_replacement]), np.sin(elev[idx_of_zone_axis_for_replacement])],
                [0, -np.sin(elev[idx_of_zone_axis_for_replacement]), np.cos(elev[idx_of_zone_axis_for_replacement])],
            ]
        )

        m3z = np.array(
            [
                [np.cos(azim[idx_of_zone_axis_for_replacement]), -np.sin(azim[idx_of_zone_axis_for_replacement]), 0],
                [np.sin(azim[idx_of_zone_axis_for_replacement]), np.cos(azim[idx_of_zone_axis_for_replacement]), 0],
                [0, 0, 1],
            ]
        )

        zone_axis_aligment_matrix = m1z @ m2x @ m3z

        in_plane_rotation_matrix_with_repsect_to_z = zone_axis_aligment_matrix.T @ input_rotation_matrix_canonical

        

        in_plane_angle_rotation_matrix_with_repsect_to_z = find_angle_from_sin_cos(in_plane_rotation_matrix_with_repsect_to_z[0,1], in_plane_rotation_matrix_with_repsect_to_z[0,0])
        
        # # crystal.orientation_rotation_matrices[idx_of_zone_axis_for_replacement, :, :] = input_rotation_matrix_canonical
        # # print("crystal.orientation_rotation_matrices\n", crystal.orientation_rotation_matrices[idx_of_zone_axis_for_replacement], "\n")
        # # crystal.orientation_rotation_angles[idx_of_zone_axis_for_replacement, :] = [azim[idx_of_zone_axis_for_replacement], elev[idx_of_zone_axis_for_replacement], in_plane_angle_rotation_matrix_with_repsect_to_z]

        # # Wrap the value to the range [0, 2pi)

        # print("initial in_plane_angle_rotation_matrix_with_repsect_to_z", in_plane_angle_rotation_matrix_with_repsect_to_z)
        # wrapped_in_plane_angle_rotation_matrix_with_repsect_to_z = in_plane_angle_rotation_matrix_with_repsect_to_z % (2 * np.pi)
        
        # # If the result is negative, adjust it to be within [0, 2pi)
        # if wrapped_in_plane_angle_rotation_matrix_with_repsect_to_z < 0:
        #     print("before wrapped_in_plane_angle_rotation_matrix_with_repsect_to_z", wrapped_in_plane_angle_rotation_matrix_with_repsect_to_z)
        #     wrapped_in_plane_angle_rotation_matrix_with_repsect_to_z += 2 * np.pi
        #     print("after wrapped_in_plane_angle_rotation_matrix_with_repsect_to_z", wrapped_in_plane_angle_rotation_matrix_with_repsect_to_z)
        
        # print("Wrapped value:", wrapped_in_plane_angle_rotation_matrix_with_repsect_to_z)
        

        # crystal.in_plane_angle_rotation_matrix_with_repsect_to_z = in_plane_angle_rotation_matrix_with_repsect_to_z
        index_closest = (np.abs(crystal.orientation_gamma - in_plane_angle_rotation_matrix_with_repsect_to_z)).argmin()
        # print("index_closest", index_closest)
        # print("crystal.orientation_gamma[index_closest]", crystal.orientation_gamma[index_closest])

        crystal.in_plane_angle_rotation_matrix_with_repsect_to_z = in_plane_angle_rotation_matrix_with_repsect_to_z
        crystal.in_plane_angle_rotation_matrix_with_repsect_to_z_idx = index_closest

        # print("crystal.in_plane_angle_rotation_matrix_with_repsect_to_z: ", crystal.in_plane_angle_rotation_matrix_with_repsect_to_z)
        # print("crystal.in_plane_angle_rotation_matrix_with_repsect_to_z_idx: ", crystal.in_plane_angle_rotation_matrix_with_repsect_to_z_idx)



    # if input_rotation_matrix is not None:
    #     m1z = np.array(
    #         [
    #             [np.cos(azim[idx_of_zone_axis_for_replacement]), np.sin(azim[idx_of_zone_axis_for_replacement]), 0],
    #             [-np.sin(azim[idx_of_zone_axis_for_replacement]), np.cos(azim[idx_of_zone_axis_for_replacement]), 0],
    #             [0, 0, 1],
    #         ]
    #     )
    #     m2x = np.array(
    #         [
    #             [1, 0, 0],
    #             [0, np.cos(elev[idx_of_zone_axis_for_replacement]), np.sin(elev[idx_of_zone_axis_for_replacement])],
    #             [0, -np.sin(elev[idx_of_zone_axis_for_replacement]), np.cos(elev[idx_of_zone_axis_for_replacement])],
    #         ]
    #     )

    #     zone_axis_aligment_matrix = m1z @ m2x

    #     in_plane_rotation_matrix_with_repsect_to_z = zone_axis_aligment_matrix.T @ input_rotation_matrix_canonical

    #     in_plane_angle_rotation_matrix_with_repsect_to_z = np.arccos(np.clip(in_plane_rotation_matrix_with_repsect_to_z[0,0], -1., 1.))
    #     crystal.orientation_rotation_matrices[idx_of_zone_axis_for_replacement, :, :] = input_rotation_matrix_canonical
    #     # print("crystal.orientation_rotation_matrices\n", crystal.orientation_rotation_matrices[idx_of_zone_axis_for_replacement], "\n")
    #     crystal.orientation_rotation_angles[idx_of_zone_axis_for_replacement, :] = [azim[idx_of_zone_axis_for_replacement], elev[idx_of_zone_axis_for_replacement], in_plane_angle_rotation_matrix_with_repsect_to_z]
        
    # print("crystal.orientation_refine\n", crystal.orientation_refine, "\n") ## KWANG DELETE

    # Calculate reference arrays for all orientations
    k0 = np.array([0.0, 0.0, -1.0 / crystal.wavelength])
    n = np.array([0.0, 0.0, -1.0])

    if calculate_correlation_array:
        # initialize empty correlation array
        crystal.orientation_ref = np.zeros(
            (
                crystal.orientation_num_zones,
                np.size(crystal.orientation_shell_radii),
                crystal.orientation_in_plane_steps,
            ),
            dtype="float",
        )

        for a0 in tqdmnd(
            np.arange(crystal.orientation_num_zones),
            desc="Orientation plan",
            unit=" zone axes",
            disable=not progress_bar,
        ):
            # reciprocal lattice spots and excitation errors
            g = crystal.orientation_rotation_matrices[a0, :, :].T @ crystal.g_vec_all
            sg = crystal.excitation_errors(g)

            # Keep only points that will contribute to this orientation plan slice
            keep = np.abs(sg) < crystal.orientation_kernel_size

            # in-plane rotation angle
            phi = np.arctan2(g[1, :], g[0, :])

            # Loop over all peaks
            for a1 in np.arange(crystal.g_vec_all.shape[1]):
                ind_radial = crystal.orientation_shell_index[a1]

                if keep[a1] and ind_radial >= 0:
                    # 2D orientation plan
                    crystal.orientation_ref[a0, ind_radial, :] += (
                        np.power(crystal.orientation_shell_radii[ind_radial], radial_power)
                        * np.power(crystal.struct_factors_int[a1], intensity_power)
                        * np.maximum(
                            1
                            - np.sqrt(
                                sg[a1] ** 2
                                + (
                                    (
                                        np.mod(
                                            crystal.orientation_gamma - phi[a1] + np.pi,
                                            2 * np.pi,
                                        )
                                        - np.pi
                                    )
                                    * crystal.orientation_shell_radii[ind_radial]
                                )
                                ** 2
                            )
                            / crystal.orientation_kernel_size,
                            0,
                        )
                    )

            orientation_ref_norm = np.sqrt(np.sum(crystal.orientation_ref[a0, :, :] ** 2))
            if orientation_ref_norm > 0:
                crystal.orientation_ref[a0, :, :] /= orientation_ref_norm

        # print("crystal.orientation_ref\n", crystal.orientation_ref)
        # print("crystal.orientation_ref.shape\n", crystal.orientation_ref.shape)
        
        # print("crystal.orientation_ref[idx_of_zone_axis_for_replacement]\n", crystal.orientation_ref[idx_of_zone_axis_for_replacement])

        # Maximum value
        crystal.orientation_ref_max = np.max(np.real(crystal.orientation_ref))

        # Fourier domain along angular axis
        if crystal.CUDA:
            crystal.orientation_ref = cp.asarray(crystal.orientation_ref)
            crystal.orientation_ref = cp.conj(cp.fft.fft(crystal.orientation_ref))
        else:
            crystal.orientation_ref = np.conj(np.fft.fft(crystal.orientation_ref))

    return idx_of_zone_axis_for_replacement, is_input_rotation_matrix_inverted_wrt_canonical



def measure_sparseCorr_of_single_pattern(
    crystal,
    bragg_peaks: PointList,
    num_matches_return: int = 1,
    min_angle_between_matches_deg=None,
    min_number_peaks=3,
    inversion_symmetry=True,
    multiple_corr_reset=True,
    plot_polar: bool = False,
    plot_corr: bool = False,
    returnfig: bool = False,
    figsize: Union[list, tuple, np.ndarray] = (12, 4),
    verbose: bool = False,
    # plot_corr_3D: bool = False,
):
    """
    This function is adapted from the open-source library py4DSTEM
    (https://github.com/py4dstem/py4DSTEM), licensed under GPL-3.0.
    
    Solve for the best fit orientation of a single diffraction pattern.

    Parameters
    --------
    bragg_peaks: PointList
        numpy array containing the Bragg positions and intensities ('qx', 'qy', 'intensity')
    num_matches_return: int
        return these many matches as 3th dim of orient (matrix)
    min_angle_between_matches_deg: int
        Minimum angle between zone axis of multiple matches, in degrees.
        Note that I haven't thought how to handle in-plane rotations, since multiple matches are possible.
    min_number_peaks: int
        Minimum number of peaks required to perform ACOM matching
    inversion_symmetry bool
        check for inversion symmetry in the matches
    multiple_corr_reset bool
        keep original correlation score for multiple matches
    subpixel_tilt: bool
        set to false for faster matching, returning the nearest corr point
    plot_polar: bool
        set to true to plot the polar transform of the diffraction pattern
    plot_corr: bool
        set to true to plot the resulting correlogram
    returnfig: bool
        return figure handles
    figsize: list
        size of figure
    verbose: bool
        Print the fitted zone axes, correlation scores
    CUDA: bool
        Enable CUDA for the FFT steps

    Returns
    --------
    orientation: Orientation
        Orientation class containing all outputs
    fig, ax: handles
        Figure handles for the plotting output
    """

    # adding assert statement for checking  crystal.orientation_ref is present
    # adding assert statement for checking  crystal.orientation_ref is present
    # if not hasattr(self, "orientation_ref"):
    #     raise ValueError(
    #         "orientation_plan must be run with 'calculate_correlation_array=True'"
    #     )

    orientation = Orientation(num_matches=num_matches_return)
    if bragg_peaks.data.shape[0] < min_number_peaks:
        return orientation

    import warnings

    # get bragg peak data
    qx = bragg_peaks.data["qx"]
    qy = bragg_peaks.data["qy"]
    intensity = bragg_peaks.data["intensity"]

    # other init
    dphi = crystal.orientation_gamma[1] - crystal.orientation_gamma[0]
    corr_value = np.zeros(crystal.orientation_num_zones)
    corr_in_plane_angle = np.zeros(crystal.orientation_num_zones)
    if inversion_symmetry:
        corr_inv = np.zeros(crystal.orientation_num_zones, dtype="bool")

    # loop over the number of matches to return
    for match_ind in range(num_matches_return):
        # Convert Bragg peaks to polar coordinates
        qr = np.sqrt(qx**2 + qy**2)
        qphi = np.arctan2(qy, qx)

        # Calculate polar Bragg peak image
        im_polar = np.zeros(
            (
                np.size(crystal.orientation_shell_radii),
                crystal.orientation_in_plane_steps,
            ),
            dtype="float",
        )

        for ind_radial, radius in enumerate(crystal.orientation_shell_radii):
            dqr = np.abs(qr - radius)
            sub = dqr < crystal.orientation_kernel_size

            if np.any(sub):
                im_polar[ind_radial, :] = np.sum(
                    np.power(radius, crystal.orientation_radial_power)
                    * np.power(
                        np.maximum(intensity[sub, None], 0.0),
                        crystal.orientation_intensity_power,
                    )
                    * np.maximum(
                        1
                        - np.sqrt(
                            dqr[sub, None] ** 2
                            + (
                                (
                                    np.mod(
                                        crystal.orientation_gamma[None, :]
                                        - qphi[sub, None]
                                        + np.pi,
                                        2 * np.pi,
                                    )
                                    - np.pi
                                )
                                * radius
                            )
                            ** 2
                        )
                        / crystal.orientation_kernel_size,
                        0,
                    ),
                    axis=0,
                )

        # Determine the RMS signal from im_polar for the first match.
        # Note that we use scaling slightly below RMS so that following matches
        # don't have higher correlating scores than previous matches.
        if multiple_corr_reset is False and num_matches_return > 1:
            if match_ind == 0:
                im_polar_scale_0 = np.mean(im_polar**2) ** 0.4
            else:
                im_polar_scale = np.mean(im_polar**2) ** 0.4
                if im_polar_scale > 0:
                    im_polar *= im_polar_scale_0 / im_polar_scale
                # im_polar /= np.sqrt(np.mean(im_polar**2))
                # im_polar *= im_polar_0_rms

        # If later refinement is performed, we need to keep the original image's polar tranform if corr reset is enabled
        if crystal.orientation_refine:
            if multiple_corr_reset:
                if match_ind == 0:
                    if crystal.CUDA:
                        im_polar_refine = cp.asarray(im_polar.copy())
                    else:
                        im_polar_refine = im_polar.copy()
            else:
                if crystal.CUDA:
                    im_polar_refine = cp.asarray(im_polar.copy())
                else:
                    im_polar_refine = im_polar.copy()

        # Plot polar space image if needed
        if plot_polar is True:  # and match_ind==0:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            ax.imshow(im_polar)
            plt.show()

        # FFT along theta
        if crystal.CUDA:
            im_polar_fft = cp.fft.fft(cp.asarray(im_polar))
        else:
            im_polar_fft = np.fft.fft(im_polar)
        if crystal.orientation_refine:
            if crystal.CUDA:
                im_polar_refine_fft = cp.fft.fft(cp.asarray(im_polar_refine))
            else:
                im_polar_refine_fft = np.fft.fft(im_polar_refine)

        # Calculate full orientation correlogram
        if crystal.orientation_refine:
            corr_full = np.zeros(
                (
                    crystal.orientation_num_zones,
                    crystal.orientation_in_plane_steps,
                )
            )
            if crystal.CUDA:
                corr_full[crystal.orientation_sieve, :] = cp.maximum(
                    cp.sum(
                        cp.real(
                            cp.fft.ifft(
                                crystal.orientation_ref[crystal.orientation_sieve_CUDA, :, :]
                                * im_polar_fft[None, :, :]
                            )
                        ),
                        axis=1,
                    ),
                    0,
                ).get()
            else:
                corr_full[crystal.orientation_sieve, :] = np.maximum(
                    np.sum(
                        np.real(
                            np.fft.ifft(
                                crystal.orientation_ref[crystal.orientation_sieve, :, :]
                                * im_polar_fft[None, :, :]
                            )
                        ),
                        axis=1,
                    ),
                    0,
                )

        else:
            if crystal.CUDA:
                corr_full = np.maximum(
                    np.sum(
                        np.real(
                            cp.fft.ifft(crystal.orientation_ref * im_polar_fft[None, :, :])
                        ),
                        axis=1,
                    ),
                    0,
                ).get()
            else:
                corr_full = np.maximum(
                    np.sum(
                        np.real(
                            np.fft.ifft(crystal.orientation_ref * im_polar_fft[None, :, :])
                        ),
                        axis=1,
                    ),
                    0,
                )

        # If minimum angle is specified and we're on a match later than the first,
        # we zero correlation values within the given range.
        if min_angle_between_matches_deg is not None:
            if match_ind > 0:
                inds_previous = orientation.inds[:match_ind, 0]
                for a0 in range(inds_previous.size):
                    mask_zero = np.arccos(
                        np.clip(
                            np.sum(
                                crystal.orientation_vecs
                                * crystal.orientation_vecs[inds_previous[a0], :],
                                axis=1,
                            ),
                            -1,
                            1,
                        )
                    ) < np.deg2rad(min_angle_between_matches_deg)
                    corr_full[mask_zero, :] = 0.0

        # Get maximum (non inverted) correlation value
        # print("corr_full.shape\n", corr_full.shape, "\n") # KWANG DELETE.
        # print("corr_full\n", corr_full, "\n") # KWANG DELETE.
        # print("")

        # if not inversion_symmetry: ## Kwang

        #     print("corr_full[crystal.idx_of_zone_axis_for_replacement,0]", corr_full[crystal.idx_of_zone_axis_for_replacement,0]) ## Kwang
        
        
        ind_phi = np.argmax(corr_full, axis=1)
        # print("ind_phi\n", ind_phi, "\n")
        # print("np.max(corr_full)", np.max(corr_full))
        # print("ind_phi.shape", ind_phi.shape, "\n")
        # print("ind_phi[crystal.idx_of_zone_axis_for_replacement]\n", ind_phi[crystal.idx_of_zone_axis_for_replacement], "\n")

        # Calculate orientation correlogram for inverse pattern (in-plane mirror)
        if inversion_symmetry:
            if crystal.orientation_refine:
                corr_full_inv = np.zeros(
                    (
                        crystal.orientation_num_zones,
                        crystal.orientation_in_plane_steps,
                    )
                )
                if crystal.CUDA:
                    corr_full_inv[crystal.orientation_sieve, :] = cp.maximum(
                        cp.sum(
                            cp.real(
                                cp.fft.ifft(
                                    crystal.orientation_ref[
                                        crystal.orientation_sieve_CUDA, :, :
                                    ]
                                    * cp.conj(im_polar_fft)[None, :, :]
                                )
                            ),
                            axis=1,
                        ),
                        0,
                    ).get()
                else:
                    corr_full_inv[crystal.orientation_sieve, :] = np.maximum(
                        np.sum(
                            np.real(
                                np.fft.ifft(
                                    crystal.orientation_ref[crystal.orientation_sieve, :, :]
                                    * np.conj(im_polar_fft)[None, :, :]
                                )
                            ),
                            axis=1,
                        ),
                        0,
                    )
            else:
                if crystal.CUDA:
                    corr_full_inv = np.maximum(
                        np.sum(
                            np.real(
                                cp.fft.ifft(
                                    crystal.orientation_ref
                                    * cp.conj(im_polar_fft)[None, :, :]
                                )
                            ),
                            axis=1,
                        ),
                        0,
                    ).get()
                else:
                    corr_full_inv = np.maximum(
                        np.sum(
                            np.real(
                                np.fft.ifft(
                                    crystal.orientation_ref
                                    * np.conj(im_polar_fft)[None, :, :]
                                )
                            ),
                            axis=1,
                        ),
                        0,
                    )

            # If minimum angle is specified and we're on a match later than the first,
            # we zero correlation values within the given range.
            if min_angle_between_matches_deg is not None:
                if match_ind > 0:
                    inds_previous = orientation.inds[:match_ind, 0]
                    for a0 in range(inds_previous.size):
                        mask_zero = np.arccos(
                            np.clip(
                                np.sum(
                                    crystal.orientation_vecs
                                    * crystal.orientation_vecs[inds_previous[a0], :],
                                    axis=1,
                                ),
                                -1,
                                1,
                            )
                        ) < np.deg2rad(min_angle_between_matches_deg)
                        corr_full_inv[mask_zero, :] = 0.0

            ind_phi_inv = np.argmax(corr_full_inv, axis=1)
            corr_inv = np.zeros(crystal.orientation_num_zones, dtype="bool")


        if inversion_symmetry:
            # print("corr_full_inv.shape\n", corr_full_inv.shape, "\n") # KWANG DELETE.
            # print("corr_full_inv\n", corr_full_inv, "\n") # KWANG DELETE.
            # print("corr_full_inv[crystal.idx_of_zone_axis_for_replacement,0]", corr_full_inv[crystal.idx_of_zone_axis_for_replacement,0])
            # print("corr_full_inv[crystal.idx_of_zone_axis_for_replacement,90]", corr_full_inv[crystal.idx_of_zone_axis_for_replacement,90])

            if crystal.in_plane_angle_rotation_matrix_with_repsect_to_z_idx > int(int(len(crystal.orientation_gamma)/2) - 1):
                index_of_interest_for_inv = crystal.in_plane_angle_rotation_matrix_with_repsect_to_z_idx - int(90)
            else:
                index_of_interest_for_inv = int(90) + crystal.in_plane_angle_rotation_matrix_with_repsect_to_z_idx

            # print("ayatsuno", index_of_interest_for_inv)

            # if index_of_interest_for_inv > 179:
            #     ndex_of_interest_for_inv = int(90) - crystal.in_plane_angle_rotation_matrix_with_repsect_to_z_idx

            ind_phi_inv[crystal.idx_of_zone_axis_for_replacement] = index_of_interest_for_inv

            # print("ind_phi_inv\n", ind_phi_inv, "\n")

        else:################################################################################################################### REMOVE IF UNNECSAARY
            # print("ataho ind_phi[crystal.idx_of_zone_axis_for_replacement]", ind_phi[crystal.idx_of_zone_axis_for_replacement])
            # print("corr_full.shape\n", corr_full.shape, "\n") # KWANG DELETE.
            # print("corr_full\n", corr_full, "\n") # KWANG DELETE.
            # print("corr_full[crystal.idx_of_zone_axis_for_replacement,0]", corr_full[crystal.idx_of_zone_axis_for_replacement,0])
            # print("corr_full[crystal.idx_of_zone_axis_for_replacement,131]", corr_full[crystal.idx_of_zone_axis_for_replacement,131])
            ind_phi[crystal.idx_of_zone_axis_for_replacement] = int(0) + crystal.in_plane_angle_rotation_matrix_with_repsect_to_z_idx
            # print("ind_phi\n", ind_phi)
            # print("ind_phi.shape", ind_phi.shape)
            

        # Find best match for each zone axis
        corr_value[:] = 0
        for a0 in range(crystal.orientation_num_zones):
            # if a0 == crystal.idx_of_zone_axis_for_replacement: ## KWANGKWANG DELETE
                # print("a0", a0)
                # print("ind_phi[a0]\n", ind_phi[a0], "\n")
                # print("corr_full[a0, ind_phi[a0]]", corr_full[a0, ind_phi[a0]], "\n")

                # if inversion_symmetry:
                #     print("corr_full_inv[a0, ind_phi_inv[a0]]", corr_full_inv[a0, ind_phi_inv[a0]])
                #     print("corr_full[a0, ind_phi[a0]]", corr_full[a0, ind_phi[a0]])
                #     print("ind_phi_inv[a0]\n", ind_phi_inv[a0], "\n")                
                #     print("corr_full_inv[a0, ind_phi_inv[a0]]\n" ,corr_full_inv[a0, ind_phi_inv[a0]] ,"\n")
            # print("######################a0")
            if (crystal.orientation_refine is False) or crystal.orientation_sieve[a0]:
                # Correlation score
                if inversion_symmetry:
                    corr_value[a0] = corr_full_inv[a0, ind_phi_inv[a0]]
                    corr_inv[a0] = True ## KWANG. Here if corr_full value of given zone axis a0 index is smaller than corr_full.
                    
                    # if corr_full_inv[a0, ind_phi_inv[a0]] > corr_full[a0, ind_phi[a0]]:
                    #     # print("charlie")
                    #     corr_value[a0] = corr_full_inv[a0, ind_phi_inv[a0]]
                    #     corr_inv[a0] = True ## KWANG. Here if corr_full value of given zone axis a0 index is smaller than corr_full. 
                    # else:
                    #     # print("park")
                    #     corr_value[a0] = corr_full[a0, ind_phi[a0]]
                else:
                    corr_value[a0] = corr_full[a0, ind_phi[a0]]

                # KEEP IT (START)

                # In-plane sub-pixel angular fit
                if inversion_symmetry and corr_inv[a0]:
                    # print("if inversion_symmetry and corr_inv[a0]") # KWANG DELETE
                    # print("ind_phi_inv[a0]", ind_phi_inv[a0])
                    inds = np.mod(
                        ind_phi_inv[a0] + np.arange(-1, 2), crystal.orientation_gamma.size
                    ).astype("int")
                    # print("inds", inds)
                    c = corr_full_inv[a0, inds]
                    # print("c", c)
                    if np.max(c) > 0:
                        if np.abs(4 * c[1] - 2 * c[0] - 2 * c[2]) > 1e-8: 
                            dc = (c[2] - c[0]) / (4 * c[1] - 2 * c[0] - 2 * c[2])
                            corr_in_plane_angle[a0] = (
                                crystal.orientation_gamma[ind_phi_inv[a0]] + dc * dphi
                            ) + np.pi
                        else:
                            corr_in_plane_angle[a0] = crystal.orientation_gamma[ind_phi_inv[a0]]
                        # print("corr_in_plane_angle[a0]", corr_in_plane_angle[a0])
                else:
                    # print("if not inversion_symmetry and corr_inv[a0]") # KWANG DELETE
                    # print("ind_phi[a0]", ind_phi[a0])
                    inds = np.mod(
                        ind_phi[a0] + np.arange(-1, 2), crystal.orientation_gamma.size
                    ).astype("int")
                    c = corr_full[a0, inds]
                    # print("c", c)
                    if np.max(c) > 0:
                        if np.abs(4 * c[1] - 2 * c[0] - 2 * c[2]) > 1e-8: 
                            dc = (c[2] - c[0]) / (4 * c[1] - 2 * c[0] - 2 * c[2])
                            corr_in_plane_angle[a0] = (
                                crystal.orientation_gamma[ind_phi[a0]] + dc * dphi
                            )
                        else:
                            corr_in_plane_angle[a0] = crystal.orientation_gamma[ind_phi[a0]]
                        # print("corr_in_plane_angle[a0]", corr_in_plane_angle[a0])

                # KEEP IT (END)


        # If needed, keep original polar image to recompute the correlations
        if (
            multiple_corr_reset
            and num_matches_return > 1
            and match_ind == 0
            and not crystal.orientation_refine
        ):
            corr_value_keep = corr_value.copy()
            corr_in_plane_angle_keep = corr_in_plane_angle.copy()
        
        # print("corr_in_plane_angle_keep", corr_in_plane_angle_keep)
        # print("corr_in_plane_angle", corr_in_plane_angle)

        # Determine the best fit orientation
        ind_best_fit = np.unravel_index(np.argmax(corr_value), corr_value.shape)[0]

        # print("ind_best_fit\n", ind_best_fit, "\n")

        ind_best_fit = crystal.idx_of_zone_axis_for_replacement
        
        # print("before crystal.orientation_refine, corr_full.shape", corr_full.shape) ## KWANG DELETE
        # print("before crystal.orientation_refine, corr_full_inv.shape", corr_full_inv.shape) ## KWANG DELETE

        ############################################################
        # If needed, perform fine step refinement of the zone axis #
        ############################################################
        if crystal.orientation_refine:
            mask_refine = np.arccos(
                np.clip(
                    np.sum(
                        crystal.orientation_vecs * crystal.orientation_vecs[ind_best_fit, :],
                        axis=1,
                    ),
                    -1,
                    1,
                )
            ) < np.deg2rad(crystal.orientation_refine_range)
            if crystal.CUDA:
                mask_refine_CUDA = cp.asarray(mask_refine)

            if crystal.CUDA:
                corr_full[mask_refine, :] = cp.maximum(
                    cp.sum(
                        cp.real(
                            cp.fft.ifft(
                                crystal.orientation_ref[mask_refine_CUDA, :, :]
                                * im_polar_refine_fft[None, :, :]
                            )
                        ),
                        axis=1,
                    ),
                    0,
                ).get()
            else:
                corr_full[mask_refine, :] = np.maximum(
                    np.sum(
                        np.real(
                            np.fft.ifft(
                                crystal.orientation_ref[mask_refine, :, :]
                                * im_polar_refine_fft[None, :, :]
                            )
                        ),
                        axis=1,
                    ),
                    0,
                )

            # Get maximum (non inverted) correlation value
            ind_phi = np.argmax(corr_full, axis=1)

            # Inversion symmetry
            if inversion_symmetry:
                if crystal.CUDA:
                    corr_full_inv[mask_refine, :] = cp.maximum(
                        cp.sum(
                            cp.real(
                                cp.fft.ifft(
                                    crystal.orientation_ref[mask_refine_CUDA, :, :]
                                    * cp.conj(im_polar_refine_fft)[None, :, :]
                                )
                            ),
                            axis=1,
                        ),
                        0,
                    ).get()
                else:
                    corr_full_inv[mask_refine, :] = np.maximum(
                        np.sum(
                            np.real(
                                np.fft.ifft(
                                    crystal.orientation_ref[mask_refine, :, :]
                                    * np.conj(im_polar_refine_fft)[None, :, :]
                                )
                            ),
                            axis=1,
                        ),
                        0,
                    )
                ind_phi_inv = np.argmax(corr_full_inv, axis=1)

            # Determine best in-plane correlation
            print("crystal.orientation_refine==True, corr_full.shape", corr_full.shape) ## KWANG DELETE
            print("crystal.orientation_refine==True, corr_full_inv.shape", corr_full_inv.shape) ## KWANG DELETE
            for a0 in np.argwhere(mask_refine):
                # Correlation score
                if inversion_symmetry:
                    if corr_full_inv[a0, ind_phi_inv[a0]] > corr_full[a0, ind_phi[a0]]:
                        corr_value[a0] = corr_full_inv[a0, ind_phi_inv[a0]]
                        corr_inv[a0] = True
                    else:
                        corr_value[a0] = corr_full[a0, ind_phi[a0]]
                else:
                    corr_value[a0] = corr_full[a0, ind_phi[a0]]

                # Subpixel angular fit
                if inversion_symmetry and corr_inv[a0]:
                    inds = np.mod(
                        ind_phi_inv[a0] + np.arange(-1, 2), crystal.orientation_gamma.size
                    ).astype("int")
                    c = corr_full_inv[a0, inds]
                    if np.max(c) > 0:
                        dc = (c[2] - c[0]) / (4 * c[1] - 2 * c[0] - 2 * c[2])
                        corr_in_plane_angle[a0] = (
                            crystal.orientation_gamma[ind_phi_inv[a0]] + dc * dphi
                        ) + np.pi
                else:
                    inds = np.mod(
                        ind_phi[a0] + np.arange(-1, 2), crystal.orientation_gamma.size
                    ).astype("int")
                    c = corr_full[a0, inds]
                    if np.max(c) > 0:
                        dc = (c[2] - c[0]) / (4 * c[1] - 2 * c[0] - 2 * c[2])
                        corr_in_plane_angle[a0] = (
                            crystal.orientation_gamma[ind_phi[a0]] + dc * dphi
                        )

            # Determine the new best fit orientation
            ind_best_fit = np.unravel_index(
                np.argmax(corr_value * mask_refine[None, :]), corr_value.shape
            )[0]

        ind_best_fit = crystal.idx_of_zone_axis_for_replacement
            
        ## KWANG GNETLY NOTE THAT CORRELATION SCORE IS ALREADY CALCULATED ABOVE NOT BELOW

        # Verify current match has a correlation > 0
        if corr_value[ind_best_fit] > 0:
            isCorrScoreZero = False;
            # Get orientation matrix
            orientation_matrix = np.squeeze(
                crystal.orientation_rotation_matrices[ind_best_fit, :, :]
            )
            
            # print("chimha orientation_matrix\n", orientation_matrix, "\n") ## KWANG
            ## KWANG. Check where does ind_best_fit comes from

            # apply in-plane rotation, and inversion if needed
            if (
                multiple_corr_reset
                and match_ind > 0
                and crystal.orientation_refine is False
            ):
                phi = corr_in_plane_angle_keep[ind_best_fit]
            else:
                phi = corr_in_plane_angle[ind_best_fit]

            try:
                v = [np.cos(phi), np.sin(phi), 0]
            except RuntimeWarning:
                print(f"RuntimeWarning at iteration, phi = {phi}")
                # print("corr_in_plane_angle_keep[ind_best_fit]\n", corr_in_plane_angle_keep[ind_best_fit], "\n")
                # print()
                break
            m3z = np.array(
                [
                    [np.cos(phi), np.sin(phi), 0],
                    [-np.sin(phi), np.cos(phi), 0],
                    [0, 0, 1],
                ]
            )
            # print("")
            # print("m3z\n", m3z, "\n")
            # print("phi\n", phi, "\n")
            orientation_matrix = orientation_matrix @ m3z
            
            if inversion_symmetry and corr_inv[ind_best_fit]:                     # KWANG NOTE HERE
                # Rotate 180 degrees around x axis for projected x-mirroring operation
                orientation_matrix[:, 1:] = -orientation_matrix[:, 1:]

            # Output best fit values into Orientation class
            orientation.matrix[match_ind] = orientation_matrix

            if crystal.orientation_refine:
                orientation.corr[match_ind] = corr_value[ind_best_fit]
            else:
                if multiple_corr_reset and match_ind > 0:
                    orientation.corr[match_ind] = corr_value_keep[ind_best_fit]
                else:
                    orientation.corr[match_ind] = corr_value[ind_best_fit]

            if inversion_symmetry and corr_inv[ind_best_fit]:
                # print("kakao")
                ind_phi = ind_phi_inv[ind_best_fit]
            else:
                # print("taotao")
                ind_phi = ind_phi[ind_best_fit]
            orientation.inds[match_ind, 0] = ind_best_fit
            orientation.inds[match_ind, 1] = ind_phi

            if inversion_symmetry:
                orientation.mirror[match_ind] = corr_inv[ind_best_fit]

            orientation.angles[match_ind, :] = crystal.orientation_rotation_angles[
                ind_best_fit, :
            ]
            orientation.angles[match_ind, 2] += phi

            # If point group is known, use pymatgen to caculate the symmetry-
            # reduced orientation matrix, producing the crystal direction family.
            if crystal.pymatgen_available:
                orientation = crystal.symmetry_reduce_directions(
                    orientation,
                    match_ind=match_ind,
                )

        else:
            isCorrScoreZero = True;
            # No more matches are detected, so output default orientation matrix and leave corr = 0
            orientation.matrix[match_ind] = np.squeeze(
                crystal.orientation_rotation_matrices[0, :, :]
            )
            orientation.corr[match_ind] = 0.0

        if verbose:
            if crystal.pymatgen_available:
                if np.abs(crystal.cell[5] - 120.0) < 1e-6:
                    x_proj_lattice = crystal.lattice_to_hexagonal(
                        crystal.cartesian_to_lattice(orientation.family[match_ind][:, 0])
                    )
                    x_proj_lattice = np.round(x_proj_lattice, decimals=3)
                    zone_axis_lattice = crystal.lattice_to_hexagonal(
                        crystal.cartesian_to_lattice(orientation.family[match_ind][:, 2])
                    )
                    zone_axis_lattice = np.round(zone_axis_lattice, decimals=3)
                else:
                    if np.max(np.abs(orientation.family)) > 0.1:
                        x_proj_lattice = crystal.cartesian_to_lattice(
                            orientation.family[match_ind][:, 0]
                        )
                        x_proj_lattice = np.round(x_proj_lattice, decimals=3)
                        zone_axis_lattice = crystal.cartesian_to_lattice(
                            orientation.family[match_ind][:, 2]
                        )
                        zone_axis_lattice = np.round(zone_axis_lattice, decimals=3)

                if orientation.corr[match_ind] > 0:
                    print(
                        "Best fit lattice directions: z axis = ("
                        + str(zone_axis_lattice)
                        + "),"
                        " x axis = ("
                        + str(x_proj_lattice)
                        + "),"
                        + " with corr value = "
                        + str(np.round(orientation.corr[match_ind], decimals=3))
                    )
                else:
                    print("No good match found for index " + str(match_ind))

            else:
                zone_axis_fit = orientation.matrix[match_ind][:, 2]
                zone_axis_lattice = crystal.cartesian_to_lattice(zone_axis_fit)
                zone_axis_lattice = np.round(zone_axis_lattice, decimals=3)
                print(
                    "Best fit zone axis (lattice) = ("
                    + str(zone_axis_lattice)
                    + "),"
                    + " with corr value = "
                    + str(np.round(orientation.corr[match_ind], decimals=3))
                )

        # if needed, delete peaks for next iteration
        if num_matches_return > 1 and corr_value[ind_best_fit] > 0:
            bragg_peaks_fit = crystal.generate_diffraction_pattern(
                orientation,
                ind_orientation=match_ind,
                sigma_excitation_error=crystal.orientation_kernel_size,
            )

            remove = np.zeros_like(qx, dtype="bool")
            scale_int = np.ones_like(qx)
            for a0 in np.arange(qx.size):
                d_2 = (bragg_peaks_fit.data["qx"] - qx[a0]) ** 2 + (
                    bragg_peaks_fit.data["qy"] - qy[a0]
                ) ** 2

                dist_min = np.sqrt(np.min(d_2))

                if dist_min < crystal.orientation_tol_peak_delete:
                    remove[a0] = True
                elif dist_min < crystal.orientation_kernel_size:
                    scale_int[a0] = (dist_min - crystal.orientation_tol_peak_delete) / (
                        crystal.orientation_kernel_size - crystal.orientation_tol_peak_delete
                    )

            intensity = intensity * scale_int
            qx = qx[~remove]
            qy = qy[~remove]
            intensity = intensity[~remove]

        # plotting correlation image
        if plot_corr is True:
            corr_plot = corr_value.copy()
            sig_in_plane = np.squeeze(corr_full[ind_best_fit, :]).copy()

            if crystal.orientation_full:
                fig, ax = plt.subplots(1, 2, figsize=figsize * np.array([2, 2]))
                cmin = np.min(corr_plot)
                cmax = np.max(corr_plot)

                im_corr_zone_axis = np.zeros(
                    (
                        2 * crystal.orientation_zone_axis_steps + 1,
                        2 * crystal.orientation_zone_axis_steps + 1,
                    )
                )

                sub = crystal.orientation_inds[:, 2] == 0
                x_inds = (
                    crystal.orientation_inds[sub, 0] - crystal.orientation_inds[sub, 1]
                ).astype("int") + crystal.orientation_zone_axis_steps
                y_inds = (
                    crystal.orientation_inds[sub, 1].astype("int")
                    + crystal.orientation_zone_axis_steps
                )
                inds_1D = np.ravel_multi_index(
                    [x_inds, y_inds], im_corr_zone_axis.shape
                )
                im_corr_zone_axis.ravel()[inds_1D] = corr_plot[sub]

                sub = crystal.orientation_inds[:, 2] == 1
                x_inds = (
                    crystal.orientation_inds[sub, 0] - crystal.orientation_inds[sub, 1]
                ).astype("int") + crystal.orientation_zone_axis_steps
                y_inds = crystal.orientation_zone_axis_steps - crystal.orientation_inds[
                    sub, 1
                ].astype("int")
                inds_1D = np.ravel_multi_index(
                    [x_inds, y_inds], im_corr_zone_axis.shape
                )
                im_corr_zone_axis.ravel()[inds_1D] = corr_plot[sub]

                sub = crystal.orientation_inds[:, 2] == 2
                x_inds = (
                    crystal.orientation_inds[sub, 1] - crystal.orientation_inds[sub, 0]
                ).astype("int") + crystal.orientation_zone_axis_steps
                y_inds = (
                    crystal.orientation_inds[sub, 1].astype("int")
                    + crystal.orientation_zone_axis_steps
                )
                inds_1D = np.ravel_multi_index(
                    [x_inds, y_inds], im_corr_zone_axis.shape
                )
                im_corr_zone_axis.ravel()[inds_1D] = corr_plot[sub]

                sub = crystal.orientation_inds[:, 2] == 3
                x_inds = (
                    crystal.orientation_inds[sub, 1] - crystal.orientation_inds[sub, 0]
                ).astype("int") + crystal.orientation_zone_axis_steps
                y_inds = crystal.orientation_zone_axis_steps - crystal.orientation_inds[
                    sub, 1
                ].astype("int")
                inds_1D = np.ravel_multi_index(
                    [x_inds, y_inds], im_corr_zone_axis.shape
                )
                im_corr_zone_axis.ravel()[inds_1D] = corr_plot[sub]

                im_plot = (im_corr_zone_axis - cmin) / (cmax - cmin)
                ax[0].imshow(im_plot, cmap="viridis", vmin=0.0, vmax=1.0)

            elif crystal.orientation_half:
                fig, ax = plt.subplots(1, 2, figsize=figsize * np.array([2, 1]))
                cmin = np.min(corr_plot)
                cmax = np.max(corr_plot)

                im_corr_zone_axis = np.zeros(
                    (
                        crystal.orientation_zone_axis_steps + 1,
                        crystal.orientation_zone_axis_steps * 2 + 1,
                    )
                )

                sub = crystal.orientation_inds[:, 2] == 0
                x_inds = (
                    crystal.orientation_inds[sub, 0] - crystal.orientation_inds[sub, 1]
                ).astype("int")
                y_inds = (
                    crystal.orientation_inds[sub, 1].astype("int")
                    + crystal.orientation_zone_axis_steps
                )
                inds_1D = np.ravel_multi_index(
                    [x_inds, y_inds], im_corr_zone_axis.shape
                )
                im_corr_zone_axis.ravel()[inds_1D] = corr_plot[sub]

                sub = crystal.orientation_inds[:, 2] == 1
                x_inds = (
                    crystal.orientation_inds[sub, 0] - crystal.orientation_inds[sub, 1]
                ).astype("int")
                y_inds = crystal.orientation_zone_axis_steps - crystal.orientation_inds[
                    sub, 1
                ].astype("int")
                inds_1D = np.ravel_multi_index(
                    [x_inds, y_inds], im_corr_zone_axis.shape
                )
                im_corr_zone_axis.ravel()[inds_1D] = corr_plot[sub]

                im_plot = (im_corr_zone_axis - cmin) / (cmax - cmin)
                ax[0].imshow(im_plot, cmap="viridis", vmin=0.0, vmax=1.0)

            else:
                fig, ax = plt.subplots(1, 2, figsize=figsize)
                cmin = np.min(corr_plot)
                cmax = np.max(corr_plot)

                im_corr_zone_axis = np.zeros(
                    (
                        crystal.orientation_zone_axis_steps + 1,
                        crystal.orientation_zone_axis_steps + 1,
                    )
                )
                im_mask = np.ones(
                    (
                        crystal.orientation_zone_axis_steps + 1,
                        crystal.orientation_zone_axis_steps + 1,
                    ),
                    dtype="bool",
                )

                # Image indices
                x_inds = (
                    crystal.orientation_inds[:, 0] - crystal.orientation_inds[:, 1]
                ).astype("int")
                y_inds = crystal.orientation_inds[:, 1].astype("int")

                # Check vertical range of the orientation triangle.
                if (
                    crystal.orientation_fiber_angles is not None
                    and np.abs(crystal.orientation_fiber_angles[0] - 180.0) > 1e-3
                ):
                    # Orientation covers only top of orientation sphere

                    inds_1D = np.ravel_multi_index(
                        [x_inds, y_inds], im_corr_zone_axis.shape
                    )
                    im_corr_zone_axis.ravel()[inds_1D] = corr_plot
                    im_mask.ravel()[inds_1D] = False

                else:
                    # Orientation covers full vertical range of orientation sphere.
                    # top half
                    sub = crystal.orientation_inds[:, 2] == 0
                    inds_1D = np.ravel_multi_index(
                        [x_inds[sub], y_inds[sub]], im_corr_zone_axis.shape
                    )
                    im_corr_zone_axis.ravel()[inds_1D] = corr_plot[sub]
                    im_mask.ravel()[inds_1D] = False
                    # bottom half
                    sub = crystal.orientation_inds[:, 2] == 1
                    inds_1D = np.ravel_multi_index(
                        [
                            crystal.orientation_zone_axis_steps - y_inds[sub],
                            crystal.orientation_zone_axis_steps - x_inds[sub],
                        ],
                        im_corr_zone_axis.shape,
                    )
                    im_corr_zone_axis.ravel()[inds_1D] = corr_plot[sub]
                    im_mask.ravel()[inds_1D] = False

                if cmax > cmin:
                    im_plot = np.ma.masked_array(
                        (im_corr_zone_axis - cmin) / (cmax - cmin), mask=im_mask
                    )
                else:
                    im_plot = im_corr_zone_axis

                ax[0].imshow(im_plot, cmap="viridis", vmin=0.0, vmax=1.0)
                ax[0].spines["left"].set_color("none")
                ax[0].spines["right"].set_color("none")
                ax[0].spines["top"].set_color("none")
                ax[0].spines["bottom"].set_color("none")

                inds_plot = np.unravel_index(
                    np.argmax(im_plot, axis=None), im_plot.shape
                )
                ax[0].scatter(
                    inds_plot[1],
                    inds_plot[0],
                    s=120,
                    linewidth=2,
                    facecolors="none",
                    edgecolors="r",
                )

                if np.abs(crystal.cell[5] - 120.0) < 1e-6:
                    label_0 = crystal.rational_ind(
                        crystal.lattice_to_hexagonal(
                            crystal.cartesian_to_lattice(
                                crystal.orientation_zone_axis_range[0, :]
                            )
                        )
                    )
                    label_1 = crystal.rational_ind(
                        crystal.lattice_to_hexagonal(
                            crystal.cartesian_to_lattice(
                                crystal.orientation_zone_axis_range[1, :]
                            )
                        )
                    )
                    label_2 = crystal.rational_ind(
                        crystal.lattice_to_hexagonal(
                            crystal.cartesian_to_lattice(
                                crystal.orientation_zone_axis_range[2, :]
                            )
                        )
                    )
                else:
                    label_0 = crystal.rational_ind(
                        crystal.cartesian_to_lattice(
                            crystal.orientation_zone_axis_range[0, :]
                        )
                    )
                    label_1 = crystal.rational_ind(
                        crystal.cartesian_to_lattice(
                            crystal.orientation_zone_axis_range[1, :]
                        )
                    )
                    label_2 = crystal.rational_ind(
                        crystal.cartesian_to_lattice(
                            crystal.orientation_zone_axis_range[2, :]
                        )
                    )

                ax[0].set_xticks([0, crystal.orientation_zone_axis_steps])
                ax[0].set_xticklabels([str(label_0), str(label_2)], size=14)
                ax[0].xaxis.tick_top()

                ax[0].set_yticks([crystal.orientation_zone_axis_steps])
                ax[0].set_yticklabels([str(label_1)], size=14)

            # In-plane rotation
            # sig_in_plane = np.squeeze(corr_full[ind_best_fit, :])
            sig_in_plane_max = np.max(sig_in_plane)
            if sig_in_plane_max > 0:
                sig_in_plane /= sig_in_plane_max
            ax[1].plot(
                crystal.orientation_gamma * 180 / np.pi,
                sig_in_plane,
            )

            # Add markers for the best fit
            tol = 0.01
            sub = sig_in_plane > 1 - tol
            ax[1].scatter(
                crystal.orientation_gamma[sub] * 180 / np.pi,
                sig_in_plane[sub],
                s=120,
                linewidth=2,
                facecolors="none",
                edgecolors="r",
            )

            ax[1].set_xlabel("In-plane rotation angle [deg]", size=16)
            ax[1].set_ylabel("Corr. of Best Fit Zone Axis", size=16)
            ax[1].set_ylim([0, 1.03])

            plt.show()

    # print("crystal.idx_of_zone_axis_for_replacement\n", crystal.idx_of_zone_axis_for_replacement)

    if isCorrScoreZero:
        final_corr_value = 0.0;

    else:
        if inversion_symmetry:
            # print("ind_phi_inv[crystal.idx_of_zone_axis_for_replacement]\n", ind_phi_inv[crystal.idx_of_zone_axis_for_replacement])
            final_corr_value = corr_full_inv[crystal.idx_of_zone_axis_for_replacement, ind_phi]
        else:
            # print("ind_phi\n", ind_phi)
            # print("crystal.idx_of_zone_axis_for_replacement\n", crystal.idx_of_zone_axis_for_replacement, "\n")
            # print("ind_phi[crystal.idx_of_zone_axis_for_replacement]\n", ind_phi[crystal.idx_of_zone_axis_for_replacement])
            final_corr_value = corr_full[crystal.idx_of_zone_axis_for_replacement, ind_phi]

    if returnfig:
        return orientation, fig, ax, final_corr_value
    else:
        return orientation, final_corr_value


def zone_axis_local_correlation_map(Z_field, real_space_grain_index, index_of_crystal=1, window_radius=2):
    """
    Compute local correlation map for zone-axis unit vectors, 
    EXCLUDING the correlation of the center pixel with itself.
    """
    H, W, _ = Z_field.shape
    local_corr = np.full((H, W), np.nan, dtype=np.float64)
    mask = (real_space_grain_index == index_of_crystal)

    for i in range(H):
        for j in range(W):
            if not mask[i, j]:
                continue

            # 1. Define local window bounds
            i_min = max(0, i - window_radius)
            i_max = min(H, i + window_radius + 1)
            j_min = max(0, j - window_radius)
            j_max = min(W, j + window_radius + 1)

            # 2. Get the local mask identifying all crystal pixels in the window
            local_mask = mask[i_min:i_max, j_min:j_max]

            # print("local_mask\n", local_mask)
            # print("local_mask.shape\n", local_mask.shape)
            
            # 3. Create a secondary mask to EXCLUDE the center pixel (i, j)
            # Find center index within the local window (i_max-i_min, j_max-j_min)
            center_row = i - i_min
            center_col = j - j_min

            # print("")
            # print("i", i)
            # print("j", j)
            # print("center_row", center_row)
            # print("center_col", center_col)
            
            # The local_mask already identifies all crystal pixels, including the center.
            # We create a 'neighbor_only' mask by copying the local_mask and setting 
            # the center position to False.
            neighbor_only_mask = local_mask.copy()
            neighbor_only_mask[center_row, center_col] = False 
            
            # 4. Extract the true neighbor vectors
            # V_window is the whole window slice of the Z_field
            V_window = Z_field[i_min:i_max, j_min:j_max]
            v_neighbors = V_window[neighbor_only_mask]

            # print("v_neighbors.shape", v_neighbors.shape)
            
            # 5. Check: If there are no valid neighbors, assign NaN and skip.
            # This handles isolated pixels and edge cases where the only crystal 
            # pixel in the window is the center itself.
            if v_neighbors.size == 0:
                continue 

            # 6. Perform correlation calculation
            v_center = Z_field[i, j]

            # Compute cosine of angle (dot product) between center and ONLY neighbors
            cos_thetas = np.dot(v_neighbors, v_center)
            cos_thetas = np.clip(cos_thetas, -1.0, 1.0)
            
            # The local correlation is the mean of the cosines with true neighbors
            local_corr[i, j] = cos_thetas.mean()

    return local_corr


def zone_axis_autocorrelation_map_masked(Z_field_i, real_space_grain_index, index_of_crystal=1):
    """
    Compute 2D autocorrelation map for a field of zone-axis unit vectors (3D),
    using dot products (cosine of angle), only over pixels in the chosen grain.

    Parameters
    ----------
    Z_field : ndarray, shape (H, W, 3)
        Zone-axis vectors (each normalized to length 1).
    real_space_grain_index : ndarray, shape (H, W)
        Integer labels for grains or background.
    index_of_crystal : int, optional
        Grain index for which to compute correlation (default=1).

    Returns
    -------
    corr_map : ndarray, shape (2H-1, 2W-1)
        Global two-point correlation map averaged over valid pairs.
        corr_map[H-1+di, W-1+dj] = average dot(Z_ij, Z_{i+di,j+dj})
    """
    H, W, _ = Z_field_i.shape
    corr_map = np.zeros((2 * H - 1, 2 * W - 1), dtype=np.float64)
    
    mask = (real_space_grain_index == index_of_crystal)

    # Ensure normalization (in case of small numerical drift)

    Z_field = Z_field_i.copy()
    norm = np.linalg.norm(Z_field, axis=-1, keepdims=True)
    
    # Normalize only where mask is True
    Z_field[mask] /= norm[mask]

    # Iterate over all spatial offsets
    for di in range(-H + 1, H):
        for dj in range(-W + 1, W):

            # Overlapping regions
            i1 = slice(max(0, -di), min(H, H - di))
            j1 = slice(max(0, -dj), min(W, W - dj))
            i2 = slice(max(0, di), min(H, H + di))
            j2 = slice(max(0, dj), min(W, W + dj))

            mask1 = mask[i1, j1]
            mask2 = mask[i2, j2]
            valid_mask = mask1 & mask2

            if not np.any(valid_mask):
                continue

            Z1 = Z_field[i1, j1][valid_mask]
            Z2 = Z_field[i2, j2][valid_mask]

            # Compute dot products between corresponding vectors
            corr_vals = np.einsum('ni,ni->n', Z1, Z2)

            corr_map[H - 1 + di, W - 1 + dj] = corr_vals.mean()

    return corr_map

def zone_axis_cross_correlation_map_masked(Z_field_1_i, Z_field_2_i, real_space_grain_index, index_of_crystal=1):
    """
    Compute 2D cross-correlation map between two fields of zone-axis unit vectors (3D),
    using dot products (cosine of angle), only over pixels in the chosen grain.

    Parameters
    ----------
    Z_field_1 : ndarray, shape (H, W, 3)
        First zone-axis vector field (each normalized to length 1).
    Z_field_2 : ndarray, shape (H, W, 3)
        Second zone-axis vector field (same shape as Z_field_1).
    real_space_grain_index : ndarray, shape (H, W)
        Integer labels for grains or background.
    index_of_crystal : int, optional
        Grain index for which to compute correlation (default=1).

    Returns
    -------
    corr_map : ndarray, shape (2H-1, 2W-1)
        Global two-point cross-correlation map averaged over valid pairs.
        corr_map[H-1+di, W-1+dj] = average dot(Z1_ij, Z2_{i+di,j+dj})
    """
    H, W, _ = Z_field_1_i.shape
    corr_map = np.zeros((2 * H - 1, 2 * W - 1), dtype=np.float64)
    
    mask = (real_space_grain_index == index_of_crystal)

    # Copy fields to avoid modifying original
    Z_field_1 = Z_field_1_i.copy()
    Z_field_2 = Z_field_2_i.copy()
    
    # Compute norms only for valid pixels
    norm1 = np.linalg.norm(Z_field_1, axis=-1, keepdims=True)
    norm2 = np.linalg.norm(Z_field_2, axis=-1, keepdims=True)
    
    # Normalize only where mask is True
    Z_field_1[mask] /= norm1[mask]
    Z_field_2[mask] /= norm2[mask]


    # Iterate over all spatial offsets
    for di in range(-H + 1, H):
        for dj in range(-W + 1, W):

            # Overlapping regions
            i1 = slice(max(0, -di), min(H, H - di))
            j1 = slice(max(0, -dj), min(W, W - dj))
            i2 = slice(max(0, di), min(H, H + di))
            j2 = slice(max(0, dj), min(W, W + dj))

            mask1 = mask[i1, j1]
            mask2 = mask[i2, j2]
            valid_mask = mask1 & mask2

            if not np.any(valid_mask):
                continue

            Z1 = Z_field_1[i1, j1][valid_mask]
            Z2 = Z_field_2[i2, j2][valid_mask]

            # Compute dot products between corresponding vectors
            corr_vals = np.einsum('ni,ni->n', Z1, Z2)

            corr_map[H - 1 + di, W - 1 + dj] = corr_vals.mean()

    return corr_map

def radial_average_corrMap(data, nbins=60, include_center=True):
    if data.shape[0] != data.shape[1]:
        raise ValueError("Input array must be square")

    N = data.shape[0]
    center = (N // 2, N // 2)

    # 1. Calculate radial distance (r) for every point
    y, x = np.indices(data.shape)
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)

    r = r.flatten()
    data_flat = data.flatten()

    # 2. Handle center exclusion
    if not include_center:
        # Create a mask to exclude the center point (where r=0)
        mask = r > 0
        r = r[mask]
        data_flat = data_flat[mask]

    # --- FIX APPLIED HERE ---
    # 3. Define the bin range based on the potentially masked data
    r_min, r_max = r.min(), r.max()
    bins = np.linspace(r_min, r_max, nbins + 1)
    # ------------------------

    # 4. Bin the radii
    bin_indices = np.digitize(r, bins) - 1
    
    # Handle the edge case where r = r_max, mapping it to the last bin
    bin_indices[bin_indices == nbins] = nbins - 1
    # Ensure indices are within bounds (0 to nbins-1)
    bin_indices[bin_indices < 0] = 0

    radial_profile = np.zeros(nbins)
    counts = np.zeros(nbins)

    # 5. Calculate sum and count for each bin
    np.add.at(radial_profile, bin_indices, data_flat)
    np.add.at(counts, bin_indices, 1)

    # 6. Calculate the average
    radial_profile /= np.maximum(counts, 1)
    radial_centers = 0.5 * (bins[:-1] + bins[1:])

    return radial_centers, radial_profile
