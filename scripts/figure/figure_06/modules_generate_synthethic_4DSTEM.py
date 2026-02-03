#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 13:40:45 2025

@author: kwang
"""
import numpy as np
from scipy.signal import fftconvolve
from scipy.spatial import distance
from scipy.ndimage import binary_dilation
from skimage.transform import AffineTransform, warp
from matplotlib.colors import ListedColormap
import py4DSTEM
import skimage.filters
from skimage.draw import disk
from scipy.ndimage import gaussian_filter


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


def find_gaussian_center_cropped(image, crop_size=8):
    """
    Finds the center of a Gaussian-like signal in a 2D image by cropping the center region.

    Parameters:
        image (2D np.array): The input image.
        crop_size (int): Half-size of the crop around the image center (i.e., total crop is 2*crop_size x 2*crop_size)

    Returns:
        (x_center, y_center): Estimated center coordinates in the original image coordinates (floats).
    """
    H, W = image.shape
    cx, cy = W // 2, H // 2  # initial rough center

    # Define crop bounds
    x_start = max(cx - crop_size, 0)
    x_end = min(cx + crop_size, W)
    y_start = max(cy - crop_size, 0)
    y_end = min(cy + crop_size, H)

    cropped = image[y_start:y_end, x_start:x_end]

    # Compute centroid in cropped coordinates
    y_idx, x_idx = np.indices(cropped.shape)
    total = np.sum(cropped)
    x_c = np.sum(x_idx * cropped) / total
    y_c = np.sum(y_idx * cropped) / total

    # Convert back to original image coordinates
    x_c_full = x_start + x_c
    y_c_full = y_start + y_c

    return x_c_full, y_c_full

def _subtract_dog(frame, min_sigma=1, max_sigma=55):
    """Background removal using difference of Gaussians.
    
    This module is from pyxem repository:
        https://github.com/pyxem/pyxem

    Parameters
    ----------
    frame : NumPy 2D array
    min_sigma : float
    max_sigma : float

    Returns
    -------
    background_removed : Numpy 2D array

    Examples
    --------
    >>> import pyxem.utils._dask as dt
    >>> s = pxm.data.dummy_data.dummy_data.get_cbed_signal()
    >>> s_rem = dt._background_removal_single_frame_dog(s.data[0, 0])

    """
    blur_max = gaussian_filter(frame, max_sigma)
    blur_min = gaussian_filter(frame, min_sigma)
    return np.maximum(np.where(blur_min > blur_max, frame, 0) - blur_max, 0)

def remove_artifacts_at_corner(image, distance_from_corner = 10):
    new_image = np.copy(image)
    for i in range(0, image.shape[0], image.shape[0] - 1):
        for j in range(0, image.shape[1], image.shape[1] - 1):

            for neighbors_of_i in range(i - distance_from_corner, i + distance_from_corner + 1):
                if neighbors_of_i > -1 and neighbors_of_i < image.shape[0]:
                    for neighbors_of_j in range(j - distance_from_corner, j + distance_from_corner + 1):
                        if neighbors_of_j > -1 and neighbors_of_j < image.shape[1]:
                            new_image[neighbors_of_i][neighbors_of_j] = np.min(image)
            
    return new_image

def calculate_direct_beam_kernel_for_correlation(aligned_data,
                                                 correlation_thresh_for_direct_beam = 8e4,
                                                 collection_pixel_nums = 6,
                                                 diffGauss_sigma1 = 2, 
                                                 diffGauss_sigma2 = 8,
                                                 diffraction_pattern_x_size = 128,
                                                 diffraction_pattern_y_size = 128,
                                                 ):
    
    DPImgcenter = int(diffraction_pattern_x_size / 2)
    
    centers = []
    
    for i in range(aligned_data.shape[0]):
        for j in range(aligned_data.shape[1]):
            current_diffPatt = aligned_data[i,j]
            current_center = current_diffPatt[DPImgcenter - collection_pixel_nums:DPImgcenter + collection_pixel_nums, DPImgcenter - collection_pixel_nums:DPImgcenter + collection_pixel_nums]
            centers.append(current_center)
    centers = np.array(centers)
    centers_av = np.average(centers, axis = 0)
    
    correlation_kernel = np.zeros((diffraction_pattern_x_size, diffraction_pattern_y_size))
    correlation_kernel[DPImgcenter - collection_pixel_nums:DPImgcenter + collection_pixel_nums, DPImgcenter - collection_pixel_nums:DPImgcenter + collection_pixel_nums] = centers_av 
    correlation_kernel = np.where(correlation_kernel > correlation_thresh_for_direct_beam, correlation_kernel, 0)
    
    return correlation_kernel

def detect_Bragg_Disks_in_4DSTEM_data(aligned_data,
                                      crystal_unit_cell_file_path = "./",
                                      correlation_thresh_for_templ_match = 10000,
                                      correlation_thresh_for_direct_beam = 8e4,
                                      diffGauss_sigma1 = 2, 
                                      diffGauss_sigma2 = 8,                                      
                                      collection_pixel_nums = 6,
                                      diffraction_pattern_x_size = 128,
                                      diffraction_pattern_y_size = 128,
                                      sampling = 8,
                                      pixel_size_inv_Ang_guess = 0.0327,
                                      q_range = (218, 252),
                                      crystal_name = "Cu_fcc",
                                      ):

    detect_params = {
        'minAbsoluteIntensity': correlation_thresh_for_templ_match,   # intensity threshold
        'minRelativeIntensity': 0.0,   # int. thresh. relative to brightest disk in each pattern
        'minPeakSpacing': 12,         # if two peaks are closer than this (in pixels), remove the dimmer peak
        'edgeBoundary': 6,           # remove peaks within this distance of the edge of the diffraction pattern
        'sigma': 0.1,                  # gaussian blur size to apply to cross correlation before finding maxima
        'maxNumPeaks': 100,          # maximum number of peaks to return, in order of intensity
        'subpixel' : 'poly',         # subpixel resolution method
        'corrPower': 1.0,            # if <1.0, performs a hybrid cross/phase correlation. More sensitive to edges and to noise
        # 'CUDA': True,              # if a GPU is configured and cuda dependencies are installed, speeds up calculation 
    }
    
    crystal_cif_file = crystal_unit_cell_file_path + crystal_name + ".cif"

    
    correlation_kernel = calculate_direct_beam_kernel_for_correlation(aligned_data,
                                                                      correlation_thresh_for_direct_beam,
                                                                      collection_pixel_nums,
                                                                      diffGauss_sigma1,
                                                                      diffGauss_sigma2,
                                                                      diffraction_pattern_x_size,
                                                                      diffraction_pattern_y_size,
    )

    shifted_correlation_kernel = np.fft.fftshift(correlation_kernel)
    customPROBE = shifted_correlation_kernel / np.linalg.norm(shifted_correlation_kernel)

    
    real_scan_space_x_size = aligned_data.shape[0]
    real_scan_space_y_size = aligned_data.shape[1]

    data_after_difference_of_gaussian = []

    for si in range(real_scan_space_x_size):
        r_data_after_difference_of_gaussian = []
        for sj in range(real_scan_space_y_size):
            temp_img_no_nomral = aligned_data[si,sj]
            temp_image_dog = _subtract_dog(temp_img_no_nomral , min_sigma = diffGauss_sigma1, max_sigma = diffGauss_sigma2)
            r_data_after_difference_of_gaussian.append(temp_image_dog)
        data_after_difference_of_gaussian.append(r_data_after_difference_of_gaussian)    
    data_after_difference_of_gaussian = np.array(data_after_difference_of_gaussian)

    datacube_aligned_data = py4DSTEM.DataCube(data_after_difference_of_gaussian)

    dp_mean = datacube_aligned_data.get_dp_mean()
    dp_max = datacube_aligned_data.get_dp_max()

    bragg_peaks = datacube_aligned_data.find_Bragg_disks(
        template = customPROBE,
        **detect_params,
    )
    
    print("Bragg Disk detection complete.\n\n")
    
    print("Now performing calibration using the detected Bragg Disks...\n")
    
    ########################################################################
    #####
    ##### USE PY4DSTEM modules for calibration.
    #####
    ########################################################################
    
    qx0_meas,qy0_meas,mask_meas = bragg_peaks.measure_origin()
    bragg_peaks.calibration.get_origin_meas()
    qx0_fit,qy0_fit,qx0_residuals,qy0_residuals = bragg_peaks.fit_origin()

    # compute
    # bvm = bragg_peaks.histogram(
    #     sampling = sampling,
    # )
    
    # bvm_r = bragg_peaks.histogram( mode='raw', sampling=sampling )
    bvm_c = bragg_peaks.histogram( mode='cal', sampling=sampling )
    
    
    p_ellipse = py4DSTEM.process.calibration.fit_ellipse_1D(
        bvm_c,
        center = bvm_c.origin,
        fitradii = q_range,
    )
    
    bragg_peaks.calibration.set_p_ellipse(p_ellipse)
    bragg_peaks.setcal()
    # bvm_e = bragg_peaks.histogram(
    #     sampling=sampling
    # )
    
    
    k_max = pixel_size_inv_Ang_guess * (diffraction_pattern_x_size / 2)
    
    crystal_fcc = py4DSTEM.process.diffraction.Crystal.from_CIF(crystal_cif_file)
    crystal_fcc.calculate_structure_factors(
        k_max,
    )
        
    # calibrate
    bragg_peaks.calibration.set_Q_pixel_size(pixel_size_inv_Ang_guess)
    bragg_peaks.calibration.set_Q_pixel_units('A^-1')
    bragg_peaks.setcal()
    
    crystal_fcc.calibrate_pixel_size(
        bragg_peaks = bragg_peaks,
        bragg_k_power = 2.0,
        plot_result = True,
    );
    
    bragg_peaks.setcal()
    # bvm_p = bragg_peaks.histogram(
    #     sampling=sampling
    # )
    
    print("\n")
    
    print("calibration complete.\n")
        
    return bragg_peaks




def generate_synthetic_diffraction_pattern(sim_diffrac_patt_img , 
                                           background_of_interest, 
                                           correlation_kernel,
                                           normalized_bgs_avg_for_masking_sim_img = None,
                                           diffraction_pattern_side_dim = 128,
                                           random_seed = 42,
                                           sim_BD_intensity_scaling_factor_mean = 0.065, 
                                           sim_BD_intensity_scaling_factor_std = 0.01,
                                           sim_BD_minimum_intensity_threshold = 1e-9,
                                           conv_img_intensity_scaling_factor_mean = 1.2,
                                           conv_img_intensity_scaling_factor_std = 0.02,
                                           conv_img_intensity_scaling_factor_threshold = 0.9,
                                           background_scaling_factor_mean = 1.0, 
                                           background_scaling_factor_std = 0.08, 
                                           background_minimum_intensity_threshold = 0.05,
                                           diffraction_pattern_boundary_cut_off_range = 4,
                                           simulated_direct_beam_removal_range = 5,
                                           gaussian_blur_sigma = 0.75,
                                          ):

    # np.random.seed(random_seed)  # For reproducibility
    # seeds = np.random.randint(0, size, size=(num_grains, 2))

    # print("np.unravel_index(np.argmax(sim_diffrac_patt_img), sim_diffrac_patt_img.shape)", np.unravel_index(np.argmax(sim_diffrac_patt_img), sim_diffrac_patt_img.shape))

    
    
    # sim_diffrac_patt_img[int(diffraction_pattern_side_dim/2)-3:int(diffraction_pattern_side_dim/2),int(diffraction_pattern_side_dim/2)-3:int(diffraction_pattern_side_dim/2)] = 0.
    # sim_diffrac_patt_img[int(diffraction_pattern_side_dim/2)-3:int(diffraction_pattern_side_dim/2),int(diffraction_pattern_side_dim/2)+1:int(diffraction_pattern_side_dim/2)+4] = 0.
    # sim_diffrac_patt_img[int(diffraction_pattern_side_dim/2)+1:int(diffraction_pattern_side_dim/2)+4,int(diffraction_pattern_side_dim/2)-3:int(diffraction_pattern_side_dim/2)] = 0.
    # sim_diffrac_patt_img[int(diffraction_pattern_side_dim/2)+1:int(diffraction_pattern_side_dim/2)+4,int(diffraction_pattern_side_dim/2)+1:int(diffraction_pattern_side_dim/2)+4] = 0.
    # print("sim_diffrac_patt_img[64,64]", sim_diffrac_patt_img[64,64])
    
    background_scaling_factor = np.random.normal(background_scaling_factor_mean, background_scaling_factor_std)
    background_scaling_factor = np.where(background_scaling_factor < background_minimum_intensity_threshold, background_minimum_intensity_threshold, background_scaling_factor)
    sim_BDs_scaling_factor = np.random.normal(sim_BD_intensity_scaling_factor_mean, sim_BD_intensity_scaling_factor_std, sim_diffrac_patt_img.shape)
    sim_BDs_scaling_factor = np.where(sim_BDs_scaling_factor < sim_BD_minimum_intensity_threshold, sim_BD_minimum_intensity_threshold, sim_BDs_scaling_factor)

    # print("background_scaling_factor", background_scaling_factor)

    # fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    # ax.set_title("sim_diffrac_patt_img")
    # im = ax.imshow(sim_diffrac_patt_img,  norm = 'log', cmap = 'gray')
    # ax.set_xlabel(r"$q_y$", fontsize = 16)
    # ax.set_ylabel(r"$q_x$", fontsize = 16)
    # plt.colorbar(im, ax = ax, fraction=0.0468, pad=0.02)
    # plt.show()

    # Find location of direct beam after convolution    
    convResult = fftconvolve(sim_diffrac_patt_img, correlation_kernel, mode='same')
    center_x_convResult, center_y_convResult = np.unravel_index(np.argmax(convResult), convResult.shape)

    sim_diffrac_patt_img = sim_diffrac_patt_img * sim_BDs_scaling_factor
    convResult = fftconvolve(sim_diffrac_patt_img, correlation_kernel, mode='same')
    # print("center_x_convResult, center_y_convResult", center_x_convResult, center_y_convResult)
    #translate_convolved_Simulated_Bragg_disks
    # translate_x_simBD = int(int(diffraction_pattern_side_dim / 2) - center_x_convResult)
    # translate_y_simBD = int(int(diffraction_pattern_side_dim / 2) - center_y_convResult)

    # print("translate_x_simBD", translate_x_simBD)
    # print("translate_y_simBD", translate_y_simBD)
    
    # result = correlate(sim_diffrac_patt_img, correlation_kernel, method = 'fft')
    # result = result[int(result.shape[0]/4):int(result.shape[0]*3/4), int(result.shape[1]/4):int(result.shape[1]*3/4)]
    
    # tform_simBD = AffineTransform(translation=(translate_x_simBD, translate_y_simBD))
    # convResult = warp(convResult, tform_simBD.inverse) ############### KKWANG
    # convResult = convResult

    # fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    # ax.set_title("convResult")
    # im = ax.imshow(convResult,  norm = 'log', cmap = 'gray')
    # ax.set_xlabel(r"$q_y$", fontsize = 16)
    # ax.set_ylabel(r"$q_x$", fontsize = 16)
    # plt.colorbar(im, ax = ax, fraction=0.0468, pad=0.02)
    # plt.show()
    
    
    convResult_intensity_max_x, convResult_intensity_max_y = np.unravel_index(np.argmax(convResult), convResult.shape)
    
    # print("convResult_intensity_max_x, convResult_intensity_max_y", convResult_intensity_max_x, convResult_intensity_max_y)
    
    convResult_center_x, convResult_center_y = int(convResult.shape[0] / 2), int(convResult.shape[1] / 2)
    convResult_after_removing_direct_beam = np.copy(convResult)
    convResult_after_removing_direct_beam[int(convResult_center_x - simulated_direct_beam_removal_range):int(convResult_center_x + simulated_direct_beam_removal_range), int(convResult_center_y - simulated_direct_beam_removal_range):int(convResult_center_y + simulated_direct_beam_removal_range)] = 0.0


    trans_conv_Img_scaling_factor = np.random.normal(conv_img_intensity_scaling_factor_mean, conv_img_intensity_scaling_factor_std, convResult_after_removing_direct_beam.shape)
    trans_conv_Img_scaling_factor = np.where(trans_conv_Img_scaling_factor < conv_img_intensity_scaling_factor_threshold, conv_img_intensity_scaling_factor_threshold, trans_conv_Img_scaling_factor)
    convResult_after_removing_direct_beam = convResult_after_removing_direct_beam * trans_conv_Img_scaling_factor
    if normalized_bgs_avg_for_masking_sim_img is not None:
        convResult_after_removing_direct_beam = convResult_after_removing_direct_beam * normalized_bgs_avg_for_masking_sim_img 

    convResult_after_removing_direct_beam = skimage.filters.gaussian(convResult_after_removing_direct_beam, sigma=gaussian_blur_sigma, preserve_range=True)

    

    # fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    # ax.set_title("convResult_after_removing_direct_beam")
    # im = ax.imshow(convResult_after_removing_direct_beam,  norm = 'log', cmap = 'gray')
    # ax.set_xlabel(r"$q_y$", fontsize = 16)
    # ax.set_ylabel(r"$q_x$", fontsize = 16)
    # plt.colorbar(im, ax = ax, fraction=0.0468, pad=0.02)
    # plt.show()
    
    ######
    
    background_intensity_max_x, background_intensity_max_y = find_gaussian_center_cropped(background_of_interest)
    
    translate_x_BG = int(diffraction_pattern_side_dim / 2) - background_intensity_max_x
    translate_y_BG = int(diffraction_pattern_side_dim / 2) - background_intensity_max_y
    tform_BG = AffineTransform(translation=(translate_x_BG, translate_y_BG))
    translated_BG = warp(background_of_interest, tform_BG.inverse)
    
    
    synthetic_diffraction_pattern = translated_BG * background_scaling_factor + convResult_after_removing_direct_beam
    synthetic_diffraction_pattern = synthetic_diffraction_pattern[diffraction_pattern_boundary_cut_off_range:int(synthetic_diffraction_pattern.shape[0] - diffraction_pattern_boundary_cut_off_range), diffraction_pattern_boundary_cut_off_range:int(synthetic_diffraction_pattern.shape[1] - diffraction_pattern_boundary_cut_off_range)]


    return synthetic_diffraction_pattern

def process_background_diffraction_pattern( 
                                            background_of_interest, 
                                            diffraction_pattern_side_dim = 128,
                                            random_seed = 42,
                                            background_scaling_factor_mean = 0.9, 
                                            background_scaling_factor_std = 0.05,
                                            direct_beam_scaling_factor = 4.0,
                                            direct_beam_radius = 4,
                                            background_minimum_intensity_threshold = 0.8,
                                            diffraction_pattern_boundary_cut_off_range = 4,
                                            simulated_direct_beam_removal_range = 5):

    # np.random.seed(random_seed)  # For reproducibility
    # seeds = np.random.randint(0, size, size=(num_grains, 2))

    # print("np.unravel_index(np.argmax(sim_diffrac_patt_img), sim_diffrac_patt_img.shape)", np.unravel_index(np.argmax(sim_diffrac_patt_img), sim_diffrac_patt_img.shape))

    
    
    # sim_diffrac_patt_img[int(diffraction_pattern_side_dim/2)-3:int(diffraction_pattern_side_dim/2),int(diffraction_pattern_side_dim/2)-3:int(diffraction_pattern_side_dim/2)] = 0.
    # sim_diffrac_patt_img[int(diffraction_pattern_side_dim/2)-3:int(diffraction_pattern_side_dim/2),int(diffraction_pattern_side_dim/2)+1:int(diffraction_pattern_side_dim/2)+4] = 0.
    # sim_diffrac_patt_img[int(diffraction_pattern_side_dim/2)+1:int(diffraction_pattern_side_dim/2)+4,int(diffraction_pattern_side_dim/2)-3:int(diffraction_pattern_side_dim/2)] = 0.
    # sim_diffrac_patt_img[int(diffraction_pattern_side_dim/2)+1:int(diffraction_pattern_side_dim/2)+4,int(diffraction_pattern_side_dim/2)+1:int(diffraction_pattern_side_dim/2)+4] = 0.
    # print("sim_diffrac_patt_img[64,64]", sim_diffrac_patt_img[64,64])
    
    background_scaling_factor = np.random.normal(background_scaling_factor_mean, background_scaling_factor_std)
    background_scaling_factor = np.where(background_scaling_factor < background_minimum_intensity_threshold, background_minimum_intensity_threshold, background_scaling_factor)
    
    background_intensity_max_x, background_intensity_max_y = find_gaussian_center_cropped(background_of_interest)
    
    translate_x_BG = int(diffraction_pattern_side_dim / 2) - background_intensity_max_x
    translate_y_BG = int(diffraction_pattern_side_dim / 2) - background_intensity_max_y
    tform_BG = AffineTransform(translation=(translate_x_BG, translate_y_BG))
    translated_BG = warp(background_of_interest, tform_BG.inverse)

    direct_beam_scale_mask = np.ones_like(translated_BG, dtype = np.float32)
    DPImgcenter = int(translated_BG.shape[0] / 2)
    rr, cc = disk((DPImgcenter, DPImgcenter), direct_beam_radius, shape=(direct_beam_scale_mask.shape[0], direct_beam_scale_mask.shape[1]))
    direct_beam_scale_mask[rr, cc] =  direct_beam_scaling_factor


    ###
    H, W = direct_beam_scale_mask.shape
    y, x = np.indices((H, W))
    r = np.sqrt((x - DPImgcenter)**2 + (y - DPImgcenter)**2)

    sigma = direct_beam_radius / 2  # adjust as needed
    direct_beam_scale_mask = 1 + (direct_beam_scaling_factor - 1) * np.exp(-(r**2) / (2 * sigma**2))
    ###
    
    
    synthetic_diffraction_pattern = translated_BG * background_scaling_factor
    synthetic_diffraction_pattern = synthetic_diffraction_pattern * direct_beam_scale_mask
    synthetic_diffraction_pattern = synthetic_diffraction_pattern[diffraction_pattern_boundary_cut_off_range:int(synthetic_diffraction_pattern.shape[0] - diffraction_pattern_boundary_cut_off_range), diffraction_pattern_boundary_cut_off_range:int(synthetic_diffraction_pattern.shape[1] - diffraction_pattern_boundary_cut_off_range)]


    return synthetic_diffraction_pattern

def generate_2D_IMG_array(qx,
                          qy,
                          intensity,
                          k_max,
                          pixel_numbers,
                          corner_centered = False,
                          remove_direct_beam = False,
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

    qx_refined = qx
    qy_refined = qy
    intensity_refined = intensity
    mask = None

    if remove_direct_beam:            
        mask = np.full((len(intensity)), True)
        index_of_direct_beam = np.argmax(intensity)
        mask[index_of_direct_beam] = False
        qx_refined = qx_refined[mask]
        qy_refined = qy_refined[mask]
        intensity_refined = intensity_refined[mask]
        
    qx_refined[np.where(np.abs(qx_refined[:]) < (1e-14))[0]] = 0.0
    qy_refined[np.where(np.abs(qy_refined[:]) < (1e-14))[0]] = 0.0

    stack_qxqy = np.stack((qx_refined, qy_refined), axis = 1)
    width_of_diff2D_img = k_max * 2.0
    pixel_size = width_of_diff2D_img / pixel_numbers

    if pixel_numbers % 2 == 0:
        pixel_bins = np.arange(-k_max, k_max + pixel_size, pixel_size) # change
        pixel_bins[np.where(np.abs(pixel_bins[:]) < (1e-14))[0]] = 0.0 # change
        
        
        digitized_bin = np.digitize(stack_qxqy, pixel_bins) - 1
    else:
        pixel_bins = np.linspace(-k_max - pixel_size/2.0, k_max + pixel_size/2.0, pixel_numbers + 1, endpoint = True)
        digitized_bin = np.digitize(stack_qxqy, pixel_bins) - 1


    diff2D_img = np.zeros((pixel_numbers, pixel_numbers))
    diff2D_img[digitized_bin[:,0],digitized_bin[:,1]] = intensity_refined

    if corner_centered:
        diff2D_img = np.roll(diff2D_img, (-int(pixel_numbers / 2), -int(pixel_numbers / 2)), axis=(0, 1))

    return diff2D_img

def generate_grains(size=256, num_grains=60, boundary_thickness=3, random_seed = 42):
    # 1. Generate random seed points
    np.random.seed(random_seed)  # For reproducibility
    seeds = np.random.randint(0, size, size=(num_grains, 2))

    # 2. Create a grid of coordinates
    xv, yv = np.meshgrid(np.arange(size), np.arange(size))
    grid_points = np.stack([xv, yv], axis=-1).reshape(-1, 2)

    # 3. Assign each pixel to the nearest seed (Voronoi)
    dist = distance.cdist(grid_points, seeds)
    voronoi_labels = np.argmin(dist, axis=1) + 1  # Labels: 1 to num_grains
    label_image = voronoi_labels.reshape((size, size))

    # 4. Detect boundaries
    boundary_mask = np.zeros_like(label_image, dtype=bool)
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
        shifted = np.roll(label_image, shift=(dy, dx), axis=(0, 1))
        boundary_mask |= shifted != label_image

    # 5. Thicken the boundary
    for _ in range(boundary_thickness - 1):
        boundary_mask = binary_dilation(boundary_mask)

    # 6. Set boundary pixels to 0 (background)
    label_image[boundary_mask] = 0

    return label_image






def generate_random_colormap(num_colors=20, background_color=(0, 0, 0)):
    """
    Generate a matplotlib ListedColormap with `num_colors` random colors,
    and optionally add a background color at index 0.
    
    Parameters:
        num_colors (int): Number of random colors (for grains).
        background_color (tuple): RGB tuple for background (index 0).
    
    Returns:
        ListedColormap: Colormap object for use in matplotlib.
    """
    rng = np.random.default_rng(seed=12)  # For reproducibility
    random_colors = rng.uniform(0.1, 0.9, size=(num_colors, 3))  # Avoid too dark/light

    # Prepend background color at index 0
    all_colors = np.vstack([background_color, random_colors])
    
    return ListedColormap(all_colors)



def sample_diffraction_patterns_from_rotation_matrices(crystal, 
                                                       rotation_matrices,
                                                       accelerating_voltage,
                                                       k_max,
                                                       thickness_input,
                                                       random_seed = 50,
                                                       thickness_num_for_sampling = 500,
                                                       max_sequence_length = 106,
                                                       ):
    crystal.setup_diffraction(accelerating_voltage)
    crystal.calculate_structure_factors(
        k_max,
    )
    
    np.random.seed(random_seed)
    
    # Convert the V_g to relativistic-corrected U_g and store in a datastructure optimized
    # for access by the Bloch code
    crystal.calculate_dynamical_structure_factors(
        300e3, "WK-CP", k_max=k_max * 2., thermal_sigma=0.08, tol_structure_factor=-1.0
    )
    
    
    thickness_lower_limit = crystal.lat_real[0][0] * 2
    thickness_upper_limit = crystal.lat_real[0][0] * 900
    
    
    k_max_radial = 2.98
    
    
    DPs_collection = []
    DP_Bragg_Disk_table_cartesian = []
    labels_collection = []
    thickness_sampled = []
    

    for RM_idx, orientation_matrix in enumerate(rotation_matrices):
        # thicknesses = np.linspace(thickness_lower_limit, thickness_upper_limit, thickness_num_for_sampling)
        
        
        thicknesses = np.array([thickness_input, thickness_input])
    
        beams = crystal.generate_diffraction_pattern(
                                    orientation_matrix = orientation_matrix,
                                    sigma_excitation_error = 0.04,
                                    tol_intensity = 0.0,
                                    k_max = k_max,
        )
    
        dynamic_patterns = crystal.generate_dynamical_diffraction_pattern(
                            beams = beams,
                            orientation_matrix = orientation_matrix,
                            thickness=thicknesses,
                        )
    
        
    
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
    
            # print("len(qx)", len(qx))
    
            # collection_together.append(np.stack((qx, qy, intensity)).T)
            if len(qx) > 1:
                positions_of_Bragg_disks = np.stack((qx, qy)).T
                k_radial_distnaces_of_BPs = np.linalg.norm(positions_of_Bragg_disks, axis = 1)
    
                indices_where_cartesian_is_smaller_than_k_max_square = np.intersect1d(np.where(np.abs(positions_of_Bragg_disks[:,0]) < k_max)[0], np.where(np.abs(positions_of_Bragg_disks[:,1]) < k_max)[0])
                indices_where_radial_distance_smaller_than_k_max = np.where(k_radial_distnaces_of_BPs < k_max_radial)[0]
                indices_where_radial_distance_smaller_than_k_max = np.intersect1d(indices_where_radial_distance_smaller_than_k_max, indices_where_cartesian_is_smaller_than_k_max_square)
    
                if len(indices_where_radial_distance_smaller_than_k_max) > 1:
                    
                    final_qx = qx[indices_where_radial_distance_smaller_than_k_max]
                    final_qy = qy[indices_where_radial_distance_smaller_than_k_max]
                    final_intensities_of_Bragg_disks = intensities_of_Bragg_disks[indices_where_radial_distance_smaller_than_k_max]
                    
                    BD_cart = np.stack((final_qx, final_qy, final_intensities_of_Bragg_disks / np.max(final_intensities_of_Bragg_disks))).T



                    numbers_to_pad = max_sequence_length - BD_cart.shape[0]
                    for numStack in range(numbers_to_pad):
                        BD_cart = np.vstack((BD_cart, np.array([[0.0, 0.0, 0.0]])))
                    DP_Bragg_Disk_table_cartesian.append(BD_cart)

                    # print("len(indices_where_radial_distance_smaller_than_k_max)", len(indices_where_radial_distance_smaller_than_k_max))
                    # print("orientation_matrix\n", orientation_matrix)
                    
    

                    
                    DP = generate_2D_IMG_array(final_qx,
                                      final_qy,
                                      final_intensities_of_Bragg_disks,
                                      k_max = k_max,
                                      pixel_numbers = 128)
                    DP[64,64] = 10
                    DPs_collection.append(DP)
                    labels_collection.append(orientation_matrix)
                    thickness_sampled.append(thicknesses[enIdx])
                    break
    
    # labels_collection = np.array(labels_collection)
    return DPs_collection, labels_collection, thickness_sampled, np.array(DP_Bragg_Disk_table_cartesian)