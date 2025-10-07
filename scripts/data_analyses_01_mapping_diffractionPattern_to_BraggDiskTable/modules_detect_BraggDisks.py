#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 13:40:45 2025

@author: kwang
"""
import numpy as np
import py4DSTEM
from scipy.ndimage import gaussian_filter




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
    
    bragg_vector_map_centered = bragg_peaks.get_bvm()
    bvm_c = bragg_vector_map_centered
    qx0_meas,qy0_meas,mask_meas = bragg_peaks.measure_origin()
    bragg_peaks.calibration.get_origin_meas()
    qx0_fit,qy0_fit,qx0_residuals,qy0_residuals = bragg_peaks.fit_origin()

    # compute
    bvm = bragg_peaks.histogram(
        sampling = sampling,
    )
    
    bvm_r = bragg_peaks.histogram( mode='raw', sampling=sampling )
    bvm_c = bragg_peaks.histogram( mode='cal', sampling=sampling )
    
    
    p_ellipse = py4DSTEM.process.calibration.fit_ellipse_1D(
        bvm_c,
        center = bvm_c.origin,
        fitradii = q_range,
    )
    bragg_peaks.calibration.set_p_ellipse(p_ellipse)
    bragg_peaks.setcal()
    bvm_e = bragg_peaks.histogram(
        sampling=sampling
    )
    

    q, intensity_radial = py4DSTEM.process.utils.radial_integral(
        bragg_vector_map_centered,
    )
    
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
    bvm_p = bragg_peaks.histogram(
        sampling=sampling
    )
    
    print("\n")
    
    print("calibration complete.\n")
        
    return bragg_peaks


