import numpy as np
import torch

from orientationMapping.LossFunctions import symmetric_orthogonalization

def digitize_radial_distance(radial_distances, radial_bins):
    return np.digitize(radial_distances, radial_bins) - 1

def digitize_polarAngle(polar_angles, angle_bins):
    return np.digitize(polar_angles, angle_bins) - 1

def digitize_braggIntensity(braggDisk_intensities, intensity_bins):
    return np.digitize(braggDisk_intensities, intensity_bins) - 1

def process_pandas_tabular_data(
                                df, 
                                num_bins_radialDistance, 
                                num_bins_polarAngle, 
                                num_bins_braggintensity, 
                                max_sequence_length,
                                min_radial_distance,
                                max_radial_distance,
                                min_braggIntensity,
                                max_braggIntensity):
    
    
    radial_bins = np.linspace(0.0, max_radial_distance + (min_radial_distance*0.05), num_bins_radialDistance + 1)
    radial_bin_centers = (radial_bins[:-1] + radial_bins[1:]) / 2
    
    angle_bins = np.arange(-np.pi - np.pi/360., np.pi + np.pi/360., np.pi/180.)
    angle_bin_centers = (angle_bins[:-1] + angle_bins[1:]) / 2
    angle_bins[-1] = np.pi + np.pi/360 # further change the last element
    
    intensity_bins = np.linspace(0.0, max_braggIntensity + (min_braggIntensity*0.05), num_bins_braggintensity + 1)
    intensity_bin_centers = (intensity_bins[:-1] + intensity_bins[1:]) / 2
    
        
    list_of_Bragg_disks_total = []

    # max_r = []
    # min_r = []
    for idx, diffractionPattern in df.items():
        np_diffractionPattern = np.array(diffractionPattern['input'])
        # max_r.append(np.max(np_diffractionPattern[:, 0]))
        # min_r.append(np.min(np_diffractionPattern[:, 0]))
        np_diffractionPattern[:, 0] = digitize_radial_distance(np_diffractionPattern[:,0], radial_bins)        
        np_diffractionPattern[:, 1] = digitize_polarAngle(np_diffractionPattern[:,1], angle_bins)
        np_diffractionPattern[:, 2] = digitize_braggIntensity(np_diffractionPattern[:,2], intensity_bins)
        np_diffractionPattern = np_diffractionPattern.astype(np.int32)   

            
        if idx == 0:
            if len(diffractionPattern['input']) < max_sequence_length:
                numbers_of_pad_tokens_to_add = max_sequence_length - len(diffractionPattern['input'])
                for recur in range(numbers_of_pad_tokens_to_add):
                    np_diffractionPattern = np.vstack((np_diffractionPattern, np.array([[0, 0, 0]])))

        # print("np_diffractionPattern after\n", np_diffractionPattern)
        
    
        list_of_Bragg_disks_total.append(torch.tensor(np_diffractionPattern))
    # max_r = np.array(max_r)
    # min_r = np.array(min_r)    
    
    radial_bins = torch.tensor(radial_bins, dtype = torch.float32)
    radial_bin_centers = torch.tensor(radial_bin_centers, dtype = torch.float32)
    
    angle_bins = torch.tensor(angle_bins, dtype = torch.float32)
    angle_bin_centers = torch.tensor(angle_bin_centers, dtype = torch.float32)
    
    intensity_bins = torch.tensor(intensity_bins, dtype = torch.float32)
    intensity_bin_centers = torch.tensor(intensity_bin_centers, dtype = torch.float32)
    
    return list_of_Bragg_disks_total, radial_bins, radial_bin_centers, angle_bins, angle_bin_centers, intensity_bins, intensity_bin_centers


def map_exp_BraggDisk_to_orienation_via_Transformer(model, exp_loader, device, PAD = 0):
    for name, loader in [("experimental", exp_loader)]:

        rotation_matrices_including_mirror = []
        
        with torch.no_grad():
            model.eval()
            for features in loader:
                features = features.to(device = device)
                pad_mask = (torch.sum(features, dim = 2) == PAD).view(features.size(0), 1, 1, features.size(1))
                pred = model(features, pad_mask)
                predicted_rotation_matrix = symmetric_orthogonalization(pred[0])


                probabilities = torch.sigmoid(pred[1])  # Apply sigmoid to get probabilities
                predicted_labels = (probabilities >= 0.5).float()  # or .int() if you want integers
                indices_of_mirr = torch.where(predicted_labels > 0.0)[0]

                predicted_rotation_matrix[indices_of_mirr,:,1:] = predicted_rotation_matrix[indices_of_mirr,:,1:] * (-1.0)

                rotation_matrices_including_mirror.append(predicted_rotation_matrix)
            
            model.train(True)
    return torch.vstack(rotation_matrices_including_mirror)

def pre_process_experimental_BraggDisk(bragg_peaks):
    scan_x_dim, scan_y_dim = bragg_peaks.Rshape
    
    table_of_BraggDisk_qx_qy_intensity_for_eachScanIndex = {}
    dict_idx = 0
    for i in range(scan_x_dim):
        for j in range(scan_y_dim):
            if len(bragg_peaks.cal[i,j].qx) > 2:

                qx = np.copy(bragg_peaks.cal[i,j].data["qx"])
                qy = np.copy(bragg_peaks.cal[i,j].data["qy"])
                intensity = np.copy(bragg_peaks.cal[i,j].data["intensity"])

                k_radial_distnaces_of_BPs = np.linalg.norm(np.stack((qx, qy)).T, axis = 1)
                index_of_direct_beam = np.argmin(k_radial_distnaces_of_BPs)

                qx = np.delete(qx, index_of_direct_beam)
                qy = np.delete(qy, index_of_direct_beam)
                intensity = np.delete(intensity, index_of_direct_beam)

                positions_of_Bragg_disks = np.stack((qx, qy)).T
                k_radial_distnaces_of_BPs = np.linalg.norm(positions_of_Bragg_disks, axis = 1)
                polar_angles = np.arctan2(positions_of_Bragg_disks[:,1], positions_of_Bragg_disks[:,0])

                table_of_BraggDisk_qx_qy_intensity_for_eachScanIndex[dict_idx] = {'input': np.stack((k_radial_distnaces_of_BPs, polar_angles, intensity/np.max(intensity))).T, 'scanIndices': [i,j]}
                    
                dict_idx += 1
    return table_of_BraggDisk_qx_qy_intensity_for_eachScanIndex