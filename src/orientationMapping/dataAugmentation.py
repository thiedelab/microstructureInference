#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 08:20:54 2025

@author: kwang
"""
import torch
from torchvision.transforms import v2
from typing import Any, Callable, List, Optional, Union, Tuple, Dict

def find_opposite_pairs_merged(points, r_tol=1, theta_tol=0.00001 + torch.pi / 180):
    """
    Find pairs of polar-coordinate points with same radius and angle difference ~ pi.
    Fully GPU-friendly (no Python loops over all N).
    
    Args:
        points: (N, 2) tensor [r, theta], theta in radians
        r_tol: tolerance for radius equality (for discretization)
        theta_tol: tolerance for pi angle difference
    
    Returns:
        pairs: list of (i, j) tuples (indices in original `points`)
        unmatched: list of indices with no partner
    """
    # print("points\n", points)
    r, theta = points[:, 0], (points[:, 1] * (torch.pi / 180.0))
    # print("r\n", r)
    # print("theta\n", theta)

    # --- bucket by radius (discretized) ---
    r_key = r
    unique_r = torch.unique(r_key)

    pairs = []
    unmatched = []

    for ur in unique_r:
        idx = (r_key == ur).nonzero(as_tuple=False).flatten()
        if idx.numel() < 2:  # nothing to pair
            unmatched.extend(idx.tolist())
            continue

        th = theta[idx]
        # compute opposite angles
        target = (th + torch.pi) % (2*torch.pi)

        # pairwise angle diffs
        diff = (th[:, None] - target[None, :]).abs()
        diff = torch.min(diff, 2*torch.pi - diff)

        # mask valid matches
        valid = (diff < theta_tol) & (~torch.eye(len(idx), device=points.device, dtype=torch.bool))

        # Greedy matching
        matched = torch.zeros(len(idx), dtype=torch.bool, device=points.device)
        for i in range(len(idx)):
            if matched[i]:
                continue
            cand = torch.nonzero(valid[i] & ~matched, as_tuple=False).flatten()
            if cand.numel() > 0:
                j = cand[0].item()
                pairs.append((idx[i].item(), idx[j].item()))
                matched[i] = True
                matched[j] = True
        # leftovers
        for k in range(len(idx)):
            if not matched[k]:
                unmatched.append(idx[k].item())

    return pairs, unmatched

class applyMirrorLabels(v2.Transform):

    def __init__(self):
        super().__init__()

    def mirrorLabel(
                            self,
                            labels:torch.Tensor,
                            ) -> torch.Tensor:

        new_labels = labels.clone().detach()
        new_labels = new_labels + 1

        return new_labels

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self.mirrorLabel(inpt)

class applyMirrorOperation(v2.Transform):

    def __init__(self, feature_angleBins = 360, feature_axis = 1):
        super().__init__()
        self.feature_angleBins = torch.tensor(feature_angleBins, dtype = torch.long)
        self.feature_angle_cutoff = torch.tensor(feature_angleBins / 2, dtype = torch.long)
        self.feature_axis = feature_axis

    def apply_mirror(
                        self, BraggDiskList:torch.Tensor, ) -> torch.Tensor :

        idx_of_BraggDisks = torch.where(torch.sum(BraggDiskList, dim = 1) > 0)[0]
        BraggDiskList_after_flipping = BraggDiskList.clone().detach()

        # print("type(BraggDiskList_after_flipping)", type(BraggDiskList_after_flipping))

        indices_with_positive_angles = torch.where(BraggDiskList_after_flipping[idx_of_BraggDisks, self.feature_axis] > self.feature_angle_cutoff)[0]
        indices_with_negative_angles = torch.where(BraggDiskList_after_flipping[idx_of_BraggDisks, self.feature_axis] < (self.feature_angle_cutoff + 1))[0]

        BraggDiskList_after_flipping[idx_of_BraggDisks[indices_with_positive_angles], self.feature_axis] = (self.feature_angle_cutoff + self.feature_angleBins) - BraggDiskList_after_flipping[idx_of_BraggDisks[indices_with_positive_angles], self.feature_axis]
        BraggDiskList_after_flipping[idx_of_BraggDisks[indices_with_negative_angles], self.feature_axis] = self.feature_angle_cutoff - BraggDiskList_after_flipping[idx_of_BraggDisks[indices_with_negative_angles], self.feature_axis]

        return BraggDiskList_after_flipping

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self.apply_mirror(inpt)


class RandomRemoval(v2.Transform):

    def __init__(self, fractionToRemove = 0.4):
        super().__init__()
        self.fractionToRemove = fractionToRemove
    
    def removeBraggDisks(
                        self, BraggDiskList:torch.Tensor, ) -> torch.Tensor :
    
        idx_of_BraggDisks = torch.where(torch.sum(BraggDiskList, dim = 1) > 0)[0]
        numberOfBraggDisks_for_removal = int(len(idx_of_BraggDisks) * self.fractionToRemove)
        
        if numberOfBraggDisks_for_removal > 1 and int(len(idx_of_BraggDisks) * (1.0 - self.fractionToRemove)) > 1:
            permuted_indices = torch.randperm(len(idx_of_BraggDisks))
            BraggDiskList_after_removal = BraggDiskList.clone().detach()
            BraggDiskList_after_removal[idx_of_BraggDisks[permuted_indices][:numberOfBraggDisks_for_removal]] = torch.tensor([0, 0, 0])
            
            idx_of_BraggDisks_after_removal = torch.where(torch.sum(BraggDiskList_after_removal, dim = 1) > 0)[0]

            pairs, unmatched = find_opposite_pairs_merged(BraggDiskList_after_removal[idx_of_BraggDisks_after_removal])
            
            if len(unmatched) == 0:
                return BraggDiskList
            else:
                return BraggDiskList_after_removal
        else:
            return BraggDiskList
            
    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self.removeBraggDisks(inpt)

class RemoveBraggDisksWithWeakInten(v2.Transform):

    def __init__(self, fractionToRemove = 0.4):
        super().__init__()
        self.fractionToRemove = fractionToRemove
    
    def removeWeakIntBraggDisks(
                        self, BraggDiskList:torch.Tensor, ) -> torch.Tensor :
    
        idx_of_BraggDisks = torch.where(torch.sum(BraggDiskList, dim = 1) > 0)[0]
        numberOfBraggDisks_for_removal = int(len(idx_of_BraggDisks) * self.fractionToRemove)
        
        if numberOfBraggDisks_for_removal > 1 and int(len(idx_of_BraggDisks) * (1.0 - self.fractionToRemove)) > 1:

            val_of_BD_intensity_ascending_order, indices_of_BD_intensity_ascending_order = torch.sort(BraggDiskList[idx_of_BraggDisks][:,2], descending=False)
            indices_of_sorted_tensor_whose_element_is_zero = torch.where(val_of_BD_intensity_ascending_order == 0)[0]
            permuted_indices = torch.randperm(len(indices_of_sorted_tensor_whose_element_is_zero))
            permuted_indices_of_sorted_tensor_whose_element_is_zero = indices_of_sorted_tensor_whose_element_is_zero[permuted_indices]
            
            indices_of_BD_intensity_ascending_order[indices_of_sorted_tensor_whose_element_is_zero] = indices_of_BD_intensity_ascending_order[permuted_indices_of_sorted_tensor_whose_element_is_zero]
            
            BraggDiskList_after_removal = BraggDiskList.clone().detach()
            BraggDiskList_after_removal[idx_of_BraggDisks[indices_of_BD_intensity_ascending_order][:numberOfBraggDisks_for_removal]] = torch.tensor([0, 0, 0])
            
            idx_of_BraggDisks_after_removal = torch.where(torch.sum(BraggDiskList_after_removal, dim = 1) > 0)[0]
            pairs, unmatched = find_opposite_pairs_merged(BraggDiskList_after_removal[idx_of_BraggDisks_after_removal])
            
            if len(unmatched) == 0:
                return BraggDiskList
            else:
                return BraggDiskList_after_removal
        else:
            return BraggDiskList
            
    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self.removeWeakIntBraggDisks(inpt)


class uniformIntensityScaling_gaussianNoise(v2.Transform):

    def __init__(self, feature_maxBinIdx = 256, mean = 1, std = 0.3):
        super().__init__()
        self.feature_maxBinIdx = feature_maxBinIdx
        self.mean = mean
        self.std = std
    
    def uniformScale(
                        self, BraggDiskList:torch.Tensor, ) -> torch.Tensor :
    
        BraggDiskList_after_scaling = BraggDiskList.clone().detach()
        
        # print("uniform before\n", BraggDiskList_after_scaling)

        uniform_scaling_factor = torch.normal(self.mean, self.std, (1,))

        BraggDiskList_after_scaling[:,2] = torch.clamp(torch.round(BraggDiskList[:,2] * uniform_scaling_factor).type(torch.int64), min = 0, max = self.feature_maxBinIdx - 1)
        # print("uniform after", BraggDiskList_after_scaling)
        
        return BraggDiskList_after_scaling
            
    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self.uniformScale(inpt)

class normalize_intensity(v2.Transform):

    def __init__(self, feature_maxBinIdx = 256):
        super().__init__()
        self.feature_maxBinIdx = feature_maxBinIdx
    
    def normalizeInt(
                        self, BraggDiskList:torch.Tensor, ) -> torch.Tensor :
    
        BraggDiskList_after_normalization = BraggDiskList.clone().detach()

        BraggDiskList_after_normalization[:,2] = torch.clamp(torch.round((self.feature_maxBinIdx - 1) * BraggDiskList[:,2] / torch.max(BraggDiskList[:,2])).type(torch.int64), min = 0, max = self.feature_maxBinIdx - 1)
        
        return BraggDiskList_after_normalization
            
    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self.normalizeInt(inpt)

class individualIntensityScaling_gaussianNoise(v2.Transform):

    def __init__(self, feature_maxBinIdx = 256, mean = 1, std = 0.3):
        super().__init__()
        self.feature_maxBinIdx = feature_maxBinIdx
        self.mean = mean
        self.std = std
    
    def individualScale(
                        self, BraggDiskList:torch.Tensor, ) -> torch.Tensor :
    
        idx_of_BraggDisks = torch.where(torch.sum(BraggDiskList, dim = 1) > 0)[0]
        BraggDiskList_after_scaling = BraggDiskList.clone().detach()
        
        # print("individual before\n", BraggDiskList_after_scaling)

        individual_scaling_factor = torch.normal(self.mean, self.std, (len(idx_of_BraggDisks),))

        BraggDiskList_after_scaling[idx_of_BraggDisks,2] = torch.clamp(torch.round(BraggDiskList[idx_of_BraggDisks,2] * individual_scaling_factor).type(torch.int64), min = 0, max = self.feature_maxBinIdx - 1)
        # print("individual after\n", BraggDiskList_after_scaling)
        return BraggDiskList_after_scaling
            
    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self.individualScale(inpt)

class displaceFeature_GaussianNoise_polar(v2.Transform):

    def __init__(self, feature_axis, feature_maxBinIdx, mean = 0.0, std = 2, fractionToAddNoise = 0.8):
        super().__init__()
        self.mean = mean
        self.std = std
        self.fractionToAddNoise = fractionToAddNoise
        self.feature_axis = feature_axis
        self.feature_maxBinIdx = feature_maxBinIdx

    def add_gaussian_noise(
                        self, BraggDiskList:torch.Tensor, ) -> torch.Tensor :

        idx_of_BraggDisks = torch.where(torch.sum(BraggDiskList, dim = 1) > 0)[0]
        numberOfBraggDisks_to_contaminate = int(len(idx_of_BraggDisks) * self.fractionToAddNoise)

        if numberOfBraggDisks_to_contaminate > 1:
            permuted_indices = torch.randperm(len(idx_of_BraggDisks))
            indices_of_BraggDisks_permuted = idx_of_BraggDisks[permuted_indices]
            BraggDiskList_after_contamination = BraggDiskList.clone().detach()
            gaussian_noise = torch.zeros((numberOfBraggDisks_to_contaminate, 3), dtype = torch.int64)
            gaussian_noise[:,self.feature_axis] = torch.round(torch.normal(self.mean, self.std, (numberOfBraggDisks_to_contaminate,))).type(torch.int64)
            new_feature_vals_after_noise_addition = BraggDiskList[indices_of_BraggDisks_permuted[:numberOfBraggDisks_to_contaminate]] + gaussian_noise

            negative_indices_for_axis_1_polar_angle = torch.where(new_feature_vals_after_noise_addition[:, self.feature_axis] < 0)[0]

            if len(negative_indices_for_axis_1_polar_angle) > 0:
                new_feature_vals_after_noise_addition[:, self.feature_axis][negative_indices_for_axis_1_polar_angle] = new_feature_vals_after_noise_addition[:, self.feature_axis][negative_indices_for_axis_1_polar_angle] + torch.tensor([int(self.feature_maxBinIdx)], dtype = torch.int64)
            new_feature_vals_after_noise_addition[:, self.feature_axis] = torch.clamp(new_feature_vals_after_noise_addition[:, self.feature_axis],  max = self.feature_maxBinIdx - 1)
            BraggDiskList_after_contamination[indices_of_BraggDisks_permuted[:numberOfBraggDisks_to_contaminate]] = new_feature_vals_after_noise_addition

            return BraggDiskList_after_contamination
        else:
            return BraggDiskList

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self.add_gaussian_noise(inpt)

class displaceFeature_GaussianNoise(v2.Transform):

    def __init__(self, feature_axis, feature_maxBinIdx, mean = 0.0, std = 2, fractionToAddNoise = 0.8, min_clamp_index = 1):
        super().__init__()
        self.mean = mean
        self.std = std
        self.fractionToAddNoise = fractionToAddNoise
        self.feature_axis = feature_axis
        self.feature_maxBinIdx = feature_maxBinIdx
        self.min_clamp_index = min_clamp_index
    
    def add_gaussian_noise(
                        self, BraggDiskList:torch.Tensor, ) -> torch.Tensor :
    
        idx_of_BraggDisks = torch.where(torch.sum(BraggDiskList, dim = 1) > 0)[0]
        numberOfBraggDisks_to_contaminate = int(len(idx_of_BraggDisks) * self.fractionToAddNoise)
        
        if numberOfBraggDisks_to_contaminate > 1:
            permuted_indices = torch.randperm(len(idx_of_BraggDisks))
            indices_of_BraggDisks_permuted = idx_of_BraggDisks[permuted_indices]
            BraggDiskList_after_contamination = BraggDiskList.clone().detach()
            gaussian_noise = torch.zeros((numberOfBraggDisks_to_contaminate, 3), dtype = torch.int64)
            gaussian_noise[:,self.feature_axis] = torch.round(torch.normal(self.mean, self.std, (numberOfBraggDisks_to_contaminate,))).type(torch.int64)
            new_feature_vals_after_noise_addition = BraggDiskList[indices_of_BraggDisks_permuted[:numberOfBraggDisks_to_contaminate]] + gaussian_noise
            new_feature_vals_after_noise_addition[:, self.feature_axis] = torch.clamp(new_feature_vals_after_noise_addition[:, self.feature_axis], min = self.min_clamp_index , max = self.feature_maxBinIdx - 1)
            BraggDiskList_after_contamination[indices_of_BraggDisks_permuted[:numberOfBraggDisks_to_contaminate]] = new_feature_vals_after_noise_addition
            
            return BraggDiskList_after_contamination
        else:
            return BraggDiskList
            
    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self.add_gaussian_noise(inpt)

class addFalsePositiveBraggDisks(v2.Transform):

    def __init__(self, feature0_maxBinIdx, feature1_maxBinIdx, feature2_maxBinIdx, std_of_intensity = 4, scaling_factor_of_intensity = 0.02, fractionToAddBraggDisks = 0.1):
        super().__init__()
        self.std_of_intensity = std_of_intensity
        self.fractionToAddBraggDisks = fractionToAddBraggDisks
        self.feature0_maxBinIdx = feature0_maxBinIdx
        self.feature1_maxBinIdx = feature1_maxBinIdx
        self.feature2_maxBinIdx = feature2_maxBinIdx
        self.scaling_factor_of_intensity = scaling_factor_of_intensity
    
    def add_false_positives(
                        self, BraggDiskList:torch.Tensor, ) -> torch.Tensor :

        # fraction of padded tokens should be lower than XX
    
        idx_of_PADDED_tokens = torch.where(torch.sum(BraggDiskList, dim = 1) == 0)[0]
        
        numberOfFalsePositives_to_add = int(len(idx_of_PADDED_tokens) * self.fractionToAddBraggDisks)

        if numberOfFalsePositives_to_add > 0:
            idx_of_BraggDisks = torch.where(torch.sum(BraggDiskList, dim = 1) > 0)[0]
            average_intensity = torch.sum(BraggDiskList[idx_of_BraggDisks][:,2]) / len(idx_of_BraggDisks)

            BraggDiskList_after_contamination = BraggDiskList.clone().detach()
            

            ## intensity of false postivies better be small.
            ## angle and radial distance can be compltely random.
            ## 

            false_positive_feature_0 = torch.randint(
                                        low = 1, 
                                        high = self.feature0_maxBinIdx, 
                                        size = (numberOfFalsePositives_to_add,)
            )
            false_positive_feature_1 = torch.randint(
                                        low = 0, 
                                        high = self.feature1_maxBinIdx, 
                                        size = (numberOfFalsePositives_to_add,)
            )
            gaussian_noise_of_intensity = torch.round(torch.normal(average_intensity * self.scaling_factor_of_intensity, self.std_of_intensity, (numberOfFalsePositives_to_add,))).type(torch.int64)
            false_positive_feature_2 = torch.clamp(gaussian_noise_of_intensity, min = 1 , max = self.feature2_maxBinIdx - 1)            

            false_positive_features = torch.stack((false_positive_feature_0, false_positive_feature_1, false_positive_feature_2)).T

            BraggDiskList_after_contamination[idx_of_PADDED_tokens[:numberOfFalsePositives_to_add]] = false_positive_features

            
            return BraggDiskList_after_contamination
        else:
            return BraggDiskList
            
    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self.add_false_positives(inpt)

class intensity_permutation(v2.Transform):

    def __init__(self, feature_maxBinIdx, feature_axis = 2, min_clamp_index = 1):
        super().__init__()
        self.feature_maxBinIdx = feature_maxBinIdx
        self.feature_axis = feature_axis
        self.min_clamp_index = min_clamp_index
    
    def apply_intensity_permutation_move(
                        self, BraggDiskList:torch.Tensor, ) -> torch.Tensor :
    
        idx_of_BraggDisks = torch.where(torch.sum(BraggDiskList, dim = 1) > 0)[0]
        numberOfBraggDisks_to_permute = int(len(idx_of_BraggDisks))

        if numberOfBraggDisks_to_permute > 1:
            permuted_indices = torch.randperm(numberOfBraggDisks_to_permute)
            indices_of_BraggDisks_permuted = idx_of_BraggDisks[permuted_indices]
            BraggDiskList_after_permutation = BraggDiskList.clone().detach()
            BraggDiskList_after_permutation[idx_of_BraggDisks[:numberOfBraggDisks_to_permute], self.feature_axis] = BraggDiskList_after_permutation[indices_of_BraggDisks_permuted[:numberOfBraggDisks_to_permute], self.feature_axis]

            return BraggDiskList_after_permutation
        else:
            return BraggDiskList
            
    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self.apply_intensity_permutation_move(inpt)

class random_intensity_assignment(v2.Transform):

    def __init__(self, feature_maxBinIdx, feature_axis = 2,std = 20, min_clamp_index = 1):
        super().__init__()
        self.feature_maxBinIdx = feature_maxBinIdx
        self.feature_axis = feature_axis
        self.min_clamp_index = min_clamp_index
        self.mean = int(feature_maxBinIdx * 0.5)
        self.std = std
    
    def apply_random_intensity_assignment(
                        self, BraggDiskList:torch.Tensor, ) -> torch.Tensor :
    
        idx_of_BraggDisks = torch.where(torch.sum(BraggDiskList, dim = 1) > 0)[0]

        random_intensities = torch.round(torch.normal(self.mean, self.std, (len(idx_of_BraggDisks),))).type(torch.int64)
        random_intensities = torch.clamp(random_intensities, min = self.min_clamp_index , max = self.feature_maxBinIdx - 1)


        BraggDiskList_after_assignment = BraggDiskList.clone().detach()
        BraggDiskList_after_assignment[idx_of_BraggDisks, self.feature_axis] = random_intensities

        return BraggDiskList_after_assignment
            
    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self.apply_random_intensity_assignment(inpt)

def custom_transforms_for_Data_Aug(num_bins_radialDistance, 
                                    num_bins_polarAngle, 
                                    num_bins_braggintensity):
    
    # uniform_intScaling = uniformIntensityScaling_gaussianNoise(feature_maxBinIdx = num_bins_braggintensity)
    individ_intScaling = individualIntensityScaling_gaussianNoise(feature_maxBinIdx = num_bins_braggintensity)
    # random_choice_of_int_Scaling = v2.RandomChoice([uniform_intScaling, individ_intScaling])
    random_int_Assignment = random_intensity_assignment(feature_maxBinIdx = num_bins_braggintensity)

    random_apply_int_Scaling = v2.RandomApply(transforms = [individ_intScaling], p = 0.90)
    random_apply_int_Assign = v2.RandomApply(transforms = [random_int_Assignment], p = 0.4)


    Displace_R = displaceFeature_GaussianNoise(feature_axis = 0, 
                                                 feature_maxBinIdx = num_bins_radialDistance, 
                                                 mean = 0.0, 
                                                 std = 1.7, 
                                                 fractionToAddNoise = 0.90,
                                                 min_clamp_index = 1)

    Displace_A = displaceFeature_GaussianNoise_polar(feature_axis = 1, 
                                                 feature_maxBinIdx = num_bins_polarAngle, 
                                                 mean = 0.0, 
                                                 std = 1.2, 
                                                 fractionToAddNoise = 0.90)

    Displace_I = displaceFeature_GaussianNoise(feature_axis = 2, 
                                                 feature_maxBinIdx = num_bins_braggintensity, 
                                                 mean = 0.0, 
                                                 std = 20, 
                                                 fractionToAddNoise = 1.0,
                                                 min_clamp_index = 1)
    
    random_apply_Displace_R =  v2.RandomApply(transforms = [Displace_R], p = 0.65)
    random_apply_Displace_A =  v2.RandomApply(transforms = [Displace_A], p = 0.65)
    random_apply_Displace_I =  v2.RandomApply(transforms = [Displace_I], p = 0.90)


    #### REMOVE BRAGG DISKS transforms
    weakBraggDiskRemoval = v2.RandomChoice([
                                            RemoveBraggDisksWithWeakInten(fractionToRemove = 0.05),
                                            RemoveBraggDisksWithWeakInten(fractionToRemove = 0.10),
                                            RemoveBraggDisksWithWeakInten(fractionToRemove = 0.15),
                                            RemoveBraggDisksWithWeakInten(fractionToRemove = 0.20),
                                            RemoveBraggDisksWithWeakInten(fractionToRemove = 0.25),
                                            RemoveBraggDisksWithWeakInten(fractionToRemove = 0.30),
                                            RemoveBraggDisksWithWeakInten(fractionToRemove = 0.35),
                                            RemoveBraggDisksWithWeakInten(fractionToRemove = 0.40),
                                            RemoveBraggDisksWithWeakInten(fractionToRemove = 0.45),
                                            RemoveBraggDisksWithWeakInten(fractionToRemove = 0.50),
                                            RemoveBraggDisksWithWeakInten(fractionToRemove = 0.55),
                                            RemoveBraggDisksWithWeakInten(fractionToRemove = 0.60),
                                            RemoveBraggDisksWithWeakInten(fractionToRemove = 0.65),
                                            RemoveBraggDisksWithWeakInten(fractionToRemove = 0.70),
                                            #RemoveBraggDisksWithWeakInten(fractionToRemove = 0.75),
                                            #RemoveBraggDisksWithWeakInten(fractionToRemove = 0.80),
                                            ])

    randomBraggDiskRemoval = v2.RandomChoice([
                                                RandomRemoval(fractionToRemove = 0.05),
                                                RandomRemoval(fractionToRemove = 0.10),
                                                RandomRemoval(fractionToRemove = 0.15),
                                                RandomRemoval(fractionToRemove = 0.20),
                                                ])

    # random_choice_of_BraggDiskRemoval = v2.RandomChoice([weakBraggDiskRemoval, randomBraggDiskRemoval])

    random_apply_weakBraggDiskRemoval = v2.RandomApply(transforms = [weakBraggDiskRemoval], p = 0.20)
    random_apply_ranDomBraggDiskRemoval = v2.RandomApply(transforms = [randomBraggDiskRemoval], p = 0.05)


    
    #### ADD FALSE POSITIVES transforms
    addFalsePositives01 = addFalsePositiveBraggDisks(num_bins_radialDistance, 
                                                     num_bins_polarAngle, 
                                                     num_bins_braggintensity, 
                                                     std_of_intensity = 1, 
                                                     fractionToAddBraggDisks = 0.02)

    addFalsePositives02 = addFalsePositiveBraggDisks(num_bins_radialDistance, 
                                                     num_bins_polarAngle, 
                                                     num_bins_braggintensity, 
                                                     std_of_intensity = 1, 
                                                     fractionToAddBraggDisks = 0.03)

    addFalsePositives03 = addFalsePositiveBraggDisks(num_bins_radialDistance, 
                                                     num_bins_polarAngle, 
                                                     num_bins_braggintensity, 
                                                     std_of_intensity = 1, 
                                                     fractionToAddBraggDisks = 0.04)

    
    # addFalsePositives04 = addFalsePositiveBraggDisks(num_bins_radialDistance, 
    #                                                  num_bins_polarAngle, 
    #                                                  num_bins_braggintensity, 
    #                                                  std_of_intensity = 1, 
    #                                                  fractionToAddBraggDisks = 0.05)

    random_choice_of_falsePositiveAddition =  v2.RandomChoice([
                                                                addFalsePositives01,
                                                                addFalsePositives02,
                                                                addFalsePositives03,
                                                             ])


    random_apply_addFalsePositives = v2.RandomApply(
                        transforms = [random_choice_of_falsePositiveAddition], 
                        p = 0.1
    )
    
    last_normalization_of_intensity = normalize_intensity(feature_maxBinIdx  = num_bins_braggintensity)
    
    composed_transforms = v2.Compose([
                                    random_apply_weakBraggDiskRemoval,
                                    random_apply_ranDomBraggDiskRemoval,
                                    random_apply_int_Assign,
                                    random_apply_int_Scaling,
                                    random_apply_Displace_R,
                                    random_apply_Displace_A,
                                    random_apply_Displace_I,
                                    random_apply_addFalsePositives,
                                    last_normalization_of_intensity,
                                    ])
    
    return composed_transforms
