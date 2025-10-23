#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 08:20:54 2025

@author: kwang
"""
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from orientationMapping.dataModules import cubic_proper_point_group_operations
from orientationMapping.LossFunctions import pointGroup_map_rotation_prediction, pointGroup_map_rotation_and_phase_prediction, symmetric_orthogonalization

def predict_rotation_sim_data_with_labels(model, dataloader, device, PAD = 0):
    point_group_op_matrices = cubic_proper_point_group_operations()
    point_group_op_matrices = point_group_op_matrices.to(device)
    
    predicted_rotation_matrices = []
    losses, count = [], 0
        

    with torch.no_grad():
        model.eval()
        for x, y in dataloader:
            features = x.to(device)
            labels_r  = y.to(device)
            pad_mask = (torch.sum(features, dim = 2) == PAD).view(features.size(0), 1, 1, features.size(1))
            
            
            
            pred = model(features, pad_mask)
            
            # print("pred", pred)
            loss = pointGroup_map_rotation_prediction(pred, labels_r, point_group_op_matrices)
            
            predicted_rotation_matrix = symmetric_orthogonalization(pred)
            predicted_rotation_matrices.append(predicted_rotation_matrix)
            
            losses.append(loss.item())
            count += 1
            
    return torch.vstack(predicted_rotation_matrices), np.mean(losses)


def predict_rotation_and_phases_sim_data_with_labels(model, dataloader, device, PAD = 0):
    point_group_op_matrices = cubic_proper_point_group_operations()
    point_group_op_matrices = point_group_op_matrices.to(device)
    
    predicted_rotation_matrices = []
    geo_error, phase_prediction_error, count = 0, 0, 0
        

    with torch.no_grad():
        model.eval()
        for x, y, z in dataloader:
            features = x.to(device)
            labels_r  = y.to(device)
            labels_phase  = z.to(device)
            pad_mask = (torch.sum(features, dim = 2) == PAD).view(features.size(0), 1, 1, features.size(1))
            
            
            
            pred = model(features, pad_mask)
            
            # print("pred", pred)
            loss, rotation_prediction_loss, phase_prediction_loss = pointGroup_map_rotation_and_phase_prediction(pred, labels_r, labels_phase, point_group_op_matrices)
            
            predicted_rotation_matrix = symmetric_orthogonalization(pred)
            predicted_rotation_matrices.append(predicted_rotation_matrix)
            
            phase_prediction_error += phase_prediction_loss
            geo_error += rotation_prediction_loss
            count += 1
            
    return torch.vstack(predicted_rotation_matrices), geo_error/count, phase_prediction_error/count

def predict_rotation_experimental_data(model, exp_dataloader, device, PAD = 0):
    
    predicted_rotation_matrices = []

    with torch.no_grad():
        model.eval()
        for x in exp_dataloader:
            features = x.to(device)
            pad_mask = (torch.sum(features, dim = 2) == PAD).view(features.size(0), 1, 1, features.size(1))            
            
            pred = model(features, pad_mask)
                        
            predicted_rotation_matrix = symmetric_orthogonalization(pred)
            predicted_rotation_matrices.append(predicted_rotation_matrix)
            
    return torch.vstack(predicted_rotation_matrices)

def point_in_spherical_triangle_oriented(p, a, b, c, tol=1e-10):
    """
    Robustly tests if a point p lies inside a spherical triangle on the unit sphere.
    
    The spherical triangle is defined by three vertices `a`, `b`, and `c`, each a 3D vector
    on the unit sphere. The function determines whether the test vector `p` lies inside the
    spherical triangle, on its edge, or outside, considering spherical geometry and orientation.
    
    Parameters
    ----------
    p : array-like, shape (3,)
        The query point vector on the unit sphere to test for inclusion in the spherical triangle.
    a, b, c : array-like, shape (3,)
        The three vertices of the spherical triangle, each normalized to unit length.
    tol : float, optional (default=1e-10)
        Numerical tolerance for floating-point comparisons to handle points near edges or vertices.
        
    Returns
    -------
    inside : bool
        True if `p` lies inside the spherical triangle (strictly inside, on boundary, or antipodal),
        otherwise False.
    status : int
        +1  : Point lies strictly inside the original spherical triangle (all dot products > tol).
        0   : Point lies exactly on an edge or vertex within tolerance.
        -1  : Point lies strictly inside the antipodal triangle.
    
    Variables
    ---------
    normalize : function
        Nested helper function to normalize 3D vectors to unit length.
    n_face : ndarray, shape (3,)
        Approximate normal vector to the spherical triangle's surface, computed as normalized sum of vertices.
    n_ab, n_bc, n_ca : ndarray, shape (3,)
        Oriented great-circle normal vectors for edges AB, BC, and CA, pointing inward toward triangle interior.
    s_ab, s_bc, s_ca : float
        Dot products between the test point `p` and each oriented edge normal, indicating side relative to edges.
    signs : ndarray, shape (3,)
        Signs of the dot products used to determine point position relative to all edges.
    """
    def normalize(v):
        v = np.asarray(v, dtype=float)
        return v / np.linalg.norm(v)

    a, b, c, p = map(normalize, [a, b, c, p])

    # Compute face normal (roughly pointing toward triangle center)
    n_face = normalize(a + b + c)

    # Oriented great-circle normals
    n_ab = np.cross(a, b)
    if np.dot(n_ab, n_face) < 0:
        n_ab = -n_ab

    n_bc = np.cross(b, c)
    if np.dot(n_bc, n_face) < 0:
        n_bc = -n_bc

    n_ca = np.cross(c, a)
    if np.dot(n_ca, n_face) < 0:
        n_ca = -n_ca

    # Dot tests (same hemisphere as triangle interior)
    s_ab = np.dot(p, n_ab)
    s_bc = np.dot(p, n_bc)
    s_ca = np.dot(p, n_ca)

    signs = np.sign([s_ab, s_bc, s_ca])
    if np.all(signs > tol):
        # strictly inside
        return True, +1
    elif np.any(np.abs(signs) <= tol) and np.all(signs >= -tol):
        # on edge or vertex (within tolerance)
        return True, 0  # boundary
    elif np.all(signs < -tol):
        # strictly inside antipode
        return True, -1
    else:
        return False, None