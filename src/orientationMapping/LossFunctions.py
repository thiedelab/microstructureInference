import torch
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    

def geodesic_distance(outputMatrices, targetMatrices, mode = 'mean'):
    # Relative rotation
    R = outputMatrices.transpose(-1, -2) @ targetMatrices

    # Compute skew-symmetric part
    skew = R - R.transpose(-1, -2)

    # vee operator (vector from skew-symmetric matrix)
    vee = torch.stack([
        skew[..., 2, 1],
        skew[..., 0, 2],
        skew[..., 1, 0]
    ], dim=-1)

    # Norm of the vee (magnitude of axis vector)
    sin_theta = 0.5 * torch.linalg.norm(vee, dim=-1)

    # Compute trace of R
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]

    # Compute angle using atan2
    theta = torch.atan2(sin_theta, 0.5 * (trace - 1))

    if mode == 'mean':
        return torch.mean(
                            theta,
                            dtype = torch.float32
                            )
    elif mode == 'sum':
        return torch.sum(
                            theta,
                            dtype = torch.float32
                            )
    else:
        raise NotImplementedError



def symmetric_orthogonalization(x):
    """Maps 9D input vectors onto SO(3) via symmetric orthogonalization.
        x: should have size [batch_size, 9]

        Output has size [batch_size, 3, 3], where each inner 3x3 matrix is in SO(3).
    """
    m = x.view(-1, 3, 3)
    u, s, v = torch.svd(m)
    vt = torch.transpose(v, 1, 2)
    
    
    det = torch.det(torch.matmul(u, vt))
    det = det.view(-1, 1, 1)
    vt = torch.cat((vt[:, :2, :], vt[:, -1:, :] * det), 1)
    r = torch.matmul(u, vt)
    return r


def geodesic_distance_min(
                            outputMatrices, 
                            TruthMatrices, 
                            point_group_operations,
                          ):
    """
    Calculates the minimum geodesic distance between each predicted rotation matrix
    and a set of ground-truth candidate matrices.

    Args:
        outputMatrices (torch.Tensor): A tensor of shape (B, 3, 3) representing
                                       the predicted rotation matrices.
        targetMatrices (torch.Tensor): A tensor of shape (B, 48, 3, 3) representing
                                       the ground-truth candidate rotation matrices.

    Returns:
        torch.Tensor: The mean of the minimum geodesic distances across the batch.
    """
    # print("ataho")
    # print("point_group_operations.device", point_group_operations.device)
    # print("TruthMatrices.device", TruthMatrices.device)

    targetMatrices = torch.matmul(point_group_operations.unsqueeze(1), TruthMatrices.unsqueeze(0)).transpose(0,1)

    B, num_candidates, _, _ = targetMatrices.shape

    # Expand outputMatrices to match the shape of targetMatrices for broadcasting
    outputMatrices_expanded = outputMatrices.unsqueeze(1).expand(-1, num_candidates, -1, -1)

    # Reshape for batch-wise calculation
    output_flat = outputMatrices_expanded.reshape(B * num_candidates, 3, 3)
    target_flat = targetMatrices.reshape(B * num_candidates, 3, 3)

    # Relative rotation
    R = torch.bmm(output_flat.transpose(1, 2), target_flat)

    # Compute skew-symmetric part
    skew = R - R.transpose(1, 2)

    # vee operator (vector from skew-symmetric matrix)
    vee = torch.stack([
        skew[:, 2, 1],
        skew[:, 0, 2],
        skew[:, 1, 0]
    ], dim=-1)

    # Norm of the vee (magnitude of axis vector)
    sin_theta = 0.5 * torch.linalg.norm(vee, dim=-1)

    # Compute trace of R
    trace = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1)

    # Compute angle using atan2
    theta = torch.atan2(sin_theta, 0.5 * (trace - 1))

    # Reshape the distances back to (B, 48)
    geodesic_dists = theta.reshape(B, num_candidates)

    # Find the minimum distance for each batch element
    min_distances, _ = torch.min(geodesic_dists, dim=1)

    # Return the mean of these minimum distances as the loss
    
    return torch.mean(min_distances, dtype=torch.float32)

def geodesic_distance_min_return_entire_geodesic_stack(
                                                        outputMatrices, 
                                                        TruthMatrices, 
                                                        point_group_operations,
                          ):
    """
    Calculates the minimum geodesic distance between each predicted rotation matrix
    and a set of ground-truth candidate matrices.

    Args:
        outputMatrices (torch.Tensor): A tensor of shape (B, 3, 3) representing
                                       the predicted rotation matrices.
        targetMatrices (torch.Tensor): A tensor of shape (B, 48, 3, 3) representing
                                       the ground-truth candidate rotation matrices.

    Returns:
        torch.Tensor: The entinre minimum geodesic distances across the batch.
    """
    # print("ataho")
    # print("point_group_operations.device", point_group_operations.device)
    # print("TruthMatrices.device", TruthMatrices.device)

    targetMatrices = torch.matmul(point_group_operations.unsqueeze(1), TruthMatrices.unsqueeze(0)).transpose(0,1)

    B, num_candidates, _, _ = targetMatrices.shape

    # Expand outputMatrices to match the shape of targetMatrices for broadcasting
    outputMatrices_expanded = outputMatrices.unsqueeze(1).expand(-1, num_candidates, -1, -1)

    # Reshape for batch-wise calculation
    output_flat = outputMatrices_expanded.reshape(B * num_candidates, 3, 3)
    target_flat = targetMatrices.reshape(B * num_candidates, 3, 3)

    # Relative rotation
    R = torch.bmm(output_flat.transpose(1, 2), target_flat)

    # Compute skew-symmetric part
    skew = R - R.transpose(1, 2)

    # vee operator (vector from skew-symmetric matrix)
    vee = torch.stack([
        skew[:, 2, 1],
        skew[:, 0, 2],
        skew[:, 1, 0]
    ], dim=-1)

    # Norm of the vee (magnitude of axis vector)
    sin_theta = 0.5 * torch.linalg.norm(vee, dim=-1)

    # Compute trace of R
    trace = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1)

    # Compute angle using atan2
    theta = torch.atan2(sin_theta, 0.5 * (trace - 1))

    # Reshape the distances back to (B, 48)
    geodesic_dists = theta.reshape(B, num_candidates)

    # Find the minimum distance for each batch element
    min_distances, _ = torch.min(geodesic_dists, dim=1)

    # Return the mean of these minimum distances as the loss
    
    return torch.mean(min_distances, dtype=torch.float32), min_distances.float()

def binry_map_objective_criterion(predictions, target_rotation, mirrorTarget, weights = [0.75, 0.25]):
    
    predicted_rotation_matrix = symmetric_orthogonalization(predictions[0])
    rotation_prediction_loss = geodesic_distance(predicted_rotation_matrix, target_rotation)
    mirrorOp_prediction_loss = F.binary_cross_entropy_with_logits(predictions[1], mirrorTarget)    
    loss =(rotation_prediction_loss * weights[0]) + (mirrorOp_prediction_loss * weights[1])

    return loss, rotation_prediction_loss, mirrorOp_prediction_loss, predicted_rotation_matrix

def pointGroup_map_rotation_prediction(predictions, target_rotation, point_group_op_matrices):
    # print("point_group_op_matrices.device", point_group_op_matrices.device)
    # print("target_rotation.device", target_rotation.device)
    
    predicted_rotation_matrix = symmetric_orthogonalization(predictions)
    rotation_prediction_loss = geodesic_distance_min(predicted_rotation_matrix, target_rotation, point_group_op_matrices)
    loss = rotation_prediction_loss
    return loss

def pointGroup_map_rotation_prediction_return_geodesic_distance_stack(predictions, target_rotation, point_group_op_matrices):
    # print("point_group_op_matrices.device", point_group_op_matrices.device)
    # print("target_rotation.device", target_rotation.device)
    
    predicted_rotation_matrix = symmetric_orthogonalization(predictions)
    average_geodesic_distance, geodesic_distance_stack = geodesic_distance_min_return_entire_geodesic_stack(predicted_rotation_matrix, target_rotation, point_group_op_matrices)
    return average_geodesic_distance, geodesic_distance_stack

def pointGroup_map_rotation_and_phase_prediction(predictions, target_rotation, target_phase, point_group_op_matrices, weights = [0.9, 0.1]):
    # print("point_group_op_matrices.device", point_group_op_matrices.device)
    # print("target_rotation.device", target_rotation.device)
    
    predicted_rotation_matrix = symmetric_orthogonalization(predictions[0])
    phase_prediction_loss = F.binary_cross_entropy_with_logits(predictions[1], target_phase)
    rotation_prediction_loss = geodesic_distance_min(predicted_rotation_matrix, target_rotation, point_group_op_matrices)
    loss =(rotation_prediction_loss * weights[0]) + (phase_prediction_loss * weights[1])

    return loss, rotation_prediction_loss, phase_prediction_loss