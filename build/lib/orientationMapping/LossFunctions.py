import torch
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def acos_grad(z):
    return -1 / torch.sqrt(1 - z**2)

class safe_acos(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        
        # protect ourselves from nan outputs in forward pass.
        threshold = 5e-9
        return torch.clamp(input, min = -1 + threshold, max=1 - threshold).acos()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()

        # protect ourselves from large gradients in backward pass.
        # outside of (-1 + epsilon, 1 - epsilon), gradient value is a 
        # fixed constant to acos'(1-epsilon)
        epsilon = 0.005
        safe_input = torch.clamp(input, min=-1 + epsilon, max=1 - epsilon)

        return acos_grad(safe_input) * grad_input

def geodesicLoss(outputMatrices, targetMatrices, mode = 'mean'):
    matrixMultiples = torch.matmul(
                                    torch.transpose(outputMatrices, 1, 2), 
                                    targetMatrices
                                    )
    
    trace = matrixMultiples.diagonal(
                                    offset=0, 
                                    dim1=-1, 
                                    dim2=-2).sum(-1)
    
    if mode == 'mean':
        return torch.mean(
                            safe_acos.apply((trace - 1.0) / 2.0), 
                            dtype = torch.float32
                            )
    elif mode == 'sum':
        return torch.sum(
                            safe_acos.apply((trace - 1.0) / 2.0), 
                            dtype = torch.float32
                            )
    else:
        raise NotImplementedError
    
    

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


def sign_consistency_loss(output, labels):
    label_signs = torch.sign(labels[:,:,2])
    label_sign_tot = torch.unsqueeze((torch.sign(torch.sum(label_signs, dim = 1)) +1)/2, dim = 1)
    return F.binary_cross_entropy_with_logits(torch.unsqueeze(output[:,0], dim = 1), label_sign_tot)

def reshapeOutputToRotationMatrix(output):
    """
    For each input data, the neural network model outputs
    6-dimensional vector (6D). This function use the 6D vector to
    create a rotation matrix with shape (3,3). The rotation matrix
    is a desired label (continuous value) for the input data.
    
    Args:
        output:
                    output of a neural network model.
                    torch.Tensor with shape (batch_size, 6)
    returns:
        rotation_matrices:
                    output further processed into 
                    rotation matrices.
                    torch.Tensor with shape (batch_size, 3, 3)        
    
    """
    
    ################## LOSS 1. GEODSIC LOSS ###############################
    isMirrorSymm = torch.unsqueeze(((output[:,0] >= 0.0).float() * 2.) - 1., dim = 1)
    
    reshaped_output = torch.reshape(
                                    output[:,1:],
                                    (output.shape[0], 2, 3)
                                    )
    reshaped_output_vectorSoftPlused = torch.stack(((F.celu(reshaped_output[:,0]) + 1.) * isMirrorSymm, reshaped_output[:,1]), dim = 1)
    normalized_reshaped_output = F.normalize(
                                    reshaped_output_vectorSoftPlused,
                                    p = 2, 
                                    dim = 2
                                    )
    R3 = F.normalize(
                    torch.cross(
                                normalized_reshaped_output[:,0], 
                                normalized_reshaped_output[:,1], 
                                dim = 1
                                ), 
                    dim = 1
                    )
   
    ## R2
    R2 = torch.cross(R3, normalized_reshaped_output[:,0], dim = 1)
    # print("R2[1]\n", R2[1], "\n")
    
    stacked = torch.stack((R2, R3, normalized_reshaped_output[:,0]), dim = 1)
    rotation_matrices = torch.transpose(stacked, 1, 2)
    # print("rotation_matrices[1]\n", rotation_matrices[1], "\n")

    
    return rotation_matrices

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

def objective_criterion(pred, target_rotation, fraction_of_geodesic_loss = 0.95, weight_of_geodesic_loss = 1.0):
    # print("ataho")
    # print("reshapeOutputToRotationMatrix(pred).type()", reshapeOutputToRotationMatrix(pred).type())
    # print("target.type()", target.type())
    #thickness_prediction_loss = F.mse_loss(torch.unsqueeze(pred[:, 0], dim = 1), target_thickness)
    predicted_rotation_matrix = symmetric_orthogonalization(pred[:, 0:])
    geodloss = geodesic_distance(predicted_rotation_matrix, target_rotation)
    # print("geodloss", geodloss)
    #signloss = sign_consistency_loss(pred, target)
    # print("signloss\n", signloss)
    # loss = (geodloss * fraction_of_geodesic_loss ) + (signloss * (1. - fraction_of_geodesic_loss))
    return  geodloss,  predicted_rotation_matrix

def mirrorMap_crossEntropy_loss(pred, mirrorTarget):
    # print("pred\n", pred, "\n")
    # print("mirrorTarget\n", mirrorTarget, "\n")
    # print("------------------------------------------")
    loss = F.binary_cross_entropy_with_logits(pred, mirrorTarget)
    return loss


def composite_objective_criterion(predictions, target_rotation, mirrorTarget, weights = [0.5, 0.5]):
    
    predicted_rotation_matrix = symmetric_orthogonalization(predictions[0])
    rotation_prediction_loss = geodesic_distance(predicted_rotation_matrix, target_rotation)
    mirrorOp_prediction_loss = F.binary_cross_entropy_with_logits(predictions[1], mirrorTarget)    
    loss =(rotation_prediction_loss * weights[0]) + (mirrorOp_prediction_loss * weights[1])

    return loss, rotation_prediction_loss, mirrorOp_prediction_loss, predicted_rotation_matrix