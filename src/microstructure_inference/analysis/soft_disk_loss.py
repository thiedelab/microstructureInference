"""
soft_disk_loss.py

Soft set-to-set matching loss for comparing two Bragg-disk sets.

The intended usage is:

    set_A = experimental Bragg disks
    set_B = simulated Bragg disks generated from a predicted orientation

Each Bragg disk is represented as:

    [x, y, intensity]

with input shapes:

    set_A : (N_A, 3)
    set_B : (N_B, 3)

The returned scalar is a soft discrepancy measure. Lower values indicate
better agreement between the experimental and simulated Bragg-disk sets.

The loss contains:

1. A positional mismatch term.
2. A weak intensity mismatch term.
3. Mild intensity-based weighting of experimental disks.
4. A soft unmatched penalty for experimental disks.
5. An optional reverse-direction term that penalizes simulated disks without
   experimental support.

The function is intended primarily as an evaluation metric for comparing
predicted orientations through their corresponding simulated diffraction
patterns. It does not search over orientations or template libraries.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import torch


ArrayLike = Union[np.ndarray, torch.Tensor]


def _as_float_tensor(
    array: ArrayLike,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Convert a NumPy array or tensor to float32 on the requested device."""
    if torch.is_tensor(array):
        return array.detach().to(device=device, dtype=torch.float32)

    return torch.as_tensor(
        np.asarray(array),
        dtype=torch.float32,
        device=device,
    )


def _validate_disk_set(disks: torch.Tensor, name: str) -> None:
    """Validate a single unpadded Bragg-disk set."""
    if disks.ndim != 2 or disks.shape[1] != 3:
        raise ValueError(
            f"{name} must have shape (N, 3), got {tuple(disks.shape)}"
        )

    if disks.shape[0] == 0:
        raise ValueError(f"{name} contains no Bragg disks.")

    if not torch.isfinite(disks).all():
        raise ValueError(f"{name} contains NaN or infinite values.")

    if (disks[:, 2] < 0).any():
        raise ValueError(f"{name} contains negative intensities.")


def soft_disk_loss(
    set_A: ArrayLike,
    set_B: ArrayLike,
    *,
    sigma_x: float = 0.15,
    gamma: float = 0.2,
    lambda_I: float = 0.02,
    alpha: float = 0.25,
    tau: float = 0.05,
    c_miss: float = 1.0,
    eta: float = 0.5,
    c_extra: float = 0.5,
    device: Union[str, torch.device] = "cuda",
    eps: float = 1e-12,
) -> float:
    """
    Compute the soft Bragg-disk matching loss between two individual disk sets.

    Parameters
    ----------
    set_A
        Experimental Bragg-disk set with shape (N_A, 3).

        Columns:
            set_A[:, 0] = x position
            set_A[:, 1] = y position
            set_A[:, 2] = intensity

    set_B
        Simulated Bragg-disk set with shape (N_B, 3).

        Columns:
            set_B[:, 0] = x position
            set_B[:, 1] = y position
            set_B[:, 2] = intensity

    sigma_x
        Positional tolerance scale. The spatial pairwise cost is

            d_ij^2 / (2 * sigma_x^2)

    gamma
        Intensity compression exponent:

            I -> I^gamma

    lambda_I
        Weight of the pairwise intensity mismatch term.

    alpha
        Floor parameter for experimental-disk weighting:

            w_i = alpha + (1 - alpha) * I_i^gamma

    tau
        Soft-min temperature.

    c_miss
        Unmatched fallback penalty for an experimental disk.

    eta
        Weight of the reverse simulated-to-experimental term.
        Set eta=0 to disable the reverse term.

    c_extra
        Unmatched fallback penalty for a simulated disk in the reverse term.

    device
        PyTorch computation device, for example "cuda" or "cpu".

    eps
        Small numerical stability constant.

    Returns
    -------
    float
        Scalar soft Bragg-disk loss.

        Lower values indicate better agreement between set_A and set_B.

    Notes
    -----
    The direction of the comparison is intentional:

        set_A = experimental disks
        set_B = simulated disks

    The loss is not generally symmetric because experimental disks receive
    intensity-based weights and the forward and reverse terms have different
    meanings.
    """
    if sigma_x <= 0:
        raise ValueError("sigma_x must be > 0")

    if gamma < 0:
        raise ValueError("gamma must be >= 0")

    if lambda_I < 0:
        raise ValueError("lambda_I must be >= 0")

    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1]")

    if tau <= 0:
        raise ValueError("tau must be > 0")

    if c_miss < 0:
        raise ValueError("c_miss must be >= 0")

    if eta < 0:
        raise ValueError("eta must be >= 0")

    if c_extra < 0:
        raise ValueError("c_extra must be >= 0")

    if eps <= 0:
        raise ValueError("eps must be > 0")

    device = torch.device(device)

    A = _as_float_tensor(set_A, device=device)
    B = _as_float_tensor(set_B, device=device)

    _validate_disk_set(A, "set_A")
    _validate_disk_set(B, "set_B")

    with torch.no_grad():
        # ------------------------------------------------------------
        # Coordinates and intensities
        # ------------------------------------------------------------
        A_xy = A[:, :2]          # (N_A, 2)
        B_xy = B[:, :2]          # (N_B, 2)

        A_I = A[:, 2]            # (N_A,)
        B_I = B[:, 2]            # (N_B,)

        A_a = A_I.clamp_min(0.0).pow(gamma)
        B_a = B_I.clamp_min(0.0).pow(gamma)

        # ------------------------------------------------------------
        # Pairwise cost
        #
        # C_ij =
        #     ||x_i - x_j||^2 / (2 sigma_x^2)
        #     + lambda_I * |I_i^gamma - I_j^gamma|
        # ------------------------------------------------------------
        A_sq = (A_xy**2).sum(dim=-1)                       # (N_A,)
        B_sq = (B_xy**2).sum(dim=-1)                       # (N_B,)
        cross = A_xy @ B_xy.T                              # (N_A, N_B)

        d2 = A_sq[:, None] + B_sq[None, :] - 2.0 * cross
        d2 = d2.clamp_min(0.0)

        spatial_cost = d2 / (2.0 * sigma_x**2 + eps)

        intensity_cost = lambda_I * (
            A_a[:, None] - B_a[None, :]
        ).abs()

        pair_cost = spatial_cost + intensity_cost          # (N_A, N_B)

        # ------------------------------------------------------------
        # Forward term: experimental -> simulated
        #
        # Each experimental disk can softly match one simulated disk
        # or remain unmatched with penalty c_miss.
        # ------------------------------------------------------------
        miss_logit = torch.full(
            (A.shape[0], 1),
            fill_value=-c_miss / tau,
            device=device,
            dtype=A.dtype,
        )

        forward_logits = torch.cat(
            [-pair_cost / tau, miss_logit],
            dim=1,
        )

        forward_per_disk = -tau * torch.logsumexp(
            forward_logits.float(),
            dim=1,
        )

        # Mild intensity-based weighting of experimental disks.
        A_weights = alpha + (1.0 - alpha) * A_a
        A_weights = A_weights / A_weights.sum().clamp_min(eps)

        forward_loss = (forward_per_disk * A_weights).sum()

        # ------------------------------------------------------------
        # Reverse term: simulated -> experimental
        #
        # Each simulated disk can softly match one experimental disk
        # or remain unsupported with penalty c_extra.
        # ------------------------------------------------------------
        if eta > 0.0:
            extra_logit = torch.full(
                (1, B.shape[0]),
                fill_value=-c_extra / tau,
                device=device,
                dtype=A.dtype,
            )

            reverse_logits = torch.cat(
                [-pair_cost / tau, extra_logit],
                dim=0,
            )

            reverse_per_disk = -tau * torch.logsumexp(
                reverse_logits.float(),
                dim=0,
            )

            reverse_loss = reverse_per_disk.mean()
            total_loss = forward_loss + eta * reverse_loss
        else:
            total_loss = forward_loss

    return float(total_loss.item())


if __name__ == "__main__":
    # Small sanity check.
    #
    # Identical disk sets should produce a lower loss than a noticeably
    # displaced simulated pattern.
    set_A = np.array(
        [
            [0.10, 0.20, 1.00],
            [0.30, 0.40, 0.60],
            [0.55, 0.25, 0.30],
        ],
        dtype=np.float32,
    )

    set_B_good = set_A.copy()

    set_B_bad = set_A.copy()
    set_B_bad[:, 0] += 0.10

    test_device = "cuda" if torch.cuda.is_available() else "cpu"

    loss_good = soft_disk_loss(
        set_A,
        set_B_good,
        device=test_device,
    )

    loss_bad = soft_disk_loss(
        set_A,
        set_B_bad,
        device=test_device,
    )

    print(f"device    : {test_device}")
    print(f"good loss : {loss_good:.6f}")
    print(f"bad loss  : {loss_bad:.6f}")
