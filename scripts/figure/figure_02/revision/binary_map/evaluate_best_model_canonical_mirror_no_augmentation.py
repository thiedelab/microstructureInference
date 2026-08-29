#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate best_model.pth on the unaugmented canonical-orientation +
binary-mirror validation set and save per-sample metrics.

Saved arrays
------------
geodesic_distance_stack
    Shape (N,). Canonical-orientation geodesic distance in radians.

binary_prediction_accuracy_stack
    Shape (N,). Per-sample mirror correctness: 1.0 = correct, 0.0 = incorrect.

Also saved for convenience
--------------------------
geodesic_distance_deg_stack
mirror_probability_stack
mirror_prediction_stack
mirror_target_stack
"""

import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataModules import digitized_bin_centers
from dataModules_canonical_mirror import (
    Dataset_canonicalOrientation_and_mirror,
)
from LossFunctions import symmetric_orthogonalization
from transformerModel import ModelConfig, make_model


def parse_args():
    parser = argparse.ArgumentParser()

    # ---------------------------------------------------------------
    # Paths
    # ---------------------------------------------------------------
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing the validation .npy files.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to best_model.pth.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="validation_per_sample_metrics_best_model.npz",
    )

    parser.add_argument(
        "--val_input_file",
        type=str,
        default="entire_Bradd_disks_padded_valid.npy",
    )
    parser.add_argument(
        "--val_rotation_file",
        type=str,
        default="orientation_canonical_labels_valid.npy",
    )
    parser.add_argument(
        "--val_mirror_file",
        type=str,
        default="entire_mirror_valid.npy",
    )

    # ---------------------------------------------------------------
    # Model/tokenization: same defaults as training script
    # ---------------------------------------------------------------
    parser.add_argument("--embed_dim", type=int, default=384)
    parser.add_argument("--max_sequence_length", type=int, default=76)

    parser.add_argument("--max_radial_distance", type=float, default=2.99000)
    parser.add_argument("--max_braggIntensity", type=float, default=1.0)

    parser.add_argument("--num_bins_radialDistance", type=int, default=256)
    parser.add_argument("--num_bins_polarAngle", type=int, default=360)
    parser.add_argument("--num_bins_braggintensity", type=int, default=64)

    parser.add_argument("--PAD", type=int, default=0)

    # ---------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=6)

    return parser.parse_args()


def resolve_path(input_dir, file_path):
    if os.path.isabs(file_path):
        return file_path
    return os.path.join(input_dir, file_path)


def geodesic_distance_per_sample(output_matrices, target_matrices):
    """
    Same geodesic-angle definition used by LossFunctions.geodesic_distance,
    but returns one value per sample rather than the batch mean.

    Parameters
    ----------
    output_matrices : (B, 3, 3)
    target_matrices : (B, 3, 3)

    Returns
    -------
    theta : (B,)
        Geodesic distance in radians.
    """
    R = output_matrices.transpose(-1, -2) @ target_matrices

    skew = R - R.transpose(-1, -2)

    vee = torch.stack(
        [
            skew[..., 2, 1],
            skew[..., 0, 2],
            skew[..., 1, 0],
        ],
        dim=-1,
    )

    sin_theta = 0.5 * torch.linalg.norm(vee, dim=-1)

    trace = (
        R[..., 0, 0]
        + R[..., 1, 1]
        + R[..., 2, 2]
    )

    theta = torch.atan2(
        sin_theta,
        0.5 * (trace - 1.0),
    )

    return theta.float()


@torch.inference_mode()
def evaluate_per_sample(model, dataloader, device, PAD=0):
    model.eval()

    geodesic_chunks = []
    mirror_correct_chunks = []

    # Extra arrays that are often useful for checking the binary classifier.
    mirror_probability_chunks = []
    mirror_prediction_chunks = []
    mirror_target_chunks = []

    for x, y_rot, y_mir in tqdm(
        dataloader,
        total=len(dataloader),
        desc="Evaluating validation set",
    ):
        features = x.to(
            device,
            non_blocking=True,
        )
        labels_r = y_rot.to(
            device,
            non_blocking=True,
        )
        labels_m = y_mir.to(
            device,
            non_blocking=True,
        )

        pad_mask = (
            torch.sum(
                features,
                dim=2,
            )
            == PAD
        ).view(
            features.size(0),
            1,
            1,
            features.size(1),
        )

        # pred[0]: (B, 9) raw canonical-orientation output
        # pred[1]: (B, 1) mirror logit
        pred = model(
            features,
            pad_mask,
        )

        predicted_rotation_matrix = (
            symmetric_orthogonalization(
                pred[0]
            )
        )

        geodesic_batch = (
            geodesic_distance_per_sample(
                predicted_rotation_matrix,
                labels_r,
            )
        )

        mirror_probability = torch.sigmoid(
            pred[1]
        ).reshape(-1)

        mirror_prediction = (
            mirror_probability >= 0.5
        ).float()

        mirror_target = labels_m.reshape(-1)

        mirror_correct = (
            mirror_prediction == mirror_target
        ).float()

        geodesic_chunks.append(
            geodesic_batch.cpu()
        )
        mirror_correct_chunks.append(
            mirror_correct.cpu()
        )

        mirror_probability_chunks.append(
            mirror_probability.cpu()
        )
        mirror_prediction_chunks.append(
            mirror_prediction.cpu()
        )
        mirror_target_chunks.append(
            mirror_target.cpu()
        )

    geodesic_distance_stack = torch.cat(
        geodesic_chunks,
        dim=0,
    ).numpy()

    binary_prediction_accuracy_stack = torch.cat(
        mirror_correct_chunks,
        dim=0,
    ).numpy()

    mirror_probability_stack = torch.cat(
        mirror_probability_chunks,
        dim=0,
    ).numpy()

    mirror_prediction_stack = torch.cat(
        mirror_prediction_chunks,
        dim=0,
    ).numpy()

    mirror_target_stack = torch.cat(
        mirror_target_chunks,
        dim=0,
    ).numpy()

    return (
        geodesic_distance_stack,
        binary_prediction_accuracy_stack,
        mirror_probability_stack,
        mirror_prediction_stack,
        mirror_target_stack,
    )


def main():
    args = parse_args()

    # ---------------------------------------------------------------
    # Reproducibility
    # ---------------------------------------------------------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("device:", device)

    # ---------------------------------------------------------------
    # Resolve paths
    # ---------------------------------------------------------------
    input_dir = os.path.abspath(
        os.path.expanduser(
            args.input_dir
        )
    )

    val_input_file = resolve_path(
        input_dir,
        args.val_input_file,
    )
    val_rotation_file = resolve_path(
        input_dir,
        args.val_rotation_file,
    )
    val_mirror_file = resolve_path(
        input_dir,
        args.val_mirror_file,
    )

    checkpoint_path = os.path.abspath(
        os.path.expanduser(
            args.checkpoint_path
        )
    )

    for path in (
        val_input_file,
        val_rotation_file,
        val_mirror_file,
        checkpoint_path,
    ):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    # ---------------------------------------------------------------
    # Same bin-center construction as training
    # ---------------------------------------------------------------
    (
        radial_bins,
        radial_bin_centers,
        angle_bins,
        angle_bin_centers,
        intensity_bin_centers,
    ) = digitized_bin_centers(
        args.num_bins_radialDistance,
        args.max_radial_distance,
        args.num_bins_polarAngle,
        args.num_bins_braggintensity,
        args.max_braggIntensity,
    )

    # ---------------------------------------------------------------
    # Validation data: NO augmentation for final evaluation
    # ---------------------------------------------------------------
    val_transforms = None
    print("validation augmentation: OFF")

    # ---------------------------------------------------------------
    # Validation dataset / loader
    # shuffle=False keeps stack index aligned with dataset index.
    # ---------------------------------------------------------------
    dataset_val = Dataset_canonicalOrientation_and_mirror(
        val_input_file,
        val_rotation_file,
        val_mirror_file,
        transform=val_transforms,
    )

    val_loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    if args.num_workers > 0:
        val_loader_kwargs.update(
            persistent_workers=True,
            prefetch_factor=2,
        )

    val_loader = DataLoader(
        dataset_val,
        **val_loader_kwargs,
    )

    print("number of validation samples:", len(dataset_val))

    # ---------------------------------------------------------------
    # Same model architecture as training
    # ---------------------------------------------------------------
    config = ModelConfig(
        d_embed=args.embed_dim,
        d_ff=2 * args.embed_dim,
        angle_bin_centers=angle_bin_centers,
        intensity_bin_centers=intensity_bin_centers,
        num_bins_radialDistance=args.num_bins_radialDistance,
        device=device,
        num_feature=9,
        h=8,
        N_encoder=3,
        max_seq_len=args.max_sequence_length,
        dropout=0.001,
        multiTask=1,
    )

    model = make_model(
        config
    )

    # ---------------------------------------------------------------
    # Load best model
    # best_model.pth is a checkpoint dictionary, not a raw state_dict.
    # ---------------------------------------------------------------
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        "checkpoint epoch:",
        checkpoint.get("epoch", "unknown"),
    )
    print(
        "checkpoint best_valid_loss:",
        checkpoint.get("best_valid_loss", "unknown"),
    )

    # ---------------------------------------------------------------
    # Per-sample evaluation
    # ---------------------------------------------------------------
    (
        geodesic_distance_stack,
        binary_prediction_accuracy_stack,
        mirror_probability_stack,
        mirror_prediction_stack,
        mirror_target_stack,
    ) = evaluate_per_sample(
        model,
        val_loader,
        device,
        PAD=args.PAD,
    )

    if (
        len(geodesic_distance_stack)
        != len(dataset_val)
    ):
        raise RuntimeError(
            "Geodesic stack length does not match validation dataset."
        )

    if (
        len(binary_prediction_accuracy_stack)
        != len(dataset_val)
    ):
        raise RuntimeError(
            "Binary-accuracy stack length does not match validation dataset."
        )

    geodesic_distance_deg_stack = np.rad2deg(
        geodesic_distance_stack
    )

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------
    output_file = os.path.abspath(
        os.path.expanduser(
            args.output_file
        )
    )

    output_dir = os.path.dirname(
        output_file
    )
    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

    np.savez(
        output_file,
        geodesic_distance_stack=geodesic_distance_stack.astype(
            np.float32
        ),
        geodesic_distance_deg_stack=geodesic_distance_deg_stack.astype(
            np.float32
        ),
        binary_prediction_accuracy_stack=binary_prediction_accuracy_stack.astype(
            np.float32
        ),
        mirror_probability_stack=mirror_probability_stack.astype(
            np.float32
        ),
        mirror_prediction_stack=mirror_prediction_stack.astype(
            np.uint8
        ),
        mirror_target_stack=mirror_target_stack.astype(
            np.uint8
        ),
    )

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print("")
    print("Saved:", output_file)
    print(
        "geodesic_distance_stack.shape:",
        geodesic_distance_stack.shape,
    )
    print(
        "binary_prediction_accuracy_stack.shape:",
        binary_prediction_accuracy_stack.shape,
    )
    print(
        "mean geodesic distance [rad]:",
        float(
            geodesic_distance_stack.mean()
        ),
    )
    print(
        "mean geodesic distance [deg]:",
        float(
            geodesic_distance_deg_stack.mean()
        ),
    )
    print(
        "mirror prediction accuracy:",
        float(
            binary_prediction_accuracy_stack.mean()
        ),
    )


if __name__ == "__main__":
    main()
