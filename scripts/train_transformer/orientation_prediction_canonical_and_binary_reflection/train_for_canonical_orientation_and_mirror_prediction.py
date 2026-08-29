#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train canonical orientation + binary mirror labels.

This script intentionally reuses the EXISTING:
    microstructure_inference.transformerModel
    microstructure_inference.LossFunctions

The current transformer is simply run with multiTask=1:
    MLP_head_0 -> 9D canonical-orientation output
    MLP_head_1 -> 1 mirror logit
"""

import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from microstructure_inference.dataAugmentation import (
    custom_transforms_for_Data_Aug,
)
from microstructure_inference.dataModules import (
    digitized_bin_centers,
)
from microstructure_inference.dataModules_canonical_mirror import (
    Dataset_canonicalOrientation_and_mirror,
)

# Reuse the CURRENT transformer implementation directly.
from microstructure_inference.transformerModel import (
    ModelConfig,
    make_model,
)

from microstructure_inference.trainer_canonical_mirror import (
    load_checkpoint,
    train,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train canonical-orientation + mirror-label model "
            "using the existing multitask transformer."
        )
    )

    # ---------------------------------------------------------------
    # Input arrays
    # ---------------------------------------------------------------
    parser.add_argument(
        "--input_dir",
        type=str,
        default=".",
        help=(
            "Directory containing the six input .npy files. "
            "This can point to node-local NVMe storage."
        ),
    )

    parser.add_argument(
        "--train_input_file",
        type=str,
        default="entire_Bradd_disks_padded_train.npy",
        help="Training Bragg-disk input filename.",
    )
    parser.add_argument(
        "--val_input_file",
        type=str,
        default="entire_Bradd_disks_padded_valid.npy",
        help="Validation Bragg-disk input filename.",
    )

    parser.add_argument(
        "--train_rotation_file",
        type=str,
        default="orientation_canonical_labels_train.npy",
        help=(
            "Canonical orientation matrices for training, "
            "shape (N, 3, 3)."
        ),
    )
    parser.add_argument(
        "--val_rotation_file",
        type=str,
        default="orientation_canonical_labels_valid.npy",
        help=(
            "Canonical orientation matrices for validation, "
            "shape (N, 3, 3)."
        ),
    )
    parser.add_argument(
        "--train_mirror_file",
        type=str,
        default="entire_mirror_train.npy",
        help=(
            "Binary mirror labels for training, "
            "shape (N,) or (N, 1)."
        ),
    )
    parser.add_argument(
        "--val_mirror_file",
        type=str,
        default="entire_mirror_valid.npy",
        help=(
            "Binary mirror labels for validation, "
            "shape (N,) or (N, 1)."
        ),
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="/tmp/kwang/canonical_mirror/",
    )

    # ---------------------------------------------------------------
    # Same model/tokenization defaults as current orientation training
    # ---------------------------------------------------------------
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=384,
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=76,
    )
    parser.add_argument(
        "--min_radial_distance",
        type=float,
        default=0.45844,
    )
    parser.add_argument(
        "--max_radial_distance",
        type=float,
        default=2.99000,
    )
    parser.add_argument(
        "--min_braggIntensity",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--max_braggIntensity",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--num_bins_radialDistance",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--num_bins_polarAngle",
        type=int,
        default=360,
    )
    parser.add_argument(
        "--num_bins_braggintensity",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--PAD",
        type=int,
        default=0,
    )

    # ---------------------------------------------------------------
    # Optimization
    # ---------------------------------------------------------------
    parser.add_argument(
        "--num_warmup_epochs",
        type=int,
        default=15,
    )
    parser.add_argument(
        "--cos_decay_epoch",
        type=int,
        default=250,
    )
    parser.add_argument(
        "--eta_intial",
        type=float,
        default=0.00007,
    )
    parser.add_argument(
        "--eta_min",
        type=float,
        default=5e-7,
    )

    # Same historical weights already used by
    # binry_map_objective_criterion in LossFunctions.py.
    parser.add_argument(
        "--rotation_loss_weight",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--mirror_loss_weight",
        type=float,
        default=0.10,
    )

    # ---------------------------------------------------------------
    # DataLoader
    # ---------------------------------------------------------------
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
    )

    # Keep current protocol by default: both train and validation
    # datasets receive custom_transforms_for_Data_Aug.
    parser.add_argument(
        "--disable_val_augmentation",
        action="store_true",
        help=(
            "Disable stochastic validation augmentation. "
            "Leave unset for direct consistency with the "
            "current orientation-training script."
        ),
    )

    # ---------------------------------------------------------------
    # Run control
    # ---------------------------------------------------------------
    parser.add_argument(
        "--initial_run",
        action="store_true",
        help=(
            "Start from epoch 0. If omitted, resume from "
            "output_path/last_updated_model.pth."
        ),
    )
    parser.add_argument(
        "--printArg",
        action="store_true",
    )
    parser.add_argument(
        "--printModelInfo",
        action="store_true",
    )

    return parser.parse_args()


def _require_file(
    path,
    description,
):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{description} not found: {path}"
        )


def _resolve_input_path(
    input_dir,
    file_path,
):
    """
    Resolve an input file against input_dir.

    Absolute paths are preserved. Relative paths are interpreted
    relative to input_dir.
    """
    if os.path.isabs(file_path):
        return file_path

    return os.path.join(
        input_dir,
        file_path,
    )


def main():
    args = parse_args()

    os.makedirs(
        args.output_path,
        exist_ok=True,
    )

    # Resolve all relative input filenames against --input_dir.
    args.input_dir = os.path.abspath(
        os.path.expanduser(
            args.input_dir
        )
    )

    args.train_input_file = _resolve_input_path(
        args.input_dir,
        args.train_input_file,
    )
    args.val_input_file = _resolve_input_path(
        args.input_dir,
        args.val_input_file,
    )
    args.train_rotation_file = _resolve_input_path(
        args.input_dir,
        args.train_rotation_file,
    )
    args.val_rotation_file = _resolve_input_path(
        args.input_dir,
        args.val_rotation_file,
    )
    args.train_mirror_file = _resolve_input_path(
        args.input_dir,
        args.train_mirror_file,
    )
    args.val_mirror_file = _resolve_input_path(
        args.input_dir,
        args.val_mirror_file,
    )

    _require_file(
        args.train_input_file,
        "training Bragg-disk input",
    )
    _require_file(
        args.val_input_file,
        "validation Bragg-disk input",
    )
    _require_file(
        args.train_rotation_file,
        "training canonical-orientation label",
    )
    _require_file(
        args.val_rotation_file,
        "validation canonical-orientation label",
    )
    _require_file(
        args.train_mirror_file,
        "training mirror label",
    )
    _require_file(
        args.val_mirror_file,
        "validation mirror label",
    )

    if (
        args.rotation_loss_weight < 0
        or args.mirror_loss_weight < 0
    ):
        raise ValueError(
            "Loss weights must be nonnegative."
        )

    if (
        args.rotation_loss_weight == 0
        and args.mirror_loss_weight == 0
    ):
        raise ValueError(
            "At least one loss weight must be positive."
        )

    # Reproducibility. Python random is included because the augmentation
    # implementation also uses the Python random module.
    torch.manual_seed(
        args.seed
    )
    np.random.seed(
        args.seed
    )
    random.seed(
        args.seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            args.seed
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "torch device",
        device,
        "\n",
    )

    if args.printArg:
        print(
            "Arguments passed:"
        )
        for arg, value in vars(args).items():
            print(
                f"  {arg}: {value}"
            )
        print()

    # Follow the current training script exactly here.
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

    train_transforms = (
        custom_transforms_for_Data_Aug(
            args.num_bins_radialDistance,
            args.num_bins_polarAngle,
            args.num_bins_braggintensity,
            radial_bins,
            radial_bin_centers,
            angle_bins,
            angle_bin_centers,
        )
    )

    if args.disable_val_augmentation:
        val_transforms = None
    else:
        val_transforms = (
            custom_transforms_for_Data_Aug(
                args.num_bins_radialDistance,
                args.num_bins_polarAngle,
                args.num_bins_braggintensity,
                radial_bins,
                radial_bin_centers,
                angle_bins,
                angle_bin_centers,
            )
        )

    dataset_train = (
        Dataset_canonicalOrientation_and_mirror(
            args.train_input_file,
            args.train_rotation_file,
            args.train_mirror_file,
            transform=train_transforms,
        )
    )

    dataset_val = (
        Dataset_canonicalOrientation_and_mirror(
            args.val_input_file,
            args.val_rotation_file,
            args.val_mirror_file,
            transform=val_transforms,
        )
    )

    train_indices = np.arange(
        len(dataset_train)
    )
    np.random.shuffle(
        train_indices
    )

    train_sampler = SubsetRandomSampler(
        train_indices
    )

    train_loader_kwargs = {
        "batch_size": args.batch_size,
        "sampler": train_sampler,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    val_loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    if args.num_workers > 0:
        train_loader_kwargs.update(
            persistent_workers=True,
            prefetch_factor=2,
        )
        val_loader_kwargs.update(
            persistent_workers=True,
            prefetch_factor=2,
        )

    train_loader = DataLoader(
        dataset_train,
        **train_loader_kwargs,
    )

    val_loader = DataLoader(
        dataset_val,
        **val_loader_kwargs,
    )

    # IMPORTANT:
    # multiTask=1 activates MLP_head_1 in the EXISTING transformer.
    #
    # MLP_head_0 -> 9 raw orientation values
    # MLP_head_1 -> 1 binary mirror logit
    config = ModelConfig(
        d_embed=args.embed_dim,
        d_ff=2 * args.embed_dim,
        angle_bin_centers=angle_bin_centers,
        intensity_bin_centers=intensity_bin_centers,
        num_bins_radialDistance=(
            args.num_bins_radialDistance
        ),
        device=device,
        num_feature=9,
        h=8,
        N_encoder=3,
        max_seq_len=(
            args.max_sequence_length
        ),
        dropout=0.001,
        multiTask=1,
    )

    model = make_model(
        config
    )

    if args.printModelInfo:
        from torchinfo import summary
        summary(
            model
        )

    print("")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.eta_intial,
    )

    linear_warmup = (
        torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=(
                1 / args.num_warmup_epochs
            ),
            end_factor=1.0,
            total_iters=(
                args.num_warmup_epochs - 1
            ),
            last_epoch=-1,
        )
    )

    cos_decay = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=args.cos_decay_epoch,
            eta_min=args.eta_min,
        )
    )

    num_epochs = int(
        args.num_warmup_epochs
        + args.cos_decay_epoch
    )

    if args.initial_run:
        start_epoch = 0
        best_valid_loss = 1000.0

    else:
        checkpoint_path = os.path.join(
            args.output_path,
            "last_updated_model.pth",
        )

        if not os.path.isfile(
            checkpoint_path
        ):
            raise FileNotFoundError(
                "Resume requested but checkpoint was not found: "
                f"{checkpoint_path}"
            )

        (
            model,
            optimizer,
            linear_warmup,
            cos_decay,
            start_epoch,
            best_valid_loss,
        ) = load_checkpoint(
            model,
            optimizer,
            linear_warmup,
            cos_decay,
            checkpoint_path,
            device,
        )

    train(
        model,
        train_loader,
        val_loader,
        num_epochs,
        optimizer,
        linear_warmup,
        cos_decay,
        args.num_warmup_epochs,
        args.cos_decay_epoch,
        device,
        args.output_path,
        PAD=args.PAD,
        start_epoch=start_epoch,
        best_valid_loss=best_valid_loss,
        loss_weights=(
            args.rotation_loss_weight,
            args.mirror_loss_weight,
        ),
    )


if __name__ == "__main__":
    main()
