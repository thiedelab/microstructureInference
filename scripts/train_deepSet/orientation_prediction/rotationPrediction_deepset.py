#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end DeepSet training launcher for point-group-aware rotation prediction.

This script is the DeepSet counterpart of rotationPrediction_transformer.py.
It keeps the same data loading, data augmentation, optimizer/scheduler setup,
trainer, and point-group geodesic rotation loss. The main model change is:

    transformerModel.ModelConfig/make_model
        -> deepSetModel_updated.ModelConfig/make_model

Expected files in --data_dir:
    entire_Bradd_disks_padded_train.npy
    orientation_original_labels_train.npy
    entire_Bradd_disks_padded_valid.npy
    orientation_original_labels_valid.npy

Important convention:
    [0, 0, 0] is treated as padding by trainer_point_group_rotation_map.py.
    Real Bragg disks should have radial_bin >= 1 after direct-beam removal.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from microstructure_inference.dataAugmentation import custom_transforms_for_Data_Aug
from microstructure_inference.dataModules import DataSetPointGroup_rotation, digitized_bin_centers
from microstructure_inference.deepSetModel_updated import ModelConfig, make_model
from microstructure_inference.trainer_point_group_rotation_map import train, load_checkpoint


def str2bool(v):
    """Robust bool parser for argparse."""
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "true", "t", "1", "y"):
        return True
    if v in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DeepSet baseline for point-group-aware rotation prediction."
    )

    # ------------------------------------------------------------------
    # Data and output paths
    # ------------------------------------------------------------------
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/tmp/kwang/tempD/",
        help="Directory containing padded Bragg-disk tables and orientation labels.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/tmp/kwang/deepSetR06/",
        help="Directory where DeepSet checkpoints and training curves are saved.",
    )

    # ------------------------------------------------------------------
    # Model / discretization settings. Keep these matched to Transformer.
    # ------------------------------------------------------------------
    parser.add_argument("--embed_dim", type=int, default=384, help="Embedding dimension.")
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=76,
        help="Maximum number of Bragg-disk tokens per pattern.",
    )
    parser.add_argument("--min_radial_distance", type=float, default=0.45844, help="Minimum radius.")
    parser.add_argument("--max_radial_distance", type=float, default=2.99000, help="Maximum radius.")
    parser.add_argument("--min_braggIntensity", type=float, default=0.001, help="Minimum intensity.")
    parser.add_argument("--max_braggIntensity", type=float, default=1.0, help="Maximum intensity.")
    parser.add_argument(
        "--num_bins_radialDistance",
        type=int,
        default=256,
        help="Number of discretized bins for radial distance.",
    )
    parser.add_argument(
        "--num_bins_polarAngle",
        type=int,
        default=360,
        help="Number of discretized bins for polar angle.",
    )
    parser.add_argument(
        "--num_bins_braggintensity",
        type=int,
        default=64,
        help="Number of discretized bins for Bragg intensity.",
    )
    parser.add_argument(
        "--N_encoder",
        type=int,
        default=3,
        help="Number of independent per-disk MLP blocks in DeepSet phi network.",
    )
    parser.add_argument("--dropout", type=float, default=0.001, help="Dropout probability.")

    # The current trainer_point_group_rotation_map.py expects a single [B, 9]
    # rotation output. Do not use multitask mode with this trainer unless the
    # trainer/loss is also updated.
    parser.add_argument(
        "--isMultitask",
        type=int,
        default=0,
        help="Must be 0 for the current point-group rotation trainer.",
    )

    # ------------------------------------------------------------------
    # Training settings
    # ------------------------------------------------------------------
    parser.add_argument("--seed", type=int, default=6, help="Random seed for NumPy and PyTorch.")
    parser.add_argument("--PAD", type=int, default=0, help="PAD token value.")
    parser.add_argument(
        "--initial_run",
        type=str2bool,
        default=True,
        help="True: train from scratch. False: resume from output_dir/last_updated_model.pth.",
    )
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of DataLoader workers.")
    parser.add_argument(
        "--num_warmup_epochs",
        type=int,
        default=15,
        help="Number of epochs for linear warm-up scheduler.",
    )
    parser.add_argument(
        "--cos_decay_epoch",
        type=int,
        default=250,
        help="Number of epochs for cosine decay scheduler.",
    )
    parser.add_argument("--eta_intial", type=float, default=0.00007, help="Initial learning rate.")
    parser.add_argument("--eta_min", type=float, default=5e-7, help="Minimum learning rate.")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    parser.add_argument("--printArg", type=str2bool, default=True, help="Print parsed arguments.")
    parser.add_argument(
        "--printModelInfo",
        type=str2bool,
        default=True,
        help="Print model architecture summary if torchinfo is available.",
    )

    return parser.parse_args()


def make_loader(dataset, batch_size, sampler=None, shuffle=False, num_workers=16, device=None):
    """DataLoader helper that works for num_workers=0 and GPU/CPU."""
    pin_memory = bool(device is not None and device.type == "cuda")
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    if num_workers > 0:
        kwargs.update(
            persistent_workers=True,
            prefetch_factor=2,
        )

    return DataLoader(dataset, **kwargs)


def main():
    args = parse_args()

    if int(args.isMultitask) != 0:
        raise ValueError(
            "The current trainer_point_group_rotation_map.py expects a single [B, 9] "
            "rotation output. Please use --isMultitask 0 for this DeepSet baseline."
        )

    data_dir = Path(args.data_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    num_bins_radialDistance = args.num_bins_radialDistance
    num_bins_polarAngle = args.num_bins_polarAngle
    num_bins_braggintensity = args.num_bins_braggintensity

    embed_dim = args.embed_dim
    max_sequence_length = args.max_sequence_length

    max_radial_distance = args.max_radial_distance
    max_braggIntensity = args.max_braggIntensity

    seed = args.seed
    PAD = args.PAD
    num_warmup_epochs = args.num_warmup_epochs
    cos_decay_epoch = args.cos_decay_epoch
    num_epochs = int(num_warmup_epochs + cos_decay_epoch)

    eta_intial = args.eta_intial
    eta_min = args.eta_min

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("torch device", device, "\n")

    if args.printArg:
        print("Arguments passed:")
        for arg, value in vars(args).items():
            print(f"  {arg}: {value}")
        print("")

    # ------------------------------------------------------------------
    # Bin centers and augmentation. This is unchanged from Transformer run.
    # ------------------------------------------------------------------
    radial_bins, radial_bin_centers, angle_bins, angle_bin_centers, intensity_bin_centers = digitized_bin_centers(
        num_bins_radialDistance,
        max_radial_distance,
        num_bins_polarAngle,
        num_bins_braggintensity,
        max_braggIntensity,
    )

    train_transforms = custom_transforms_for_Data_Aug(
        num_bins_radialDistance,
        num_bins_polarAngle,
        num_bins_braggintensity,
        radial_bins,
        radial_bin_centers,
        angle_bins,
        angle_bin_centers,
    )

    val_transforms = custom_transforms_for_Data_Aug(
        num_bins_radialDistance,
        num_bins_polarAngle,
        num_bins_braggintensity,
        radial_bins,
        radial_bin_centers,
        angle_bins,
        angle_bin_centers,
    )

    # ------------------------------------------------------------------
    # Datasets. Same file names as your Transformer script, but data_dir is
    # now an argument so the script is easier to rerun.
    # ------------------------------------------------------------------
    train_table_path = data_dir / "entire_Bradd_disks_padded_train.npy"
    train_label_path = data_dir / "orientation_original_labels_train.npy"
    valid_table_path = data_dir / "entire_Bradd_disks_padded_valid.npy"
    valid_label_path = data_dir / "orientation_original_labels_valid.npy"

    for path in [train_table_path, train_label_path, valid_table_path, valid_label_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required input file not found: {path}")

    kwang_dataset_train = DataSetPointGroup_rotation(
        str(train_table_path),
        str(train_label_path),
        num_bins_polarAngle,
        transform=train_transforms,
    )
    kwang_dataset_val = DataSetPointGroup_rotation(
        str(valid_table_path),
        str(valid_label_path),
        num_bins_polarAngle,
        transform=val_transforms,
    )

    train_indices = np.arange(len(kwang_dataset_train))
    np.random.shuffle(train_indices)
    train_sampler = SubsetRandomSampler(train_indices)

    train_loader = make_loader(
        kwang_dataset_train,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        device=device,
    )

    val_loader = make_loader(
        kwang_dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device=device,
    )

    # ------------------------------------------------------------------
    # DeepSet model. This is the main change from Transformer training.
    # h is included only for config compatibility and is ignored by DeepSet.
    # ------------------------------------------------------------------
    config = ModelConfig(
        d_embed=embed_dim,
        d_ff=2 * embed_dim,
        angle_bin_centers=angle_bin_centers,
        intensity_bin_centers=intensity_bin_centers,
        num_bins_radialDistance=num_bins_radialDistance,
        device=device,
        num_feature=9,
        h=8,
        N_encoder=args.N_encoder,
        max_seq_len=max_sequence_length,
        dropout=args.dropout,
        multiTask=0,
    )

    model = make_model(config)

    if args.printModelInfo:
        try:
            from torchinfo import summary

            summary(model)
        except ImportError:
            print("torchinfo is not installed; skipping model summary.")

    print("")

    optimizer = torch.optim.AdamW(model.parameters(), lr=eta_intial)

    linear_warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1 / num_warmup_epochs,
        end_factor=1.0,
        total_iters=num_warmup_epochs - 1,
        last_epoch=-1,
    )

    cos_decay = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=cos_decay_epoch,
        eta_min=eta_min,
    )

    if args.initial_run:
        start_epoch = 0
    else:
        checkpoint_path = data_dir / "last_updated_model.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Cannot resume because checkpoint was not found: {checkpoint_path}"
            )
        model, optimizer, linear_warmup, cos_decay, start_epoch = load_checkpoint(
            model,
            optimizer,
            linear_warmup,
            cos_decay,
            str(checkpoint_path),
            device,
        )

    # ------------------------------------------------------------------
    # Train with the same trainer/loss as Transformer.
    # trainer_point_group_rotation_map.train() will save:
    #   best_model.pth
    #   last_updated_model.pth
    # inside output_dir.
    # ------------------------------------------------------------------
    train_error, valid_error = train(
        model,
        train_loader,
        val_loader,
        num_epochs,
        optimizer,
        linear_warmup,
        cos_decay,
        num_warmup_epochs,
        cos_decay_epoch,
        device,
        str(output_dir) + os.sep,
        PAD=PAD,
        start_epoch=start_epoch,
    )

    np.save(output_dir / "train_error.npy", np.array(train_error))
    np.save(output_dir / "valid_error.npy", np.array(valid_error))

    final_path = output_dir / "supervisedRegression_deepset_last.pt"
    torch.save(model.state_dict(), final_path)

    print("DeepSet training complete.")
    print(f"Saved final state_dict to: {final_path}")
    print(f"Saved training outputs to: {output_dir}")


if __name__ == "__main__":
    main()
