#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Predict canonical orientation matrices and binary mirror labels from
experimental Bragg-disk data using the canonical-orientation + mirror model.

No geodesic distance or prediction accuracy is computed because experimental
ground-truth orientation/mirror labels are not available.

Outputs are aligned sample-by-sample with the experimental scan ordering:
    - predicted_canonical_rotation_matrices: (N, 3, 3)
    - predicted_mirror_labels:              (N,)
    - predicted_mirror_probabilities:       (N,)
    - predicted_mirror_logits:              (N,)
"""

import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd
import py4DSTEM
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Use the local model/data/loss implementations used for the new formulation.
from dataModules import ExpDataset
from LossFunctions import symmetric_orthogonalization
from transformerModel import ModelConfig, make_model

# Reuse the same experimental Bragg-disk preprocessing as the original script.
from microstructure_inference.dataProcessing import (
    pre_process_experimental_BraggDisk,
    process_pandas_tabular_data,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Predict canonical orientations and binary mirror labels "
            "from experimental Bragg-disk data."
        )
    )

    # ------------------------------------------------------------------
    # Experimental Bragg-disk input
    # ------------------------------------------------------------------
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/kwang/Desktop/Storage/project/p03_orientation_mapping/figure/figure_04_and_05",
        help=(
            "Directory containing the experimental Bragg-disk .h5 file. "
            "Ignored for the file location when --bragg_disk_file is an "
            "absolute path."
        ),
    )
    parser.add_argument(
        "--bragg_disk_file",
        type=str,
        default=None,
        help=(
            "Experimental Bragg-disk .h5 file. If omitted, the filename "
            "is constructed from --correlationThresholdTemplateMatch."
        ),
    )
    parser.add_argument(
        "--correlationThresholdTemplateMatch",
        type=int,
        default=14000,
    )

    # ------------------------------------------------------------------
    # Model / output
    # ------------------------------------------------------------------
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        # required=True,
        help="Path to best_model.pth from the canonical+mirror training.",
        default="/home/kwang/Desktop/Storage/project/p03_orientation_mapping/figure/figure_02/revision_binary_map/independent_run_02/final_output/best_model.pth"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="processed_data",
        help="Directory in which prediction arrays are saved.",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="canonical_mirror",
        help="Prefix used for saved prediction files.",
    )

    # ------------------------------------------------------------------
    # Same model/tokenization settings as training
    # ------------------------------------------------------------------
    parser.add_argument("--embed_dim", type=int, default=384)
    parser.add_argument("--max_sequence_length", type=int, default=76)
    parser.add_argument("--max_radial_distance", type=float, default=2.99000)
    parser.add_argument("--max_braggIntensity", type=float, default=1.0)
    parser.add_argument("--num_bins_radialDistance", type=int, default=256)
    parser.add_argument("--num_bins_polarAngle", type=int, default=360)
    parser.add_argument("--num_bins_braggintensity", type=int, default=64)
    parser.add_argument("--PAD", type=int, default=0)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--mirror_threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--printArg", action="store_true")
    parser.add_argument("--printModelInfo", action="store_true")

    return parser.parse_args()


def resolve_path(base_dir, path):
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.abspath(
        os.path.join(
            os.path.expanduser(base_dir),
            path,
        )
    )


@torch.inference_mode()
def predict_canonical_orientation_and_mirror(
    model,
    dataloader,
    device,
    PAD=0,
    mirror_threshold=0.5,
):
    """
    Run the canonical-orientation + mirror model on experimental data.

    Returns
    -------
    rotation_matrix_stack : np.ndarray, shape (N, 3, 3)
        SO(3)-projected canonical orientation prediction.

    mirror_label_stack : np.ndarray, shape (N,)
        Binary prediction obtained from sigmoid(logit) >= mirror_threshold.

    mirror_probability_stack : np.ndarray, shape (N,)
        Sigmoid probability from the mirror head.

    mirror_logit_stack : np.ndarray, shape (N,)
        Raw output of the mirror head.
    """
    model.eval()

    rotation_chunks = []
    mirror_label_chunks = []
    mirror_probability_chunks = []
    mirror_logit_chunks = []

    for features in tqdm(
        dataloader,
        total=len(dataloader),
        desc="Predicting experimental data",
    ):
        features = features.to(
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

        # For multiTask=1:
        # pred[0] -> raw 9D canonical-orientation output
        # pred[1] -> binary mirror logit
        pred = model(
            features,
            pad_mask,
        )

        rotation_matrix = symmetric_orthogonalization(
            pred[0]
        )

        mirror_logit = pred[1].reshape(-1)
        mirror_probability = torch.sigmoid(
            mirror_logit
        )
        mirror_label = (
            mirror_probability >= mirror_threshold
        ).to(torch.uint8)

        rotation_chunks.append(
            rotation_matrix.detach().cpu()
        )
        mirror_logit_chunks.append(
            mirror_logit.detach().cpu()
        )
        mirror_probability_chunks.append(
            mirror_probability.detach().cpu()
        )
        mirror_label_chunks.append(
            mirror_label.detach().cpu()
        )

    rotation_matrix_stack = torch.cat(
        rotation_chunks,
        dim=0,
    ).numpy().astype(np.float32)

    mirror_label_stack = torch.cat(
        mirror_label_chunks,
        dim=0,
    ).numpy().astype(np.uint8)

    mirror_probability_stack = torch.cat(
        mirror_probability_chunks,
        dim=0,
    ).numpy().astype(np.float32)

    mirror_logit_stack = torch.cat(
        mirror_logit_chunks,
        dim=0,
    ).numpy().astype(np.float32)

    return (
        rotation_matrix_stack,
        mirror_label_stack,
        mirror_probability_stack,
        mirror_logit_stack,
    )


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("")
    print("torch device:", device)
    print("")

    if args.printArg:
        print("Arguments passed:")
        for arg, value in vars(args).items():
            print(f"  {arg}: {value}")
        print("")

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    data_dir = os.path.abspath(
        os.path.expanduser(
            args.data_dir
        )
    )

    checkpoint_path = os.path.abspath(
        os.path.expanduser(
            args.checkpoint_path
        )
    )

    output_dir = os.path.abspath(
        os.path.expanduser(
            args.output_dir
        )
    )
    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    if args.bragg_disk_file is None:
        bragg_file_stem = (
            "bragg_disks_corThForK80000_dog_sig1_2.00_"
            "sig2_6.00_cortThForTemp_%d"
            % args.correlationThresholdTemplateMatch
        )
        bragg_disk_path = os.path.join(
            data_dir,
            bragg_file_stem + ".h5",
        )
    else:
        bragg_disk_path = resolve_path(
            data_dir,
            args.bragg_disk_file,
        )
        bragg_file_stem = os.path.splitext(
            os.path.basename(
                bragg_disk_path
            )
        )[0]

    if not os.path.isfile(bragg_disk_path):
        raise FileNotFoundError(
            f"Experimental Bragg-disk file not found: {bragg_disk_path}"
        )

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    output_stem = (
        f"{args.output_prefix}_{bragg_file_stem}"
    )

    # ------------------------------------------------------------------
    # Load experimental Bragg disks
    # ------------------------------------------------------------------
    start_perf = time.perf_counter()

    bragg_disks = py4DSTEM.read(
        bragg_disk_path
    )

    elapsed_perf = (
        time.perf_counter()
        - start_perf
    )

    print(
        f"Loading Bragg disk list: "
        f"{elapsed_perf:.6f} seconds"
    )
    print("")

    # ------------------------------------------------------------------
    # Same preprocessing used in the original experimental script
    # ------------------------------------------------------------------
    start_perf = time.perf_counter()

    table_of_BraggDisk_qx_qy_intensity_for_eachScanIdx = (
        pre_process_experimental_BraggDisk(
            bragg_disks
        )
    )

    preprocessed_pickle_path = os.path.join(
        output_dir,
        output_stem + "_preProcessed_dictionary.pkl",
    )

    with open(
        preprocessed_pickle_path,
        "wb",
    ) as f:
        pickle.dump(
            table_of_BraggDisk_qx_qy_intensity_for_eachScanIdx,
            f,
        )

    df = pd.DataFrame(
        table_of_BraggDisk_qx_qy_intensity_for_eachScanIdx
    )

    preprocessed_json_path = os.path.join(
        output_dir,
        output_stem + "_preProcessed_df.json",
    )

    df.to_json(
        preprocessed_json_path,
        index=True,
    )

    (
        list_of_Bragg_disks_total,
        radial_bins,
        radial_bin_centers,
        angle_bins,
        angle_bin_centers,
        intensity_bins,
        intensity_bin_centers,
    ) = process_pandas_tabular_data(
        df,
        args.num_bins_radialDistance,
        args.num_bins_polarAngle,
        args.num_bins_braggintensity,
        args.max_sequence_length,
        args.max_radial_distance,
        args.max_braggIntensity,
    )

    list_of_Bragg_disks_total = (
        torch.nn.utils.rnn.pad_sequence(
            list_of_Bragg_disks_total,
            batch_first=True,
            padding_value=args.PAD,
        )
    )

    input_table_path = os.path.join(
        output_dir,
        output_stem + "_table.pt",
    )

    torch.save(
        list_of_Bragg_disks_total,
        input_table_path,
    )

    elapsed_perf = (
        time.perf_counter()
        - start_perf
    )

    print(
        "Mapping Bragg disk list to torch table tensor inputs: "
        f"{elapsed_perf:.6f} seconds"
    )
    print(
        "experimental input stack shape:",
        tuple(list_of_Bragg_disks_total.shape),
    )
    print("")

    # ------------------------------------------------------------------
    # Experimental loader
    # No augmentation.
    # shuffle=False preserves scan ordering.
    # ------------------------------------------------------------------
    experimental_dataset = ExpDataset(
        list_of_Bragg_disks_total,
        transform=None,
    )

    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    if args.num_workers > 0:
        loader_kwargs.update(
            persistent_workers=True,
            prefetch_factor=2,
        )

    exp_loader = DataLoader(
        experimental_dataset,
        **loader_kwargs,
    )

    # ------------------------------------------------------------------
    # Same architecture used for canonical-orientation + mirror training
    # multiTask=1 activates both heads.
    # ------------------------------------------------------------------
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

    if args.printModelInfo:
        from torchinfo import summary
        summary(model)

    # ------------------------------------------------------------------
    # Load best checkpoint
    # ------------------------------------------------------------------
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        "checkpoint epoch:",
        checkpoint.get(
            "epoch",
            "unknown",
        ),
    )
    print(
        "checkpoint best_valid_loss:",
        checkpoint.get(
            "best_valid_loss",
            "unknown",
        ),
    )
    print("")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    start_perf = time.perf_counter()

    (
        predicted_canonical_rotation_matrices,
        predicted_mirror_labels,
        predicted_mirror_probabilities,
        predicted_mirror_logits,
    ) = predict_canonical_orientation_and_mirror(
        model,
        exp_loader,
        device,
        PAD=args.PAD,
        mirror_threshold=args.mirror_threshold,
    )

    elapsed_perf = (
        time.perf_counter()
        - start_perf
    )

    n_samples = len(
        experimental_dataset
    )

    if (
        predicted_canonical_rotation_matrices.shape[0]
        != n_samples
    ):
        raise RuntimeError(
            "Rotation prediction count does not match "
            "experimental dataset length."
        )

    if (
        predicted_mirror_labels.shape[0]
        != n_samples
    ):
        raise RuntimeError(
            "Mirror prediction count does not match "
            "experimental dataset length."
        )

    # ------------------------------------------------------------------
    # Save separate .npy arrays
    # ------------------------------------------------------------------
    rotation_path = os.path.join(
        output_dir,
        output_stem
        + "_predicted_canonical_rotation_matrices.npy",
    )

    mirror_label_path = os.path.join(
        output_dir,
        output_stem
        + "_predicted_mirror_labels.npy",
    )

    mirror_probability_path = os.path.join(
        output_dir,
        output_stem
        + "_predicted_mirror_probabilities.npy",
    )

    mirror_logit_path = os.path.join(
        output_dir,
        output_stem
        + "_predicted_mirror_logits.npy",
    )

    np.save(
        rotation_path,
        predicted_canonical_rotation_matrices,
    )
    np.save(
        mirror_label_path,
        predicted_mirror_labels,
    )
    np.save(
        mirror_probability_path,
        predicted_mirror_probabilities,
    )
    np.save(
        mirror_logit_path,
        predicted_mirror_logits,
    )

    # ------------------------------------------------------------------
    # Also save one combined file for convenient downstream analysis.
    # ------------------------------------------------------------------
    combined_path = os.path.join(
        output_dir,
        output_stem
        + "_canonical_mirror_predictions.npz",
    )

    np.savez(
        combined_path,
        predicted_canonical_rotation_matrices=(
            predicted_canonical_rotation_matrices
        ),
        predicted_mirror_labels=(
            predicted_mirror_labels
        ),
        predicted_mirror_probabilities=(
            predicted_mirror_probabilities
        ),
        predicted_mirror_logits=(
            predicted_mirror_logits
        ),
        mirror_threshold=np.asarray(
            args.mirror_threshold,
            dtype=np.float32,
        ),
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("")
    print(
        "predicted_canonical_rotation_matrices.shape:",
        predicted_canonical_rotation_matrices.shape,
    )
    print(
        "predicted_mirror_labels.shape:",
        predicted_mirror_labels.shape,
    )
    print(
        "fraction predicted mirror label = 1:",
        float(
            predicted_mirror_labels.mean()
        ),
    )
    print("")
    print(
        f"Prediction time: {elapsed_perf:.6f} seconds"
    )
    print("")
    print("Saved:")
    print(" ", rotation_path)
    print(" ", mirror_label_path)
    print(" ", mirror_probability_path)
    print(" ", mirror_logit_path)
    print(" ", combined_path)
    print("")
    print("JOB DONE")
    print("")


if __name__ == "__main__":
    main()
