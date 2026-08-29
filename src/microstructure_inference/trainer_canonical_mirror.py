#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Trainer for canonical-orientation + binary mirror prediction.

The loss is imported directly from the existing
microstructure_inference.LossFunctions module.

Tracked during training/validation:
    - total joint loss
    - canonical-orientation geodesic loss
    - mirror BCE loss
    - mirror classification accuracy

Rotation-matrix MSE is intentionally not computed.
"""

import os

import numpy as np
import torch
from tqdm import tqdm

from LossFunctions import (
    binry_map_objective_criterion,
)


def save_checkpoint(
    model,
    optimizer,
    linear_warmup,
    cos_decay,
    epoch,
    checkpoint_path,
    best_valid_loss,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "linear_warmup_state_dict": linear_warmup.state_dict(),
        "cos_decay_state_dict": cos_decay.state_dict(),
        "best_valid_loss": float(best_valid_loss),
    }

    torch.save(
        checkpoint,
        checkpoint_path,
    )


def load_checkpoint(
    model,
    optimizer,
    linear_warmup,
    cos_decay,
    checkpoint_path,
    device,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )
    linear_warmup.load_state_dict(
        checkpoint["linear_warmup_state_dict"]
    )
    cos_decay.load_state_dict(
        checkpoint["cos_decay_state_dict"]
    )

    start_epoch = int(checkpoint["epoch"]) + 1
    best_valid_loss = float(
        checkpoint.get(
            "best_valid_loss",
            1000.0,
        )
    )

    return (
        model,
        optimizer,
        linear_warmup,
        cos_decay,
        start_epoch,
        best_valid_loss,
    )


def _mirror_accuracy(
    mirror_logits,
    mirror_target,
):
    mirror_prediction = (
        torch.sigmoid(mirror_logits) >= 0.5
    ).float()

    return (
        mirror_prediction == mirror_target
    ).float().mean()


def train_epoch(
    model,
    dataloader,
    optimizer,
    device,
    PAD=0,
    loss_weights=(0.75, 0.25),
):
    model.train()

    losses = []
    geo_errors = []
    mirror_errors = []
    mirror_accuracies = []

    pbar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
    )

    for idx, (x, y_rot, y_mir) in pbar:
        optimizer.zero_grad()

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

        # With multiTask=1 in the existing transformerModel.py:
        # pred[0] = MLP_head_0 output, shape (B, 9)
        # pred[1] = MLP_head_1 output, shape (B, 1)
        pred = model(
            features,
            pad_mask,
        )

        # Directly use the existing objective from LossFunctions.py.
        (
            loss,
            rotation_loss,
            mirror_loss,
            _,
        ) = binry_map_objective_criterion(
            pred,
            labels_r,
            labels_m,
            weights=list(loss_weights),
        )

        loss.backward()
        optimizer.step()

        mirror_acc = _mirror_accuracy(
            pred[1].detach(),
            labels_m,
        )

        losses.append(
            loss.item()
        )
        geo_errors.append(
            rotation_loss.item()
        )
        mirror_errors.append(
            mirror_loss.item()
        )
        mirror_accuracies.append(
            mirror_acc.item()
        )

        if idx > 0 and idx % 50 == 0:
            pbar.set_description(
                f"train loss={loss.item():.4f}, "
                f"rotLoss={rotation_loss.item():.7f}, "
                f"mirLoss={mirror_loss.item():.7f}, "
                f"mirAcc={mirror_acc.item():.4f}"
            )

    return (
        float(np.mean(losses)),
        float(np.mean(geo_errors)),
        float(np.mean(mirror_errors)),
        float(np.mean(mirror_accuracies)),
    )


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device,
    PAD=0,
    loss_weights=(0.75, 0.25),
):
    model.eval()

    losses = []
    geo_errors = []
    mirror_errors = []
    mirror_accuracies = []

    for x, y_rot, y_mir in dataloader:
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

        pred = model(
            features,
            pad_mask,
        )

        (
            loss,
            rotation_loss,
            mirror_loss,
            _,
        ) = binry_map_objective_criterion(
            pred,
            labels_r,
            labels_m,
            weights=list(loss_weights),
        )

        mirror_acc = _mirror_accuracy(
            pred[1],
            labels_m,
        )

        losses.append(
            loss.item()
        )
        geo_errors.append(
            rotation_loss.item()
        )
        mirror_errors.append(
            mirror_loss.item()
        )
        mirror_accuracies.append(
            mirror_acc.item()
        )

    return (
        float(np.mean(losses)),
        float(np.mean(geo_errors)),
        float(np.mean(mirror_errors)),
        float(np.mean(mirror_accuracies)),
    )


def train(
    model,
    train_loader,
    test_loader,
    epochs,
    optimizer,
    linear_warmup,
    cos_decay,
    num_warmup_epochs,
    cos_decay_epoch,
    device,
    file_path,
    PAD=0,
    start_epoch=0,
    best_valid_loss=1000.0,
    loss_weights=(0.75, 0.25),
):
    """
    Training structure intentionally follows the current orientation trainer
    and the historical binary-mirror trainer.
    """

    os.makedirs(
        file_path,
        exist_ok=True,
    )

    train_error = []
    valid_error = []

    train_rotation_error = []
    valid_rotation_error = []

    train_mirror_error = []
    valid_mirror_error = []

    train_mirror_accuracy = []
    valid_mirror_accuracy = []

    for ep in range(
        start_epoch,
        epochs,
    ):
        (
            train_loss,
            train_rotLoss,
            train_mirrLoss,
            train_mirrAcc,
        ) = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            PAD=PAD,
            loss_weights=loss_weights,
        )

        train_error.append(
            train_loss
        )
        train_rotation_error.append(
            train_rotLoss
        )
        train_mirror_error.append(
            train_mirrLoss
        )
        train_mirror_accuracy.append(
            train_mirrAcc
        )

        print("")
        print(
            f"ep {ep}: "
            f"tra_loss={train_loss:.7f}, "
            f"tra_geod={train_rotLoss:.7f}, "
            f"tra_mirr={train_mirrLoss:.7f}, "
            f"tra_mirrAcc={train_mirrAcc:.7f}"
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        (
            val_loss,
            val_rotLoss,
            val_mirrLoss,
            val_mirrAcc,
        ) = evaluate(
            model,
            test_loader,
            device,
            PAD=PAD,
            loss_weights=loss_weights,
        )

        valid_error.append(
            val_loss
        )
        valid_rotation_error.append(
            val_rotLoss
        )
        valid_mirror_error.append(
            val_mirrLoss
        )
        valid_mirror_accuracy.append(
            val_mirrAcc
        )

        print("")
        print(
            f"ep {ep}: "
            f"val_loss={val_loss:.7f}, "
            f"val_geod={val_rotLoss:.7f}, "
            f"val_mirr={val_mirrLoss:.7f}, "
            f"val_mirrAcc={val_mirrAcc:.7f}"
        )

        # Same scheduler scheme as the current trainer.
        if ep < num_warmup_epochs:
            linear_warmup.step()
            print(
                "linear_warmup.get_last_lr()",
                linear_warmup.get_last_lr(),
            )
        else:
            cos_decay.step()
            print(
                "cos_decay.get_last_lr()",
                cos_decay.get_last_lr(),
            )

        if val_loss < best_valid_loss:
            best_valid_loss = val_loss

            checkpoint_path = os.path.join(
                file_path,
                "best_model.pth",
            )

            save_checkpoint(
                model,
                optimizer,
                linear_warmup,
                cos_decay,
                ep,
                checkpoint_path,
                best_valid_loss,
            )

            print("")
            print(
                "ep",
                ep,
                " val_loss",
                val_loss,
                ", new best model saved",
            )
            print("")

        most_recent_model_path = os.path.join(
            file_path,
            "last_updated_model.pth",
        )

        save_checkpoint(
            model,
            optimizer,
            linear_warmup,
            cos_decay,
            ep,
            most_recent_model_path,
            best_valid_loss,
        )

        np.savez(
            os.path.join(
                file_path,
                "training_history.npz",
            ),
            train_loss=np.asarray(
                train_error
            ),
            valid_loss=np.asarray(
                valid_error
            ),
            train_geodesic=np.asarray(
                train_rotation_error
            ),
            valid_geodesic=np.asarray(
                valid_rotation_error
            ),
            train_mirror_bce=np.asarray(
                train_mirror_error
            ),
            valid_mirror_bce=np.asarray(
                valid_mirror_error
            ),
            train_mirror_accuracy=np.asarray(
                train_mirror_accuracy
            ),
            valid_mirror_accuracy=np.asarray(
                valid_mirror_accuracy
            ),
        )

    return (
        train_error,
        valid_error,
    )
