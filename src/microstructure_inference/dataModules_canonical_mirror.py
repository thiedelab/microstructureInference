#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataset for canonical-orientation + binary mirror-label prediction.

All arrays are opened with NumPy memory mapping in read-only mode.
Only the requested sample is copied into writable PyTorch tensor memory.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class Dataset_canonicalOrientation_and_mirror(Dataset):
    """
    Parameters
    ----------
    input_file : str
        Padded Bragg-disk token array with shape (N, S, 3).

    rot_file : str
        Canonical orientation matrices with shape (N, 3, 3).

    mir_file : str
        Binary mirror labels with shape (N,) or (N, 1).

    transform : callable, optional
        Augmentation applied only to the Bragg-disk token tensor.
    """

    def __init__(
        self,
        input_file,
        rot_file,
        mir_file,
        transform=None,
    ):
        # Keep all large arrays memory-mapped and read-only.
        self.data = np.load(
            input_file,
            mmap_mode="r",
        )
        self.target_rot = np.load(
            rot_file,
            mmap_mode="r",
        )
        self.target_mir = np.load(
            mir_file,
            mmap_mode="r",
        )

        self.transform = transform

        n = len(self.data)

        if len(self.target_rot) != n:
            raise ValueError(
                "Input and canonical-orientation arrays have different "
                f"lengths: {n} versus {len(self.target_rot)}."
            )

        if len(self.target_mir) != n:
            raise ValueError(
                "Input and mirror-label arrays have different "
                f"lengths: {n} versus {len(self.target_mir)}."
            )

        if self.target_rot.shape[1:] != (3, 3):
            raise ValueError(
                "Canonical orientation labels must have shape (N, 3, 3). "
                f"Received {self.target_rot.shape}."
            )

    def __getitem__(self, index):
        """
        Read only one sample from each memory-mapped array.

        torch.tensor(...) intentionally makes a small per-sample copy.
        This avoids sharing read-only NumPy mmap memory with PyTorch and
        prevents the non-writable-array warning.
        """

        x = torch.tensor(
            self.data[index],
            dtype=torch.int64,
        )

        y_rot = torch.tensor(
            self.target_rot[index],
            dtype=torch.float32,
        )

        y_mir = torch.tensor(
            self.target_mir[index],
            dtype=torch.float32,
        ).reshape(1)

        if self.transform is not None:
            x = self.transform(x)

        return x, y_rot, y_mir

    def __len__(self):
        return len(self.data)


# Optional CamelCase alias.
DatasetCanonicalOrientationAndMirror = (
    Dataset_canonicalOrientation_and_mirror
)
