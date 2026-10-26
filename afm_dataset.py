#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
afm_dataset.py

Turns the output of sweep.py (data/data_table.csv plus the clean/ and noise/
tiff stacks) into (noisy, clean) image pairs that PyTorch can train on.

One training sample = one frame: a noisy 64x64 image (input) and the clean
64x64 image it came from (target). Each row of data_table.csv is one noisy
stack of n_frames frames, so the dataset has (rows x n_frames) samples.

Splitting into train / val / test (split_table_ratio) is done on whole
groups, never on single frames: frames of one stack differ only by a 0.1 nm
jitter, so a frame-level split would put near copies of training images in
the test set.

@author: lukethomas
"""

from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import torch
from torch.utils.data import Dataset

from tools import add_noise


def load_table(data_root):
    """Read data_table.csv written by sweep.py. Without index_col the
    'protein' index sweep.py saved comes back as a normal column, which is
    what we want so we can filter on it."""
    return pd.read_csv(Path(data_root) / "data_table.csv")


def split_table(table, split_by, val_values):
    """
    Split the table into train and validation rows.

    split_by   : column name to split on ('protein', 'tip_radius', ...)
    val_values : values of that column that go to validation; the rest train.
    """
    is_val = table[split_by].isin(val_values)
    train = table[~is_val].reset_index(drop=True)
    val = table[is_val].reset_index(drop=True)
    if len(train) == 0 or len(val) == 0:
        raise ValueError(f"Split on {split_by}={val_values} leaves an empty "
                         f"set. Values available: {sorted(table[split_by].unique())}")
    return train, val


def split_table_ratio(table, group_by="clean_path", ratios=(0.70, 0.15, 0.15),
                      seed=0):
    """
    Split into train / validation / test by WHOLE GROUPS, aiming for `ratios`.

    Why groups and not individual frames: frames in one stack differ only by
    a 0.1 nm jitter, and every noise level of a stack shares the same clean
    image. If those were scattered across train and test, the test set would
    contain near copies of training targets and the score would be too good
    to be true (data leakage). So everything that shares a group goes to the
    same set.

    group_by:
    - "clean_path" (now): a group is one protein + tip radius + cone angle,
      with all its noise levels and frames
    - "protein" (later, with several proteins): test on unseen molecules

    Groups are shuffled with `seed` (same seed -> same split every run), then
    at least one group goes to val and one to test. With few groups the ratio
    can only be approximate: 6 groups -> 4/1/1 = 67/17/17.

    Returns (train, val, test) tables.
    """
    groups = np.array(sorted(table[group_by].unique()))
    n = len(groups)
    if n < 3:
        raise ValueError(f"Need at least 3 '{group_by}' groups for a "
                         f"train/val/test split, found {n}.")
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)

    n_val = max(1, round(ratios[1] * n))
    n_test = max(1, round(ratios[2] * n))
    n_train = n - n_val - n_test

    train_g = groups[:n_train]
    val_g = groups[n_train:n_train + n_val]
    test_g = groups[n_train + n_val:]

    pick = lambda g: table[table[group_by].isin(g)].reset_index(drop=True)
    return pick(train_g), pick(val_g), pick(test_g)


class AFMDenoiseDataset(Dataset):
    """
    PyTorch Dataset of (noisy, clean) frame pairs.

    Parameters
    ----------
    table : the rows (from data_table.csv) to use
    data_root : folder that the paths in the table are relative to ('data/')
    scale : number of nm that maps to 1.0. Both noisy and clean are divided
            by it. Neural nets train best on values around 0-1, and using one
            fixed scale (not per-image normalisation) keeps the physical
            height information, so outputs convert straight back to nm.
    fresh_noise : how the noise is made.
            False (validation / test): the FIXED noise for each row, rebuilt
            from its noise_seed with tools.add_noise. Same images every run,
            so scores are repeatable, and identical to what export_noisy.py
            saves if you want to look at them.
            True (training): draw NEW Gaussian noise with the same rms each
            time a frame is used. Each epoch then sees different noise on the
            same molecules, which is effectively unlimited training data and
            stops the network memorising one fixed noise pattern. Same noise
            model (clean + randn * noise), so nothing about the physics
            changes.
    augment : random flips and 90 degree rotations. A rotated AFM image is
            still a valid AFM image (with the tip approximated as symmetric),
            so this gives 8x more variety for free. Training only.
    """

    def __init__(self, table, data_root, scale, fresh_noise=False,
                 augment=False, seed=0):
        self.data_root = Path(data_root)
        self.scale = float(scale)
        self.fresh_noise = fresh_noise
        self.augment = augment
        self.rng = np.random.default_rng(seed)

        noisy_frames, clean_frames, sigmas, rows = [], [], [], []
        clean_cache = {}   # several noise levels share one clean stack

        for i, row in table.iterrows():
            cpath = self.data_root / row["clean_path"]
            if cpath not in clean_cache:
                clean_cache[cpath] = tifffile.imread(cpath).astype(np.float32)
            clean = clean_cache[cpath]
            # fixed, seeded noise for this row (nothing is read from disk)
            noisy = add_noise(clean, row["noise"], int(row["noise_seed"]))

            # a 2D tiff (single frame) comes back as (H, W); make it (1, H, W)
            if clean.ndim == 2:
                clean, noisy = clean[None], noisy[None]

            for f in range(clean.shape[0]):
                clean_frames.append(clean[f])
                noisy_frames.append(noisy[f])
                sigmas.append(row["noise"])
                rows.append(i)

        # (N, H, W) arrays, already divided by the scale
        self.clean = np.stack(clean_frames) / self.scale
        self.noisy = np.stack(noisy_frames) / self.scale
        self.sigma = np.array(sigmas, dtype=np.float32)  # rms noise in nm
        self.row = np.array(rows)                        # which table row

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, idx):
        clean = self.clean[idx]

        if self.fresh_noise:
            # new noise each call, same rms as this row (sigma is in nm)
            noise = self.rng.standard_normal(clean.shape).astype(np.float32)
            noisy = clean + noise * (self.sigma[idx] / self.scale)
        else:
            noisy = self.noisy[idx]

        if self.augment:
            # apply the SAME flip/rotation to input and target
            k = self.rng.integers(4)
            noisy, clean = np.rot90(noisy, k), np.rot90(clean, k)
            if self.rng.random() < 0.5:
                noisy, clean = noisy[:, ::-1], clean[:, ::-1]

        # add the channel dimension: (H, W) -> (1, H, W).
        # np.ascontiguousarray because rot90/flip give views torch can't take
        noisy = torch.from_numpy(np.ascontiguousarray(noisy))[None]
        clean = torch.from_numpy(np.ascontiguousarray(clean))[None]
        return noisy, clean, idx
