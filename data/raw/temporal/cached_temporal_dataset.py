"""
=========================================================
Cached Temporal Dataset
---------------------------------------------------------
Reuses the SAME raw face-crop cache built for the
Appearance Branch (data/cache/train, val, test — built by
data/processed/build_all_cache.py). Those .npy files store
raw, un-normalized, uint8 16-frame crops: [T, H, W, 3].

This avoids re-decoding video entirely — the single biggest
bottleneck in the 150-group temporal test run. This dataset
applies the Temporal Branch's OWN split-aware transform
(ImageNet normalization, and for train, one random
augmentation replayed identically across all 16 frames —
see data/raw/temporal/transforms.py) on top of the cached
crops.

IMPORTANT ASSUMPTION — confirm before using:
    This assumes data/split/train.csv (etc.) are the SAME
    rows, in the SAME order, as whatever CSV was used to
    build the appearance cache. If the row order/content
    differs even slightly, index i's cached crop will NOT
    match index i's label/bbox here, and this will silently
    train on mismatched data. Verify row counts match (see
    build_all_cache.py output) before trusting this for a
    real run.
=========================================================
"""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .transforms import get_temporal_transforms


class CachedTemporalClipDataset(Dataset):

    def __init__(self, csv_file, cache_dir, split="train", image_size=224):
        self.df = pd.read_csv(csv_file)
        self.cache_dir = cache_dir
        self.split = split
        self.image_size = image_size
        self.is_train = split == "train"

        self.label_map = {"real": 0, "fake": 1}
        self.transform = get_temporal_transforms(split, image_size)

        lab = (
            self.df["label"]
            .astype(str).str.lower().str.strip()
            .map(self.label_map)
        )
        if lab.isna().any():
            raise ValueError(f"Unknown labels in {csv_file}")
        self.labels = lab.to_numpy(dtype=np.int64)

        # Sanity check: confirm the cache exists before training starts,
        # rather than failing deep into epoch 1.
        missing = [
            i for i in range(min(len(self.df), 200))
            if not os.path.exists(self._path(i))
        ]
        if missing:
            raise FileNotFoundError(
                f"Cache incomplete or mismatched in {cache_dir}, "
                f"e.g. row {missing[0]} missing. Either run "
                f"data/processed/build_all_cache.py first, or confirm "
                f"this split's CSV matches the cached rows exactly."
            )

        print(f"CachedTemporalClipDataset ({split}): "
              f"{len(self.df):,} clips from {cache_dir}")

    def _path(self, i):
        return os.path.join(self.cache_dir, f"{i:06d}.npy")

    def __len__(self):
        return len(self.df)

    def _apply_consistent_augmentation(self, raw_faces):
        """
        Train: one random augmentation, replayed identically on
        every frame (prevents fake temporal discontinuities from
        per-frame-random augmentation — see transforms.py).
        Val/test: deterministic transform, no replay needed.
        """
        frames = []
        if self.is_train:
            first = self.transform(image=raw_faces[0])
            frames.append(first["image"])
            replay_params = first["replay"]
            for face in raw_faces[1:]:
                replayed = self.transform.replay(replay_params, image=face)
                frames.append(replayed["image"])
        else:
            for face in raw_faces:
                frames.append(self.transform(image=face)["image"])
        return frames

    def __getitem__(self, i):
        raw = np.load(self._path(i))  # [T, H, W, 3] uint8, RGB, already cropped

        if raw.ndim != 4 or raw.shape[-1] != 3:
            raise RuntimeError(
                f"Unexpected cache shape at row {i}: {raw.shape}. "
                f"Expected [T, H, W, 3]."
            )

        raw_faces = [raw[t] for t in range(raw.shape[0])]

        frames = self._apply_consistent_augmentation(raw_faces)
        clip = torch.stack(frames, dim=0)  # [T, 3, H, W], normalized float

        label = torch.tensor(self.labels[i], dtype=torch.long)
        return {"clip": clip, "label": label}

    def get_labels(self):
        return self.labels