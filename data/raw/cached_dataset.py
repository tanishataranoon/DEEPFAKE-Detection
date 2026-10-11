"""
Returns:
    clip  : uint8 tensor [T, 3, 224, 224]   (NOT divided by 255 - the trainer does that on the GPU)
    label : long tensor
"""
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CachedClipDataset(Dataset):
    def __init__(self, csv_file, cache_dir):
        self.df = pd.read_csv(csv_file)
        self.cache_dir = cache_dir
        lab = self.df["label"].astype(str).str.lower().str.strip().map({"real": 0, "fake": 1, "0": 0, "1": 1})
        if lab.isna().any():
            raise ValueError("unknown labels in CSV")
        self.labels = lab.to_numpy(dtype=np.int64)

        missing = [i for i in range(min(len(self.df), 200)) if not os.path.exists(self._path(i))]
        if missing:
            raise FileNotFoundError(f"cache incomplete in {cache_dir}, e.g. row {missing[0]}. Run cache_clips.py first.")
        print(f"CachedClipDataset: {len(self.df):,} clips from {cache_dir}")

    def _path(self, i):
        return os.path.join(self.cache_dir, f"{i:06d}.npy")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        x = np.load(self._path(i))                       # [T,H,W,3] uint8
        x = torch.from_numpy(x).permute(0, 3, 1, 2)      # [T,3,H,W] uint8
        return x, torch.tensor(self.labels[i], dtype=torch.long)
