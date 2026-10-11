"""
=========================================================
TemporalDeepfakeDataset
---------------------------------------------------------
Same clip-reconstruction logic as the Appearance Branch's
DeepfakeDataset (data/raw/appearanch/dataset.py). The only
difference: augmentation is computed ONCE per clip and
replayed identically on all 16 frames (see transforms.py).
=========================================================
"""

import os
import ast

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .transforms import get_temporal_transforms


class TemporalDeepfakeDataset(Dataset):

    def __init__(self, metadata_csv, dataset_root, split="train", image_size=224):
        self.df = pd.read_csv(metadata_csv)
        self.dataset_root = dataset_root
        self.split = split
        self.image_size = image_size

        self.label_map = {"real": 0, "fake": 1}

        self.video_dirs = {
            "real": "DFD_original sequences",
            "fake": os.path.join(
                "DFD_manipulated_sequences", "DFD_manipulated_sequences"
            ),
        }

        self.transform = get_temporal_transforms(split, image_size)
        self.is_train = split == "train"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        video_name = row["video_name"]
        label = row["label"]

        video_path = self._get_video_path(video_name, label)

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found:\n{video_path}")

        frame_ids = self._parse_frame_ids(row["frame_ids"])
        bboxes = self._parse_bboxes(row["bboxes"])

        if len(frame_ids) != len(bboxes):
            raise ValueError(
                f"frame_ids/bboxes length mismatch for {video_name} "
                f"track {row['track_id']} clip {row['clip_id']}: "
                f"{len(frame_ids)} vs {len(bboxes)}"
            )

        frames = self._extract_clip(video_path, frame_ids, bboxes)
        clip_tensor = torch.stack(frames, dim=0)  # (T, C, H, W)

        return {
            "clip": clip_tensor,
            "label": self.label_map[label],
            "video_name": video_name,
            "track_id": int(row["track_id"]),
            "clip_id": int(row["clip_id"]),
        }

    def _get_video_path(self, video_name, label):
        folder = self.video_dirs[label]
        return os.path.join(self.dataset_root, folder, video_name)

    def _parse_frame_ids(self, frame_ids_str):
        return [int(f) for f in str(frame_ids_str).split(",")]

    def _parse_bboxes(self, bboxes_str):
        return ast.literal_eval(bboxes_str)

    def _extract_clip(self, video_path, frame_ids, bboxes):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        start_pos = frame_ids[0] - 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_pos)

        raw_faces = []
        target_idx = 0
        current_pos = start_pos

        while target_idx < len(frame_ids):
            ret, frame = cap.read()
            if not ret:
                break

            target_pos = frame_ids[target_idx] - 1
            if current_pos == target_pos:
                bbox = bboxes[target_idx]
                face = self._crop_face(frame, bbox)
                raw_faces.append(face)
                target_idx += 1

            current_pos += 1

        cap.release()

        if len(raw_faces) == 0:
            raw_faces = [
                np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
                for _ in frame_ids
            ]
        while len(raw_faces) < len(frame_ids):
            raw_faces.append(raw_faces[-1].copy())

        return self._apply_consistent_augmentation(raw_faces)

    def _apply_consistent_augmentation(self, raw_faces):
        """
        Train split: transform the first frame, capture the random
        parameters it used, then REPLAY those exact parameters on
        every other frame in the clip.
        Val/test: transform is already deterministic, no replay needed.
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

    def _crop_face(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))

        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            face = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        return face

    def get_labels(self):
        return self.df["label"].map(self.label_map).values


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = TemporalDeepfakeDataset(
        metadata_csv="data/split/test_150/train.csv",
        dataset_root="data/raw",
        split="train",
        image_size=224,
    )

    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Clip shape : {sample['clip'].shape}")
    print(f"Label      : {sample['label']}")

    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    print(f"Batch clip shape : {batch['clip'].shape}")
    print(f"Batch labels     : {batch['label']}")