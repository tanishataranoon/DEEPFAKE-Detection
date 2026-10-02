
"""
=========================================================
Deepfake Dataset
---------------------------------------------------------
Loads 16-frame face clips from the metadata CSV.

CSV columns used:
    video_name
    label
    frame_ids
    bboxes

Dataset output:
    frames : [T, 3, 224, 224]
    label  : int
        0 = Real
        1 = Fake

Raw video layout:
    data/raw/
        DFD_original sequences/
            *.mp4

        DFD_manipulated_sequences/
            DFD_manipulated_sequences/
                *.mp4
=========================================================
"""

from pathlib import Path
import ast

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class DeepfakeDataset(Dataset):
    """
    PyTorch dataset for the appearance branch.

    Each CSV row represents one 16-frame face clip.
    Faces are cropped using the stored bounding boxes and
    resized to 224x224.
    """

    def __init__(
        self,
        csv_file=None,
        video_root="data/raw",
        image_size=224,
        num_frames=16,
        metadata_csv=None,
    ):
        # Support both names:
        #   csv_file
        #   metadata_csv
        if metadata_csv is not None:
            csv_file = metadata_csv

        if csv_file is None:
            raise ValueError(
                "Either 'csv_file' or 'metadata_csv' must be provided."
            )
        self.csv_file = Path(csv_file)
        self.video_root = Path(video_root)
        self.image_size = image_size
        self.num_frames = num_frames

        if not self.csv_file.exists():
            raise FileNotFoundError(
                f"CSV file not found: {self.csv_file.resolve()}"
            )

        if not self.video_root.exists():
            raise FileNotFoundError(
                f"Video root not found: {self.video_root.resolve()}"
            )

        self.df = pd.read_csv(self.csv_file)

        required_columns = [
            "video_name",
            "label",
            "frame_ids",
            "bboxes",
        ]

        missing = [
            col for col in required_columns
            if col not in self.df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required CSV columns: {missing}"
            )

        # -------------------------------------------------
        # Labels
        # -------------------------------------------------

        self.label_map = {
            "real": 0,
            "fake": 1,
        }

        # -------------------------------------------------
        # Normalize label values
        # -------------------------------------------------

        self.df["label"] = (
            self.df["label"]
            .astype(str)
            .str.lower()
            .str.strip()
        )

        unknown_labels = set(self.df["label"]) - set(self.label_map)

        if unknown_labels:
            raise ValueError(
                f"Unknown labels found in CSV: {unknown_labels}"
            )

        print("=" * 60)
        print("DeepfakeDataset")
        print("=" * 60)
        print("CSV          :", self.csv_file)
        print("Video root   :", self.video_root)
        print("Samples      :", len(self.df))
        print(
            "Real samples :",
            int((self.df["label"] == "real").sum())
        )
        print(
            "Fake samples :",
            int((self.df["label"] == "fake").sum())
        )
        print("Frame count  :", self.num_frames)
        print("Image size   :", f"{self.image_size}x{self.image_size}")
        print("=" * 60)

    # =====================================================
    # LENGTH
    # =====================================================

    def __len__(self):
        return len(self.df)

    # =====================================================
    # VIDEO PATH
    # =====================================================

    def _find_video(self, video_name, label):
        """
        Find the video in the appropriate DFD folder.
        """

        if label == "real":
            candidates = [
                self.video_root
                / "DFD_original sequences"
                / video_name,

                self.video_root
                / video_name,
            ]

        else:
            candidates = [
                self.video_root
                / "DFD_manipulated_sequences"
                / "DFD_manipulated_sequences"
                / video_name,

                self.video_root
                / "DFD_manipulated_sequences"
                / video_name,

                self.video_root
                / video_name,
            ]

        for path in candidates:
            if path.exists():
                return path

        # -------------------------------------------------
        # Last-resort recursive search.
        # This is only used when the normal paths fail.
        # -------------------------------------------------

        matches = list(
            self.video_root.rglob(video_name)
        )

        if matches:
            return matches[0]

        raise FileNotFoundError(
            "\nVideo not found.\n"
            f"Video      : {video_name}\n"
            f"Label      : {label}\n"
            f"Video root : {self.video_root.resolve()}\n"
        )

    # =====================================================
    # PARSE FRAME IDS
    # =====================================================

    @staticmethod
    def _parse_frame_ids(value):
        """
        Convert:

            '1,2,3,...,16'

        into:

            [1,2,3,...,16]
        """

        if isinstance(value, list):
            return [int(x) for x in value]

        if pd.isna(value):
            raise ValueError("frame_ids is empty")

        return [
            int(x.strip())
            for x in str(value).split(",")
            if x.strip()
        ]

    # =====================================================
    # PARSE BOUNDING BOXES
    # =====================================================

    @staticmethod
    def _parse_bboxes(value):
        """
        Convert the CSV string representation of bboxes
        into a list of [x1,y1,x2,y2].
        """

        if isinstance(value, list):
            return value

        if pd.isna(value):
            raise ValueError("bboxes is empty")

        parsed = ast.literal_eval(str(value))

        return parsed

    # =====================================================
    # LOAD ONE FRAME
    # =====================================================

    def _read_frame(self, cap, frame_id, bbox):
        """
        Read one frame and crop the detected face.

        CSV frame IDs are 1-based.
        OpenCV frame positions are 0-based.

        Therefore:
            OpenCV position = frame_id - 1
        """

        frame_position = int(frame_id) - 1

        if frame_position < 0:
            raise ValueError(
                f"Invalid frame_id: {frame_id}"
            )

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)

        success, frame = cap.read()

        if not success or frame is None:
            raise RuntimeError(
                f"Could not read frame {frame_id}"
            )

        # OpenCV: BGR -> RGB
        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        height, width = frame.shape[:2]

        # -------------------------------------------------
        # Bounding box
        # -------------------------------------------------

        x1, y1, x2, y2 = [
            float(v) for v in bbox
        ]

        # Clamp bbox to image boundaries
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))

        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))

        if x2 <= x1 or y2 <= y1:
            raise RuntimeError(
                f"Invalid bounding box for frame {frame_id}: "
                f"{bbox}"
            )

        face = frame[y1:y2, x1:x2]

        if face.size == 0:
            raise RuntimeError(
                f"Empty face crop for frame {frame_id}"
            )

        # -------------------------------------------------
        # Resize to AIMv2 input size
        # -------------------------------------------------

        face = cv2.resize(
            face,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )

        return face

    # =====================================================
    # GET ITEM
    # =====================================================

    def __getitem__(self, index):

        row = self.df.iloc[index]

        video_name = str(row["video_name"])

        label_name = str(row["label"]).lower().strip()

        label = self.label_map[label_name]

        frame_ids = self._parse_frame_ids(
            row["frame_ids"]
        )

        bboxes = self._parse_bboxes(
            row["bboxes"]
        )

        if len(frame_ids) != len(bboxes):
            raise RuntimeError(
                f"Frame/bbox mismatch for {video_name}, "
                f"clip {row.get('clip_id', index)}: "
                f"{len(frame_ids)} frames vs "
                f"{len(bboxes)} bboxes"
            )

        if len(frame_ids) != self.num_frames:
            raise RuntimeError(
                f"Expected {self.num_frames} frames but got "
                f"{len(frame_ids)} for {video_name}, "
                f"clip {row.get('clip_id', index)}"
            )

        # -------------------------------------------------
        # Locate video
        # -------------------------------------------------

        video_path = self._find_video(
            video_name,
            label_name
        )

        cap = cv2.VideoCapture(
            str(video_path)
        )

        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open video:\n{video_path}"
            )

        frames = []

        try:
            for frame_id, bbox in zip(
                frame_ids,
                bboxes
            ):
                face = self._read_frame(
                    cap,
                    frame_id,
                    bbox
                )

                frames.append(face)

        finally:
            cap.release()

        # -------------------------------------------------
        # Convert:
        #
        # [T, H, W, C]
        #
        # to:
        #
        # [T, C, H, W]
        # -------------------------------------------------

        frames = np.stack(
            frames,
            axis=0
        )

        frames = torch.from_numpy(
            frames.copy()
        ).float()

        frames = frames.permute(
            0, 3, 1, 2
        )

        # -------------------------------------------------
        # Normalize pixel values
        #
        # AIMv2/vision models expect floating point
        # image tensors.
        # -------------------------------------------------

        frames = frames / 255.0

        return frames, torch.tensor(
            label,
            dtype=torch.long
        )

