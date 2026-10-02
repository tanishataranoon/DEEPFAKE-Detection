"""
=========================================================
Data Structures
---------------------------------------------------------
Shared data classes used throughout the preprocessing
pipeline.
=========================================================
"""

from dataclasses import dataclass
from typing import List
import numpy as np


# =========================================================
# Frame
# =========================================================

@dataclass
class FrameData:
    """
    One frame extracted from a video.
    """

    frame_id: int
    image: np.ndarray


# =========================================================
# Face Detection
# =========================================================

@dataclass
class Detection:
    """
    YOLO face detection.
    """

    bbox: List[float]          # [x1,y1,x2,y2]
    confidence: float


# =========================================================
# Face Track
# =========================================================

@dataclass
class Track:
    """
    ByteTrack result.
    """

    track_id: int
    frame_id: int
    bbox: List[float]
    confidence: float


# =========================================================
# Cropped Face
# =========================================================

@dataclass
class FaceCrop:
    """
    Metadata of one tracked face in one frame.
    """

    track_id: int

    frame_id: int

    bbox: List[float]

    confidence: float

    face_width: int

    face_height: int


# =========================================================
# Face Clip
# =========================================================

@dataclass
class FaceClip:

    track_id: int

    clip_id: int

    video_name: str

    label: str

    start_frame: int

    end_frame: int

    frame_ids: List[int]

    bboxes: List[List[float]]

    num_frames: int

    confidences: List[float]

    face_width: int

    face_height: int
    fps: float
    clip_duration: float

    video_duration: float

# =========================================================
# Metadata Row
# (Future)
# =========================================================
