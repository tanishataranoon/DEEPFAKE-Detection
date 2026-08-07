"""
==========================================================
Configuration File
----------------------------------------------------------
This file stores all paths and parameters used throughout
the DeepFake Detection project.

If you need to change a directory or parameter,
change it here instead of editing multiple files.
==========================================================
"""

from pathlib import Path

# ==========================================================
# Project Root
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

# ==========================================================
# Data Directories
# ==========================================================

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

REAL_DATA_DIR = RAW_DATA_DIR / "DFD_original sequences"

FAKE_DATA_DIR = RAW_DATA_DIR / "DFD_manipulated sequences"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

CLIPS_DIR = PROCESSED_DATA_DIR / "clips"

METADATA_FILE = PROCESSED_DATA_DIR / "metadata.csv"

SPLIT_DIR = DATA_DIR / "split"

TRAIN_CSV = SPLIT_DIR / "train.csv"

VAL_CSV = SPLIT_DIR / "val.csv"

TEST_CSV = SPLIT_DIR / "test.csv"

# ==========================================================
# Model Directory
# ==========================================================

MODELS_DIR = PROJECT_ROOT / "models"

WEIGHTS_DIR = PROJECT_ROOT / "weights"

YOLO_MODEL = WEIGHTS_DIR / "model.pt"

# ==========================================================
# Detection Parameters
# ==========================================================

CONFIDENCE_THRESHOLD = 0.10

IMAGE_SIZE = 640

DEVICE = "cuda" # automatically changed later if needed

# ==========================================================
# Video Processing
# ==========================================================

FRAME_SKIP = 1

CLIP_LENGTH = 16

FACE_SIZE = 224

# ==========================================================
# Tracking
# ==========================================================

MAX_TRACK_LOST = 30

# ==========================================================
# Random Seed
# ==========================================================

RANDOM_SEED = 42

#======================================# Evalution
IOU_THRESHOLD=0.50
SAVE_VISUALIZATION = True