"""
=========================================================
Dataset Verification Script
---------------------------------------------------------
Checks:
1. Unique videos
2. Train / Val / Test counts
3. Class distribution
4. Data leakage
=========================================================
"""

import os
import re
import pandas as pd


# -------------------------------------------------------
# Change this if needed
# -------------------------------------------------------

PROCESSED = "data/processed/metadata.csv"

TRAIN = "data/split/train.csv"
VAL = "data/split/val.csv"
TEST = "data/split/test.csv"


# -------------------------------------------------------
# Same function used by DatasetSplitter
# -------------------------------------------------------

def get_source_group(video_name):

    name = video_name.replace(".mp4", "")

    # Original video
    if re.match(r"^\d+__", name):
        if name.count("__") == 1:
            return name.split("__", 1)[1]

    # Fake video
    parts = name.split("__")

    if len(parts) >= 3:
        return parts[1]

    return name


# =======================================================
# 1. Check metadata
# =======================================================

print("=" * 60)
print("CHECK 1 : METADATA")
print("=" * 60)

metadata = pd.read_csv(PROCESSED)

unique_videos = metadata["video_name"].nunique()

print(f"Metadata Rows    : {len(metadata):,}")
print(f"Unique Videos    : {unique_videos}")

print()


# =======================================================
# 2. Read splits
# =======================================================

train = pd.read_csv(TRAIN)
val = pd.read_csv(VAL)
test = pd.read_csv(TEST)

train_videos = set(train["video_name"].unique())
val_videos = set(val["video_name"].unique())
test_videos = set(test["video_name"].unique())

print("=" * 60)
print("CHECK 2 : VIDEO SPLITS")
print("=" * 60)

print(f"Train Videos : {len(train_videos)}")
print(f"Val Videos   : {len(val_videos)}")
print(f"Test Videos  : {len(test_videos)}")
print()

print(
    f"Total Videos : {len(train_videos)+len(val_videos)+len(test_videos)}"
)

print()


# =======================================================
# 3. Class distribution
# =======================================================

print("=" * 60)
print("CHECK 3 : CLASS DISTRIBUTION")
print("=" * 60)

for name, df in zip(
    ["Train", "Validation", "Test"],
    [train, val, test]
):

    counts = (
        df[["video_name", "label"]]
        .drop_duplicates()["label"]
        .value_counts()
    )

    print(name)

    print(counts)

    print()


# =======================================================
# 4. Leakage check
# =======================================================

print("=" * 60)
print("CHECK 4 : SOURCE LEAKAGE")
print("=" * 60)

train_groups = set(
    map(get_source_group, train_videos)
)

val_groups = set(
    map(get_source_group, val_videos)
)

test_groups = set(
    map(get_source_group, test_videos)
)

train_val = train_groups & val_groups
train_test = train_groups & test_groups
val_test = val_groups & test_groups

if len(train_val) == 0 \
   and len(train_test) == 0 \
   and len(val_test) == 0:

    print("✓ NO DATA LEAKAGE DETECTED")

else:

    print("✗ DATA LEAKAGE FOUND")

    print()

    print("Train-Val :", len(train_val))
    print("Train-Test:", len(train_test))
    print("Val-Test  :", len(val_test))

print()


# =======================================================
# 5. Clip counts
# =======================================================

print("=" * 60)
print("CHECK 5 : CLIPS")
print("=" * 60)

print(f"Train Clips : {len(train):,}")
print(f"Val Clips   : {len(val):,}")
print(f"Test Clips  : {len(test):,}")

print()

print(f"Total Clips : {len(train)+len(val)+len(test):,}")

print()

print("=" * 60)
print("CHECK COMPLETE")
print("=" * 60)