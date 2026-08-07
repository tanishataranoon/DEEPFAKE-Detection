"""
=========================================================
Dataset Split Leakage Checker
=========================================================

Checks whether the same original video appears in
multiple dataset splits.

Expected output:

Train videos : XXXXX
Val videos   : XXXXX
Test videos  : XXXXX

Train-Val overlap   : 0
Train-Test overlap  : 0
Val-Test overlap    : 0

✓ No video leakage detected
=========================================================
"""

import pandas as pd


# =========================================================
# Configuration
# =========================================================

TRAIN_CSV = "data/split/train.csv"
VAL_CSV = "data/split/val.csv"
TEST_CSV = "data/split/test.csv"


# =========================================================
# Main
# =========================================================

def check_split():

    # -----------------------------------------------------
    # Load splits
    # -----------------------------------------------------

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)

    # -----------------------------------------------------
    # Get unique videos
    # -----------------------------------------------------

    train_videos = set(train_df["video_name"].unique())
    val_videos = set(val_df["video_name"].unique())
    test_videos = set(test_df["video_name"].unique())

    # -----------------------------------------------------
    # Find overlaps
    # -----------------------------------------------------

    train_val_overlap = train_videos & val_videos
    train_test_overlap = train_videos & test_videos
    val_test_overlap = val_videos & test_videos

    # -----------------------------------------------------
    # Print summary
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("DATASET SPLIT LEAKAGE CHECK")
    print("=" * 60)

    print()

    print(f"Train videos : {len(train_videos)}")
    print(f"Val videos   : {len(val_videos)}")
    print(f"Test videos  : {len(test_videos)}")

    print()

    print(f"Train-Val overlap   : {len(train_val_overlap)}")
    print(f"Train-Test overlap  : {len(train_test_overlap)}")
    print(f"Val-Test overlap    : {len(val_test_overlap)}")

    # -----------------------------------------------------
    # Safety check
    # -----------------------------------------------------

    leakage = (
        len(train_val_overlap)
        + len(train_test_overlap)
        + len(val_test_overlap)
    )

    print()

    if leakage == 0:

        print("✓ No video leakage detected")

    else:

        print("✗ VIDEO LEAKAGE DETECTED")

        print()

        if train_val_overlap:
            print("Train ↔ Val:")
            for video in sorted(train_val_overlap):
                print(f"  {video}")

        if train_test_overlap:
            print("Train ↔ Test:")
            for video in sorted(train_test_overlap):
                print(f"  {video}")

        if val_test_overlap:
            print("Val ↔ Test:")
            for video in sorted(val_test_overlap):
                print(f"  {video}")

        raise AssertionError(
            "Dataset split contains video-level leakage!"
        )

    print()


# =========================================================
# Entry Point
# =========================================================

if __name__ == "__main__":
    check_split()