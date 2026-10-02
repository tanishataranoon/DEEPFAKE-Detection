"""
=========================================================
Leakage-Free Group-Level Dataset Splitter
---------------------------------------------------------
Every original video and ALL manipulated versions derived
from it are placed into the same split.

Output:
    train.csv
    val.csv
    test.csv
=========================================================
"""

import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split


# =========================================================
# Group Builder
# =========================================================

def get_source_group(video_name):
    """
    Convert every filename into its original source video.

    Examples
    --------
    Real:
        01__podium_speech_happy.mp4
            -> 01__podium_speech_happy

    Fake:
        28_01__podium_speech_happy__ABC123.mp4
            -> 01__podium_speech_happy
    """

    name = os.path.splitext(video_name)[0]

    # ------------------------------
    # Real video
    # ------------------------------

    m = re.match(r"^(\d+)__(.+)$", name)

    if m:
        return f"{m.group(1)}__{m.group(2)}"

    # ------------------------------
    # Fake video
    # ------------------------------

    m = re.match(r"^\d+_(\d+)__(.+?)__[^_]+$", name)

    if m:
        source_id = m.group(1)
        scenario = m.group(2)
        return f"{source_id}__{scenario}"

    raise ValueError(f"Unknown filename format: {video_name}")


# =========================================================
# Splitter
# =========================================================

class DatasetSplitter:

    def __init__(
        self,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
    ):

        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state

    # =====================================================
    # Split
    # =====================================================

    def split(self, metadata_csv, output_dir):

        print("\nGenerating dataset splits...\n")

        df = pd.read_csv(metadata_csv)

        # ---------------------------------------------
        # Build source group for EVERY clip
        # ---------------------------------------------

        df["group"] = df["video_name"].apply(get_source_group)

        # ---------------------------------------------
        # One unique row per source group
        # ---------------------------------------------

        groups = (
            df["group"]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        print(f"Unique Groups : {len(groups)}")

        # ---------------------------------------------
        # Split groups ONLY
        # ---------------------------------------------

        train_groups, temp_groups = train_test_split(
            groups,
            train_size=self.train_ratio,
            random_state=self.random_state,
            shuffle=True,
        )

        val_size = self.val_ratio / (
            self.val_ratio + self.test_ratio
        )

        val_groups, test_groups = train_test_split(
            temp_groups,
            train_size=val_size,
            random_state=self.random_state,
            shuffle=True,
        )

        train_groups = set(train_groups)
        val_groups = set(val_groups)
        test_groups = set(test_groups)

        # ---------------------------------------------
        # Safety checks
        # ---------------------------------------------

        assert len(train_groups & val_groups) == 0
        assert len(train_groups & test_groups) == 0
        assert len(val_groups & test_groups) == 0

        # ---------------------------------------------
        # Assign clips
        # ---------------------------------------------

        train_df = df[df["group"].isin(train_groups)].copy()

        val_df = df[df["group"].isin(val_groups)].copy()

        test_df = df[df["group"].isin(test_groups)].copy()

        # Remove helper column

        train_df.drop(columns=["group"], inplace=True)
        val_df.drop(columns=["group"], inplace=True)
        test_df.drop(columns=["group"], inplace=True)

        # ---------------------------------------------
        # Duplicate safety
        # ---------------------------------------------

        key = ["video_name", "track_id", "clip_id"]

        assert train_df.duplicated(key).sum() == 0
        assert val_df.duplicated(key).sum() == 0
        assert test_df.duplicated(key).sum() == 0

        total = (
            len(train_df)
            + len(val_df)
            + len(test_df)
        )

        assert total == len(df), (
            f"Split error! {total} != {len(df)}"
        )

        # ---------------------------------------------
        # Save
        # ---------------------------------------------

        os.makedirs(output_dir, exist_ok=True)

        train_df.to_csv(
            os.path.join(output_dir, "train.csv"),
            index=False,
        )

        val_df.to_csv(
            os.path.join(output_dir, "val.csv"),
            index=False,
        )

        test_df.to_csv(
            os.path.join(output_dir, "test.csv"),
            index=False,
        )

        # ---------------------------------------------
        # Summary
        # ---------------------------------------------

        print("=" * 60)
        print("GROUP-LEVEL DATASET SPLIT")
        print("=" * 60)

        print(f"Groups")
        print(f"  Train : {len(train_groups)}")
        print(f"  Val   : {len(val_groups)}")
        print(f"  Test  : {len(test_groups)}")

        print()

        print(f"Clips")
        print(f"  Train : {len(train_df):,}")
        print(f"  Val   : {len(val_df):,}")
        print(f"  Test  : {len(test_df):,}")
        print(f"  Total : {len(df):,}")

        print()

        print("Video counts")

        print("Train")
        print(train_df["video_name"].nunique())

        print("Validation")
        print(val_df["video_name"].nunique())

        print("Test")
        print(test_df["video_name"].nunique())

        print()

        print("Label distribution")

        print("\nTrain")
        print(train_df["label"].value_counts())

        print("\nValidation")
        print(val_df["label"].value_counts())

        print("\nTest")
        print(test_df["label"].value_counts())

        print("\nDataset split complete.")


# =========================================================
# Standalone
# =========================================================

if __name__ == "__main__":

    splitter = DatasetSplitter()

    splitter.split(
        metadata_csv="data/processed/metadata.csv",
        output_dir="data/split",
    )