"""
Create a 150-source-group leakage-free test dataset.

Purpose
-------
Create a small end-to-end experiment without re-running preprocessing.

Split:
    100 source groups -> train
     25 source groups -> validation
     25 source groups -> test

IMPORTANT
---------
The grouping unit is the ORIGINAL SOURCE VIDEO.

A source group can contain:
    - 1 real/original video
    - multiple fake/manipulated versions

The complete source group stays in exactly one split.

Only metadata rows are selected. No video/image preprocessing is performed.
"""

import os
import re
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

METADATA_CSV = "data/processed/metadata.csv"

OUTPUT_DIR = "data/split/test_150"

TRAIN_GROUPS = 100
VAL_GROUPS = 25
TEST_GROUPS = 25

RANDOM_STATE = 42


# ============================================================
# SOURCE GROUP BUILDER
# ============================================================

def get_source_group(video_name):
    """
    Convert a video filename into its original source group.

    Real:
        28__podium_speech_happy.mp4
        -> 28__podium_speech_happy

    Fake:
        05_28__podium_speech_happy__2DK5X0Y0.mp4
        -> 28__podium_speech_happy
    """

    name = os.path.splitext(
        os.path.basename(str(video_name))
    )[0]

    # Real / original video
    match = re.match(
        r"^(\d+)__(.+)$",
        name
    )

    if match:
        source_id = match.group(1)
        scenario = match.group(2)
        return f"{source_id}__{scenario}"

    # Fake / manipulated video
    match = re.match(
        r"^\d+_(\d+)__(.+?)__[^_]+$",
        name
    )

    if match:
        source_id = match.group(1)
        scenario = match.group(2)
        return f"{source_id}__{scenario}"

    raise ValueError(
        f"Unknown filename format: {video_name}"
    )


# ============================================================
# HELPERS
# ============================================================

def print_video_distribution(name, df):
    print()
    print(name)
    print("-" * 40)
    print(f"Unique videos : {df['video_name'].nunique():,}")
    print(
        f"Real videos   : "
        f"{df[df['label'] == 'real']['video_name'].nunique():,}"
    )
    print(
        f"Fake videos   : "
        f"{df[df['label'] == 'fake']['video_name'].nunique():,}"
    )


def print_label_distribution(name, df):
    print()
    print(name)
    print("-" * 40)
    print(
        df["label"]
        .value_counts()
        .sort_index()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("150-SOURCE-GROUP LEAKAGE-FREE TEST DATASET")
    print("=" * 70)

    print()
    print(f"Metadata : {METADATA_CSV}")
    print(f"Output   : {OUTPUT_DIR}")

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    if not os.path.exists(METADATA_CSV):
        raise FileNotFoundError(
            f"Metadata file not found:\n{METADATA_CSV}"
        )

    df = pd.read_csv(METADATA_CSV)

    print()
    print(f"Total metadata clips : {len(df):,}")

    # --------------------------------------------------------
    # Validate required columns
    # --------------------------------------------------------

    required_columns = [
        "video_name",
        "label",
        "track_id",
        "clip_id",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "Metadata is missing required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )

    # --------------------------------------------------------
    # Validate labels
    # --------------------------------------------------------

    valid_labels = {"real", "fake"}

    df["label"] = (
        df["label"]
        .astype(str)
        .str.lower()
    )

    actual_labels = set(
        df["label"].unique()
    )

    invalid_labels = actual_labels - valid_labels

    if invalid_labels:
        raise RuntimeError(
            "Unexpected labels found:\n"
            f"{invalid_labels}\n"
            f"Expected only: {valid_labels}"
        )

    # --------------------------------------------------------
    # Build source groups
    # --------------------------------------------------------

    print()
    print("Building source groups...")

    df["group"] = (
        df["video_name"]
        .apply(get_source_group)
    )

    total_groups = df["group"].nunique()

    print(
        f"Total source groups : "
        f"{total_groups:,}"
    )

    # --------------------------------------------------------
    # Only use groups containing a real source video
    # --------------------------------------------------------

    real_groups = (
        df.loc[
            df["label"] == "real",
            "group"
        ]
        .drop_duplicates()
        .tolist()
    )

    print(
        f"Source groups with real video : "
        f"{len(real_groups):,}"
    )

    groups_without_real = sorted(
        set(df["group"].unique())
        - set(real_groups)
    )

    if groups_without_real:
        print()
        print("WARNING:")
        print(
            f"{len(groups_without_real)} source groups "
            "contain no real video."
        )
        print("They will not be selected.")

    # --------------------------------------------------------
    # Check requested number
    # --------------------------------------------------------

    total_requested = (
        TRAIN_GROUPS
        + VAL_GROUPS
        + TEST_GROUPS
    )

    if total_requested > len(real_groups):
        raise RuntimeError(
            f"Requested {total_requested} source groups, "
            f"but only {len(real_groups)} groups containing "
            f"real videos are available."
        )

    # --------------------------------------------------------
    # Randomly select exactly 150 source groups
    # --------------------------------------------------------

    selected_groups = (
        pd.Series(real_groups)
        .sample(
            n=total_requested,
            random_state=RANDOM_STATE
        )
        .tolist()
    )

    # --------------------------------------------------------
    # Split exact counts
    # --------------------------------------------------------

    selected_groups = pd.Series(
        selected_groups
    ).sample(
        frac=1.0,
        random_state=RANDOM_STATE
    ).tolist()

    train_groups = set(
        selected_groups[:TRAIN_GROUPS]
    )

    val_groups = set(
        selected_groups[
            TRAIN_GROUPS:
            TRAIN_GROUPS + VAL_GROUPS
        ]
    )

    test_groups = set(
        selected_groups[
            TRAIN_GROUPS + VAL_GROUPS:
        ]
    )

    print()
    print(
        f"Selected source groups : "
        f"{len(selected_groups):,}"
    )
    print(f"Train groups            : {len(train_groups):,}")
    print(f"Validation groups       : {len(val_groups):,}")
    print(f"Test groups             : {len(test_groups):,}")

    # --------------------------------------------------------
    # Create selected dataframe
    # --------------------------------------------------------

    small_df = df[
        df["group"].isin(selected_groups)
    ].copy()

    # --------------------------------------------------------
    # Create split dataframes
    # --------------------------------------------------------

    train_df = small_df[
        small_df["group"].isin(train_groups)
    ].copy()

    val_df = small_df[
        small_df["group"].isin(val_groups)
    ].copy()

    test_df = small_df[
        small_df["group"].isin(test_groups)
    ].copy()

    # --------------------------------------------------------
    # Leakage checks
    # --------------------------------------------------------

    train_group_check = set(
        train_df["video_name"]
        .apply(get_source_group)
        .unique()
    )

    val_group_check = set(
        val_df["video_name"]
        .apply(get_source_group)
        .unique()
    )

    test_group_check = set(
        test_df["video_name"]
        .apply(get_source_group)
        .unique()
    )

    train_val_overlap = (
        train_group_check & val_group_check
    )

    train_test_overlap = (
        train_group_check & test_group_check
    )

    val_test_overlap = (
        val_group_check & test_group_check
    )

    if train_val_overlap:
        raise RuntimeError(
            "GROUP LEAKAGE DETECTED: Train/Validation overlap."
        )

    if train_test_overlap:
        raise RuntimeError(
            "GROUP LEAKAGE DETECTED: Train/Test overlap."
        )

    if val_test_overlap:
        raise RuntimeError(
            "GROUP LEAKAGE DETECTED: Validation/Test overlap."
        )

    # --------------------------------------------------------
    # Duplicate clip checks
    # --------------------------------------------------------

    key = [
        "video_name",
        "track_id",
        "clip_id",
    ]

    if train_df.duplicated(key).any():
        raise RuntimeError(
            "Duplicate clips detected in train."
        )

    if val_df.duplicated(key).any():
        raise RuntimeError(
            "Duplicate clips detected in validation."
        )

    if test_df.duplicated(key).any():
        raise RuntimeError(
            "Duplicate clips detected in test."
        )

    # --------------------------------------------------------
    # Row-count check
    # --------------------------------------------------------

    total_split_rows = (
        len(train_df)
        + len(val_df)
        + len(test_df)
    )

    if total_split_rows != len(small_df):
        raise RuntimeError(
            "Split row count mismatch!\n"
            f"Expected : {len(small_df):,}\n"
            f"Got      : {total_split_rows:,}"
        )

    # --------------------------------------------------------
    # Remove helper column
    # --------------------------------------------------------

    train_df.drop(
        columns=["group"],
        inplace=True
    )

    val_df.drop(
        columns=["group"],
        inplace=True
    )

    test_df.drop(
        columns=["group"],
        inplace=True
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    train_path = os.path.join(
        OUTPUT_DIR,
        "train.csv"
    )

    val_path = os.path.join(
        OUTPUT_DIR,
        "val.csv"
    )

    test_path = os.path.join(
        OUTPUT_DIR,
        "test.csv"
    )

    train_df.to_csv(
        train_path,
        index=False
    )

    val_df.to_csv(
        val_path,
        index=False
    )

    test_df.to_csv(
        test_path,
        index=False
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 70)
    print("150-SOURCE-GROUP DATASET CREATED")
    print("=" * 70)

    print()
    print("SOURCE GROUPS")
    print("-" * 40)
    print(f"Selected : {len(selected_groups):,}")
    print(f"Train    : {len(train_groups):,}")
    print(f"Val      : {len(val_groups):,}")
    print(f"Test     : {len(test_groups):,}")

    print()
    print("CLIPS")
    print("-" * 40)
    print(f"Train : {len(train_df):,}")
    print(f"Val   : {len(val_df):,}")
    print(f"Test  : {len(test_df):,}")
    print(f"Total : {total_split_rows:,}")

    print()
    print("VIDEOS")
    print("-" * 40)

    print_video_distribution(
        "TRAIN",
        train_df
    )

    print_video_distribution(
        "VALIDATION",
        val_df
    )

    print_video_distribution(
        "TEST",
        test_df
    )

    print()
    print("LABEL DISTRIBUTION")
    print("-" * 40)

    print_label_distribution(
        "TRAIN",
        train_df
    )

    print_label_distribution(
        "VALIDATION",
        val_df
    )

    print_label_distribution(
        "TEST",
        test_df
    )

    print()
    print("LEAKAGE CHECK")
    print("-" * 40)

    print(
        f"Train groups : {len(train_group_check):,}"
    )
    print(
        f"Val groups   : {len(val_group_check):,}"
    )
    print(
        f"Test groups  : {len(test_group_check):,}"
    )

    print()
    print(
        f"Train ∩ Val  : {len(train_val_overlap)}"
    )
    print(
        f"Train ∩ Test : {len(train_test_overlap)}"
    )
    print(
        f"Val ∩ Test   : {len(val_test_overlap)}"
    )

    if (
        len(train_val_overlap) == 0
        and len(train_test_overlap) == 0
        and len(val_test_overlap) == 0
    ):
        print()
        print(
            "✓ NO SOURCE VIDEO LEAKAGE DETECTED"
        )

    print()
    print("OUTPUT FILES")
    print("-" * 40)
    print(f"Train : {train_path}")
    print(f"Val   : {val_path}")
    print(f"Test  : {test_path}")

    print()
    print("=" * 70)
    print("Dataset generation complete.")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()