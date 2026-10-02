
"""
Create Small Leakage-Free Dataset

Purpose
-------
Create a smaller dataset from the existing metadata.csv without
re-running preprocessing.

IMPORTANT
---------
The grouping unit is the ORIGINAL SOURCE VIDEO.

One source group can contain:
    - 1 real/original video
    - multiple fake/manipulated versions

Example
-------
28__podium_speech_happy.mp4
05_28__podium_speech_happy__ABC123.mp4
08_28__podium_speech_happy__XYZ789.mp4

All belong to:

    28__podium_speech_happy

Therefore, the complete group must remain in the same
train / validation / test split.

The resulting CSV files can be shared by:

    Appearance Branch
    Temporal Branch
    Fusion

No image/video preprocessing is performed here.
Only metadata rows are selected.
"""

import os
import re
import pandas as pd

from sklearn.model_selection import train_test_split


# ============================================================
# CONFIG
# ============================================================

METADATA_CSV = "data/processed/metadata.csv"

OUTPUT_DIR = "data/split/small"

# Number of ORIGINAL SOURCE GROUPS to keep.
#
# Example:
#     363 source groups
#
# Every selected group includes:
#     real source video
#     + all fake versions belonging to that source
#
NUM_SOURCE_GROUPS = 363

# Train / validation / test split
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

RANDOM_STATE = 42


# ============================================================
# SOURCE GROUP BUILDER
# ============================================================

def get_source_group(video_name):
    """
    Convert a video filename into its original source group.

    Real example
    ------------
    28__podium_speech_happy.mp4

        ->
        
    28__podium_speech_happy


    Fake example
    ------------
    05_28__podium_speech_happy__2DK5X0Y0.mp4

        ->
        
    28__podium_speech_happy


    Another fake
    ------------
    08_28__podium_speech_happy__GGCUYQA4.mp4

        ->
        
    28__podium_speech_happy


    IMPORTANT
    ---------
    A group is allowed to contain both:

        real
        fake

    This is intentional.

    The purpose of the group is leakage prevention.
    """

    name = os.path.splitext(
        os.path.basename(str(video_name))
    )[0]

    # --------------------------------------------------------
    # Real / original video
    #
    # Example:
    #
    # 28__podium_speech_happy
    # --------------------------------------------------------

    match = re.match(
        r"^(\d+)__(.+)$",
        name
    )

    if match:
        source_id = match.group(1)
        scenario = match.group(2)

        return f"{source_id}__{scenario}"

    # --------------------------------------------------------
    # Fake / manipulated video
    #
    # Example:
    #
    # 05_28__podium_speech_happy__2DK5X0Y0
    #
    # First number  = manipulation/source index
    # Second number = original source ID
    # --------------------------------------------------------

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

def print_label_distribution(name, df):
    """Print real/fake distribution for a dataframe."""

    print()
    print(name)
    print("-" * 40)

    print(
        df["label"]
        .value_counts()
        .sort_index()
    )


def print_video_distribution(name, df):
    """Print unique video counts."""

    print()
    print(name)
    print("-" * 40)

    print(
        f"Unique videos : "
        f"{df['video_name'].nunique():,}"
    )

    print(
        f"Real videos   : "
        f"{df[df['label'] == 'real']['video_name'].nunique():,}"
    )

    print(
        f"Fake videos   : "
        f"{df[df['label'] == 'fake']['video_name'].nunique():,}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("SMALL LEAKAGE-FREE DATASET CREATOR")
    print("=" * 70)

    print()
    print(f"Metadata : {METADATA_CSV}")
    print(f"Output   : {OUTPUT_DIR}")

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    if not os.path.exists(METADATA_CSV):

        raise FileNotFoundError(
            f"Metadata file not found:\n"
            f"{METADATA_CSV}"
        )

    df = pd.read_csv(
        METADATA_CSV
    )

    print()
    print(
        f"Total metadata clips : "
        f"{len(df):,}"
    )

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

    actual_labels = set(
        df["label"]
        .astype(str)
        .str.lower()
        .unique()
    )

    invalid_labels = (
        actual_labels - valid_labels
    )

    if invalid_labels:

        raise RuntimeError(
            "Unexpected labels found:\n"
            f"{invalid_labels}\n"
            f"Expected only: {valid_labels}"
        )

    df["label"] = (
        df["label"]
        .astype(str)
        .str.lower()
    )

    # --------------------------------------------------------
    # Build source group
    # --------------------------------------------------------

    print()
    print("Building source groups...")

    df["group"] = (
        df["video_name"]
        .apply(get_source_group)
    )

    total_groups = (
        df["group"]
        .nunique()
    )

    print(
        f"Total source groups : "
        f"{total_groups:,}"
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # DO NOT reject groups containing both real/fake.
    #
    # This is expected:
    #
    # group
    #   real source
    #   fake 1
    #   fake 2
    #   fake 3
    #
    # All stay together.
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Determine source groups containing real videos
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

    # --------------------------------------------------------
    # Safety check:
    #
    # Every source group selected for this dataset should
    # have an original real video.
    #
    # If some groups have only fake videos, they are not
    # valid source groups for this sampling strategy.
    # --------------------------------------------------------

    groups_without_real = sorted(
        set(
            df["group"].unique()
        )
        -
        set(real_groups)
    )

    if groups_without_real:

        print()
        print(
            "WARNING:"
        )

        print(
            f"{len(groups_without_real)} "
            "source groups contain no real video."
        )

        print(
            "They will not be selected."
        )

    # --------------------------------------------------------
    # Check requested number
    # --------------------------------------------------------

    if NUM_SOURCE_GROUPS > len(real_groups):

        raise RuntimeError(
            f"Requested {NUM_SOURCE_GROUPS} "
            f"source groups, but only "
            f"{len(real_groups)} groups containing "
            f"real videos are available."
        )

    # --------------------------------------------------------
    # Randomly select source groups
    # --------------------------------------------------------

    groups = pd.Series(
        real_groups
    )

    selected_groups = (
        groups
        .sample(
            n=NUM_SOURCE_GROUPS,
            random_state=RANDOM_STATE
        )
        .tolist()
    )

    selected_groups = set(
        selected_groups
    )

    print()
    print(
        f"Selected source groups : "
        f"{len(selected_groups):,}"
    )

    # --------------------------------------------------------
    # Create selected dataframe
    #
    # IMPORTANT:
    #
    # We select by GROUP, not by individual video.
    #
    # Therefore:
    #
    # real source
    # +
    # all fake versions
    #
    # are kept together.
    # --------------------------------------------------------

    small_df = df[
        df["group"].isin(
            selected_groups
        )
    ].copy()

    print()
    print(
        f"Selected clips : "
        f"{len(small_df):,}"
    )

    # --------------------------------------------------------
    # Remove helper column
    # --------------------------------------------------------

    small_df.drop(
        columns=["group"],
        inplace=True
    )

    # --------------------------------------------------------
    # Split source groups
    #
    # IMPORTANT:
    #
    # We must split the GROUPS, not metadata rows.
    # --------------------------------------------------------

    selected_group_list = list(
        selected_groups
    )

    train_groups, temp_groups = train_test_split(
        selected_group_list,
        train_size=TRAIN_RATIO,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    # Validation fraction within remaining 20%
    val_fraction_of_temp = (
        VAL_RATIO /
        (VAL_RATIO + TEST_RATIO)
    )

    val_groups, test_groups = train_test_split(
        temp_groups,
        train_size=val_fraction_of_temp,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    train_groups = set(train_groups)
    val_groups = set(val_groups)
    test_groups = set(test_groups)

    # --------------------------------------------------------
    # Split safety checks
    # --------------------------------------------------------

    assert (
        len(train_groups & val_groups) == 0
    )

    assert (
        len(train_groups & test_groups) == 0
    )

    assert (
        len(val_groups & test_groups) == 0
    )

    assert (
        len(
            train_groups
            | val_groups
            | test_groups
        )
        ==
        len(selected_groups)
    )

    # --------------------------------------------------------
    # Rebuild group column temporarily
    # --------------------------------------------------------

    small_df["group"] = (
        small_df["video_name"]
        .apply(get_source_group)
    )

    train_df = small_df[
        small_df["group"].isin(
            train_groups
        )
    ].copy()

    val_df = small_df[
        small_df["group"].isin(
            val_groups
        )
    ].copy()

    test_df = small_df[
        small_df["group"].isin(
            test_groups
        )
    ].copy()

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
    # DATA LEAKAGE CHECK
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
        train_group_check
        &
        val_group_check
    )

    train_test_overlap = (
        train_group_check
        &
        test_group_check
    )

    val_test_overlap = (
        val_group_check
        &
        test_group_check
    )

    if train_val_overlap:
        raise RuntimeError(
            "GROUP LEAKAGE DETECTED: "
            "Train/Validation overlap."
        )

    if train_test_overlap:
        raise RuntimeError(
            "GROUP LEAKAGE DETECTED: "
            "Train/Test overlap."
        )

    if val_test_overlap:
        raise RuntimeError(
            "GROUP LEAKAGE DETECTED: "
            "Validation/Test overlap."
        )

    # --------------------------------------------------------
    # Duplicate checks
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
    # Total-row safety check
    # --------------------------------------------------------

    total_split_rows = (
        len(train_df)
        +
        len(val_df)
        +
        len(test_df)
    )

    if total_split_rows != len(small_df):

        raise RuntimeError(
            "Split row count mismatch!\n"
            f"Expected : {len(small_df):,}\n"
            f"Got      : {total_split_rows:,}"
        )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

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
    print("SMALL DATASET CREATED")
    print("=" * 70)

    print()
    print("SOURCE GROUPS")
    print("-" * 40)

    print(
        f"Selected : "
        f"{len(selected_groups):,}"
    )

    print(
        f"Train    : "
        f"{len(train_groups):,}"
    )

    print(
        f"Val      : "
        f"{len(val_groups):,}"
    )

    print(
        f"Test     : "
        f"{len(test_groups):,}"
    )

    print()
    print("CLIPS")
    print("-" * 40)

    print(
        f"Train : "
        f"{len(train_df):,}"
    )

    print(
        f"Val   : "
        f"{len(val_df):,}"
    )

    print(
        f"Test  : "
        f"{len(test_df):,}"
    )

    print(
        f"Total : "
        f"{total_split_rows:,}"
    )

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
        f"Train groups : "
        f"{len(train_group_check):,}"
    )

    print(
        f"Val groups   : "
        f"{len(val_group_check):,}"
    )

    print(
        f"Test groups  : "
        f"{len(test_group_check):,}"
    )

    print()
    print(
        "Train ∩ Val  : "
        f"{len(train_val_overlap)}"
    )

    print(
        "Train ∩ Test : "
        f"{len(train_test_overlap)}"
    )

    print(
        "Val ∩ Test   : "
        f"{len(val_test_overlap)}"
    )

    print()

    if (
        len(train_val_overlap) == 0
        and
        len(train_test_overlap) == 0
        and
        len(val_test_overlap) == 0
    ):

        print(
            "✓ NO SOURCE VIDEO LEAKAGE DETECTED"
        )

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    print()
    print("OUTPUT FILES")
    print("-" * 40)

    print(
        f"Train : {train_path}"
    )

    print(
        f"Val   : {val_path}"
    )

    print(
        f"Test  : {test_path}"
    )

    print()
    print("=" * 70)
    print(
        "Small dataset generation complete."
    )
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()

