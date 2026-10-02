"""
=========================================================
VERIFY TEST_150 DATASET SPLIT
---------------------------------------------------------
Checks:

1. Train / validation / test sample counts
2. Label distribution
3. Source-group distribution
4. Source-group overlap
5. Video overlap
6. Duplicate rows
7. Label consistency
8. Basic split sanity checks

NO MODEL IS LOADED.
NO GPU IS USED.
=========================================================
"""

import pandas as pd
from pathlib import Path


# =========================================================
# CONFIG
# =========================================================

SPLIT_DIR = Path("data/split/test_150")

TRAIN_CSV = SPLIT_DIR / "train.csv"
VAL_CSV = SPLIT_DIR / "val.csv"
TEST_CSV = SPLIT_DIR / "test.csv"


# =========================================================
# LOAD
# =========================================================

def load_csv(path):

    print()
    print(f"Loading: {path}")

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    df = pd.read_csv(path)

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Columns: {list(df.columns)}"
    )

    return df


# =========================================================
# FIND COLUMN
# =========================================================

def find_column(df, candidates):

    lower_map = {
        str(col).lower(): col
        for col in df.columns
    }

    for candidate in candidates:

        if candidate.lower() in lower_map:

            return lower_map[
                candidate.lower()
            ]

    return None


# =========================================================
# COLUMN DETECTION
# =========================================================

def detect_columns(df):

    source_col = find_column(
        df,
        [
            "source_group",
            "source_group_id",
            "group",
            "group_id",
            "source",
            "source_id",
        ],
    )

    video_col = find_column(
        df,
        [
            "video_id",
            "video",
            "video_path",
            "video_name",
            "filename",
            "file",
        ],
    )

    label_col = find_column(
        df,
        [
            "label",
            "class",
            "target",
        ],
    )

    return (
        source_col,
        video_col,
        label_col,
    )


# =========================================================
# BASIC SUMMARY
# =========================================================

def summarize(name, df):

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(
        f"Samples : {len(df):,}"
    )

    source_col, video_col, label_col = (
        detect_columns(df)
    )

    print()
    print("Detected columns:")

    print(
        f"  Source : {source_col}"
    )

    print(
        f"  Video  : {video_col}"
    )

    print(
        f"  Label  : {label_col}"
    )

    # -----------------------------------------------------
    # Labels
    # -----------------------------------------------------

    if label_col is not None:

        print()
        print("Label distribution:")

        print(
            df[label_col]
            .value_counts(dropna=False)
        )

        print()
        print("Label percentages:")

        percentages = (
            df[label_col]
            .value_counts(
                normalize=True,
                dropna=False,
            )
            * 100
        )

        for label, percentage in percentages.items():

            print(
                f"  {label}: "
                f"{percentage:.2f}%"
            )

    # -----------------------------------------------------
    # Source groups
    # -----------------------------------------------------

    if source_col is not None:

        print()
        print(
            f"Unique source groups : "
            f"{df[source_col].nunique():,}"
        )

        print()
        print("Samples per source group:")

        group_counts = (
            df[source_col]
            .value_counts()
        )

        print(
            group_counts.to_string()
        )

    # -----------------------------------------------------
    # Videos
    # -----------------------------------------------------

    if video_col is not None:

        print()
        print(
            f"Unique videos : "
            f"{df[video_col].nunique():,}"
        )

    # -----------------------------------------------------
    # Duplicate rows
    # -----------------------------------------------------

    duplicate_rows = df.duplicated().sum()

    print()
    print(
        f"Duplicate rows : "
        f"{duplicate_rows:,}"
    )


# =========================================================
# OVERLAP CHECK
# =========================================================

def check_overlap(
    train,
    val,
    test,
    column,
    name,
):

    print()
    print("=" * 70)
    print(f"{name} OVERLAP CHECK")
    print("=" * 70)

    if column is None:

        print(
            "SKIPPED - column not found."
        )

        return

    train_values = set(
        train[column]
        .dropna()
        .astype(str)
    )

    val_values = set(
        val[column]
        .dropna()
        .astype(str)
    )

    test_values = set(
        test[column]
        .dropna()
        .astype(str)
    )

    train_val = (
        train_values
        & val_values
    )

    train_test = (
        train_values
        & test_values
    )

    val_test = (
        val_values
        & test_values
    )

    print()
    print(
        f"Train unique : "
        f"{len(train_values):,}"
    )

    print(
        f"Val unique   : "
        f"{len(val_values):,}"
    )

    print(
        f"Test unique  : "
        f"{len(test_values):,}"
    )

    print()

    print(
        f"Train ∩ Val  : "
        f"{len(train_val):,}"
    )

    print(
        f"Train ∩ Test : "
        f"{len(train_test):,}"
    )

    print(
        f"Val ∩ Test   : "
        f"{len(val_test):,}"
    )

    # -----------------------------------------------------
    # Show leaked values
    # -----------------------------------------------------

    if train_val:

        print()
        print("WARNING: Train/Val overlap:")

        for value in sorted(
            train_val
        )[:20]:

            print(
                f"  {value}"
            )

    if train_test:

        print()
        print("WARNING: Train/Test overlap:")

        for value in sorted(
            train_test
        )[:20]:

            print(
                f"  {value}"
            )

    if val_test:

        print()
        print("WARNING: Val/Test overlap:")

        for value in sorted(
            val_test
        )[:20]:

            print(
                f"  {value}"
            )

    # -----------------------------------------------------
    # Final status
    # -----------------------------------------------------

    if not train_val and not train_test and not val_test:

        print()
        print(
            f"PASS: No {name.lower()} leakage detected."
        )

    else:

        print()
        print(
            f"FAIL: {name.lower()} leakage detected!"
        )


# =========================================================
# LABEL CONSISTENCY
# =========================================================

def check_label_consistency(
    name,
    df,
):

    source_col, video_col, label_col = (
        detect_columns(df)
    )

    if video_col is None or label_col is None:

        print()
        print(
            f"{name}: Label consistency check skipped."
        )

        return

    print()
    print("=" * 70)
    print(
        f"{name} LABEL CONSISTENCY"
    )
    print("=" * 70)

    grouped = (
        df.groupby(video_col)[label_col]
        .nunique()
    )

    inconsistent = grouped[
        grouped > 1
    ]

    print(
        f"Videos checked : "
        f"{len(grouped):,}"
    )

    print(
        f"Inconsistent videos : "
        f"{len(inconsistent):,}"
    )

    if len(inconsistent) == 0:

        print(
            "PASS: Every video has one consistent label."
        )

    else:

        print(
            "FAIL: Some videos have multiple labels!"
        )

        print()

        for video in inconsistent.index[:20]:

            labels = (
                df.loc[
                    df[video_col] == video,
                    label_col,
                ]
                .unique()
            )

            print(
                f"  {video} -> {labels}"
            )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print(
        "TEST_150 SPLIT VERIFICATION"
    )
    print("=" * 70)

    print()
    print(
        "NO MODEL WILL BE LOADED."
    )

    # -----------------------------------------------------
    # Load
    # -----------------------------------------------------

    train = load_csv(
        TRAIN_CSV
    )

    val = load_csv(
        VAL_CSV
    )

    test = load_csv(
        TEST_CSV
    )

    # -----------------------------------------------------
    # Summaries
    # -----------------------------------------------------

    summarize(
        "TRAIN",
        train,
    )

    summarize(
        "VALIDATION",
        val,
    )

    summarize(
        "TEST",
        test,
    )

    # -----------------------------------------------------
    # Detect columns
    # -----------------------------------------------------

    train_source, train_video, train_label = (
        detect_columns(train)
    )

    val_source, val_video, val_label = (
        detect_columns(val)
    )

    test_source, test_video, test_label = (
        detect_columns(test)
    )

    # -----------------------------------------------------
    # Source overlap
    # -----------------------------------------------------

    source_col = train_source

    if (
        val_source != source_col
        or test_source != source_col
    ):

        print()
        print(
            "WARNING: Source-group column names "
            "are not identical across CSVs."
        )

    check_overlap(
        train,
        val,
        test,
        source_col,
        "SOURCE GROUP",
    )

    # -----------------------------------------------------
    # Video overlap
    # -----------------------------------------------------

    video_col = train_video

    if (
        val_video != video_col
        or test_video != video_col
    ):

        print()
        print(
            "WARNING: Video column names "
            "are not identical across CSVs."
        )

    check_overlap(
        train,
        val,
        test,
        video_col,
        "VIDEO",
    )

    # -----------------------------------------------------
    # Label consistency
    # -----------------------------------------------------

    check_label_consistency(
        "TRAIN",
        train,
    )

    check_label_consistency(
        "VALIDATION",
        val,
    )

    check_label_consistency(
        "TEST",
        test,
    )

    # =====================================================
    # FINAL
    # =====================================================

    print()
    print("=" * 70)
    print(
        "SPLIT VERIFICATION COMPLETE"
    )
    print("=" * 70)

    print()
    print(
        "If SOURCE GROUP overlap = 0"
    )

    print(
        "and VIDEO overlap = 0"
    )

    print(
        "then the split is structurally clean."
    )

    print()
    print(
        "Only after this should we diagnose"
    )

    print(
        "the Appearance Branch model."
    )

    print("=" * 70)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()