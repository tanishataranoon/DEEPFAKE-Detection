"""
=========================================================
Build All Dataset Caches
---------------------------------------------------------
Creates the train, validation, and test caches using the
existing cache_clips.py script.

Expected project structure:

DEEPFAKE-Detection/
│
├── data/
│   ├── raw/
│   │   └── ... mp4 files
│   │
│   ├── processed/
│   │   ├── cache_clips.py
│   │   └── build_all_cache.py       <-- this file
│   │
│   ├── split/
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   │
│   └── cache/
│       ├── train/
│       ├── val/
│       └── test/

Usage from project root:

    python data/processed/build_all_cache.py

Optional:

    python data/processed/build_all_cache.py --workers 20

Timing test:

    python data/processed/build_all_cache.py --limit 200

The script calls cache_clips.py three times:

    train.csv -> data/cache/train
    val.csv   -> data/cache/val
    test.csv  -> data/cache/test

The generated .npy cache files are NOT part of Git.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


# =========================================================
# PROJECT PATHS
# =========================================================

# Project root:
# E:\Depfake\DEEPFAKE-Detection
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

SPLIT_DIR = DATA_DIR / "split"
CACHE_DIR = DATA_DIR / "cache"
PROCESSED_DIR = DATA_DIR / "processed"

CACHE_SCRIPT = PROCESSED_DIR / "cache_clips.py"


# =========================================================
# DATASET SPLITS
# =========================================================

SPLITS = [
    ("TRAIN", SPLIT_DIR / "train.csv", CACHE_DIR / "train"),
    ("VALIDATION", SPLIT_DIR / "val.csv", CACHE_DIR / "val"),
    ("TEST", SPLIT_DIR / "test.csv", CACHE_DIR / "test"),
]


# =========================================================
# RUN ONE CACHE
# =========================================================

def build_cache(name, csv_path, output_dir, args):
    print()
    print("=" * 70)
    print(f"{name} CACHE")
    print("=" * 70)

    print(f"CSV       : {csv_path}")
    print(f"OUTPUT    : {output_dir}")
    print(f"VIDEO ROOT: {args.video_root}")
    print(f"WORKERS   : {args.workers}")

    if args.limit:
        print(f"LIMIT     : {args.limit:,} rows")

    print()

    if not csv_path.exists():
        print(f"ERROR: CSV file not found:")
        print(f"       {csv_path}")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(CACHE_SCRIPT),
        "--csv",
        str(csv_path),
        "--out",
        str(output_dir),
        "--video-root",
        str(args.video_root),
        "--workers",
        str(args.workers),
    ]

    if args.size != 224:
        command.extend([
            "--size",
            str(args.size),
        ])

    if args.frames != 16:
        command.extend([
            "--frames",
            str(args.frames),
        ])

    if args.limit is not None:
        command.extend([
            "--limit",
            str(args.limit),
        ])

    print("Running:")
    print(" ".join(f'"{x}"' if " " in str(x) else str(x) for x in command))
    print()

    start = time.time()

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
    )

    elapsed = time.time() - start

    print()
    print("-" * 70)

    if result.returncode != 0:
        print(f"{name} CACHE FAILED")
        print(f"Time: {elapsed / 60:.1f} minutes")
        return False

    print(f"{name} CACHE COMPLETE")
    print(f"Time: {elapsed / 60:.1f} minutes")

    return True


# =========================================================
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description="Build train, validation, and test caches."
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=20,
        help="Number of cache worker processes. Default: 20",
    )

    parser.add_argument(
        "--video-root",
        type=Path,
        default=DATA_DIR / "raw",
        help="Root directory containing the raw MP4 videos.",
    )

    parser.add_argument(
        "--size",
        type=int,
        default=224,
        help="Face crop size. Default: 224",
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=16,
        help="Frames per clip. Default: 16",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process first N rows of each split. Useful for testing.",
    )

    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip train cache.",
    )

    parser.add_argument(
        "--skip-val",
        action="store_true",
        help="Skip validation cache.",
    )

    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip test cache.",
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # Validate cache script
    # -----------------------------------------------------

    if not CACHE_SCRIPT.exists():
        print()
        print("ERROR: cache_clips.py was not found.")
        print()
        print(f"Expected location:")
        print(f"  {CACHE_SCRIPT}")
        print()
        sys.exit(1)

    # -----------------------------------------------------
    # Validate video root
    # -----------------------------------------------------

    if not args.video_root.exists():
        print()
        print("ERROR: video root does not exist.")
        print()
        print(f"Expected:")
        print(f"  {args.video_root}")
        print()
        sys.exit(1)

    # -----------------------------------------------------
    # Header
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("DEEPFAKE DETECTION - CACHE BUILDER")
    print("=" * 70)
    print()
    print(f"PROJECT ROOT : {PROJECT_ROOT}")
    print(f"VIDEO ROOT   : {args.video_root}")
    print(f"SPLIT DIR    : {SPLIT_DIR}")
    print(f"CACHE DIR    : {CACHE_DIR}")
    print(f"CACHE SCRIPT : {CACHE_SCRIPT}")
    print(f"WORKERS      : {args.workers}")
    print(f"FRAME SIZE   : {args.size} x {args.size}")
    print(f"CLIP FRAMES  : {args.frames}")

    if args.limit:
        print(f"LIMIT        : {args.limit:,} rows per split")

    print()
    print("=" * 70)

    total_start = time.time()

    results = {}

    # -----------------------------------------------------
    # TRAIN
    # -----------------------------------------------------

    if args.skip_train:
        print()
        print("Skipping TRAIN cache.")
        results["train"] = "SKIPPED"
    else:
        success = build_cache(
            "TRAIN",
            SPLITS[0][1],
            SPLITS[0][2],
            args,
        )
        results["train"] = "OK" if success else "FAILED"

        if not success:
            print()
            print("TRAIN cache failed.")
            print("Stopping so you can fix the problem before continuing.")
            sys.exit(1)

    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    if args.skip_val:
        print()
        print("Skipping VALIDATION cache.")
        results["val"] = "SKIPPED"
    else:
        success = build_cache(
            "VALIDATION",
            SPLITS[1][1],
            SPLITS[1][2],
            args,
        )
        results["val"] = "OK" if success else "FAILED"

        if not success:
            print()
            print("VALIDATION cache failed.")
            print("Stopping so you can fix the problem before continuing.")
            sys.exit(1)

    # -----------------------------------------------------
    # TEST
    # -----------------------------------------------------

    if args.skip_test:
        print()
        print("Skipping TEST cache.")
        results["test"] = "SKIPPED"
    else:
        success = build_cache(
            "TEST",
            SPLITS[2][1],
            SPLITS[2][2],
            args,
        )
        results["test"] = "OK" if success else "FAILED"

        if not success:
            print()
            print("TEST cache failed.")
            print("Stopping.")
            sys.exit(1)

    # -----------------------------------------------------
    # FINAL SUMMARY
    # -----------------------------------------------------

    total_elapsed = time.time() - total_start

    print()
    print()
    print("=" * 70)
    print("CACHE BUILD SUMMARY")
    print("=" * 70)

    print()
    print(f"TRAIN       : {results['train']}")
    print(f"VALIDATION  : {results['val']}")
    print(f"TEST        : {results['test']}")

    print()
    print(f"Total time  : {total_elapsed / 60:.1f} minutes")

    print()
    print("Cache locations:")
    print(f"  Train : {CACHE_DIR / 'train'}")
    print(f"  Val   : {CACHE_DIR / 'val'}")
    print(f"  Test  : {CACHE_DIR / 'test'}")

    print()
    print("=" * 70)
    print("ALL REQUESTED CACHES COMPLETED")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()