"""
=========================================================
Temporal Branch (VideoMAE) - Full Test Evaluation
---------------------------------------------------------
Evaluates the best full-dataset Temporal Branch checkpoint
using the cached test dataset.

Input:
    data/cache/test/
    data/split/test.csv
    checkpoints/temporal_full/best.pt

Output:
    runs/temporal_full/
        test_results.json
        confusion_matrix.png
=========================================================
"""

import sys
import json
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.amp import autocast


# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================
# IMPORTS
# =========================================================

from data.raw.temporal.cached_dataset import CachedClipDataset
from models.temporal.temporal_classifier import TemporalClassifier
from models.heads.evidence_head import compute_dirichlet


# =========================================================
# PATHS
# =========================================================

TEST_CSV = PROJECT_ROOT / "data" / "split" / "test.csv"

TEST_CACHE = PROJECT_ROOT / "data" / "cache" / "test"

CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "temporal_full"
    / "best.pt"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "runs"
    / "temporal_full"
)


# =========================================================
# SETTINGS
# =========================================================

BATCH_SIZE = 16
NUM_WORKERS = 8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# MAIN
# =========================================================

def main():

    device = torch.device(DEVICE)

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("TEMPORAL BRANCH (VideoMAE) - FULL TEST EVALUATION")
    print("=" * 70)

    print(f"Device       : {DEVICE}")
    print(f"Checkpoint   : {CHECKPOINT}")
    print(f"Test CSV     : {TEST_CSV}")
    print(f"Test cache   : {TEST_CACHE}")
    print(f"Result dir   : {RESULT_DIR}")
    print(f"Batch size   : {BATCH_SIZE}")

    if DEVICE == "cuda":
        print(
            f"GPU          : "
            f"{torch.cuda.get_device_name(0)}"
        )

    # =====================================================
    # CHECK FILES
    # =====================================================

    if not TEST_CSV.exists():
        raise FileNotFoundError(
            f"Test CSV not found:\n{TEST_CSV}"
        )

    if not TEST_CACHE.exists():
        raise FileNotFoundError(
            f"Test cache not found:\n{TEST_CACHE}"
        )

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT}"
        )

    # =====================================================
    # DATASET
    # =====================================================

    print()
    print("Loading cached test dataset...")

    dataset = CachedClipDataset(
        csv_file=TEST_CSV,
        cache_dir=TEST_CACHE,
    )

    print(f"\nTest samples : {len(dataset):,}")

    print("\nLabel distribution:")

    print(
        dataset.df["label"]
        .value_counts()
    )

    # =====================================================
    # DATALOADER
    # =====================================================

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        drop_last=False,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )

    print(
        f"\nEvaluation batches : "
        f"{len(loader):,}"
    )

    # =====================================================
    # MODEL
    # =====================================================

    print("\nLoading Temporal Classifier...")

    model = TemporalClassifier(
        num_classes=2,
        freeze_backbone=True
    ).to(device)

    print("Temporal Classifier created.")

    # =====================================================
    # CHECKPOINT
    # =====================================================

    print("\nLoading checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    print("\nCheckpoint keys:")

    for key in checkpoint.keys():
        print(f"  {key}")

    if "model_state" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain "
            "'model_state'. "
            f"Keys: {list(checkpoint.keys())}"
        )

    print("\nLoading trained model weights...")

    missing_keys, unexpected_keys = (
        model.load_state_dict(
            checkpoint["model_state"],
            strict=False
        )
    )

    if missing_keys:

        print("\nWARNING - Missing keys:")

        for key in missing_keys:
            print(f"  {key}")

    if unexpected_keys:

        print("\nWARNING - Unexpected keys:")

        for key in unexpected_keys:
            print(f"  {key}")

    model.eval()

    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    print("\nCheckpoint loaded successfully.")

    # =====================================================
    # STORAGE
    # =====================================================

    all_predictions = []
    all_labels = []
    all_uncertainties = []

    correct_uncertainties = []
    incorrect_uncertainties = []

    confusion = np.zeros(
        (2, 2),
        dtype=np.int64
    )

    # =====================================================
    # EVALUATION
    # =====================================================

    print()
    print("=" * 70)
    print("STARTING FULL TEST EVALUATION")
    print("=" * 70)
    print()

    with torch.no_grad():

        for step, batch in enumerate(loader):

            # CachedClipDataset returns:
            #
            # clip  -> [B, T, 3, 224, 224]
            # label -> [B]

            clip, labels = batch

            clip = clip.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            # Cached dataset returns uint8.
            # Convert to float and normalize.
            clip = clip.float() / 255.0

            with autocast(
                device_type="cuda",
                enabled=(DEVICE == "cuda")
            ):

                outputs = model(clip)

            evidence = outputs["evidence"]

            dirichlet = compute_dirichlet(
                evidence
            )

            probabilities = (
                dirichlet["probability"]
            )

            uncertainty = (
                dirichlet["uncertainty"]
            )

            predictions = torch.argmax(
                probabilities,
                dim=1
            )

            prediction_values = (
                predictions
                .detach()
                .cpu()
                .numpy()
            )

            label_values = (
                labels
                .detach()
                .cpu()
                .numpy()
            )

            uncertainty_values = (
                uncertainty
                .view(-1)
                .detach()
                .cpu()
                .numpy()
            )

            all_predictions.extend(
                prediction_values
            )

            all_labels.extend(
                label_values
            )

            all_uncertainties.extend(
                uncertainty_values
            )

            # -------------------------------------------------
            # UNCERTAINTY
            # -------------------------------------------------

            for i in range(
                len(label_values)
            ):

                if (
                    prediction_values[i]
                    == label_values[i]
                ):

                    correct_uncertainties.append(
                        uncertainty_values[i]
                    )

                else:

                    incorrect_uncertainties.append(
                        uncertainty_values[i]
                    )

            # -------------------------------------------------
            # CONFUSION MATRIX
            # -------------------------------------------------

            for true, pred in zip(
                label_values,
                prediction_values
            ):

                confusion[
                    int(true),
                    int(pred)
                ] += 1

            # -------------------------------------------------
            # PROGRESS
            # -------------------------------------------------

            if (
                (step + 1) % 25 == 0
                or (step + 1) == len(loader)
            ):

                print(
                    f"Evaluated batch "
                    f"{step + 1:,}/"
                    f"{len(loader):,}"
                )

    # =====================================================
    # METRICS
    # =====================================================

    labels = np.asarray(
        all_labels
    )

    predictions = np.asarray(
        all_predictions
    )

    total = len(labels)

    correct = np.sum(
        predictions == labels
    )

    accuracy = (
        correct / total
        if total > 0
        else 0.0
    )

    tn = int(confusion[0, 0])
    fp = int(confusion[0, 1])
    fn = int(confusion[1, 0])
    tp = int(confusion[1, 1])

    real_recall = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    fake_recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    balanced_accuracy = (
        real_recall + fake_recall
    ) / 2.0

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    f1 = (
        2.0
        * precision
        * fake_recall
        / (precision + fake_recall)
        if (precision + fake_recall) > 0
        else 0.0
    )

    avg_uncertainty = (
        np.mean(all_uncertainties)
        if all_uncertainties
        else 0.0
    )

    correct_uncertainty = (
        np.mean(correct_uncertainties)
        if correct_uncertainties
        else 0.0
    )

    incorrect_uncertainty = (
        np.mean(incorrect_uncertainties)
        if incorrect_uncertainties
        else 0.0
    )

    # =====================================================
    # PRINT RESULTS
    # =====================================================

    print()
    print("=" * 70)
    print("TEMPORAL BRANCH - FULL TEST RESULTS")
    print("=" * 70)

    print(
        f"\nTotal samples        : "
        f"{total:,}"
    )

    print(
        f"Correct predictions  : "
        f"{correct:,}"
    )

    print(
        f"Incorrect predictions: "
        f"{total - correct:,}"
    )

    print("\nOverall Metrics")
    print("-" * 35)

    print(
        f"Accuracy             : "
        f"{accuracy:.4f}"
    )

    print(
        f"Balanced Accuracy    : "
        f"{balanced_accuracy:.4f}"
    )

    print(
        f"Precision            : "
        f"{precision:.4f}"
    )

    print(
        f"Recall               : "
        f"{fake_recall:.4f}"
    )

    print(
        f"F1                   : "
        f"{f1:.4f}"
    )

    print("\nClass Recall")
    print("-" * 35)

    print(
        f"Real Recall          : "
        f"{real_recall:.4f}"
    )

    print(
        f"Fake Recall          : "
        f"{fake_recall:.4f}"
    )

    print("\nConfusion Matrix")
    print("-" * 35)

    print("\n                 Predicted")
    print("                 Real    Fake")

    print(
        f"Actual Real      "
        f"{tn:6d}  {fp:6d}"
    )

    print(
        f"Actual Fake      "
        f"{fn:6d}  {tp:6d}"
    )

    print("\nEDL Uncertainty")
    print("-" * 35)

    print(
        f"Average uncertainty   : "
        f"{avg_uncertainty:.4f}"
    )

    print(
        f"Correct predictions   : "
        f"{correct_uncertainty:.4f}"
    )

    print(
        f"Incorrect predictions : "
        f"{incorrect_uncertainty:.4f}"
    )

    if "epoch" in checkpoint:

        print(
            f"\nCheckpoint epoch     : "
            f"{checkpoint['epoch']}"
        )

    if "best_val_acc" in checkpoint:

        print(
            f"Best validation acc  : "
            f"{checkpoint['best_val_acc']:.4f}"
        )

    if "val_acc" in checkpoint:

        print(
            f"Checkpoint val acc   : "
            f"{checkpoint['val_acc']:.4f}"
        )

    # =====================================================
    # SAVE JSON RESULTS
    # =====================================================

    results = {

        "total_samples": int(total),

        "correct_predictions": int(correct),

        "incorrect_predictions":
            int(total - correct),

        "accuracy": float(accuracy),

        "balanced_accuracy":
            float(balanced_accuracy),

        "precision":
            float(precision),

        "recall":
            float(fake_recall),

        "f1":
            float(f1),

        "real_recall":
            float(real_recall),

        "fake_recall":
            float(fake_recall),

        "confusion_matrix": {

            "actual_real": {
                "predicted_real": tn,
                "predicted_fake": fp,
            },

            "actual_fake": {
                "predicted_real": fn,
                "predicted_fake": tp,
            },
        },

        "edl_uncertainty": {

            "average": float(
                avg_uncertainty
            ),

            "correct_predictions":
                float(correct_uncertainty),

            "incorrect_predictions":
                float(incorrect_uncertainty),
        },

        "checkpoint": str(
            CHECKPOINT
        ),
    }

    if "epoch" in checkpoint:

        results["checkpoint_epoch"] = int(
            checkpoint["epoch"]
        )

    if "best_val_acc" in checkpoint:

        results["best_validation_accuracy"] = (
            float(checkpoint["best_val_acc"])
        )

    if "val_acc" in checkpoint:

        results["checkpoint_validation_accuracy"] = (
            float(checkpoint["val_acc"])
        )

    results_path = (
        RESULT_DIR
        / "test_results.json"
    )

    with open(
        results_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    # =====================================================
    # SAVE CONFUSION MATRIX IMAGE
    # =====================================================

    confusion_path = (
        RESULT_DIR
        / "confusion_matrix.png"
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    image = ax.imshow(
        confusion,
        interpolation="nearest"
    )

    ax.set_title(
        "Temporal Branch - Confusion Matrix"
    )

    ax.set_xlabel(
        "Predicted Label"
    )

    ax.set_ylabel(
        "Actual Label"
    )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(
        ["Real", "Fake"]
    )

    ax.set_yticklabels(
        ["Real", "Fake"]
    )

    for i in range(2):

        for j in range(2):

            ax.text(
                j,
                i,
                f"{confusion[i, j]:,}",
                ha="center",
                va="center"
            )

    fig.colorbar(
        image,
        ax=ax
    )

    fig.tight_layout()

    fig.savefig(
        confusion_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    # =====================================================
    # FINAL
    # =====================================================

    print()
    print("=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)

    print(
        f"\nJSON results       : "
        f"{results_path}"
    )

    print(
        f"Confusion matrix   : "
        f"{confusion_path}"
    )

    print()
    print("=" * 70)
    print("Evaluation complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()