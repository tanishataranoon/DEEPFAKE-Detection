"""
Final Appearance Branch Evaluation (uses pre-extracted test clips).

Evaluates the held-out TEST split and reports:

Binary EDL metrics:
    Accuracy
    Balanced Accuracy
    Precision
    Recall
    F1
    Real Recall
    Fake Recall

Three-state EDL output:
    REAL
    FAKE
    UNCERTAIN

Uncertainty is derived from EDL evidence; there is no third
learned class.

Before running, create the test cache once:
    python preprocessing/cache_clips.py --csv data/split/test.csv --out data/cache/test
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from data.raw.cached_dataset import CachedClipDataset
from models.appearence.appearance_classifier import AppearanceClassifier
from models.heads.evidence_head import (
    EvidenceHead,
    compute_dirichlet,
)


TEST_CSV = "data/split/test.csv"
TEST_CACHE = "data/cache/test"
CHECKPOINT = "checkpoints/appearance_full/best.pt"

BATCH_SIZE = 32
NUM_WORKERS = 8
UNCERTAINTY_THRESHOLD = 0.50
OUTPUT_JSON = "checkpoints/appearance_full/test_results.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    device = torch.device(DEVICE)

    print("=" * 72)
    print("FINAL APPEARANCE BRANCH TEST EVALUATION")
    print("=" * 72)
    print(f"Device      : {device}")
    print(f"Test CSV    : {TEST_CSV}")
    print(f"Test cache  : {TEST_CACHE}")
    print(f"Checkpoint  : {CHECKPOINT}")
    print(f"Uncertainty : {UNCERTAINTY_THRESHOLD}")
    print("=" * 72)

    dataset = CachedClipDataset(TEST_CSV, TEST_CACHE)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        drop_last=False,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=4 if NUM_WORKERS > 0 else None,
    )

    model = AppearanceClassifier(
        num_classes=2,
        freeze_backbone=True,
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    labels_all = []
    binary_preds_all = []
    three_preds_all = []
    uncertainty_all = []
    correct_uncertainty = []
    incorrect_uncertainty = []

    confusion = np.zeros((2, 2), dtype=np.int64)

    with torch.no_grad():
        for step, batch in enumerate(loader):
            clip, labels = batch

            clip = clip.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Cached dataset returns uint8 [B,T,3,H,W]; the model expects
            # float in [0, 1] (same as the original DeepfakeDataset).
            if clip.dtype == torch.uint8:
                clip = clip.float().div_(255.0)

            with autocast(
                device_type="cuda" if DEVICE == "cuda" else "cpu",
                enabled=(DEVICE == "cuda"),
            ):
                outputs = model(clip)
                evidence = outputs["evidence"]

            result = compute_dirichlet(
                evidence,
                uncertainty_threshold=UNCERTAINTY_THRESHOLD,
            )

            binary_preds = result["binary_prediction"]
            three_preds = result["prediction"]
            uncertainty = result["uncertainty"]

            labels_np = labels.cpu().numpy()
            binary_np = binary_preds.cpu().numpy()
            three_np = three_preds.cpu().numpy()
            uncertainty_np = uncertainty.cpu().numpy().reshape(-1)

            labels_all.extend(labels_np)
            binary_preds_all.extend(binary_np)
            three_preds_all.extend(three_np)
            uncertainty_all.extend(uncertainty_np)

            for y, p, u in zip(labels_np, binary_np, uncertainty_np):
                if y == p:
                    correct_uncertainty.append(float(u))
                else:
                    incorrect_uncertainty.append(float(u))
                confusion[int(y), int(p)] += 1

            if (step + 1) % 100 == 0:
                print(
                    f"Evaluated {step + 1}/{len(loader)} batches"
                )

    labels = np.asarray(labels_all)
    binary_preds = np.asarray(binary_preds_all)
    three_preds = np.asarray(three_preds_all)
    uncertainties = np.asarray(uncertainty_all)

    total = len(labels)

    accuracy = float(
        (binary_preds == labels).mean()
    ) if total else 0.0

    tn, fp = confusion[0]
    fn, tp = confusion[1]

    real_recall = (
        tn / (tn + fp)
        if (tn + fp) else 0.0
    )

    fake_recall = (
        tp / (tp + fn)
        if (tp + fn) else 0.0
    )

    balanced_accuracy = (
        real_recall + fake_recall
    ) / 2.0

    precision = (
        tp / (tp + fp)
        if (tp + fp) else 0.0
    )

    f1 = (
        2 * precision * fake_recall
        / (precision + fake_recall)
        if (precision + fake_recall) else 0.0
    )

    uncertain_mask = three_preds == 2
    decisive_mask = ~uncertain_mask

    uncertain_count = int(uncertain_mask.sum())
    uncertain_rate = (
        uncertain_count / total
        if total else 0.0
    )

    decisive_accuracy = (
        float(
            (binary_preds[decisive_mask] == labels[decisive_mask]).mean()
        )
        if decisive_mask.any()
        else 0.0
    )

    avg_uncertainty = (
        float(uncertainties.mean())
        if total else 0.0
    )

    correct_uncertainty_mean = (
        float(np.mean(correct_uncertainty))
        if correct_uncertainty else 0.0
    )

    incorrect_uncertainty_mean = (
        float(np.mean(incorrect_uncertainty))
        if incorrect_uncertainty else 0.0
    )

    results = {
        "total_samples": total,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall_fake": fake_recall,
        "f1": f1,
        "real_recall": real_recall,
        "fake_recall": fake_recall,
        "confusion_matrix": confusion.tolist(),
        "average_uncertainty": avg_uncertainty,
        "correct_prediction_uncertainty": correct_uncertainty_mean,
        "incorrect_prediction_uncertainty": incorrect_uncertainty_mean,
        "uncertainty_threshold": UNCERTAINTY_THRESHOLD,
        "uncertain_count": uncertain_count,
        "uncertain_rate": uncertain_rate,
        "decisive_count": int(decisive_mask.sum()),
        "decisive_accuracy": decisive_accuracy,
        "three_state_counts": {
            "real": int((three_preds == 0).sum()),
            "fake": int((three_preds == 1).sum()),
            "uncertain": uncertain_count,
        },
        "checkpoint_epoch": checkpoint.get("epoch"),
        "best_validation_balanced_accuracy": checkpoint.get(
            "best_balanced_accuracy"
        ),
    }

    print("\n" + "=" * 72)
    print("FINAL RESULTS")
    print("=" * 72)

    print(f"Total samples             : {total:,}")
    print(f"Accuracy                  : {accuracy:.4f}")
    print(f"Balanced Accuracy         : {balanced_accuracy:.4f}")
    print(f"Precision                 : {precision:.4f}")
    print(f"Fake Recall               : {fake_recall:.4f}")
    print(f"F1                        : {f1:.4f}")
    print(f"Real Recall               : {real_recall:.4f}")
    print(f"Fake Recall               : {fake_recall:.4f}")

    print("\nConfusion Matrix")
    print("                 Pred Real   Pred Fake")
    print(f"Actual Real      {tn:9d}   {fp:9d}")
    print(f"Actual Fake      {fn:9d}   {tp:9d}")

    print("\nEDL / Three-State Output")
    print(f"Average uncertainty       : {avg_uncertainty:.4f}")
    print(f"Uncertainty threshold     : {UNCERTAINTY_THRESHOLD:.2f}")
    print(f"REAL predictions          : {(three_preds == 0).sum():,}")
    print(f"FAKE predictions          : {(three_preds == 1).sum():,}")
    print(f"UNCERTAIN predictions     : {uncertain_count:,}")
    print(f"UNCERTAIN rate            : {uncertain_rate:.4f}")
    print(f"Accuracy on decisive only : {decisive_accuracy:.4f}")

    print("\nInterpretation:")
    print("  REAL      = highest expected probability is Real and uncertainty is below threshold")
    print("  FAKE      = highest expected probability is Fake and uncertainty is below threshold")
    print("  UNCERTAIN = EDL uncertainty is at/above threshold")

    out = Path(OUTPUT_JSON)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(f"\nResults saved -> {out}")
    print("=" * 72)


if __name__ == "__main__":
    main()