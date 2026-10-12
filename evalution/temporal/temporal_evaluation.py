"""
Full-test evaluation for the Temporal DeepFake Detection branch (VideoMAE).

Run from the project root:
    python -m evalution.temporal.temporal_evaluation

Uses:
    data/split/test.csv
    data/cache/test/
    checkpoints/temporal_full_ft2/best.pt

Outputs (runs/temporal_full_ft2/):
    test_results.json
    predictions.csv
    confusion_matrix.png

IMPORTANT:
    CachedTemporalClipDataset already applies ImageNet normalization.
    Do NOT divide by 255 or normalize again here.
    The test set must use split="test" (no random augmentation).
"""

import json
import sys
from pathlib import Path
from collections.abc import Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.raw.temporal.cached_temporal_dataset import CachedTemporalClipDataset
from models.temporal.temporal_classifier import TemporalClassifier
from models.heads.evidence_head import compute_dirichlet


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

TEST_CSV = PROJECT_ROOT / "data" / "split" / "test.csv"
TEST_CACHE = PROJECT_ROOT / "data" / "cache" / "test"
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "temporal_full_ft2" / "best.pt"
OUTPUT_DIR = PROJECT_ROOT / "runs" / "temporal_full_ft2"

BATCH_SIZE = 16
NUM_WORKERS = 8
UNCERTAINTY_THRESHOLD = 0.50
CLASS_NAMES = ["Real", "Fake"]  # 0 = Real, 1 = Fake


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def unpack_batch(batch):
    """CachedTemporalClipDataset returns {"clip": ..., "label": ...}."""
    if isinstance(batch, Mapping):
        return batch["clip"], batch["label"]
    clip, label = batch
    return clip, label


def prepare_clip(clip):
    """
    Only convert to float. NO /255 and NO second normalization:
    the dataset already returns ImageNet-normalized pixels.
    """
    if not torch.is_tensor(clip):
        raise TypeError(f"Expected Tensor, got {type(clip).__name__}")
    if clip.ndim != 5:
        raise ValueError(
            f"Expected clip shape [B,T,C,H,W], received {tuple(clip.shape)}"
        )
    return clip.float()


def safe_float(value):
    return None if value is None else float(value)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

@torch.inference_mode()
def main():
    print("=" * 76)
    print("TEMPORAL BRANCH - FULL TEST EVALUATION")
    print("=" * 76)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device       : {device}")
    if device.type == "cuda":
        print(f"GPU          : {torch.cuda.get_device_name(0)}")

    for path, description in (
        (TEST_CSV, "Test CSV"),
        (TEST_CACHE, "Test cache directory"),
        (CHECKPOINT, "Best checkpoint"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{description} not found: {path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Test CSV     : {TEST_CSV}")
    print(f"Test cache   : {TEST_CACHE}")
    print(f"Checkpoint   : {CHECKPOINT}")
    print(f"Output dir   : {OUTPUT_DIR}")

    # split="test" -> deterministic transform (resize + normalize only)
    dataset = CachedTemporalClipDataset(
        str(TEST_CSV),
        str(TEST_CACHE),
        split="test",
    )
    print(f"Test samples : {len(dataset):,}")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
    )

    model = TemporalClassifier(
        num_classes=2,
        freeze_backbone=True,
        gradient_checkpointing=False,
    ).to(device)

    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state", checkpoint.get("state_dict"))
    if state_dict is None:
        raise KeyError(f"No 'model_state' found in checkpoint: {CHECKPOINT}")

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    epoch = checkpoint.get("epoch", "unknown")
    best_balanced_accuracy = safe_float(checkpoint.get("best_balanced_accuracy"))
    checkpoint_val_accuracy = safe_float(checkpoint.get("val_accuracy"))
    print(f"Checkpoint epoch           : {epoch}")
    print(f"Best val balanced accuracy : {best_balanced_accuracy}")
    print("Checkpoint loaded successfully.")

    all_labels = []
    all_predictions = []
    all_uncertainties = []
    all_probabilities = []

    for batch_index, batch in enumerate(loader, start=1):
        clip, labels = unpack_batch(batch)
        clip = prepare_clip(clip).to(device, non_blocking=True)
        labels = torch.as_tensor(labels, dtype=torch.long).view(-1).to(device)

        # Sanity check on the first batch: normalized pixels should be
        # roughly between -2.2 and +2.7. If you see values near 0.00-0.01,
        # the input is being corrupted again.
        if batch_index == 1:
            print(
                f"First batch pixel range    : "
                f"min={clip.min().item():.3f}, max={clip.max().item():.3f}"
            )

        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            outputs = model(clip)

        evidence = outputs["evidence"].float()
        dirichlet = compute_dirichlet(evidence)

        probabilities = dirichlet["probability"]
        uncertainty = dirichlet["uncertainty"].reshape(-1)
        predictions = probabilities.argmax(dim=1)

        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())
        all_uncertainties.extend(uncertainty.cpu().tolist())
        all_probabilities.extend(probabilities.cpu().tolist())

        if batch_index % 100 == 0 or batch_index == len(loader):
            print(f"Evaluated batches: {batch_index:,}/{len(loader):,}")

    y_true = np.asarray(all_labels, dtype=np.int64)
    y_pred = np.asarray(all_predictions, dtype=np.int64)
    uncertainties = np.asarray(all_uncertainties, dtype=np.float64)
    probabilities = np.asarray(all_probabilities, dtype=np.float64)

    if y_true.size == 0:
        raise RuntimeError("The test DataLoader returned no samples.")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]

    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    real_mask = y_true == 0
    fake_mask = y_true == 1
    real_recall = float((y_pred[real_mask] == 0).mean()) if real_mask.any() else 0.0
    fake_recall = float((y_pred[fake_mask] == 1).mean()) if fake_mask.any() else 0.0

    correct_mask = y_true == y_pred
    uncertain_mask = uncertainties >= UNCERTAINTY_THRESHOLD

    avg_uncertainty = float(uncertainties.mean())
    correct_uncertainty = (
        float(uncertainties[correct_mask].mean()) if correct_mask.any() else 0.0
    )
    incorrect_uncertainty = (
        float(uncertainties[~correct_mask].mean()) if (~correct_mask).any() else 0.0
    )

    metrics = {
        "checkpoint": str(CHECKPOINT),
        "checkpoint_epoch": epoch,
        "best_validation_balanced_accuracy": best_balanced_accuracy,
        "best_validation_accuracy": checkpoint_val_accuracy,
        "total_samples": int(y_true.size),
        "test_samples": int(y_true.size),
        "class_distribution": {
            "real": int(real_mask.sum()),
            "fake": int(fake_mask.sum()),
        },
        "correct_predictions": int(correct_mask.sum()),
        "incorrect_predictions": int((~correct_mask).sum()),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "real_recall": real_recall,
        "fake_recall": fake_recall,
        "edl_uncertainty": {
            "average": avg_uncertainty,
            "correct_predictions": correct_uncertainty,
            "incorrect_predictions": incorrect_uncertainty,
        },
        "uncertainty_threshold": UNCERTAINTY_THRESHOLD,
        "uncertain_count": int(uncertain_mask.sum()),
        "uncertain_rate": float(uncertain_mask.mean()),
        "confusion_matrix": {
            "actual_real": {"predicted_real": tn, "predicted_fake": fp},
            "actual_fake": {"predicted_real": fn, "predicted_fake": tp},
        },
        "confusion_matrix_array": cm.tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
    }

    print("\n" + "=" * 76)
    print("TEST RESULTS")
    print("=" * 76)
    print(f"Samples              : {y_true.size:,}")
    print(f"Accuracy             : {accuracy:.4f}")
    print(f"Balanced accuracy    : {balanced_accuracy:.4f}")
    print(f"Fake precision       : {precision:.4f}")
    print(f"Fake F1              : {f1:.4f}")
    print(f"Real recall          : {real_recall:.4f}")
    print(f"Fake recall          : {fake_recall:.4f}")
    print(f"Average uncertainty  : {avg_uncertainty:.4f}")
    print(f"  correct preds      : {correct_uncertainty:.4f}")
    print(f"  incorrect preds    : {incorrect_uncertainty:.4f}")
    print(f"Uncertain rate       : {uncertain_mask.mean():.2%}")
    print("\nConfusion matrix (rows=actual, columns=predicted):")
    print("                 Pred Real   Pred Fake")
    print(f"Actual Real      {tn:9,d}   {fp:9,d}")
    print(f"Actual Fake      {fn:9,d}   {tp:9,d}")

    metrics_path = OUTPUT_DIR / "test_results.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    predictions_df = pd.DataFrame({
        "actual_label": y_true,
        "actual_class": [CLASS_NAMES[int(x)] for x in y_true],
        "predicted_label": y_pred,
        "predicted_class": [CLASS_NAMES[int(x)] for x in y_pred],
        "real_probability": probabilities[:, 0],
        "fake_probability": probabilities[:, 1],
        "uncertainty": uncertainties,
        "is_uncertain": uncertain_mask,
    })
    predictions_path = OUTPUT_DIR / "predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Temporal Branch - Test Confusion Matrix")
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=np.arange(2),
        yticks=np.arange(2),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted label",
        ylabel="Actual label",
    )
    threshold = cm.max() / 2.0 if cm.size else 0
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(
                col, row, f"{cm[row, col]:,}",
                ha="center", va="center",
                color="white" if cm[row, col] > threshold else "black",
            )
    fig.tight_layout()
    confusion_path = OUTPUT_DIR / "confusion_matrix.png"
    fig.savefig(confusion_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("\nSaved outputs:")
    print(f"  Metrics          : {metrics_path}")
    print(f"  Predictions CSV  : {predictions_path}")
    print(f"  Confusion matrix : {confusion_path}")
    print("=" * 76)


if __name__ == "__main__":
    main()