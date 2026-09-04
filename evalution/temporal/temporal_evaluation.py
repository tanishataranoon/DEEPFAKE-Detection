"""
=========================================================
Temporal Branch (VideoMAE) - Portion Test Evaluation
---------------------------------------------------------
Evaluates checkpoints/temporal_test_150/best.pt on
data/split/test_150/test.csv. Same report format as
150_appreanch_evaluate.py for direct comparison.
=========================================================
"""

import sys
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.amp import autocast

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.raw.temporal.temporal_dataset import TemporalDeepfakeDataset
from models.temporal.temporal_classifier import TemporalClassifier
from models.heads.evidence_head import compute_dirichlet

TEST_CSV = "data/split/test_150/test.csv"
DATASET_ROOT = "data/raw"
CHECKPOINT = "checkpoints/temporal_test_150/best.pt"

BATCH_SIZE = 4
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    device = torch.device(DEVICE)

    print("=" * 70)
    print("TEMPORAL BRANCH (VideoMAE) - PORTION TEST EVALUATION")
    print("=" * 70)
    print(f"Device      : {DEVICE}")
    print(f"Checkpoint  : {CHECKPOINT}")
    print(f"Test CSV    : {TEST_CSV}")

    if DEVICE == "cuda":
        print(f"GPU         : {torch.cuda.get_device_name(0)}")

    # -------------------------------------------------
    # split="test" selects the deterministic (no-augmentation)
    # transform from transforms.py automatically — no separate
    # transform argument needed here.
    # -------------------------------------------------

    dataset = TemporalDeepfakeDataset(
        metadata_csv=TEST_CSV,
        dataset_root=DATASET_ROOT,
        split="test",
        image_size=224,
    )

    print(f"\nTest samples : {len(dataset):,}")
    print("\nLabel distribution:")
    print(dataset.df["label"].value_counts())

    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"), drop_last=False,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )

    print(f"\nEvaluation batches : {len(loader):,}")

    print("\nLoading Temporal Classifier...")
    model = TemporalClassifier(num_classes=2, freeze_backbone=True).to(device)
    print("Temporal Classifier created.")

    print("\nLoading checkpoint...")
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)

    print("\nCheckpoint keys:")
    for key in checkpoint.keys():
        print(f"  {key}")

    if "model_state" not in checkpoint:
        raise KeyError(f"Checkpoint does not contain 'model_state'. Keys: {list(checkpoint.keys())}")

    print("\nLoading trained model weights...")
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model_state"], strict=False)

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

    all_predictions, all_labels, all_uncertainties = [], [], []
    correct_uncertainties, incorrect_uncertainties = [], []
    confusion = np.zeros((2, 2), dtype=np.int64)

    print()
    print("=" * 70)
    print("STARTING TEST EVALUATION")
    print("=" * 70)
    print()

    with torch.no_grad():
        for step, batch in enumerate(loader):
            clip = batch["clip"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            with autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
                outputs = model(clip)

            evidence = outputs["evidence"]
            dirichlet = compute_dirichlet(evidence)
            probabilities = dirichlet["probability"]
            uncertainty = dirichlet["uncertainty"]

            predictions = torch.argmax(probabilities, dim=1)
            prediction_values = predictions.detach().cpu().numpy()
            label_values = labels.detach().cpu().numpy()
            uncertainty_values = uncertainty.view(-1).detach().cpu().numpy()

            all_predictions.extend(prediction_values)
            all_labels.extend(label_values)
            all_uncertainties.extend(uncertainty_values)

            for i in range(len(label_values)):
                if prediction_values[i] == label_values[i]:
                    correct_uncertainties.append(uncertainty_values[i])
                else:
                    incorrect_uncertainties.append(uncertainty_values[i])

            for true, pred in zip(label_values, prediction_values):
                confusion[int(true), int(pred)] += 1

            if (step + 1) % 25 == 0 or (step + 1) == len(loader):
                print(f"Evaluated batch {step + 1:,}/{len(loader):,}")

    labels = np.asarray(all_labels)
    predictions = np.asarray(all_predictions)
    total = len(labels)
    correct = np.sum(predictions == labels)
    accuracy = correct / total if total > 0 else 0.0

    tn, fp = int(confusion[0, 0]), int(confusion[0, 1])
    fn, tp = int(confusion[1, 0]), int(confusion[1, 1])

    real_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fake_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    balanced_accuracy = (real_recall + fake_recall) / 2.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2.0 * precision * fake_recall / (precision + fake_recall)
          if (precision + fake_recall) > 0 else 0.0)

    avg_uncertainty = np.mean(all_uncertainties) if all_uncertainties else 0.0
    correct_uncertainty = np.mean(correct_uncertainties) if correct_uncertainties else 0.0
    incorrect_uncertainty = np.mean(incorrect_uncertainties) if incorrect_uncertainties else 0.0

    print()
    print("=" * 70)
    print("TEMPORAL BRANCH - PORTION TEST RESULTS")
    print("=" * 70)
    print(f"\nTotal samples        : {total:,}")
    print(f"Correct predictions  : {correct:,}")
    print(f"Incorrect predictions: {total - correct:,}")
    print("\nOverall Metrics")
    print("-" * 35)
    print(f"Accuracy             : {accuracy:.4f}")
    print(f"Balanced Accuracy    : {balanced_accuracy:.4f}")
    print(f"Precision            : {precision:.4f}")
    print(f"Recall               : {fake_recall:.4f}")
    print(f"F1                   : {f1:.4f}")
    print("\nClass Recall")
    print("-" * 35)
    print(f"Real Recall          : {real_recall:.4f}")
    print(f"Fake Recall          : {fake_recall:.4f}")
    print("\nConfusion Matrix")
    print("-" * 35)
    print("\n                 Predicted")
    print("                 Real    Fake")
    print(f"Actual Real      {tn:6d}  {fp:6d}")
    print(f"Actual Fake      {fn:6d}  {tp:6d}")
    print("\nEDL Uncertainty")
    print("-" * 35)
    print(f"Average uncertainty   : {avg_uncertainty:.4f}")
    print(f"Correct predictions   : {correct_uncertainty:.4f}")
    print(f"Incorrect predictions : {incorrect_uncertainty:.4f}")

    if "epoch" in checkpoint:
        print(f"\nCheckpoint epoch     : {checkpoint['epoch']}")
    if "best_val_acc" in checkpoint:
        print(f"Best validation acc  : {checkpoint['best_val_acc']:.4f}")
    if "val_acc" in checkpoint:
        print(f"Checkpoint val acc   : {checkpoint['val_acc']:.4f}")

    print()
    print("=" * 70)
    print("Evaluation complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()