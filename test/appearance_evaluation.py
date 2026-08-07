"""
Appearance Branch Evaluation

Evaluates the trained Appearance Branch on the validation set.

Metrics:
- Accuracy
- Balanced Accuracy
- Precision
- Recall
- F1
- Confusion Matrix
- Real Recall
- Fake Recall
- EDL Uncertainty
"""

import torch
import numpy as np

from torch.utils.data import DataLoader

from data.dataset.appearance.deepfake_dataset import DeepfakeDataset
from models.appearence.appearance_classifier import AppearanceClassifier
from models.heads.evidence_head import compute_dirichlet


# ============================================================
# CONFIG
# ============================================================

VAL_CSV = "data/split/val.csv"
DATASET_ROOT = "data/raw"

CHECKPOINT = "checkpoints/appearance/best.pt"

BATCH_SIZE = 16
NUM_WORKERS = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# MAIN
# ============================================================

def main():

    device = torch.device(DEVICE)

    print("=" * 60)
    print("APPEARANCE BRANCH EVALUATION")
    print("=" * 60)

    print(f"Device     : {DEVICE}")
    print(f"Checkpoint : {CHECKPOINT}")

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = DeepfakeDataset(
        metadata_csv=VAL_CSV,
        dataset_root=DATASET_ROOT,
        split="val",
        image_size=224,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        drop_last=False,
    )

    print(f"Validation samples : {len(dataset)}")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = AppearanceClassifier(
        num_classes=2,
        freeze_backbone=True,
    ).to(device)

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    print()
    print("Checkpoint keys:")

    if isinstance(checkpoint, dict):
        for key in checkpoint.keys():
            print(f"  {key}")

    # Trainer checkpoint format:
    #
    # {
    #     "epoch": ...,
    #     "model_state": ...,
    #     "optimizer_state": ...,
    #     "scaler_state": ...,
    #     "best_val_acc": ...,
    #     "val_acc": ...
    # }

    if "model_state" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain 'model_state'. "
            f"Available keys: {list(checkpoint.keys())}"
        )

    state_dict = checkpoint["model_state"]

    # --------------------------------------------------------
    # Load model weights
    # --------------------------------------------------------

    model.load_state_dict(state_dict)

    model.eval()

    print()
    print("Checkpoint loaded successfully.")
    print()

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    all_predictions = []
    all_labels = []
    all_uncertainties = []

    correct_uncertainties = []
    incorrect_uncertainties = []

    # Confusion matrix:
    #
    #              Predicted
    #             Real  Fake
    #
    # Actual Real
    # Actual Fake
    #
    confusion = np.zeros((2, 2), dtype=np.int64)

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    with torch.no_grad():

        for step, batch in enumerate(loader):

            clip = batch["clip"].to(
                device,
                non_blocking=True,
            )

            labels = batch["label"].to(device)

            # -----------------------------------------------
            # Forward
            # -----------------------------------------------

            outputs = model(clip)

            evidence = outputs["evidence"]
            attention = outputs["attention"]

            # -----------------------------------------------
            # EDL -> Dirichlet
            # -----------------------------------------------

            dirichlet = compute_dirichlet(evidence)

            probabilities = dirichlet["probability"]

            uncertainty = dirichlet["uncertainty"]

            # -----------------------------------------------
            # Prediction
            # -----------------------------------------------

            predictions = torch.argmax(
                probabilities,
                dim=1,
            )

            # -----------------------------------------------
            # Store predictions and labels
            # -----------------------------------------------

            prediction_values = predictions.cpu().numpy()
            label_values = labels.cpu().numpy()

            all_predictions.extend(
                prediction_values
            )

            all_labels.extend(
                label_values
            )

            # -----------------------------------------------
            # Uncertainty
            # -----------------------------------------------

            uncertainty_values = (
                uncertainty
                .view(-1)
                .cpu()
                .numpy()
            )

            all_uncertainties.extend(
                uncertainty_values
            )

            # -----------------------------------------------
            # Correct / incorrect uncertainty
            # -----------------------------------------------

            for i in range(len(label_values)):

                if prediction_values[i] == label_values[i]:

                    correct_uncertainties.append(
                        uncertainty_values[i]
                    )

                else:

                    incorrect_uncertainties.append(
                        uncertainty_values[i]
                    )

            # -----------------------------------------------
            # Confusion matrix
            # -----------------------------------------------

            for true, pred in zip(
                label_values,
                prediction_values,
            ):

                confusion[true, pred] += 1

            # -----------------------------------------------
            # Progress
            # -----------------------------------------------

            if (step + 1) % 100 == 0:

                print(
                    f"Evaluated batch "
                    f"{step + 1}/{len(loader)}"
                )

    # ========================================================
    # Convert
    # ========================================================

    labels = np.asarray(all_labels)
    predictions = np.asarray(all_predictions)

    # ========================================================
    # Metrics
    # ========================================================

    total = len(labels)

    correct = np.sum(
        predictions == labels
    )

    accuracy = (
        correct / total
        if total > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    tn = confusion[0, 0]
    fp = confusion[0, 1]
    fn = confusion[1, 0]
    tp = confusion[1, 1]

    # --------------------------------------------------------
    # Real Recall
    #
    # Real = class 0
    # --------------------------------------------------------

    real_recall = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Fake Recall
    #
    # Fake = class 1
    # --------------------------------------------------------

    fake_recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Balanced Accuracy
    # --------------------------------------------------------

    balanced_accuracy = (
        real_recall + fake_recall
    ) / 2.0

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    # --------------------------------------------------------
    # F1
    # --------------------------------------------------------

    if precision + fake_recall > 0:

        f1 = (
            2
            * precision
            * fake_recall
            / (precision + fake_recall)
        )

    else:

        f1 = 0.0

    # ========================================================
    # EDL Uncertainty
    # ========================================================

    avg_uncertainty = (
        np.mean(all_uncertainties)
        if len(all_uncertainties) > 0
        else 0.0
    )

    correct_uncertainty = (
        np.mean(correct_uncertainties)
        if len(correct_uncertainties) > 0
        else 0.0
    )

    incorrect_uncertainty = (
        np.mean(incorrect_uncertainties)
        if len(incorrect_uncertainties) > 0
        else 0.0
    )

    # ========================================================
    # Print Results
    # ========================================================

    print()
    print("=" * 60)
    print("APPEARANCE BRANCH RESULTS")
    print("=" * 60)

    print()
    print(f"Total samples      : {total}")

    print()
    print("Overall Metrics")
    print("-" * 30)

    print(f"Accuracy            : {accuracy:.4f}")
    print(
        f"Balanced Accuracy   : "
        f"{balanced_accuracy:.4f}"
    )
    print(f"Precision           : {precision:.4f}")
    print(f"Recall              : {fake_recall:.4f}")
    print(f"F1                  : {f1:.4f}")

    print()
    print("Class Recall")
    print("-" * 30)

    print(
        f"Real Recall         : "
        f"{real_recall:.4f}"
    )

    print(
        f"Fake Recall         : "
        f"{fake_recall:.4f}"
    )

    print()
    print("Confusion Matrix")
    print("-" * 30)

    print()
    print("              Predicted")
    print("              Real    Fake")

    print(
        f"Actual Real   "
        f"{tn:6d}  {fp:6d}"
    )

    print(
        f"Actual Fake   "
        f"{fn:6d}  {tp:6d}"
    )

    print()
    print("EDL Uncertainty")
    print("-" * 30)

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

    print()
    print("=" * 60)
    print("Evaluation complete.")
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()