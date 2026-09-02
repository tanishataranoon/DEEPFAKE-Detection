
"""
=========================================================
Appearance Branch - 150 Source Group Evaluation
---------------------------------------------------------
Evaluates the trained Appearance Branch on:

    data/split/test_150/test.csv

Checkpoint:

    checkpoints/appearance_test_150/best.pt

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

Important:
    This script contains a compatibility patch for the
    pinned remote AIMv2 implementation used by the project
    with the installed Transformers version.
=========================================================
"""

# =========================================================
# IMPORTS
# =========================================================

import torch
import numpy as np

from torch.utils.data import DataLoader
from torch.amp import autocast

import transformers
import sys
sys.path.append(".")


# =========================================================
# AIMv2 / TRANSFORMERS COMPATIBILITY PATCH
# =========================================================
#
# Your pinned AIMv2 implementation uses:
#
#     _tied_weights_keys
#
# while the installed Transformers version expects:
#
#     all_tied_weights_keys
#
# The failure happens inside:
#
#     _move_missing_keys_from_meta_to_device()
#
# before our checkpoint is loaded.
#
# Therefore patch the Transformers method itself.
# =========================================================

_original_move_missing_keys = (
    transformers.modeling_utils.PreTrainedModel
    ._move_missing_keys_from_meta_to_device
)


def _patched_move_missing_keys_from_meta_to_device(
    self,
    missing_keys,
    *args,
    **kwargs,
):

    # -----------------------------------------------------
    # Compatibility alias
    # -----------------------------------------------------

    if not hasattr(self, "all_tied_weights_keys"):

        tied_keys = getattr(
            self,
            "_tied_weights_keys",
            {},
        )

        # Set an instance-level compatibility attribute.
        #
        # Transformers expects a dictionary here.
        self.all_tied_weights_keys = (
            tied_keys if tied_keys is not None else {}
        )

    return _original_move_missing_keys(
        self,
        missing_keys,
        *args,
        **kwargs,
    )


transformers.modeling_utils.PreTrainedModel \
    ._move_missing_keys_from_meta_to_device = (
        _patched_move_missing_keys_from_meta_to_device
    )


# =========================================================
# PROJECT IMPORTS
# =========================================================

from data.raw.appearanch.dataset import (
    DeepfakeDataset,
)

from models.appearence.appearance_classifier import (
    AppearanceClassifier,
)

from models.heads.evidence_head import (
    compute_dirichlet,
)


# =========================================================
# CONFIG
# =========================================================

VAL_CSV = "data/split/test_150/test.csv"

DATASET_ROOT = "data/raw"

CHECKPOINT = (
    "checkpoints/appearance_test_150/best.pt"
)

BATCH_SIZE = 16

NUM_WORKERS = 4

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =========================================================
# MAIN
# =========================================================

def main():

    device = torch.device(DEVICE)

    print("=" * 70)
    print(
        "APPEARANCE BRANCH - "
        "150 SOURCE GROUP TEST EVALUATION"
    )
    print("=" * 70)

    print()
    print(f"Device       : {DEVICE}")
    print(f"Checkpoint   : {CHECKPOINT}")
    print(f"Test CSV     : {VAL_CSV}")

    # -----------------------------------------------------
    # GPU information
    # -----------------------------------------------------

    if DEVICE == "cuda":

        print(
            f"GPU          : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print()

    # =====================================================
    # DATASET
    # =====================================================

    dataset = DeepfakeDataset(
        metadata_csv=VAL_CSV,
        dataset_root=DATASET_ROOT,
        split="test",
        image_size=224,
    )

    print(
        f"Test samples : {len(dataset):,}"
    )

    print()
    print("Label distribution:")
    print(
        dataset.df["label"].value_counts()
    )

    # -----------------------------------------------------
    # DataLoader
    # -----------------------------------------------------

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        drop_last=False,
        persistent_workers=(
            NUM_WORKERS > 0
        ),
        prefetch_factor=2
        if NUM_WORKERS > 0
        else None,
    )

    print()
    print(
        f"Evaluation batches : {len(loader):,}"
    )

    # =====================================================
    # MODEL
    # =====================================================

    print()
    print("Loading Appearance Classifier...")

    model = AppearanceClassifier(
        num_classes=2,
        freeze_backbone=True,
    ).to(device)

    print("Appearance Classifier created.")

    # =====================================================
    # CHECKPOINT
    # =====================================================

    print()
    print("Loading checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    print()
    print("Checkpoint keys:")

    if isinstance(checkpoint, dict):

        for key in checkpoint.keys():

            print(
                f"  {key}"
            )

    # -----------------------------------------------------
    # Validate checkpoint format
    # -----------------------------------------------------

    if not isinstance(
        checkpoint,
        dict,
    ):

        raise RuntimeError(
            "Checkpoint is not a dictionary."
        )

    if "model_state" not in checkpoint:

        raise KeyError(
            "Checkpoint does not contain "
            "'model_state'.\n"
            f"Available keys: "
            f"{list(checkpoint.keys())}"
        )

    state_dict = checkpoint[
        "model_state"
    ]

    # =====================================================
    # LOAD MODEL WEIGHTS
    # =====================================================

    print()
    print("Loading trained model weights...")

    missing_keys, unexpected_keys = (
        model.load_state_dict(
            state_dict,
            strict=False,
        )
    )

    # -----------------------------------------------------
    # Check loading
    # -----------------------------------------------------

    if missing_keys:

        print()
        print(
            "WARNING - Missing keys:"
        )

        for key in missing_keys:

            print(
                f"  {key}"
            )

    if unexpected_keys:

        print()
        print(
            "WARNING - Unexpected keys:"
        )

        for key in unexpected_keys:

            print(
                f"  {key}"
            )

    # -----------------------------------------------------
    # Evaluation mode
    # -----------------------------------------------------

    model.eval()

    if DEVICE == "cuda":

        torch.cuda.empty_cache()

    print()
    print(
        "Checkpoint loaded successfully."
    )

    # =====================================================
    # STORAGE
    # =====================================================

    all_predictions = []

    all_labels = []

    all_uncertainties = []

    correct_uncertainties = []

    incorrect_uncertainties = []

    # -----------------------------------------------------
    # Confusion matrix
    #
    # class 0 = Real
    # class 1 = Fake
    #
    #              Predicted
    #             Real    Fake
    #
    # Actual Real
    # Actual Fake
    # -----------------------------------------------------

    confusion = np.zeros(
        (2, 2),
        dtype=np.int64,
    )

    # =====================================================
    # EVALUATION
    # =====================================================

    print()
    print("=" * 70)
    print("STARTING TEST EVALUATION")
    print("=" * 70)
    print()

    with torch.no_grad():

        for step, batch in enumerate(
            loader
        ):

            # -------------------------------------------------
            # Move data to GPU
            # -------------------------------------------------

            clip = batch[
                "clip"
            ].to(
                device,
                non_blocking=True,
            )

            labels = batch[
                "label"
            ].to(
                device,
                non_blocking=True,
            )

            # -------------------------------------------------
            # Forward
            # -------------------------------------------------

            with autocast(
                device_type="cuda",
                enabled=(
                    DEVICE == "cuda"
                ),
            ):

                outputs = model(
                    clip
                )

            # -------------------------------------------------
            # Evidence
            # -------------------------------------------------

            evidence = outputs[
                "evidence"
            ]

            attention = outputs[
                "attention"
            ]

            # -------------------------------------------------
            # Evidence -> Dirichlet
            # -------------------------------------------------

            dirichlet = compute_dirichlet(
                evidence
            )

            probabilities = (
                dirichlet[
                    "probability"
                ]
            )

            uncertainty = (
                dirichlet[
                    "uncertainty"
                ]
            )

            # -------------------------------------------------
            # Prediction
            # -------------------------------------------------

            predictions = torch.argmax(
                probabilities,
                dim=1,
            )

            # -------------------------------------------------
            # CPU conversion
            # -------------------------------------------------

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

            # -------------------------------------------------
            # Store
            # -------------------------------------------------

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
            # Correct / incorrect uncertainty
            # -------------------------------------------------

            for i in range(
                len(label_values)
            ):

                if (
                    prediction_values[i]
                    ==
                    label_values[i]
                ):

                    correct_uncertainties.append(
                        uncertainty_values[i]
                    )

                else:

                    incorrect_uncertainties.append(
                        uncertainty_values[i]
                    )

            # -------------------------------------------------
            # Confusion matrix
            # -------------------------------------------------

            for true, pred in zip(
                label_values,
                prediction_values,
            ):

                confusion[
                    int(true),
                    int(pred),
                ] += 1

            # -------------------------------------------------
            # Progress
            # -------------------------------------------------

            if (
                (step + 1) % 25 == 0
                or
                (step + 1)
                == len(loader)
            ):

                print(
                    f"Evaluated batch "
                    f"{step + 1:,}/"
                    f"{len(loader):,}"
                )

    # =====================================================
    # CONVERT
    # =====================================================

    labels = np.asarray(
        all_labels
    )

    predictions = np.asarray(
        all_predictions
    )

    # =====================================================
    # BASIC METRICS
    # =====================================================

    total = len(labels)

    correct = np.sum(
        predictions == labels
    )

    accuracy = (
        correct / total
        if total > 0
        else 0.0
    )

    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    tn = int(
        confusion[0, 0]
    )

    fp = int(
        confusion[0, 1]
    )

    fn = int(
        confusion[1, 0]
    )

    tp = int(
        confusion[1, 1]
    )

    # =====================================================
    # REAL RECALL
    #
    # Real = class 0
    # =====================================================

    real_recall = (

        tn / (tn + fp)

        if (tn + fp) > 0

        else 0.0
    )

    # =====================================================
    # FAKE RECALL
    #
    # Fake = class 1
    # =====================================================

    fake_recall = (

        tp / (tp + fn)

        if (tp + fn) > 0

        else 0.0
    )

    # =====================================================
    # BALANCED ACCURACY
    # =====================================================

    balanced_accuracy = (
        real_recall
        +
        fake_recall
    ) / 2.0

    # =====================================================
    # PRECISION
    # =====================================================

    precision = (

        tp / (tp + fp)

        if (tp + fp) > 0

        else 0.0
    )

    # =====================================================
    # F1
    # =====================================================

    if (
        precision + fake_recall
    ) > 0:

        f1 = (

            2.0
            *
            precision
            *
            fake_recall
            /
            (
                precision
                +
                fake_recall
            )
        )

    else:

        f1 = 0.0

    # =====================================================
    # EDL UNCERTAINTY
    # =====================================================

    avg_uncertainty = (

        np.mean(
            all_uncertainties
        )

        if all_uncertainties

        else 0.0
    )

    correct_uncertainty = (

        np.mean(
            correct_uncertainties
        )

        if correct_uncertainties

        else 0.0
    )

    incorrect_uncertainty = (

        np.mean(
            incorrect_uncertainties
        )

        if incorrect_uncertainties

        else 0.0
    )

    # =====================================================
    # RESULTS
    # =====================================================

    print()
    print("=" * 70)
    print(
        "APPEARANCE BRANCH - "
        "150 SOURCE GROUP TEST RESULTS"
    )
    print("=" * 70)

    print()
    print(
        f"Total samples        : "
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

    # -----------------------------------------------------
    # Overall metrics
    # -----------------------------------------------------

    print()
    print("Overall Metrics")
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

    # -----------------------------------------------------
    # Class recall
    # -----------------------------------------------------

    print()
    print("Class Recall")
    print("-" * 35)

    print(
        f"Real Recall          : "
        f"{real_recall:.4f}"
    )

    print(
        f"Fake Recall          : "
        f"{fake_recall:.4f}"
    )

    # -----------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------

    print()
    print("Confusion Matrix")
    print("-" * 35)

    print()
    print(
        "                 Predicted"
    )

    print(
        "                 Real    Fake"
    )

    print(
        f"Actual Real      "
        f"{tn:6d}  {fp:6d}"
    )

    print(
        f"Actual Fake      "
        f"{fn:6d}  {tp:6d}"
    )

    # -----------------------------------------------------
    # EDL uncertainty
    # -----------------------------------------------------

    print()
    print("EDL Uncertainty")
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

    # -----------------------------------------------------
    # Checkpoint information
    # -----------------------------------------------------

    if "epoch" in checkpoint:

        print()
        print(
            f"Checkpoint epoch     : "
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
    # FINISH
    # =====================================================

    print()
    print("=" * 70)
    print("Evaluation complete.")
    print("=" * 70)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()
