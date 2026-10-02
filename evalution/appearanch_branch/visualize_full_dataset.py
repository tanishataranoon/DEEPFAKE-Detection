
"""
=========================================================
Appearance Branch - Full Dataset Visualization
---------------------------------------------------------

Creates report-ready visualizations from the FINAL
Appearance Branch evaluation.

Training:
    data/split/train.csv

Validation:
    data/split/val.csv

Final Test:
    data/split/test.csv

Checkpoint:
    checkpoints/appearance_full/best.pt

Evaluation results:
    checkpoints/appearance_full/test_results.json

Output:
    runs/appearance_full/
        figures/
            01_class_distribution.png
            02_confusion_matrix.png
            03_accuracy_comparison.png
            04_class_recall.png
            05_uncertainty_comparison.png
            06_three_state_predictions.png
            07_model_vs_all_fake_baseline.png

        full_dataset_metrics.json
        README.txt

Important:
    UNCERTAIN is not a learned third class.
    It is derived from EDL uncertainty.
=========================================================
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# =========================================================
# PATHS
# =========================================================

TEST_CSV = (
    PROJECT_ROOT
    / "data"
    / "split"
    / "test.csv"
)

RESULTS_JSON = (
    PROJECT_ROOT
    / "checkpoints"
    / "appearance_full"
    / "test_results.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "runs"
    / "appearance_full"
)

FIGURES_DIR = OUTPUT_DIR / "figures"


# =========================================================
# CREATE DIRECTORIES
# =========================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# HEADER
# =========================================================

print("=" * 72)
print("APPEARANCE BRANCH - FULL DATASET VISUALIZATION")
print("=" * 72)


# =========================================================
# CHECK FILES
# =========================================================

if not TEST_CSV.exists():
    raise FileNotFoundError(
        f"Test CSV not found:\n{TEST_CSV}"
    )

if not RESULTS_JSON.exists():
    raise FileNotFoundError(
        "Evaluation results not found:\n"
        f"{RESULTS_JSON}\n\n"
        "Run appearance_evaluation.py first."
    )


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(TEST_CSV)

with open(
    RESULTS_JSON,
    "r",
    encoding="utf-8"
) as f:
    results = json.load(f)


print("\nTest CSV:")
print(TEST_CSV)

print(f"\nTest samples: {len(df):,}")

print("\nEvaluation results:")
print(RESULTS_JSON)


# =========================================================
# ACTUAL CLASS DISTRIBUTION
# =========================================================

real_samples = int(
    (df["label"].astype(str).str.lower() == "real").sum()
)

fake_samples = int(
    (df["label"].astype(str).str.lower() == "fake").sum()
)

total_samples = len(df)


# =========================================================
# READ EVALUATION METRICS
# =========================================================

accuracy = float(
    results.get("accuracy", 0.0)
)

balanced_accuracy = float(
    results.get("balanced_accuracy", 0.0)
)

precision = float(
    results.get("precision", 0.0)
)

fake_recall = float(
    results.get("fake_recall", 0.0)
)

real_recall = float(
    results.get("real_recall", 0.0)
)

f1 = float(
    results.get("f1", 0.0)
)

avg_uncertainty = float(
    results.get("average_uncertainty", 0.0)
)

correct_uncertainty = float(
    results.get(
        "correct_prediction_uncertainty",
        0.0
    )
)

incorrect_uncertainty = float(
    results.get(
        "incorrect_prediction_uncertainty",
        0.0
    )
)

uncertainty_threshold = float(
    results.get(
        "uncertainty_threshold",
        0.50
    )
)

uncertain_count = int(
    results.get("uncertain_count", 0)
)

uncertain_rate = float(
    results.get("uncertain_rate", 0.0)
)

decisive_count = int(
    results.get("decisive_count", 0)
)

decisive_accuracy = float(
    results.get("decisive_accuracy", 0.0)
)

checkpoint_epoch = results.get(
    "checkpoint_epoch"
)

best_validation_balanced_accuracy = results.get(
    "best_validation_balanced_accuracy"
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = np.asarray(
    results.get(
        "confusion_matrix",
        [[0, 0], [0, 0]]
    ),
    dtype=np.int64
)

if cm.shape != (2, 2):
    raise ValueError(
        f"Expected 2x2 confusion matrix, got {cm.shape}"
    )

tn, fp = cm[0]
fn, tp = cm[1]


# =========================================================
# THREE-STATE COUNTS
# =========================================================

three_state = results.get(
    "three_state_counts",
    {}
)

real_predictions = int(
    three_state.get("real", 0)
)

fake_predictions = int(
    three_state.get("fake", 0)
)

uncertain_predictions = int(
    three_state.get("uncertain", uncertain_count)
)


# =========================================================
# PRINT SUMMARY
# =========================================================

print("\n" + "=" * 72)
print("FINAL METRICS")
print("=" * 72)

print(f"Total samples             : {total_samples:,}")
print(f"Real samples              : {real_samples:,}")
print(f"Fake samples              : {fake_samples:,}")

print(f"\nAccuracy                  : {accuracy:.4f}")
print(f"Balanced Accuracy         : {balanced_accuracy:.4f}")
print(f"Precision                 : {precision:.4f}")
print(f"Real Recall               : {real_recall:.4f}")
print(f"Fake Recall               : {fake_recall:.4f}")
print(f"F1                        : {f1:.4f}")

print("\nThree-State EDL Output")
print(f"REAL                      : {real_predictions:,}")
print(f"FAKE                      : {fake_predictions:,}")
print(f"UNCERTAIN                 : {uncertain_predictions:,}")
print(f"Uncertain Rate            : {uncertain_rate:.4f}")

print("\nConfusion Matrix")
print("                 Pred Real   Pred Fake")
print(
    f"Actual Real      {tn:9,d}   {fp:9,d}"
)
print(
    f"Actual Fake      {fn:9,d}   {tp:9,d}"
)


# =========================================================
# SAVE CONSOLIDATED METRICS
# =========================================================

metrics = {
    "model": "Appearance Branch",
    "experiment": "Full Dataset",
    "test_csv": str(TEST_CSV),

    "checkpoint_epoch": checkpoint_epoch,
    "best_validation_balanced_accuracy":
        best_validation_balanced_accuracy,

    "test_samples": total_samples,

    "real_samples": real_samples,
    "fake_samples": fake_samples,

    "accuracy": accuracy,
    "balanced_accuracy": balanced_accuracy,
    "precision": precision,
    "real_recall": real_recall,
    "fake_recall": fake_recall,
    "f1": f1,

    "confusion_matrix": {
        "actual_real_pred_real": int(tn),
        "actual_real_pred_fake": int(fp),
        "actual_fake_pred_real": int(fn),
        "actual_fake_pred_fake": int(tp),
    },

    "average_uncertainty": avg_uncertainty,
    "correct_prediction_uncertainty":
        correct_uncertainty,
    "incorrect_prediction_uncertainty":
        incorrect_uncertainty,

    "uncertainty_threshold":
        uncertainty_threshold,

    "three_state_predictions": {
        "real": real_predictions,
        "fake": fake_predictions,
        "uncertain": uncertain_predictions,
    },

    "uncertain_count": uncertain_count,
    "uncertain_rate": uncertain_rate,

    "decisive_count": decisive_count,
    "decisive_accuracy": decisive_accuracy,

    "all_fake_baseline_accuracy": (
        fake_samples / total_samples
        if total_samples
        else 0.0
    ),
}


metrics_path = (
    OUTPUT_DIR
    / "full_dataset_metrics.json"
)

with open(
    metrics_path,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        metrics,
        f,
        indent=4
    )

print(f"\nSaved metrics:")
print(metrics_path)


# =========================================================
# 1. CLASS DISTRIBUTION
# =========================================================

plt.figure(figsize=(8, 6))

classes = [
    "Real",
    "Fake"
]

counts = [
    real_samples,
    fake_samples
]

bars = plt.bar(
    classes,
    counts
)

plt.title(
    "Appearance Branch Full Test Set Class Distribution",
    fontsize=14
)

plt.ylabel("Number of Samples")
plt.xlabel("Class")

for bar, value in zip(bars, counts):
    plt.text(
        bar.get_x()
        + bar.get_width() / 2,
        bar.get_height(),
        f"{value:,}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

path = (
    FIGURES_DIR
    / "01_class_distribution.png"
)

plt.savefig(
    path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {path}")


# =========================================================
# 2. CONFUSION MATRIX
# =========================================================

plt.figure(figsize=(7, 6))

plt.imshow(cm)

plt.title(
    "Appearance Branch Full Test Confusion Matrix",
    fontsize=14
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.xticks(
    [0, 1],
    ["Real", "Fake"]
)

plt.yticks(
    [0, 1],
    ["Real", "Fake"]
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            f"{cm[i, j]:,}",
            ha="center",
            va="center",
            fontsize=14
        )

plt.colorbar(
    label="Number of Samples"
)

plt.tight_layout()

path = (
    FIGURES_DIR
    / "02_confusion_matrix.png"
)

plt.savefig(
    path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {path}")


# =========================================================
# 3. ACCURACY COMPARISON
# =========================================================

plt.figure(figsize=(8, 6))

labels = [
    "Accuracy",
    "Balanced Accuracy"
]

values = [
    accuracy,
    balanced_accuracy
]

bars = plt.bar(
    labels,
    values
)

plt.ylim(0, 1.0)

plt.ylabel("Score")

plt.title(
    "Appearance Branch Accuracy Metrics",
    fontsize=14
)

for bar, value in zip(bars, values):
    plt.text(
        bar.get_x()
        + bar.get_width() / 2,
        value,
        f"{value:.2%}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

path = (
    FIGURES_DIR
    / "03_accuracy_comparison.png"
)

plt.savefig(
    path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {path}")


# =========================================================
# 4. CLASS RECALL
# =========================================================

plt.figure(figsize=(8, 6))

labels = [
    "Real Recall",
    "Fake Recall"
]

values = [
    real_recall,
    fake_recall
]

bars = plt.bar(
    labels,
    values
)

plt.ylim(0, 1.05)

plt.ylabel("Recall")

plt.title(
    "Appearance Branch Class-wise Recall",
    fontsize=14
)

for bar, value in zip(bars, values):
    plt.text(
        bar.get_x()
        + bar.get_width() / 2,
        value,
        f"{value:.2%}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

path = (
    FIGURES_DIR
    / "04_class_recall.png"
)

plt.savefig(
    path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {path}")


# =========================================================
# 5. EDL UNCERTAINTY
# =========================================================

plt.figure(figsize=(8, 6))

labels = [
    "All Samples",
    "Correct",
    "Incorrect"
]

values = [
    avg_uncertainty,
    correct_uncertainty,
    incorrect_uncertainty
]

bars = plt.bar(
    labels,
    values
)

plt.ylabel("Average Uncertainty")

plt.title(
    "EDL Uncertainty Analysis",
    fontsize=14
)

for bar, value in zip(bars, values):
    plt.text(
        bar.get_x()
        + bar.get_width() / 2,
        value,
        f"{value:.4f}",
        ha="center",
        va="bottom"
    )

plt.axhline(
    uncertainty_threshold,
    linestyle="--",
    label=(
        f"Threshold = "
        f"{uncertainty_threshold:.2f}"
    )
)

plt.legend()

plt.tight_layout()

path = (
    FIGURES_DIR
    / "05_uncertainty_comparison.png"
)

plt.savefig(
    path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {path}")


# =========================================================
# 6. THREE-STATE EDL OUTPUT
# =========================================================

plt.figure(figsize=(8, 6))

labels = [
    "REAL",
    "FAKE",
    "UNCERTAIN"
]

values = [
    real_predictions,
    fake_predictions,
    uncertain_predictions
]

bars = plt.bar(
    labels,
    values
)

plt.ylabel("Number of Predictions")

plt.title(
    "Appearance Branch Three-State EDL Output",
    fontsize=14
)

for bar, value in zip(bars, values):
    plt.text(
        bar.get_x()
        + bar.get_width() / 2,
        bar.get_height(),
        f"{value:,}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

path = (
    FIGURES_DIR
    / "06_three_state_predictions.png"
)

plt.savefig(
    path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {path}")


# =========================================================
# 7. MODEL VS ALL-FAKE BASELINE
# =========================================================

all_fake_accuracy = (
    fake_samples / total_samples
    if total_samples
    else 0.0
)

plt.figure(figsize=(8, 6))

labels = [
    "All-Fake Baseline",
    "Appearance Branch"
]

values = [
    all_fake_accuracy,
    accuracy
]

bars = plt.bar(
    labels,
    values
)

plt.ylim(0.0, 1.0)

plt.ylabel("Accuracy")

plt.title(
    "Appearance Branch vs All-Fake Baseline",
    fontsize=14
)

for bar, value in zip(bars, values):
    plt.text(
        bar.get_x()
        + bar.get_width() / 2,
        value,
        f"{value:.2%}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

path = (
    FIGURES_DIR
    / "07_model_vs_all_fake_baseline.png"
)

plt.savefig(
    path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {path}")


# =========================================================
# README
# =========================================================

readme = f"""
APPEARANCE BRANCH - FULL DATASET EXPERIMENT
==================================================

This folder contains the final visualization and metrics
for the Appearance Branch trained on the full training
split.

DATA
--------------------------------------------------

Training:
    data/split/train.csv

Validation:
    data/split/val.csv

Final test:
    data/split/test.csv

Test samples:
    {total_samples:,}

Real:
    {real_samples:,}

Fake:
    {fake_samples:,}


MODEL
--------------------------------------------------

Checkpoint:
    checkpoints/appearance_full/best.pt

Checkpoint epoch:
    {checkpoint_epoch}

Best validation balanced accuracy:
    {best_validation_balanced_accuracy}


FINAL TEST RESULTS
--------------------------------------------------

Accuracy:
    {accuracy:.4f} ({accuracy:.2%})

Balanced Accuracy:
    {balanced_accuracy:.4f} ({balanced_accuracy:.2%})

Precision:
    {precision:.4f} ({precision:.2%})

Real Recall:
    {real_recall:.4f} ({real_recall:.2%})

Fake Recall:
    {fake_recall:.4f} ({fake_recall:.2%})

F1:
    {f1:.4f} ({f1:.2%})


CONFUSION MATRIX
--------------------------------------------------

                 Predicted
              Real       Fake

Actual Real   {tn:8,d}   {fp:8,d}
Actual Fake   {fn:8,d}   {tp:8,d}


EDL UNCERTAINTY
--------------------------------------------------

Average uncertainty:
    {avg_uncertainty:.4f}

Correct prediction uncertainty:
    {correct_uncertainty:.4f}

Incorrect prediction uncertainty:
    {incorrect_uncertainty:.4f}

Uncertainty threshold:
    {uncertainty_threshold:.2f}


THREE-STATE OUTPUT
--------------------------------------------------

REAL:
    {real_predictions:,}

FAKE:
    {fake_predictions:,}

UNCERTAIN:
    {uncertain_predictions:,}

Uncertain rate:
    {uncertain_rate:.4f} ({uncertain_rate:.2%})

Decisive predictions:
    {decisive_count:,}

Accuracy on decisive predictions:
    {decisive_accuracy:.4f} ({decisive_accuracy:.2%})


ALL-FAKE BASELINE
--------------------------------------------------

If every test sample were classified as Fake:

    Accuracy = {all_fake_accuracy:.4f}
    ({all_fake_accuracy:.2%})


INTERPRETATION
--------------------------------------------------

The test split is used only for final evaluation.

The validation split is used during training for model
selection and early stopping.

UNCERTAIN is NOT a learned third class.

The model learns two classes:

    0 = Real
    1 = Fake

UNCERTAIN is derived from the EDL uncertainty value
using the configured uncertainty threshold.

The confusion matrix and balanced accuracy should be
considered together with raw accuracy because the dataset
contains unequal numbers of Real and Fake samples.


GENERATED FIGURES
--------------------------------------------------

01_class_distribution.png
02_confusion_matrix.png
03_accuracy_comparison.png
04_class_recall.png
05_uncertainty_comparison.png
06_three_state_predictions.png
07_model_vs_all_fake_baseline.png
"""


readme_path = (
    OUTPUT_DIR
    / "README.txt"
)

with open(
    readme_path,
    "w",
    encoding="utf-8"
) as f:
    f.write(readme.strip())

print(f"Saved: {readme_path}")


# =========================================================
# FINAL
# =========================================================

print("\n" + "=" * 72)
print("FULL DATASET VISUALIZATION COMPLETE")
print("=" * 72)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nFigures:")

for file in sorted(
    FIGURES_DIR.glob("*.png")
):
    print(f"  - {file.name}")

print("\nMetrics:")
print(f"  - {metrics_path.name}")

print("\nDocumentation:")
print(f"  - {readme_path.name}")

print("\nEverything is ready for the full-dataset report.")
print("=" * 72)

