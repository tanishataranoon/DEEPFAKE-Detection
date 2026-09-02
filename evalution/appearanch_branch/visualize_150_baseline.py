"""
=========================================================
Appearance Branch - 150 Source Group Baseline Visualization
---------------------------------------------------------
Creates report-ready visualizations from the CURRENT
Appearance Branch evaluation result.

IMPORTANT:
    This script represents the model BEFORE any fixes or
    architecture/training changes.

Output:
    runs/appearance_test_150/baseline/
        figures/
            01_class_distribution.png
            02_confusion_matrix.png
            03_accuracy_comparison.png
            04_class_recall.png
            05_uncertainty_comparison.png
            06_model_vs_all_fake_baseline.png

        baseline_metrics.json
        README.txt
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

TEST_CSV = (
    PROJECT_ROOT
    / "data"
    / "split"
    / "test_150"
    / "test.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "runs"
    / "appearance_test_150"
    / "baseline"
)

FIGURES_DIR = OUTPUT_DIR / "figures"


# =========================================================
# CURRENT EVALUATION RESULTS
# =========================================================
#
# These values come directly from the evaluation you just ran.
#
# DO NOT change these values unless you intentionally rerun
# evaluation and want to create a new baseline.
# =========================================================

TOTAL_SAMPLES = 11151

REAL_SAMPLES = 1539
FAKE_SAMPLES = 9612

CORRECT = 9613
INCORRECT = 1538

ACCURACY = 0.8621
BALANCED_ACCURACY = 0.5003

FAKE_PRECISION = 0.8621
FAKE_RECALL = 1.0000
FAKE_F1 = 0.9259

REAL_RECALL = 0.0006

AVG_UNCERTAINTY = 0.1559
CORRECT_UNCERTAINTY = 0.1494
INCORRECT_UNCERTAINTY = 0.1965

CHECKPOINT_EPOCH = 1
BEST_VAL_ACC = 0.8570


# =========================================================
# CREATE DIRECTORIES
# =========================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# LOAD TEST CSV
# =========================================================

print("=" * 60)
print("APPEARANCE BRANCH BASELINE VISUALIZATION")
print("=" * 60)

if not TEST_CSV.exists():
    raise FileNotFoundError(
        f"Test CSV not found:\n{TEST_CSV}"
    )

df = pd.read_csv(TEST_CSV)

print(f"\nTest CSV:")
print(TEST_CSV)

print(f"\nCSV rows: {len(df):,}")

if "label" in df.columns:
    print("\nActual CSV label distribution:")

    label_counts = df["label"].value_counts().sort_index()
    print(label_counts)

    for label, count in label_counts.items():
        label_name = "Real" if label == "real" else "Fake"

        print(f"  {label_name}: {count:,}")


# =========================================================
# METRICS DICTIONARY
# =========================================================

metrics = {
    "model": "Appearance Branch",
    "experiment": "150 Source Group Test",
    "checkpoint_epoch": CHECKPOINT_EPOCH,

    "test_samples": TOTAL_SAMPLES,

    "real_samples": REAL_SAMPLES,
    "fake_samples": FAKE_SAMPLES,

    "correct_predictions": CORRECT,
    "incorrect_predictions": INCORRECT,

    "accuracy": ACCURACY,
    "balanced_accuracy": BALANCED_ACCURACY,

    "fake_precision": FAKE_PRECISION,
    "fake_recall": FAKE_RECALL,
    "fake_f1": FAKE_F1,

    "real_recall": REAL_RECALL,

    "average_uncertainty": AVG_UNCERTAINTY,
    "correct_uncertainty": CORRECT_UNCERTAINTY,
    "incorrect_uncertainty": INCORRECT_UNCERTAINTY,

    "best_validation_accuracy": BEST_VAL_ACC,

    "confusion_matrix": {
        "actual_real_pred_real": 1,
        "actual_real_pred_fake": 1538,
        "actual_fake_pred_real": 0,
        "actual_fake_pred_fake": 9612,
    },

    "all_fake_baseline_accuracy": FAKE_SAMPLES / TOTAL_SAMPLES,
}


# =========================================================
# SAVE METRICS
# =========================================================

metrics_path = OUTPUT_DIR / "baseline_metrics.json"

with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

print(f"\nSaved metrics:")
print(metrics_path)


# =========================================================
# 1. CLASS DISTRIBUTION
# =========================================================

plt.figure(figsize=(8, 6))

classes = ["Real", "Fake"]
counts = [REAL_SAMPLES, FAKE_SAMPLES]

bars = plt.bar(classes, counts)

plt.title(
    "Appearance Branch Test Set Class Distribution",
    fontsize=14
)

plt.ylabel("Number of Samples")
plt.xlabel("Class")

for bar, value in zip(bars, counts):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:,}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

path = FIGURES_DIR / "01_class_distribution.png"
plt.savefig(path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved: {path}")


# =========================================================
# 2. CONFUSION MATRIX
# =========================================================

cm = np.array([
    [1, 1538],
    [0, 9612]
])

plt.figure(figsize=(7, 6))

plt.imshow(cm)

plt.title(
    "Appearance Branch Confusion Matrix",
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

plt.colorbar(label="Number of Samples")

plt.tight_layout()

path = FIGURES_DIR / "02_confusion_matrix.png"
plt.savefig(path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved: {path}")


# =========================================================
# 3. ACCURACY VS BALANCED ACCURACY
# =========================================================

plt.figure(figsize=(8, 6))

labels = [
    "Accuracy",
    "Balanced Accuracy"
]

values = [
    ACCURACY,
    BALANCED_ACCURACY
]

bars = plt.bar(
    labels,
    values
)

plt.ylim(0, 1.0)

plt.ylabel("Score")
plt.title(
    "Accuracy vs Balanced Accuracy",
    fontsize=14
)

for bar, value in zip(bars, values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value,
        f"{value:.2%}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

path = FIGURES_DIR / "03_accuracy_comparison.png"
plt.savefig(path, dpi=300, bbox_inches="tight")
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
    REAL_RECALL,
    FAKE_RECALL
]

bars = plt.bar(
    labels,
    values
)

plt.ylim(0, 1.05)

plt.ylabel("Recall")
plt.title(
    "Class-wise Recall",
    fontsize=14
)

for bar, value in zip(bars, values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value,
        f"{value:.2%}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

path = FIGURES_DIR / "04_class_recall.png"
plt.savefig(path, dpi=300, bbox_inches="tight")
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
    AVG_UNCERTAINTY,
    CORRECT_UNCERTAINTY,
    INCORRECT_UNCERTAINTY
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
        bar.get_x() + bar.get_width() / 2,
        value,
        f"{value:.4f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

path = FIGURES_DIR / "05_uncertainty_comparison.png"
plt.savefig(path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved: {path}")


# =========================================================
# 6. MODEL VS ALL-FAKE BASELINE
# =========================================================

all_fake_accuracy = FAKE_SAMPLES / TOTAL_SAMPLES

plt.figure(figsize=(8, 6))

labels = [
    "All-Fake Baseline",
    "Appearance Branch"
]

values = [
    all_fake_accuracy,
    ACCURACY
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
        bar.get_x() + bar.get_width() / 2,
        value,
        f"{value:.2%}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

path = FIGURES_DIR / "06_model_vs_all_fake_baseline.png"
plt.savefig(path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved: {path}")


# =========================================================
# CREATE README
# =========================================================

readme = f"""
APPEARANCE BRANCH - 150 SOURCE GROUP BASELINE
==================================================

This folder contains the visualization and metrics from
the Appearance Branch BEFORE making any model/training fixes.

Checkpoint:
    Epoch: {CHECKPOINT_EPOCH}

Test samples:
    {TOTAL_SAMPLES:,}

Class distribution:
    Real: {REAL_SAMPLES:,}
    Fake: {FAKE_SAMPLES:,}

MODEL RESULTS
--------------------------------------------------

Accuracy:
    {ACCURACY:.4f} ({ACCURACY:.2%})

Balanced Accuracy:
    {BALANCED_ACCURACY:.4f} ({BALANCED_ACCURACY:.2%})

Real Recall:
    {REAL_RECALL:.4f} ({REAL_RECALL:.2%})

Fake Recall:
    {FAKE_RECALL:.4f} ({FAKE_RECALL:.2%})

Fake Precision:
    {FAKE_PRECISION:.4f} ({FAKE_PRECISION:.2%})

Fake F1:
    {FAKE_F1:.4f} ({FAKE_F1:.2%})


CONFUSION MATRIX
--------------------------------------------------

                 Predicted
              Real       Fake

Actual Real      1        1538
Actual Fake      0        9612


EDL UNCERTAINTY
--------------------------------------------------

Average:
    {AVG_UNCERTAINTY:.4f}

Correct predictions:
    {CORRECT_UNCERTAINTY:.4f}

Incorrect predictions:
    {INCORRECT_UNCERTAINTY:.4f}


IMPORTANT INTERPRETATION
--------------------------------------------------

The model predicts almost every test sample as FAKE.

This produces a high raw accuracy because the test set is
strongly imbalanced toward fake samples.

However:

    Accuracy            = {ACCURACY:.2%}
    Balanced Accuracy  = {BALANCED_ACCURACY:.2%}
    Real Recall         = {REAL_RECALL:.2%}
    Fake Recall         = {FAKE_RECALL:.2%}

The all-fake baseline achieves:

    {all_fake_accuracy:.2%} accuracy

Therefore the Appearance Branch is currently performing
approximately at the majority-class baseline.

This indicates class-collapse / majority-class prediction
rather than successful real-vs-fake discrimination.

DO NOT DELETE THIS BASELINE.

It should be retained for comparison after fixing:
    - EDL loss
    - class imbalance handling
    - evidence head
    - label mapping
    - checkpoint selection
    - training configuration
    - or other verified issues.

Figures:
    01_class_distribution.png
    02_confusion_matrix.png
    03_accuracy_comparison.png
    04_class_recall.png
    05_uncertainty_comparison.png
    06_model_vs_all_fake_baseline.png
"""

readme_path = OUTPUT_DIR / "README.txt"

with open(readme_path, "w", encoding="utf-8") as f:
    f.write(readme.strip())

print(f"Saved: {readme_path}")


# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n" + "=" * 60)
print("BASELINE VISUALIZATION COMPLETE")
print("=" * 60)

print(f"\nOutput directory:")
print(OUTPUT_DIR)

print("\nFigures:")

for file in sorted(FIGURES_DIR.glob("*.png")):
    print(f"  - {file.name}")

print("\nMetrics:")
print(f"  - {metrics_path.name}")

print("\nDocumentation:")
print(f"  - {readme_path.name}")

print("\nCurrent conclusion:")
print("  The Appearance Branch is effectively predicting FAKE")
print("  for almost every sample.")

print("\nIMPORTANT:")
print("  Do NOT change the model yet.")
print("  Preserve this as the BEFORE/FIX baseline.")