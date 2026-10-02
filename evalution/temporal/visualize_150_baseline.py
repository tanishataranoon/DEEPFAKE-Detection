"""
========================================================
Temporal Branch - 150 Source Group Baseline Visualization
---------------------------------------------------------
Creates report-ready visualizations from the CURRENT
Temporal Branch (VideoMAE) evaluation result.

Output:
    runs/temporal_test_150/baseline/
        figures/
            01_class_distribution.png
            02_confusion_matrix.png
            03_accuracy_comparison.png
            04_class_recall.png
            05_uncertainty_comparison.png
            06_model_vs_all_fake_baseline.png
            07_appearance_vs_temporal.png

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
    PROJECT_ROOT / "data" / "split" / "test_150" / "test.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "runs" / "temporal_test_150" / "baseline"
)

FIGURES_DIR = OUTPUT_DIR / "figures"

# Path to the appearance baseline, for the comparison chart.
APPEARANCE_METRICS_PATH = (
    PROJECT_ROOT
    / "runs"
    / "appearance_test_150"
    / "baseline"
    / "baseline_metrics.json"
)


# =========================================================
# CURRENT EVALUATION RESULTS (from your temporal eval run)
#
# DO NOT change these values unless you intentionally rerun
# evaluation and want to create a new baseline.
# =========================================================

TOTAL_SAMPLES = 11151

REAL_SAMPLES = 1539
FAKE_SAMPLES = 9612

CORRECT = 8749
INCORRECT = 2402

ACCURACY = 0.7846
BALANCED_ACCURACY = 0.7774

FAKE_PRECISION = 0.9548
FAKE_RECALL = 0.7873
FAKE_F1 = 0.8630

REAL_RECALL = 0.7674

AVG_UNCERTAINTY = 0.4161
CORRECT_UNCERTAINTY = 0.3925
INCORRECT_UNCERTAINTY = 0.5021

CHECKPOINT_EPOCH = 4
BEST_VAL_ACC = 0.7842

CONFUSION = {
    "actual_real_pred_real": 1181,
    "actual_real_pred_fake": 358,
    "actual_fake_pred_real": 2044,
    "actual_fake_pred_fake": 7568,
}


# =========================================================
# CREATE DIRECTORIES
# =========================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# LOAD TEST CSV
# =========================================================

print("=" * 60)
print("TEMPORAL BRANCH BASELINE VISUALIZATION")
print("=" * 60)

if not TEST_CSV.exists():
    raise FileNotFoundError(f"Test CSV not found:\n{TEST_CSV}")

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
    "model": "Temporal Branch (VideoMAE)",
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

    "confusion_matrix": CONFUSION,

    "all_fake_baseline_accuracy": FAKE_SAMPLES / TOTAL_SAMPLES,
}

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
plt.title("Temporal Branch Test Set Class Distribution", fontsize=14)
plt.ylabel("Number of Samples")
plt.xlabel("Class")
for bar, value in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
              f"{value:,}", ha="center", va="bottom")
plt.tight_layout()
path = FIGURES_DIR / "01_class_distribution.png"
plt.savefig(path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {path}")


# =========================================================
# 2. CONFUSION MATRIX
# =========================================================

cm = np.array([
    [CONFUSION["actual_real_pred_real"], CONFUSION["actual_real_pred_fake"]],
    [CONFUSION["actual_fake_pred_real"], CONFUSION["actual_fake_pred_fake"]],
])

plt.figure(figsize=(7, 6))
plt.imshow(cm)
plt.title("Temporal Branch Confusion Matrix", fontsize=14)
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.xticks([0, 1], ["Real", "Fake"])
plt.yticks([0, 1], ["Real", "Fake"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=14)
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
labels = ["Accuracy", "Balanced Accuracy"]
values = [ACCURACY, BALANCED_ACCURACY]
bars = plt.bar(labels, values)
plt.ylim(0, 1.0)
plt.ylabel("Score")
plt.title("Accuracy vs Balanced Accuracy", fontsize=14)
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2, value,
              f"{value:.2%}", ha="center", va="bottom")
plt.tight_layout()
path = FIGURES_DIR / "03_accuracy_comparison.png"
plt.savefig(path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {path}")


# =========================================================
# 4. CLASS RECALL
# =========================================================

plt.figure(figsize=(8, 6))
labels = ["Real Recall", "Fake Recall"]
values = [REAL_RECALL, FAKE_RECALL]
bars = plt.bar(labels, values)
plt.ylim(0, 1.05)
plt.ylabel("Recall")
plt.title("Class-wise Recall", fontsize=14)
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2, value,
              f"{value:.2%}", ha="center", va="bottom")
plt.tight_layout()
path = FIGURES_DIR / "04_class_recall.png"
plt.savefig(path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {path}")


# =========================================================
# 5. EDL UNCERTAINTY
# =========================================================

plt.figure(figsize=(8, 6))
labels = ["All Samples", "Correct", "Incorrect"]
values = [AVG_UNCERTAINTY, CORRECT_UNCERTAINTY, INCORRECT_UNCERTAINTY]
bars = plt.bar(labels, values)
plt.ylabel("Average Uncertainty")
plt.title("EDL Uncertainty Analysis", fontsize=14)
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2, value,
              f"{value:.4f}", ha="center", va="bottom")
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
labels = ["All-Fake Baseline", "Temporal Branch"]
values = [all_fake_accuracy, ACCURACY]
bars = plt.bar(labels, values)
plt.ylim(0.0, 1.0)
plt.ylabel("Accuracy")
plt.title("Temporal Branch vs All-Fake Baseline", fontsize=14)
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2, value,
              f"{value:.2%}", ha="center", va="bottom")
plt.tight_layout()
path = FIGURES_DIR / "06_model_vs_all_fake_baseline.png"
plt.savefig(path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {path}")


# =========================================================
# 7. APPEARANCE VS TEMPORAL (only if appearance baseline exists)
# =========================================================

if APPEARANCE_METRICS_PATH.exists():
    with open(APPEARANCE_METRICS_PATH, "r", encoding="utf-8") as f:
        appearance_metrics = json.load(f)

    plt.figure(figsize=(9, 6))

    metric_names = ["Accuracy", "Balanced Accuracy", "Real Recall", "Fake Recall"]
    appearance_values = [
        appearance_metrics["accuracy"],
        appearance_metrics["balanced_accuracy"],
        appearance_metrics["real_recall"],
        appearance_metrics["fake_recall"],
    ]
    temporal_values = [ACCURACY, BALANCED_ACCURACY, REAL_RECALL, FAKE_RECALL]

    x = np.arange(len(metric_names))
    width = 0.35

    plt.bar(x - width / 2, appearance_values, width, label="Appearance Branch")
    plt.bar(x + width / 2, temporal_values, width, label="Temporal Branch")

    plt.xticks(x, metric_names)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Appearance vs Temporal Branch (150-Group Test)", fontsize=14)
    plt.legend()

    for i, (a_val, t_val) in enumerate(zip(appearance_values, temporal_values)):
        plt.text(i - width / 2, a_val, f"{a_val:.2%}", ha="center", va="bottom", fontsize=9)
        plt.text(i + width / 2, t_val, f"{t_val:.2%}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = FIGURES_DIR / "07_appearance_vs_temporal.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
else:
    print(
        f"\nSkipped appearance-vs-temporal chart — "
        f"{APPEARANCE_METRICS_PATH} not found."
    )


# =========================================================
# CREATE README
# =========================================================

readme = f"""
TEMPORAL BRANCH (VideoMAE) - 150 SOURCE GROUP BASELINE
==================================================

This folder contains the visualization and metrics from
the Temporal Branch's first working checkpoint, before
fusion with the Appearance Branch.

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

Actual Real      {CONFUSION['actual_real_pred_real']}        {CONFUSION['actual_real_pred_fake']}
Actual Fake      {CONFUSION['actual_fake_pred_real']}        {CONFUSION['actual_fake_pred_fake']}


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

Unlike the Appearance Branch (which collapsed to predicting
FAKE for nearly every sample, Real Recall = 0.06%), the
Temporal Branch shows genuine class discrimination:

    Real Recall  = {REAL_RECALL:.2%}
    Fake Recall  = {FAKE_RECALL:.2%}

Both recalls are close in value and well above the
all-fake baseline's Real Recall of ~0%, confirming the
WeightedRandomSampler and EDL setup are working as intended
on this branch.

Correct predictions carry lower average uncertainty
({CORRECT_UNCERTAINTY:.4f}) than incorrect ones
({INCORRECT_UNCERTAINTY:.4f}), meaning the evidential
head's uncertainty signal is informative — this matters for
the eventual Dempster-Shafer fusion with the Appearance
Branch.

Most errors ({CONFUSION['actual_fake_pred_real']} of
{INCORRECT} total) are Fake videos misclassified as Real,
not the reverse. Worth accounting for in fusion design.

DO NOT DELETE THIS BASELINE.

It should be retained for comparison after:
    - full-dataset training (not just 150 groups)
    - partial backbone unfreezing
    - fusion with the Appearance Branch

Figures:
    01_class_distribution.png
    02_confusion_matrix.png
    03_accuracy_comparison.png
    04_class_recall.png
    05_uncertainty_comparison.png
    06_model_vs_all_fake_baseline.png
    07_appearance_vs_temporal.png (if appearance baseline found)
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
print("  The Temporal Branch shows genuine real-vs-fake")
print("  discrimination, unlike the Appearance Branch's")
print("  all-fake collapse. Ready for full-dataset training")
print("  or partial unfreezing before fusion.")