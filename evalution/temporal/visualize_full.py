"""
=========================================================
Temporal Branch (VideoMAE) - Full Dataset Visualization
---------------------------------------------------------
Reads the CURRENT full-dataset Temporal Branch evaluation
results and creates report-ready visualizations.

Input:
    runs/temporal_full/test_results.json
    data/split/test.csv

Output:
    runs/temporal_full/
        figures/
            01_class_distribution.png
            02_confusion_matrix.png
            03_accuracy_comparison.png
            04_class_recall.png
            05_uncertainty_comparison.png
            06_model_vs_all_fake_baseline.png
            07_appearance_vs_temporal.png
        visualization_metrics.json
        README.txt

This file does NOT perform training or evaluation.
It only visualizes the saved evaluation results.
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

RESULT_DIR = (
    PROJECT_ROOT
    / "runs"
    / "temporal_full_ft2"
)

RESULTS_JSON = (
    RESULT_DIR
    / "test_results.json"
)

FIGURES_DIR = (
    RESULT_DIR
    / "figures"
)


# Appearance branch results
APPEARANCE_RESULTS_JSON = (
    PROJECT_ROOT
    / "runs"
    / "appearance_full"
    / "test_results.json"
)


# =========================================================
# CREATE DIRECTORIES
# =========================================================

RESULT_DIR.mkdir(
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

print("=" * 70)
print("TEMPORAL BRANCH - FULL DATASET VISUALIZATION")
print("=" * 70)


# =========================================================
# CHECK REQUIRED FILES
# =========================================================

if not TEST_CSV.exists():

    raise FileNotFoundError(
        f"Test CSV not found:\n{TEST_CSV}"
    )


if not RESULTS_JSON.exists():

    raise FileNotFoundError(
        f"Evaluation results not found:\n{RESULTS_JSON}\n\n"
        "Run temporal_evaluation.py first."
    )


# =========================================================
# LOAD TEST CSV
# =========================================================

print()
print("Loading test CSV...")

df = pd.read_csv(TEST_CSV)

print(f"Test CSV : {TEST_CSV}")
print(f"CSV rows : {len(df):,}")


# =========================================================
# CLASS DISTRIBUTION
# =========================================================

if "label" not in df.columns:

    raise ValueError(
        "The test CSV does not contain a 'label' column."
    )


label_series = (
    df["label"]
    .astype(str)
    .str.lower()
    .str.strip()
)


real_samples = int(
    (label_series == "real").sum()
)

fake_samples = int(
    (label_series == "fake").sum()
)

# Also support numeric labels if needed.
if real_samples == 0 and fake_samples == 0:

    real_samples = int(
        (label_series == "0").sum()
    )

    fake_samples = int(
        (label_series == "1").sum()
    )


total_samples = (
    real_samples
    + fake_samples
)


print()
print("Class distribution:")
print(f"  Real : {real_samples:,}")
print(f"  Fake : {fake_samples:,}")


# =========================================================
# LOAD EVALUATION RESULTS
# =========================================================

print()
print("Loading temporal evaluation results...")

with open(
    RESULTS_JSON,
    "r",
    encoding="utf-8"
) as f:

    results = json.load(f)


print(f"Results : {RESULTS_JSON}")


# =========================================================
# READ METRICS
# =========================================================

total = int(
    results["total_samples"]
)

correct = int(
    results["correct_predictions"]
)

incorrect = int(
    results["incorrect_predictions"]
)

accuracy = float(
    results["accuracy"]
)

balanced_accuracy = float(
    results["balanced_accuracy"]
)

precision = float(
    results["precision"]
)

recall = float(
    results["recall"]
)

f1 = float(
    results["f1"]
)

real_recall = float(
    results["real_recall"]
)

fake_recall = float(
    results["fake_recall"]
)


# =========================================================
# UNCERTAINTY
# =========================================================

uncertainty = results.get(
    "edl_uncertainty",
    {}
)

avg_uncertainty = float(
    uncertainty.get(
        "average",
        0.0
    )
)

correct_uncertainty = float(
    uncertainty.get(
        "correct_predictions",
        0.0
    )
)

incorrect_uncertainty = float(
    uncertainty.get(
        "incorrect_predictions",
        0.0
    )
)


# =========================================================
# CHECKPOINT INFORMATION
# =========================================================

checkpoint_epoch = results.get(
    "checkpoint_epoch",
    None
)

best_val_acc = results.get(
    "best_validation_accuracy",
    None
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

cm_data = results["confusion_matrix"]

tn = int(
    cm_data["actual_real"]["predicted_real"]
)

fp = int(
    cm_data["actual_real"]["predicted_fake"]
)

fn = int(
    cm_data["actual_fake"]["predicted_real"]
)

tp = int(
    cm_data["actual_fake"]["predicted_fake"]
)


confusion = np.array(
    [
        [tn, fp],
        [fn, tp]
    ]
)


# =========================================================
# ALL-FAKE BASELINE
# =========================================================

all_fake_accuracy = (
    fake_samples / total_samples
    if total_samples > 0
    else 0.0
)


# =========================================================
# SAVE VISUALIZATION METRICS
# =========================================================

visualization_metrics = {

    "model":
        "Temporal Branch (VideoMAE)",

    "experiment":
        "Full Dataset Test",

    "checkpoint_epoch":
        checkpoint_epoch,

    "test_samples":
        total,

    "real_samples":
        real_samples,

    "fake_samples":
        fake_samples,

    "correct_predictions":
        correct,

    "incorrect_predictions":
        incorrect,

    "accuracy":
        accuracy,

    "balanced_accuracy":
        balanced_accuracy,

    "precision":
        precision,

    "recall":
        recall,

    "f1":
        f1,

    "real_recall":
        real_recall,

    "fake_recall":
        fake_recall,

    "average_uncertainty":
        avg_uncertainty,

    "correct_uncertainty":
        correct_uncertainty,

    "incorrect_uncertainty":
        incorrect_uncertainty,

    "best_validation_accuracy":
        best_val_acc,

    "confusion_matrix":
        {
            "actual_real_pred_real": tn,
            "actual_real_pred_fake": fp,
            "actual_fake_pred_real": fn,
            "actual_fake_pred_fake": tp
        },

    "all_fake_baseline_accuracy":
        all_fake_accuracy
}


metrics_path = (
    RESULT_DIR
    / "visualization_metrics.json"
)

with open(
    metrics_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        visualization_metrics,
        f,
        indent=4
    )


print()
print(f"Saved metrics:")
print(metrics_path)


# =========================================================
# 1. CLASS DISTRIBUTION
# =========================================================

print()
print("Creating figure 1...")

plt.figure(
    figsize=(8, 6)
)

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
    "Temporal Branch Full Test Set Class Distribution",
    fontsize=14
)

plt.ylabel(
    "Number of Samples"
)

plt.xlabel(
    "Class"
)

for bar, value in zip(
    bars,
    counts
):

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

print()
print("Creating figure 2...")

plt.figure(
    figsize=(7, 6)
)

plt.imshow(
    confusion,
    interpolation="nearest"
)

plt.title(
    "Temporal Branch Full Dataset Confusion Matrix",
    fontsize=14
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "Actual Label"
)

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
            f"{confusion[i, j]:,}",
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
# 3. ACCURACY VS BALANCED ACCURACY
# =========================================================

print()
print("Creating figure 3...")

plt.figure(
    figsize=(8, 6)
)

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

plt.ylim(
    0,
    1.0
)

plt.ylabel(
    "Score"
)

plt.title(
    "Temporal Branch Accuracy vs Balanced Accuracy",
    fontsize=14
)

for bar, value in zip(
    bars,
    values
):

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

print()
print("Creating figure 4...")

plt.figure(
    figsize=(8, 6)
)

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

plt.ylim(
    0,
    1.05
)

plt.ylabel(
    "Recall"
)

plt.title(
    "Temporal Branch Class-wise Recall",
    fontsize=14
)

for bar, value in zip(
    bars,
    values
):

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

print()
print("Creating figure 5...")

plt.figure(
    figsize=(8, 6)
)

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

plt.ylabel(
    "Average Uncertainty"
)

plt.title(
    "Temporal Branch EDL Uncertainty Analysis",
    fontsize=14
)

for bar, value in zip(
    bars,
    values
):

    plt.text(
        bar.get_x()
        + bar.get_width() / 2,
        value,
        f"{value:.4f}",
        ha="center",
        va="bottom"
    )

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
# 6. MODEL VS ALL-FAKE BASELINE
# =========================================================

print()
print("Creating figure 6...")

plt.figure(
    figsize=(8, 6)
)

labels = [
    "All-Fake Baseline",
    "Temporal Branch"
]

values = [
    all_fake_accuracy,
    accuracy
]

bars = plt.bar(
    labels,
    values
)

plt.ylim(
    0.0,
    1.0
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "Temporal Branch vs All-Fake Baseline",
    fontsize=14
)

for bar, value in zip(
    bars,
    values
):

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
    / "06_model_vs_all_fake_baseline.png"
)

plt.savefig(
    path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {path}")


# =========================================================
# 7. APPEARANCE VS TEMPORAL
# =========================================================

print()
print("Creating figure 7...")

if APPEARANCE_RESULTS_JSON.exists():

    with open(
        APPEARANCE_RESULTS_JSON,
        "r",
        encoding="utf-8"
    ) as f:

        appearance_results = json.load(f)

    appearance_accuracy = float(
        appearance_results["accuracy"]
    )

    appearance_balanced_accuracy = float(
        appearance_results["balanced_accuracy"]
    )

    appearance_real_recall = float(
        appearance_results["real_recall"]
    )

    appearance_fake_recall = float(
        appearance_results["fake_recall"]
    )

    metric_names = [
        "Accuracy",
        "Balanced Accuracy",
        "Real Recall",
        "Fake Recall"
    ]

    appearance_values = [
        appearance_accuracy,
        appearance_balanced_accuracy,
        appearance_real_recall,
        appearance_fake_recall
    ]

    temporal_values = [
        accuracy,
        balanced_accuracy,
        real_recall,
        fake_recall
    ]

    x = np.arange(
        len(metric_names)
    )

    width = 0.35

    plt.figure(
        figsize=(9, 6)
    )

    bars1 = plt.bar(
        x - width / 2,
        appearance_values,
        width,
        label="Appearance Branch"
    )

    bars2 = plt.bar(
        x + width / 2,
        temporal_values,
        width,
        label="Temporal Branch"
    )

    plt.xticks(
        x,
        metric_names
    )

    plt.ylim(
        0,
        1.05
    )

    plt.ylabel(
        "Score"
    )

    plt.title(
        "Appearance vs Temporal Branch - Full Dataset",
        fontsize=14
    )

    plt.legend()

    for bar, value in zip(
        bars1,
        appearance_values
    ):

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            value,
            f"{value:.2%}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    for bar, value in zip(
        bars2,
        temporal_values
    ):

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            value,
            f"{value:.2%}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()

    path = (
        FIGURES_DIR
        / "07_appearance_vs_temporal.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {path}")

else:

    print(
        "Skipped appearance-vs-temporal chart."
    )

    print(
        f"Appearance results not found:\n"
        f"{APPEARANCE_RESULTS_JSON}"
    )


# =========================================================
# CREATE README
# =========================================================

print()
print("Creating README...")


checkpoint_text = (
    str(checkpoint_epoch)
    if checkpoint_epoch is not None
    else "Not available"
)

best_val_text = (
    f"{best_val_acc:.4f}"
    if best_val_acc is not None
    else "Not available"
)


readme = f"""
TEMPORAL BRANCH (VideoMAE) - FULL DATASET RESULTS
==================================================

This folder contains the visualizations generated from
the final full-dataset Temporal Branch evaluation.

The evaluation results were read directly from:

    runs/temporal_full/test_results.json


CHECKPOINT
--------------------------------------------------

Checkpoint epoch:
    {checkpoint_text}

Best validation accuracy:
    {best_val_text}


TEST DATASET
--------------------------------------------------

Total samples:
    {total:,}

Real:
    {real_samples:,}

Fake:
    {fake_samples:,}


MODEL RESULTS
--------------------------------------------------

Accuracy:
    {accuracy:.4f} ({accuracy:.2%})

Balanced Accuracy:
    {balanced_accuracy:.4f} ({balanced_accuracy:.2%})

Precision:
    {precision:.4f} ({precision:.2%})

Recall:
    {recall:.4f} ({recall:.2%})

F1:
    {f1:.4f} ({f1:.2%})


CLASS RECALL
--------------------------------------------------

Real Recall:
    {real_recall:.4f} ({real_recall:.2%})

Fake Recall:
    {fake_recall:.4f} ({fake_recall:.2%})


CONFUSION MATRIX
--------------------------------------------------

                 Predicted
              Real       Fake

Actual Real    {tn:6d}     {fp:6d}
Actual Fake    {fn:6d}     {tp:6d}


EDL UNCERTAINTY
--------------------------------------------------

Average:
    {avg_uncertainty:.4f}

Correct predictions:
    {correct_uncertainty:.4f}

Incorrect predictions:
    {incorrect_uncertainty:.4f}


ALL-FAKE BASELINE
--------------------------------------------------

All-Fake baseline accuracy:
    {all_fake_accuracy:.4f} ({all_fake_accuracy:.2%})

Temporal Branch accuracy:
    {accuracy:.4f} ({accuracy:.2%})


INTERPRETATION
--------------------------------------------------

The Temporal Branch is evaluated on the complete test set.

The confusion matrix shows how well the model distinguishes
Real and Fake videos.

Real Recall measures how many actual Real videos are correctly
identified as Real.

Fake Recall measures how many actual Fake videos are correctly
identified as Fake.

Balanced Accuracy is especially useful because the Real and Fake
classes may be imbalanced.

The EDL uncertainty values show the average uncertainty of the
evidential model. Lower uncertainty for correct predictions than
incorrect predictions is generally a useful sign that the
uncertainty estimate contains meaningful information.


FIGURES
--------------------------------------------------

01_class_distribution.png
02_confusion_matrix.png
03_accuracy_comparison.png
04_class_recall.png
05_uncertainty_comparison.png
06_model_vs_all_fake_baseline.png
07_appearance_vs_temporal.png


IMPORTANT
--------------------------------------------------

These figures are generated from the saved evaluation results.

If the Temporal Branch is retrained and evaluated again, simply
run temporal_evaluation.py first and then run this visualization
script again.

The visualization script does not perform training.
"""


readme_path = (
    RESULT_DIR
    / "README.txt"
)

with open(
    readme_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        readme.strip()
    )

print(
    f"Saved: {readme_path}"
)


# =========================================================
# FINAL SUMMARY
# =========================================================

print()
print("=" * 70)
print("FULL DATASET VISUALIZATION COMPLETE")
print("=" * 70)

print()
print("Output directory:")
print(RESULT_DIR)

print()
print("Figures:")

for file in sorted(
    FIGURES_DIR.glob("*.png")
):

    print(
        f"  - {file.name}"
    )

print()
print("Metrics:")
print(
    f"  - {metrics_path.name}"
)

print()
print("Documentation:")
print(
    f"  - {readme_path.name}"
)

print()
print("=" * 70)
print("Visualization complete.")
print("=" * 70)