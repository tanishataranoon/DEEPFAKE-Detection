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
    29,915

Real:
    3,148

Fake:
    26,767


MODEL
--------------------------------------------------

Checkpoint:
    checkpoints/appearance_full/best.pt

Checkpoint epoch:
    12

Best validation balanced accuracy:
    0.8556942163977888


FINAL TEST RESULTS
--------------------------------------------------

Accuracy:
    0.9499 (94.99%)

Balanced Accuracy:
    0.8980 (89.80%)

Precision:
    0.9799 (97.99%)

Real Recall:
    0.8323 (83.23%)

Fake Recall:
    0.9637 (96.37%)

F1:
    0.9717 (97.17%)


CONFUSION MATRIX
--------------------------------------------------

                 Predicted
              Real       Fake

Actual Real      2,620        528
Actual Fake        972     25,795


EDL UNCERTAINTY
--------------------------------------------------

Average uncertainty:
    0.0268

Correct prediction uncertainty:
    0.0216

Incorrect prediction uncertainty:
    0.1259

Uncertainty threshold:
    0.50


THREE-STATE OUTPUT
--------------------------------------------------

REAL:
    3,520

FAKE:
    26,256

UNCERTAIN:
    139

Uncertain rate:
    0.0046 (0.46%)

Decisive predictions:
    29,776

Accuracy on decisive predictions:
    0.9521 (95.21%)


ALL-FAKE BASELINE
--------------------------------------------------

If every test sample were classified as Fake:

    Accuracy = 0.8948
    (89.48%)


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