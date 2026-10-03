Appearance Branch XAI
=====================

Checkpoint:
    E:\Depfake\DEEPFAKE-Detection\checkpoints\appearance_full\best.pt

Method:
    gradcam

Target:
    pred

Backbone:
    AIMv2-Large patch14-224

Input:
    16 frames
    224 x 224

Patch grid:
    16 x 16

Patch count:
    256

EDL uncertainty threshold:
    0.5

Important:
    Heatmaps represent model attribution/sensitivity.
    They are not proof of causal reasoning.

Frame attention comes directly from the model's
AttentionPooling layer.

CAM importance is computed from the spatial attribution
mass of each frame.

Generated files:
    *.png
        Visual explanation figures.

    *_cam.npy
        Raw CAM arrays.

    summary.csv
        Prediction, probability, uncertainty,
        evidence and frame-level information.

The original checkpoint is never modified.