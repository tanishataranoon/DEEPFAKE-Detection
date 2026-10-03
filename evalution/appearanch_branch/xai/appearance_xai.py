"""
=========================================================
Appearance Branch XAI
---------------------------------------------------------
Project-specific Grad-CAM / LayerCAM explanation for the
trained Appearance Branch.

Architecture:

    16 face frames
          |
          v
    AIMv2-Large
    patch14-224
          |
          v
    256 patch tokens
          |
          v
    Mean pooling
          |
          v
    1024-D frame features
          |
          v
    AttentionPooling
          |
          +----> 16 frame attention weights
          |
          v
    EvidenceHead
          |
          +----> Real evidence
          +----> Fake evidence
          |
          v
    EDL / Dirichlet
          |
          +----> P(Real)
          +----> P(Fake)
          +----> Uncertainty
          |
          v
    REAL / FAKE / UNCERTAIN

XAI target:
    - pred
    - fake
    - real
    - uncertainty

Methods:
    - Grad-CAM
    - LayerCAM

IMPORTANT:
    This file DOES NOT train or modify best.pt.

    For XAI, the AIMv2 backbone must allow gradients.
    Therefore freeze_backbone=False is intentionally used
    when loading the already-trained checkpoint.

Run from project root:

    python evalution/appearanch_branch/xai/appearance_xai.py --inspect

Then:

    python evalution/appearanch_branch/xai/appearance_xai.py --auto-pick 4 --scan 400

Or specific test indices:

    python evalution/appearanch_branch/xai/appearance_xai.py --indices 10 250 4001

Example with uncertainty maps:

    python evalution/appearanch_branch/xai/appearance_xai.py ^
        --auto-pick 4 ^
        --scan 400 ^
        --with-uncertainty ^
        --method gradcam ^
        --tag first_xai

Outputs:

    runs/
        appearance_xai/
            first_xai/
                correct_real_....png
                correct_fake_....png
                false_fake_....png
                false_real_....png
                uncertain_....png
                *_cam.npy
                summary.csv
                README.txt
=========================================================
"""

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# =========================================================
# PROJECT ROOT
# =========================================================

# File:
# evalution/appearanch_branch/xai/appearance_xai.py
#
# parents[0] = xai
# parents[1] = appearanch_branch
# parents[2] = evalution
# parents[3] = project root

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================
# PROJECT IMPORTS
# =========================================================

from data.raw.cached_dataset import CachedClipDataset

from models.appearence.appearance_classifier import (
    AppearanceClassifier,
)

from models.heads.evidence_head import (
    compute_dirichlet,
)


# =========================================================
# CONFIGURATION
# =========================================================

CKPT_PATH = PROJECT_ROOT / "checkpoints" / "appearance_full" / "best.pt"

TEST_CSV = PROJECT_ROOT / "data" / "split" / "test.csv"

TEST_CACHE = PROJECT_ROOT / "data" / "cache" / "test"

OUT_ROOT = PROJECT_ROOT / "runs" / "appearance_xai"

UNCERTAINTY_THRESHOLD = 0.50

CLASS_NAMES = [
    "REAL",
    "FAKE",
]

NUM_CLASSES = 2

EXPECTED_FRAMES = 16

IMAGE_SIZE = 224

PATCH_SIZE = 14

EXPECTED_PATCH_GRID = IMAGE_SIZE // PATCH_SIZE

# 224 / 14 = 16
# 16 x 16 = 256 patches

EXPECTED_PATCHES = (
    EXPECTED_PATCH_GRID * EXPECTED_PATCH_GRID
)


# =========================================================
# MODEL
# =========================================================

def build_model(device):
    """
    Build the exact AppearanceClassifier used by the project
    and load checkpoints/appearance_full/best.pt.

    IMPORTANT:
        freeze_backbone=False is intentional.

    The normal evaluator uses freeze_backbone=True and therefore
    wraps AIMv2 inference in torch.no_grad().

    Grad-CAM requires gradients through AIMv2, so XAI must use
    freeze_backbone=False.

    This DOES NOT retrain the model.
    """

    print("\n[model] building AppearanceClassifier...")

    model = AppearanceClassifier(
        num_classes=2,
        freeze_backbone=False,
        uncertainty_threshold=UNCERTAINTY_THRESHOLD,
    ).to(device)

    if not CKPT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CKPT_PATH}"
        )

    checkpoint = torch.load(
        CKPT_PATH,
        map_location=device,
        weights_only=False,
    )

    if isinstance(checkpoint, dict):
        if "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]

        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            # Sometimes the checkpoint itself is a state dict.
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    # Remove DataParallel prefix if present.
    cleaned_state = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[7:]

        cleaned_state[key] = value

    result = model.load_state_dict(
        cleaned_state,
        strict=True,
    )

    print(
        f"[model] checkpoint loaded: {CKPT_PATH}"
    )

    print(
        f"[model] missing keys   : {len(result.missing_keys)}"
    )

    print(
        f"[model] unexpected keys: {len(result.unexpected_keys)}"
    )

    # XAI requires gradients.
    for parameter in model.backbone.backbone.parameters():
        parameter.requires_grad = True

    # IMPORTANT:
    #
    # AppearanceBackbone.forward() only uses no_grad when
    # self.freeze == True.
    #
    # Therefore explicitly disable its freeze flag.
    model.backbone.backbone.freeze = False

    model.eval()

    # Make absolutely sure AIMv2 remains in eval mode.
    model.backbone.backbone.eval()

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"[model] parameters        : {total:,}"
    )

    print(
        f"[model] gradient-enabled   : {trainable:,}"
    )

    print(
        "[model] XAI gradient mode  : ENABLED"
    )

    return model


# =========================================================
# DATASET
# =========================================================

def build_dataset():
    """
    Build exactly the same test dataset used by the final
    Appearance evaluation.
    """

    if not TEST_CSV.exists():
        raise FileNotFoundError(
            f"Test CSV not found:\n{TEST_CSV}"
        )

    if not TEST_CACHE.exists():
        raise FileNotFoundError(
            f"Test cache not found:\n{TEST_CACHE}"
        )

    print("\n[dataset] loading test dataset...")

    dataset = CachedClipDataset(
        str(TEST_CSV),
        str(TEST_CACHE),
    )

    print(
        f"[dataset] test samples: {len(dataset):,}"
    )

    return dataset


# =========================================================
# MODEL FORWARD
# =========================================================

def run_model(model, clip):
    """
    clip:
        (1,T,3,H,W)

    Returns:
        AppearanceClassifier dictionary.
    """

    return model(clip)


# =========================================================
# CHECKPOINT / EDL HELPERS
# =========================================================

def extract_evidence(output):
    """
    Extract the actual EDL evidence from the AppearanceClassifier
    output.

    Your AppearanceClassifier.forward() returns:

        {
            "evidence": evidence,
            "features": features,
            "attention": attention
        }
    """

    if not isinstance(output, dict):
        raise RuntimeError(
            "AppearanceClassifier output is not a dictionary."
        )

    if "evidence" not in output:
        raise RuntimeError(
            "AppearanceClassifier output does not contain 'evidence'."
        )

    evidence = output["evidence"]

    if evidence.ndim == 1:
        evidence = evidence.unsqueeze(0)

    if evidence.shape[-1] != NUM_CLASSES:
        raise RuntimeError(
            "Unexpected evidence shape: "
            f"{tuple(evidence.shape)}"
        )

    return evidence


def calculate_edl(output):
    """
    Use the project's actual compute_dirichlet() implementation.
    """

    evidence = extract_evidence(output)

    result = compute_dirichlet(
        evidence,
        uncertainty_threshold=UNCERTAINTY_THRESHOLD,
    )

    return evidence, result


# =========================================================
# TRANSFORMER BLOCK DISCOVERY
# =========================================================

def get_transformer_blocks(model):
    """
    Use the exact block discovery logic from AppearanceModel.

    Primary:
        model.backbone.backbone.backbone.trunk.blocks

    Fallback:
        model.backbone.backbone.backbone.encoder.layers
    """

    aimv2 = model.backbone.backbone.backbone

    if hasattr(aimv2, "trunk") and hasattr(
        aimv2.trunk,
        "blocks",
    ):
        blocks = aimv2.trunk.blocks

        print(
            f"[blocks] using AIMv2 trunk.blocks "
            f"({len(blocks)} blocks)"
        )

        return blocks

    if hasattr(aimv2, "encoder") and hasattr(
        aimv2.encoder,
        "layers",
    ):
        blocks = aimv2.encoder.layers

        print(
            f"[blocks] using AIMv2 encoder.layers "
            f"({len(blocks)} layers)"
        )

        return blocks

    raise RuntimeError(
        "Could not locate AIMv2 transformer blocks.\n"
        "Expected either:\n"
        "  model.backbone.backbone.backbone.trunk.blocks\n"
        "or:\n"
        "  model.backbone.backbone.backbone.encoder.layers"
    )


# =========================================================
# GRADIENT HOOK
# =========================================================

class TransformerCAMHook:
    """
    Captures the output of one AIMv2 transformer block and
    retains its gradient.

    The expected output is approximately:

        (B*T, N, D)

    where:

        N = 256 spatial patch tokens
        D = 1024 feature dimension

    The class is intentionally tolerant of tuple/list outputs.
    """

    def __init__(
        self,
        model,
        block_index=-1,
    ):

        blocks = get_transformer_blocks(model)

        if len(blocks) == 0:
            raise RuntimeError(
                "AIMv2 has no transformer blocks."
            )

        self.blocks = blocks

        self.block_index = (
            block_index
            if block_index >= 0
            else len(blocks) + block_index
        )

        if not (
            0 <= self.block_index < len(blocks)
        ):
            raise IndexError(
                f"Invalid block index {block_index}. "
                f"AIMv2 has {len(blocks)} blocks."
            )

        self.block = blocks[self.block_index]

        self.name = (
            f"AIMv2 block "
            f"{self.block_index}/{len(blocks)-1}"
        )

        self.activation = None
        self.gradient = None

        self.handle = self.block.register_forward_hook(
            self._forward_hook
        )

        print(
            f"[hook] registered on {self.name}"
        )

    def _forward_hook(
        self,
        module,
        inputs,
        output,
    ):

        tensor = output

        if isinstance(
            output,
            (tuple, list),
        ):
            tensor = output[0]

        if not torch.is_tensor(tensor):
            raise RuntimeError(
                "Transformer block output is not a tensor."
            )

        self.activation = tensor

        if tensor.requires_grad:

            tensor.register_hook(
                self._gradient_hook
            )

    def _gradient_hook(
        self,
        gradient,
    ):

        self.gradient = gradient

    def clear(self):

        self.activation = None
        self.gradient = None

    def remove(self):

        self.handle.remove()


# =========================================================
# TOKEN SHAPE HANDLING
# =========================================================

def normalize_transformer_tokens(
    tensor,
    batch_frames,
):
    """
    Convert transformer output into:

        (B*T, N, D)

    Expected AIMv2 case:

        (B*T, 256, 1024)

    The helper also handles a few common alternative layouts.
    """

    if tensor is None:
        raise RuntimeError(
            "Transformer activation/gradient is None."
        )

    if tensor.ndim == 3:

        # Standard AIMv2 case:
        # (B*T, N, D)
        if tensor.shape[0] == batch_frames:
            return tensor

        # Possible:
        # (B, T*N, D)
        if tensor.shape[1] % batch_frames == 0:

            n = tensor.shape[1] // batch_frames

            return tensor.reshape(
                -1,
                n,
                tensor.shape[2],
            )

    if tensor.ndim == 4:

        # (B,T,N,D)
        if tensor.shape[1] == batch_frames:

            return tensor.reshape(
                -1,
                tensor.shape[2],
                tensor.shape[3],
            )

        # (B,T,H,W,D)
        if tensor.shape[1] == batch_frames:

            return tensor.reshape(
                -1,
                tensor.shape[2] * tensor.shape[3],
                tensor.shape[4],
            )

    raise RuntimeError(
        "Unexpected transformer tensor shape "
        f"{tuple(tensor.shape)} for "
        f"{batch_frames} frames."
    )


def select_patch_tokens(
    activation,
    gradient,
):
    """
    Extract the spatial patch tokens.

    AIMv2 patch14-224 gives:

        224 / 14 = 16
        16 x 16 = 256 patches

    If there are extra non-spatial tokens, the final 256 tokens
    are used.

    Returns:

        activation: (B*T,256,D)
        gradient:   (B*T,256,D)
    """

    num_tokens = activation.shape[1]

    if num_tokens < EXPECTED_PATCHES:
        raise RuntimeError(
            f"Transformer output has only {num_tokens} tokens, "
            f"but {EXPECTED_PATCHES} patch tokens are expected."
        )

    if num_tokens != EXPECTED_PATCHES:

        print(
            f"[cam] transformer has {num_tokens} tokens; "
            f"using final {EXPECTED_PATCHES} patch tokens."
        )

    activation = activation[
        :, -EXPECTED_PATCHES:, :
    ]

    gradient = gradient[
        :, -EXPECTED_PATCHES:, :
    ]

    return activation, gradient


# =========================================================
# GRAD-CAM / LAYERCAM
# =========================================================

def make_cam(
    hook,
    batch_frames,
    method="gradcam",
):
    """
    Generate one 16x16 attribution map per frame.

    Grad-CAM:

        weights = mean(gradient over patches)

        CAM = ReLU(sum(weights * activation))

    LayerCAM:

        CAM = ReLU(gradient) * activation

    Returns:

        Tensor (T,16,16)
    """

    if hook.activation is None:
        raise RuntimeError(
            "No transformer activation captured."
        )

    if hook.gradient is None:
        raise RuntimeError(
            "No transformer gradient captured."
        )

    activation = normalize_transformer_tokens(
        hook.activation.detach(),
        batch_frames,
    )

    gradient = normalize_transformer_tokens(
        hook.gradient.detach(),
        batch_frames,
    )

    activation, gradient = select_patch_tokens(
        activation,
        gradient,
    )

    if method == "gradcam":

        weights = gradient.mean(
            dim=1,
            keepdim=True,
        )

        raw = (
            weights * activation
        ).sum(dim=-1)

    elif method == "layercam":

        raw = (
            F.relu(gradient) * activation
        ).sum(dim=-1)

    else:
        raise ValueError(
            f"Unknown CAM method: {method}"
        )

    cam = F.relu(raw)

    # Transformer attribution maps can sometimes become
    # entirely zero after ReLU. Use magnitude as a fallback.
    if float(cam.max()) <= 0:

        cam = raw.abs()

    cam = cam.reshape(
        -1,
        EXPECTED_PATCH_GRID,
        EXPECTED_PATCH_GRID,
    )

    return cam


# =========================================================
# FRAME DISPLAY
# =========================================================

def to_display(frames):
    """
    Convert cached frames into displayable RGB images.

    CachedClipDataset returns uint8 [0,255], and the evaluator
    converts them to float [0,1].

    This function therefore handles both forms safely.
    """

    x = frames.detach().cpu().float()

    if x.max() > 1.01:
        x = x / 255.0

    # If a normalized tensor somehow reaches this function,
    # handle the project's CLIP normalization.
    elif x.min() < -0.01:

        mean = torch.tensor(
            [
                0.48145466,
                0.4578275,
                0.40821073,
            ]
        ).view(1, 3, 1, 1)

        std = torch.tensor(
            [
                0.26862954,
                0.26130258,
                0.27577711,
            ]
        ).view(1, 3, 1, 1)

        x = x * std + mean

    x = x.clamp(
        0.0,
        1.0,
    )

    return x.permute(
        0,
        2,
        3,
        1,
    ).numpy()


# =========================================================
# CAM UPSAMPLING
# =========================================================

def upsample_cams(
    cam,
    height,
    width,
):
    """
    Convert 16x16 CAM into 224x224 heatmaps.

    A single global maximum is used across all frames so that
    frame-to-frame intensities remain comparable.
    """

    cam_tensor = cam.float()

    upsampled = F.interpolate(
        cam_tensor[:, None],
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )[:, 0]

    maximum = upsampled.max()

    if float(maximum) > 0:

        upsampled = (
            upsampled / (maximum + 1e-8)
        )

    return upsampled.clamp(
        0,
        1,
    ).numpy()


# =========================================================
# FRAME IMPORTANCE
# =========================================================

def calculate_frame_importance(
    cam,
):
    """
    Calculate spatial CAM mass for each frame.

    Returns a normalized T-length vector.
    """

    mass = cam.reshape(
        cam.shape[0],
        -1,
    ).sum(dim=1)

    total = mass.sum()

    if float(total) > 0:
        mass = mass / total

    return mass


def normalize_attention(attention):
    """
    Normalize the model's AttentionPooling weights.

    Your AttentionPooling already uses softmax, so the values
    should sum to 1.
    """

    attention = attention.detach().float().cpu().view(-1)

    total = attention.sum()

    if float(total) > 0:
        attention = attention / total

    return attention


# =========================================================
# STATE
# =========================================================

def state_of(
    probability_fake,
    uncertainty,
):
    """
    Project's three-state reporting:

        uncertainty >= 0.50 -> UNCERTAIN
        otherwise:
            P(fake) >= 0.5 -> FAKE
            else -> REAL
    """

    if uncertainty >= UNCERTAINTY_THRESHOLD:
        return "UNCERTAIN"

    if probability_fake >= 0.5:
        return "FAKE"

    return "REAL"


# =========================================================
# EXPLANATION
# =========================================================

def explain(
    model,
    hook,
    clip,
    target="pred",
    method="gradcam",
):
    """
    Run one explanation.

    Input:

        clip = (T,3,H,W)

    Target:

        pred
            Explain evidence for the model's predicted class.

        fake
            Explain Fake evidence.

        real
            Explain Real evidence.

        uncertainty
            Explain the EDL uncertainty quantity.

    Returns:

        cam
        scores
        frame_attention
        frame_cam_importance
    """

    device = next(
        model.parameters()
    ).device

    clip = clip.float()

    # Cached dataset normally gives uint8.
    if clip.max() > 1.01:
        clip = clip / 255.0

    clip = clip.to(
        device,
        non_blocking=True,
    )

    clip = clip.unsqueeze(0)

    T = clip.shape[1]

    hook.clear()

    model.zero_grad(
        set_to_none=True
    )

    with torch.enable_grad():

        output = run_model(
            model,
            clip,
        )

        evidence, edl = calculate_edl(
            output
        )

        probability = edl["probability"]

        uncertainty = (
            edl["uncertainty"]
            .view(-1)
        )

        binary_prediction = (
            edl["binary_prediction"]
        )

        if target == "uncertainty":

            target_quantity = (
                uncertainty[0]
            )

        else:

            if target == "real":
                target_class = 0

            elif target == "fake":
                target_class = 1

            elif target == "pred":
                target_class = int(
                    binary_prediction[0].item()
                )

            else:
                raise ValueError(
                    f"Invalid target: {target}"
                )

            target_quantity = (
                evidence[0, target_class]
            )

        target_quantity.backward()

    if hook.gradient is None:

        raise RuntimeError(
            "No gradient reached the AIMv2 block.\n"
            "Possible causes:\n"
            "  - wrong hook layer\n"
            "  - model is running under no_grad\n"
            "  - AIMv2 freeze flag is still True"
        )

    cam = make_cam(
        hook,
        batch_frames=T,
        method=method,
    )

    attention = output.get(
        "attention",
        None,
    )

    if attention is None:
        frame_attention = torch.zeros(T)

    else:
        frame_attention = normalize_attention(
            attention[0]
        )

    frame_cam_importance = calculate_frame_importance(
        cam
    )

    scores = {

        "probability_real": float(
            probability[0, 0].detach().cpu()
        ),

        "probability_fake": float(
            probability[0, 1].detach().cpu()
        ),

        "uncertainty": float(
            uncertainty[0].detach().cpu()
        ),

        "evidence_real": float(
            evidence[0, 0].detach().cpu()
        ),

        "evidence_fake": float(
            evidence[0, 1].detach().cpu()
        ),

        "binary_prediction": int(
            binary_prediction[0].detach().cpu()
        ),

        "target_quantity": float(
            target_quantity.detach().cpu()
        ),
    }

    return (
        cam.detach().cpu(),
        scores,
        frame_attention,
        frame_cam_importance,
    )


# =========================================================
# SAVE FIGURE
# =========================================================

def save_figure(
    path,
    images,
    cam,
    attention,
    cam_importance,
    title,
    method,
    with_uncertainty_cam=None,
    topk=4,
):
    """
    Create a report-friendly figure.

    Row 1:
        Original frames

    Row 2:
        Grad-CAM / LayerCAM

    Row 3:
        Optional uncertainty attribution

    The displayed frames are selected using the model's
    AttentionPooling weights.
    """

    T = images.shape[0]

    # Select the frames the model attended to most.
    top_frames = torch.argsort(
        attention,
        descending=True,
    )[:topk]

    top_frames = sorted(
        top_frames.tolist()
    )

    rows = 2

    if with_uncertainty_cam is not None:
        rows += 1

    fig, axes = plt.subplots(
        rows,
        len(top_frames),
        figsize=(
            3.0 * len(top_frames),
            2.9 * rows,
        ),
        squeeze=False,
    )

    upsampled = upsample_cams(
        cam,
        images.shape[1],
        images.shape[2],
    )

    uncertainty_upsampled = None

    if with_uncertainty_cam is not None:

        uncertainty_upsampled = upsample_cams(
            with_uncertainty_cam,
            images.shape[1],
            images.shape[2],
        )

    for column, frame_index in enumerate(
        top_frames
    ):

        # ---------------------------------------------
        # Original
        # ---------------------------------------------

        axes[0, column].imshow(
            images[frame_index]
        )

        axes[0, column].set_title(
            "Frame "
            f"{frame_index + 1}\n"
            f"Attention "
            f"{attention[frame_index]:.3f}",
            fontsize=9,
        )

        # ---------------------------------------------
        # Main XAI
        # ---------------------------------------------

        axes[1, column].imshow(
            images[frame_index]
        )

        axes[1, column].imshow(
            upsampled[frame_index],
            cmap="jet",
            alpha=0.45,
            vmin=0,
            vmax=1,
        )

        axes[1, column].set_title(
            f"{method.upper()}\n"
            f"CAM importance "
            f"{cam_importance[frame_index]:.3f}",
            fontsize=8,
        )

        # ---------------------------------------------
        # Uncertainty XAI
        # ---------------------------------------------

        if (
            uncertainty_upsampled is not None
        ):

            axes[2, column].imshow(
                images[frame_index]
            )

            axes[2, column].imshow(
                uncertainty_upsampled[
                    frame_index
                ],
                cmap="jet",
                alpha=0.45,
                vmin=0,
                vmax=1,
            )

            axes[2, column].set_title(
                "UNCERTAINTY",
                fontsize=8,
            )

    for row in range(rows):

        for axis in axes[row]:

            axis.set_xticks([])
            axis.set_yticks([])

    axes[0, 0].set_ylabel(
        "Original",
        fontsize=9,
    )

    axes[1, 0].set_ylabel(
        method.upper(),
        fontsize=9,
    )

    if (
        with_uncertainty_cam is not None
    ):

        axes[2, 0].set_ylabel(
            "EDL Uncertainty",
            fontsize=9,
        )

    fig.suptitle(
        title,
        fontsize=10,
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.95,
        ]
    )

    fig.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    return top_frames


# =========================================================
# INSPECTION
# =========================================================

def inspect(
    model,
    dataset,
    hook,
):
    """
    Run a single forward pass and print architecture/tensor
    information before doing XAI.
    """

    print("\n")
    print("=" * 72)
    print("APPEARANCE XAI INSPECTION")
    print("=" * 72)

    print(
        f"Project root : {PROJECT_ROOT}"
    )

    print(
        f"Checkpoint   : {CKPT_PATH}"
    )

    print(
        f"Test CSV     : {TEST_CSV}"
    )

    print(
        f"Test cache   : {TEST_CACHE}"
    )

    print(
        f"Dataset size : {len(dataset):,}"
    )

    print(
        f"CAM block    : {hook.name}"
    )

    print(
        f"Patch grid   : "
        f"{EXPECTED_PATCH_GRID} x "
        f"{EXPECTED_PATCH_GRID}"
    )

    print(
        f"Patch count  : "
        f"{EXPECTED_PATCHES}"
    )

    print(
        "\n--- MODEL MODULES ---"
    )

    for name, module in model.named_children():

        print(
            f"{name:20s} "
            f"{type(module).__name__}"
        )

    print(
        "\n--- AIMv2 BLOCKS ---"
    )

    blocks = get_transformer_blocks(
        model
    )

    print(
        f"Number of blocks: {len(blocks)}"
    )

    print(
        f"Selected block : {hook.block_index}"
    )

    print(
        f"Block type     : "
        f"{type(hook.block).__name__}"
    )

    # ---------------------------------------------
    # Dataset sample
    # ---------------------------------------------

    item = dataset[0]

    if isinstance(item, dict):

        clip = item.get(
            "clip",
            item.get("frames"),
        )

        label = item.get(
            "label",
            item.get("target"),
        )

    else:

        clip = item[0]
        label = item[1]

    print(
        "\n--- DATASET SAMPLE ---"
    )

    print(
        f"Clip shape : {tuple(clip.shape)}"
    )

    print(
        f"Clip dtype : {clip.dtype}"
    )

    print(
        f"Clip min   : {float(clip.min()):.3f}"
    )

    print(
        f"Clip max   : {float(clip.max()):.3f}"
    )

    print(
        f"Label      : {int(label)}"
    )

    if clip.shape[0] != EXPECTED_FRAMES:

        print(
            f"[warning] expected "
            f"{EXPECTED_FRAMES} frames but got "
            f"{clip.shape[0]}"
        )

    # ---------------------------------------------
    # Forward pass
    # ---------------------------------------------

    device = next(
        model.parameters()
    ).device

    clip_input = clip.float()

    if clip_input.max() > 1.01:
        clip_input = (
            clip_input / 255.0
        )

    clip_input = (
        clip_input
        .unsqueeze(0)
        .to(device)
    )

    hook.clear()

    with torch.enable_grad():

        output = model(
            clip_input
        )

    evidence, edl = calculate_edl(
        output
    )

    print(
        "\n--- MODEL OUTPUT ---"
    )

    print(
        f"Output keys : "
        f"{list(output.keys())}"
    )

    print(
        f"Evidence    : "
        f"{evidence.detach().cpu().tolist()}"
    )

    print(
        f"P(Real)     : "
        f"{float(edl['probability'][0,0]):.6f}"
    )

    print(
        f"P(Fake)     : "
        f"{float(edl['probability'][0,1]):.6f}"
    )

    print(
        f"Uncertainty : "
        f"{float(edl['uncertainty'][0]):.6f}"
    )

    print(
        f"Prediction  : "
        f"{CLASS_NAMES[int(edl['binary_prediction'][0])]}"
    )

    print(
        "\n--- FRAME ATTENTION ---"
    )

    attention = output[
        "attention"
    ][0].detach().cpu()

    for frame_index, weight in enumerate(
        attention
    ):

        print(
            f"Frame {frame_index + 1:02d}: "
            f"{float(weight):.6f}"
        )

    print(
        "\n--- HOOKED ACTIVATION ---"
    )

    if hook.activation is None:

        print(
            "Activation: NONE"
        )

    else:

        print(
            "Activation shape: "
            f"{tuple(hook.activation.shape)}"
        )

    print(
        "\nInspection complete."
    )

    print(
        "If the activation shape is approximately "
        "(16,256,1024), the CAM token mapping is correct."
    )

    print("=" * 72)


# =========================================================
# DATASET ITEM HELPER
# =========================================================

def unpack_item(item):
    """
    Convert CachedClipDataset item into:

        clip, label
    """

    if isinstance(item, dict):

        clip = None

        for key in (
            "clip",
            "frames",
            "video",
            "x",
            "images",
        ):

            if key in item:

                clip = item[key]
                break

        label = None

        for key in (
            "label",
            "labels",
            "y",
            "target",
        ):

            if key in item:

                label = item[key]
                break

    else:

        clip = item[0]

        label = (
            item[1]
            if len(item) > 1
            else None
        )

    if clip is None:

        raise RuntimeError(
            "Could not find clip tensor."
        )

    if torch.is_tensor(label):

        label = int(
            label.item()
        )

    elif label is not None:

        label = int(label)

    return clip, label


# =========================================================
# AUTO PICK
# =========================================================

def scan_and_pick(
    model,
    dataset,
    n_scan,
    per_category,
    seed,
):
    """
    Search a subset of the test set and select:

        correct_real
        correct_fake
        false_fake
        false_real
        uncertain

    Each category gets up to per_category examples.
    """

    rng = random.Random(
        seed
    )

    scan_count = min(
        n_scan,
        len(dataset),
    )

    indices = rng.sample(
        range(len(dataset)),
        scan_count,
    )

    device = next(
        model.parameters()
    ).device

    found = {

        "correct_real": [],

        "correct_fake": [],

        "false_fake": [],

        "false_real": [],

        "uncertain": [],
    }

    print(
        "\n[pick] scanning "
        f"{scan_count} test samples..."
    )

    for count, index in enumerate(
        indices,
        start=1,
    ):

        clip, label = unpack_item(
            dataset[index]
        )

        clip_input = clip.float()

        if clip_input.max() > 1.01:

            clip_input = (
                clip_input / 255.0
            )

        clip_input = (
            clip_input
            .unsqueeze(0)
            .to(device)
        )

        with torch.no_grad():

            output = model(
                clip_input
            )

            _, edl = calculate_edl(
                output
            )

        probability_fake = float(
            edl["probability"][0, 1]
        )

        uncertainty = float(
            edl["uncertainty"][0]
        )

        state = state_of(
            probability_fake,
            uncertainty,
        )

        if state == "UNCERTAIN":

            category = "uncertain"

        elif label is None:

            continue

        elif state == "REAL":

            if label == 0:

                category = "correct_real"

            else:

                category = "false_real"

        else:

            if label == 1:

                category = "correct_fake"

            else:

                category = "false_fake"

        if len(
            found[category]
        ) < per_category:

            found[category].append(
                index
            )

        complete = all(
            len(v) >= per_category
            for v in found.values()
        )

        if complete:
            break

        if count % 50 == 0:

            print(
                f"  scanned {count}/{scan_count}"
            )

    print(
        "\n[pick] results:"
    )

    for category, indices_found in found.items():

        print(
            f"  {category:15s}: "
            f"{indices_found}"
        )

    return [
        (category, index)
        for category, indices_found in found.items()
        for index in indices_found
    ]


# =========================================================
# RANDOMIZED SANITY CHECK
# =========================================================

def randomize_weights(model):
    """
    Reinitialize model parameters for a basic XAI sanity check.

    This DOES NOT modify best.pt.

    The model only exists in memory during this run.
    """

    print(
        "\n[sanity] randomizing model weights..."
    )

    for module in model.modules():

        if hasattr(
            module,
            "reset_parameters",
        ):

            try:

                module.reset_parameters()

            except Exception:

                pass

    print(
        "[sanity] randomized model is "
        "memory-only; best.pt is untouched."
    )


# =========================================================
# SAVE README
# =========================================================

def save_readme(
    output_directory,
    method,
    target,
):
    """
    Save a small explanation of the generated files.
    """

    text = f"""
Appearance Branch XAI
=====================

Checkpoint:
    {CKPT_PATH}

Method:
    {method}

Target:
    {target}

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
    {UNCERTAINTY_THRESHOLD}

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
"""

    with open(
        output_directory / "README.txt",
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            text.strip()
        )


# =========================================================
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "XAI for the trained Appearance Branch"
        )
    )

    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Inspect model, dataset and CAM tensor shapes.",
    )

    parser.add_argument(
        "--indices",
        type=int,
        nargs="*",
        default=[],
        help="Specific test dataset indices.",
    )

    parser.add_argument(
        "--auto-pick",
        type=int,
        default=0,
        help=(
            "Number of examples per category. "
            "Categories: correct_real, correct_fake, "
            "false_fake, false_real, uncertain."
        ),
    )

    parser.add_argument(
        "--scan",
        type=int,
        default=400,
        help="Number of test samples to scan for auto-pick.",
    )

    parser.add_argument(
        "--target",
        default="pred",
        choices=[
            "pred",
            "fake",
            "real",
            "uncertainty",
        ],
        help="Quantity to explain.",
    )

    parser.add_argument(
        "--with-uncertainty",
        action="store_true",
        help="Generate a second CAM explaining EDL uncertainty.",
    )

    parser.add_argument(
        "--method",
        default="gradcam",
        choices=[
            "gradcam",
            "layercam",
        ],
        help="Attribution method.",
    )

    parser.add_argument(
        "--block-index",
        type=int,
        default=-1,
        help=(
            "AIMv2 transformer block. "
            "-1 = final block."
        ),
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=4,
        help="Number of highest-attention frames to visualize.",
    )

    parser.add_argument(
        "--randomize",
        action="store_true",
        help=(
            "Randomize weights for a basic XAI sanity check."
        ),
    )

    parser.add_argument(
        "--tag",
        default="day1",
        help="Output subdirectory name.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # Device
    # -----------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 72)

    print(
        "APPEARANCE BRANCH XAI"
    )

    print("=" * 72)

    print(
        f"Project root : {PROJECT_ROOT}"
    )

    print(
        f"Device       : {device}"
    )

    if device.type == "cuda":

        print(
            f"GPU          : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"Checkpoint   : {CKPT_PATH}"
    )

    print(
        f"Method       : {args.method}"
    )

    print(
        f"Target       : {args.target}"
    )

    print(
        f"Threshold    : "
        f"{UNCERTAINTY_THRESHOLD}"
    )

    print("=" * 72)

    # -----------------------------------------------------
    # Reproducibility
    # -----------------------------------------------------

    random.seed(
        args.seed
    )

    np.random.seed(
        args.seed
    )

    torch.manual_seed(
        args.seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            args.seed
        )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = build_model(
        device
    )

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    dataset = build_dataset()

    # -----------------------------------------------------
    # CAM hook
    # -----------------------------------------------------

    hook = TransformerCAMHook(
        model,
        args.block_index,
    )

    # -----------------------------------------------------
    # Randomized sanity check
    # -----------------------------------------------------

    if args.randomize:

        randomize_weights(
            model
        )

        args.tag += "_RANDOMIZED"

    # -----------------------------------------------------
    # Inspection mode
    # -----------------------------------------------------

    if args.inspect:

        try:

            inspect(
                model,
                dataset,
                hook,
            )

        finally:

            hook.remove()

        return

    # -----------------------------------------------------
    # Jobs
    # -----------------------------------------------------

    jobs = [
        ("manual", index)
        for index in args.indices
    ]

    if args.auto_pick > 0:

        jobs.extend(
            scan_and_pick(
                model,
                dataset,
                args.scan,
                args.auto_pick,
                args.seed,
            )
        )

    if not jobs:

        hook.remove()

        raise SystemExit(
            "\nNothing to do.\n\n"
            "Use either:\n\n"
            "  --inspect\n\n"
            "or:\n\n"
            "  --auto-pick 4 --scan 400\n\n"
            "or:\n\n"
            "  --indices 10 250 4001"
        )

    # -----------------------------------------------------
    # Output
    # -----------------------------------------------------

    output_directory = (
        OUT_ROOT / args.tag
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"\n[output] {output_directory}"
    )

    save_readme(
        output_directory,
        args.method,
        args.target,
    )

    rows = []

    # -----------------------------------------------------
    # Explain each sample
    # -----------------------------------------------------

    for job_number, (
        category,
        index,
    ) in enumerate(
        jobs,
        start=1,
    ):

        print(
            "\n"
            + "-" * 72
        )

        print(
            f"[{job_number}/{len(jobs)}] "
            f"category={category} "
            f"index={index}"
        )

        clip, label = unpack_item(
            dataset[index]
        )

        # Display images.
        images = to_display(
            clip
        )

        # ---------------------------------------------
        # Main explanation
        # ---------------------------------------------

        cam, scores, attention, cam_importance = explain(
            model,
            hook,
            clip,
            target=args.target,
            method=args.method,
        )

        # ---------------------------------------------
        # Optional uncertainty explanation
        # ---------------------------------------------

        uncertainty_cam = None

        if args.with_uncertainty:

            uncertainty_cam, _, _, _ = explain(
                model,
                hook,
                clip,
                target="uncertainty",
                method=args.method,
            )

        # ---------------------------------------------
        # State
        # ---------------------------------------------

        prediction = state_of(
            scores["probability_fake"],
            scores["uncertainty"],
        )

        if label is None:

            true_label = "UNKNOWN"

        else:

            true_label = CLASS_NAMES[
                label
            ]

        # ---------------------------------------------
        # Figure title
        # ---------------------------------------------

        title = (
            f"Test index {index} | "
            f"True: {true_label} | "
            f"Prediction: {prediction} | "
            f"P(Fake): "
            f"{scores['probability_fake']:.3f} | "
            f"Uncertainty: "
            f"{scores['uncertainty']:.3f}"
        )

        figure_path = (
            output_directory
            / f"{category}_{index}.png"
        )

        shown_frames = save_figure(
            figure_path,
            images,
            cam,
            attention,
            cam_importance,
            title,
            args.method,
            with_uncertainty_cam=(
                uncertainty_cam
            ),
            topk=min(
                args.topk,
                len(clip),
            ),
        )

        # ---------------------------------------------
        # Save raw CAM
        # ---------------------------------------------

        np.save(
            output_directory
            / f"{category}_{index}_cam.npy",
            cam.numpy(),
        )

        if uncertainty_cam is not None:

            np.save(
                output_directory
                / f"{category}_{index}_uncertainty_cam.npy",
                uncertainty_cam.numpy(),
            )

        # ---------------------------------------------
        # Frame CSV information
        # ---------------------------------------------

        attention_list = [
            float(x)
            for x in attention
        ]

        cam_importance_list = [
            float(x)
            for x in cam_importance
        ]

        top_attention_frame = int(
            torch.argmax(
                attention
            ).item()
        )

        top_cam_frame = int(
            torch.argmax(
                cam_importance
            ).item()
        )

        # ---------------------------------------------
        # Summary row
        # ---------------------------------------------

        row = {

            "index":
                index,

            "category":
                category,

            "true_label":
                true_label,

            "prediction":
                prediction,

            "binary_prediction":
                (
                    CLASS_NAMES[
                        scores[
                            "binary_prediction"
                        ]
                    ]
                ),

            "probability_real":
                round(
                    scores[
                        "probability_real"
                    ],
                    6,
                ),

            "probability_fake":
                round(
                    scores[
                        "probability_fake"
                    ],
                    6,
                ),

            "evidence_real":
                round(
                    scores[
                        "evidence_real"
                    ],
                    6,
                ),

            "evidence_fake":
                round(
                    scores[
                        "evidence_fake"
                    ],
                    6,
                ),

            "uncertainty":
                round(
                    scores[
                        "uncertainty"
                    ],
                    6,
                ),

            "target_quantity":
                round(
                    scores[
                        "target_quantity"
                    ],
                    6,
                ),

            "top_attention_frame":
                top_attention_frame + 1,

            "top_cam_frame":
                top_cam_frame + 1,

            "frames_shown":
                json.dumps(
                    [
                        f + 1
                        for f in shown_frames
                    ]
                ),

            "attention":
                json.dumps(
                    [
                        round(
                            x,
                            6,
                        )
                        for x in attention_list
                    ]
                ),

            "cam_importance":
                json.dumps(
                    [
                        round(
                            x,
                            6,
                        )
                        for x in cam_importance_list
                    ]
                ),

            "figure":
                figure_path.name,
        }

        rows.append(
            row
        )

        print(
            f"[ok] true={true_label} "
            f"pred={prediction} "
            f"P(fake)="
            f"{scores['probability_fake']:.4f} "
            f"uncertainty="
            f"{scores['uncertainty']:.4f}"
        )

        print(
            "[frames] top attention: "
            + ", ".join(
                str(f + 1)
                for f in shown_frames
            )
        )

        print(
            f"[saved] {figure_path}"
        )

    # -----------------------------------------------------
    # Summary CSV
    # -----------------------------------------------------

    summary_path = (
        output_directory
        / "summary.csv"
    )

    if rows:

        fieldnames = list(
            rows[0].keys()
        )

        with open(
            summary_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            writer.writerows(
                rows
            )

    # -----------------------------------------------------
    # Finish
    # -----------------------------------------------------

    hook.remove()

    print("\n")
    print("=" * 72)

    print(
        "XAI COMPLETE"
    )

    print("=" * 72)

    print(
        f"Examples : {len(rows)}"
    )

    print(
        f"Output   : {output_directory}"
    )

    print(
        f"Summary  : {summary_path}"
    )

    print(
        "\nThe trained checkpoint was NOT modified."
    )

    print("=" * 72)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()