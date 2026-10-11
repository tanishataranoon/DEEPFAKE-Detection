"""
=========================================================
Temporal Transforms
---------------------------------------------------------
Train split returns a ReplayCompose instead of a Compose,
so the exact same random augmentation parameters (flip,
brightness, etc.) can be REPLAYED identically across every
frame of a clip.

Why this matters here specifically: the Appearance Branch
scores each frame independently, so per-frame-random
augmentation is harmless there. The Temporal Branch is
trying to learn genuine frame-to-frame inconsistencies --
if one frame gets flipped and its neighbor doesn't, that
creates a fake temporal discontinuity unrelated to real
deepfake artifacts, which the model could learn as a
shortcut. Replaying the same augmentation across the whole
clip avoids that.
=========================================================
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_temporal_transforms(split: str, image_size: int = 224):
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown split '{split}'. Expected train, val or test.")

    if split == "train":
        return A.ReplayCompose(
            [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.10, contrast_limit=0.10, p=0.25
                ),
                A.HueSaturationValue(
                    hue_shift_limit=8, sat_shift_limit=10, val_shift_limit=8, p=0.15,
                ),
                A.GaussianBlur(blur_limit=(3, 3), p=0.1),
                A.ImageCompression(quality_range=(70, 100), p=0.15),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ]
        )

    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )