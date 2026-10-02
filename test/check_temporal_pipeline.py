"""
=========================================================
Temporal Branch Pipeline Check (VideoMAE)
=========================================================
"""

import torch

from data.raw.appearanch.dataset import DeepfakeDataset
from models.temporal.temporal_classifier import TemporalClassifier


def main():
    print("=" * 60)
    print("Temporal Branch Pipeline Check (VideoMAE)")
    print("=" * 60)

    dataset = DeepfakeDataset(
        metadata_csv="data/split/small/train.csv",
        dataset_root="data/raw",
        split="train",
        image_size=224,
    )

    print(f"\nDataset size : {len(dataset)}")

    sample = dataset[0]
    clip = sample["clip"].unsqueeze(0)          # (1, T, C, H, W) = (1, 16, 3, 224, 224)
    label = torch.tensor([sample["label"]])

    print(f"Clip shape   : {clip.shape}")
    print(f"Label        : {label.item()}")
    print(f"Video        : {sample['video_name']}")

    print("\nLoading VideoMAE (downloads weights on first run)...")
    model = TemporalClassifier(num_classes=2, freeze_backbone=True)
    model.train()

    outputs = model(clip)
    evidence = outputs["evidence"]

    print(f"\nEvidence shape : {evidence.shape}")   # expect (1, 2)
    print(f"Evidence       : {evidence}")
    print(f"Contains NaN   : {torch.isnan(evidence).any().item()}")

    print("\n" + "=" * 60)
    print("✓ Pipeline check passed")
    print("=" * 60)


if __name__ == "__main__":
    main()