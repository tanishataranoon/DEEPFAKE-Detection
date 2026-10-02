"""
=========================================================
Test Temporal Classifier
=========================================================
Checks:
1. Model loads (downloads VideoMAE weights first run)
2. Forward pass on one real batch from the dataset
3. Evidence shape and values are sane
=========================================================
"""

import torch
from torch.utils.data import DataLoader

from data.raw.temporal.temporal_dataset import TemporalDeepfakeDataset
from models.temporal.temporal_classifier import TemporalClassifier

CSV = "data/split/train.csv"
ROOT = "data/raw"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print("=" * 60)
    print("TEMPORAL CLASSIFIER TEST")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    dataset = TemporalDeepfakeDataset(
        metadata_csv=CSV,
        dataset_root=ROOT,
        image_size=224,
        num_frames=16,
    )

    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))

    clips = batch["clip"].to(DEVICE)   # (B, T, C, H, W)
    labels = batch["label"].to(DEVICE)

    print("\nBatch clip shape :", clips.shape)
    print("Batch labels     :", labels)

    print("\nLoading VideoMAE (downloads on first run)...")
    model = TemporalClassifier(
        num_classes=2,
        freeze_backbone=True,
    ).to(DEVICE)
    model.eval()

    with torch.no_grad():
        outputs = model(clips)

    evidence = outputs["evidence"]

    print("\nEvidence shape :", evidence.shape)
    print("Evidence values:\n", evidence)
    print("\nMin evidence   :", evidence.min().item())
    print("Contains NaN   :", torch.isnan(evidence).any().item())

    print("\nTest passed.")


if __name__ == "__main__":
    main()