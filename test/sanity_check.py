"""
=========================================================
Appearance Branch Sanity Check

Verifies that:

Dataset
    ↓
AppearanceClassifier
    ↓
Evidence
    ↓
Dirichlet
    ↓
EDL Loss
    ↓
Backward Pass

works correctly before starting full training.
=========================================================
"""

import torch

from data.dataset.appearance.deepfake_dataset import DeepfakeDataset
from models.appearence.appearance_classifier import AppearanceClassifier
from models.heads.evidence_head import compute_dirichlet
from losses.edl_loss import compute_edl_loss

def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("Appearance Branch Sanity Check")
    print("=" * 60)
    print(f"Device: {device}")

    # -------------------------------------------------
    # Dataset
    # -------------------------------------------------

    dataset = DeepfakeDataset(
        metadata_csv="data/split/train.csv",   # ✅ Correct
        dataset_root="data/raw",
        split="train",
        image_size=224,
    )

    sample = dataset[0]

    clip = sample["clip"].unsqueeze(0).to(device)

    label = torch.tensor(
        [sample["label"]],
        dtype=torch.long,
        device=device,
    )

    print("\nDataset")
    print("-" * 60)
    print(f"Clip Shape : {clip.shape}")
    print(f"Label      : {label.item()}")

    # -------------------------------------------------
    # Model
    # -------------------------------------------------

    model = AppearanceClassifier(
        num_classes=2,
        freeze_backbone=True,
    ).to(device)

    model.train()

    outputs = model(clip)

    evidence = outputs["evidence"]

    print("\nModel")
    print("-" * 60)
    print(f"Evidence Shape : {evidence.shape}")

    # -------------------------------------------------
    # Dirichlet
    # -------------------------------------------------

    dirichlet = compute_dirichlet(evidence)

    print("\nDirichlet")
    print("-" * 60)
    print(f"Alpha Shape       : {dirichlet['alpha'].shape}")
    print(f"Belief Shape      : {dirichlet['belief'].shape}")
    print(f"Probability Shape : {dirichlet['probability'].shape}")
    print(f"Uncertainty Shape : {dirichlet['uncertainty'].shape}")

    # -------------------------------------------------
    # Loss
    # -------------------------------------------------

    loss = compute_edl_loss(
        evidence=evidence,
        target=label,
        epoch=1,
        num_classes=2,
        annealing_step=10,
        device=device,
    )

    print("\nLoss")
    print("-" * 60)
    print(f"Loss : {loss.item():.6f}")

    # -------------------------------------------------
    # Backward
    # -------------------------------------------------

    loss.backward()

    print("\nBackward Pass")
    print("-" * 60)
    print("✓ Success")

    print("\nEverything is working correctly.")


if __name__ == "__main__":
    main()