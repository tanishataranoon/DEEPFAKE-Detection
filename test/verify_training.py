"""
Quick verification before long training.

Checks:
1. Dataset labels
2. Model output
3. EvidenceHead update
4. EDL loss
"""

import torch
from torch.utils.data import DataLoader

from data.dataset.appearance.deepfake_dataset import DeepfakeDataset
from models.appearence.appearance_classifier import AppearanceClassifier
from losses.edl_loss import compute_edl_loss

CSV = "data/split/train.csv"
ROOT = "data/raw"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():

    print("="*60)
    print("APPEARANCE TRAINING VERIFICATION")
    print("="*60)


    dataset = DeepfakeDataset(
        metadata_csv=CSV,
        dataset_root=ROOT,
        split="train",
        image_size=224
    )


    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4
    )


    print()
    print("Dataset size:", len(dataset))


    # -----------------------------
    # Label check
    # -----------------------------
    import pandas as pd

    df = pd.read_csv(CSV)

    print("\nCSV label distribution")
    print("======================")

    print(df["label"].value_counts())
    print(
        "Real:",
        (df["label"] == "real").sum()
    )

    print(
        "Fake:",
        (df["label"] == "fake").sum()
    )




    # -----------------------------
    # Model
    # -----------------------------


    model = AppearanceClassifier(
        num_classes=2,
        freeze_backbone=True
    ).to(DEVICE)


    model.train()


    for i in range(3):

        sample = dataset[i]

        print(
            "Sample",
            i,
            "label:",
            sample["label"],
            "clip shape:",
            sample["clip"].shape
        )
        # -----------------------------
        # Get one batch
        # -----------------------------

        loader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=False,
            num_workers=0
        )


        batch = next(iter(loader))


        images = batch["clip"].to(DEVICE)

        labels = batch["label"].to(DEVICE)


        print()
        print("Batch shape:")
        print(images.shape)


        print()
        print("Batch labels:")
        print(labels)

    # images = batch["clip"].to(DEVICE)

    # labels = batch["label"].to(DEVICE)


    print()
    print("Batch labels:")
    print(labels)


    # Forward

    output = model(images)


    evidence = output["evidence"]


    print()
    print("Evidence:")
    print(evidence[:5])


    print()
    print("Evidence mean:")
    print(
        evidence.mean(dim=0)
    )


    # Prediction

    alpha = evidence + 1

    prob = alpha / alpha.sum(
        dim=1,
        keepdim=True
    )


    pred = torch.argmax(
        prob,
        dim=1
    )


    print()

    print("Predictions")

    print(
        "Real:",
        (pred==0).sum().item()
    )

    print(
        "Fake:",
        (pred==1).sum().item()
    )


    # -----------------------------
    # Gradient check
    # -----------------------------
    # -----------------------------
    # EDL loss
    # -----------------------------

    loss = compute_edl_loss(
        evidence,
        labels,
        epoch=0,
        num_classes=2,
        annealing_step=10
    )


    loss.backward()


    print()

    print("Loss:")
    print(loss.item())


    print()

    print("Evidence Head gradients")


    for name,p in model.evidence_head.named_parameters():

        if p.grad is not None:

            print(
                name,
                p.grad.abs().mean().item()
            )

        else:

            print(
                name,
                "NO GRAD"
            )


    print()
    print("="*60)
    print("Verification finished")
    print("="*60)



if __name__=="__main__":
    main()