"""
=========================================================
Train Temporal Branch (VideoMAE)
=========================================================
"""

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoImageProcessor
import torchvision.transforms as T

from data.dataset.temporal.temporal_dataset import TemporalDeepfakeDataset
from models.temporal.temporal_classifier import TemporalClassifier
from losses.edl_loss import compute_edl_loss

TRAIN_CSV = "data/split/train.csv"
VAL_CSV = "data/split/val.csv"
DATASET_ROOT = "data/raw"

PRETRAINED_NAME = "MCG-NJU/videomae-base"
NUM_FRAMES = 16
IMAGE_SIZE = 224
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
EPOCHS = 1
LR = 1e-4
FREEZE_BACKBONE = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoImageProcessor.from_pretrained(PRETRAINED_NAME)
mean, std = processor.image_mean, processor.image_std

frame_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=mean, std=std),
])


def make_weighted_sampler(csv_path):
    import pandas as pd

    df = pd.read_csv(csv_path)
    counts = df["label"].value_counts()
    weight_per_class = {label: 1.0 / count for label, count in counts.items()}
    sample_weights = df["label"].map(weight_per_class).values
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def main():
    print("=" * 60)
    print("TEMPORAL BRANCH TRAINING")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    train_dataset = TemporalDeepfakeDataset(
        metadata_csv=TRAIN_CSV,
        dataset_root=DATASET_ROOT,
        image_size=IMAGE_SIZE,
        num_frames=NUM_FRAMES,
        transform=frame_transform,
    )

    val_dataset = TemporalDeepfakeDataset(
        metadata_csv=VAL_CSV,
        dataset_root=DATASET_ROOT,
        image_size=IMAGE_SIZE,
        num_frames=NUM_FRAMES,
        transform=frame_transform,
    )

    sampler = make_weighted_sampler(TRAIN_CSV)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )

    print(f"Train clips: {len(train_dataset)}")
    print(f"Val clips  : {len(val_dataset)}")

    model = TemporalClassifier(
        num_classes=2,
        pretrained_name=PRETRAINED_NAME,
        freeze_backbone=FREEZE_BACKBONE,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))
    best_val_balanced_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            clips = batch["clip"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                outputs = model(clips)
                evidence = outputs["evidence"]
                loss = compute_edl_loss(
                    evidence=evidence,
                    target=labels,
                    epoch=epoch,
                    num_classes=2,
                    annealing_step=10,
                    device=DEVICE,
                )
                loss = loss / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * GRAD_ACCUM_STEPS

            if step % 20 == 0:
                print(f"Epoch {epoch} Step {step}/{len(train_loader)} "
                      f"Loss: {running_loss / (step + 1):.4f}")

        val_balanced_acc = validate(model, val_loader)
        print(f"Epoch {epoch} — Val balanced accuracy: {val_balanced_acc:.4f}")

        if val_balanced_acc > best_val_balanced_acc:
            best_val_balanced_acc = val_balanced_acc
            torch.save(model.state_dict(), "checkpoints/temporal_best.pt")
            print("  ✓ Saved new best checkpoint")


def validate(model, loader):
    model.eval()
    correct_per_class = {0: 0, 1: 0}
    total_per_class = {0: 0, 1: 0}

    with torch.no_grad():
        for batch in loader:
            clips = batch["clip"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            outputs = model(clips)
            evidence = outputs["evidence"]
            alpha = evidence + 1
            probs = alpha / alpha.sum(dim=1, keepdim=True)
            preds = torch.argmax(probs, dim=1)

            for c in (0, 1):
                mask = labels == c
                total_per_class[c] += mask.sum().item()
                correct_per_class[c] += (preds[mask] == c).sum().item()

    recalls = [
        correct_per_class[c] / total_per_class[c]
        for c in (0, 1)
        if total_per_class[c] > 0
    ]
    balanced_acc = sum(recalls) / len(recalls)
    return balanced_acc


if __name__ == "__main__":
    main()