"""
=========================================================
Temporal Branch (VideoMAE) - 150 Source Group Test Training
=========================================================

Mirrors training/Appreach/150_appreanch_train.py exactly:
    - full test_150/train.csv and test_150/val.csv, no subsetting
    - batch_size=8, accumulation_steps=2 (effective batch 16)
    - lr=1e-4
    - freeze_backbone=True
    - num_workers=8
    - early_stopping_patience=2

Only difference: num_epochs=5 (instead of 10), and
gradient_checkpointing=True kept ON as a VRAM safety net,
since VideoMAE is much heavier per-sample than AIMv2 and
batch 8 is a real OOM risk on an 8GB GPU. This does not
change training behaviour or results, only memory usage.

Checkpoint directory:
    checkpoints/temporal_test_150/
=========================================================
"""

import os
import time
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.raw.temporal.temporal_dataset import TemporalDeepfakeDataset
from models.temporal.temporal_classifier import TemporalClassifier
from models.heads.evidence_head import compute_dirichlet
from losses.edl_loss import compute_edl_loss

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


class TemporalTestTrainer:

    def __init__(
        self,
        train_csv="data/split/test_150/train.csv",
        val_csv="data/split/test_150/val.csv",
        dataset_root="data/raw",
        checkpoint_dir="checkpoints/temporal_test_150",
        batch_size=8,
        accumulation_steps=2,
        lr=1e-4,
        weight_decay=1e-4,
        device=None,
        num_epochs=5,
        num_classes=2,
        annealing_step=10,
        freeze_backbone=True,
        gradient_checkpointing=True,
        num_workers=8,
        early_stopping_patience=2,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps
        self.num_epochs = num_epochs
        self.num_classes = num_classes
        self.annealing_step = annealing_step

        self.early_stopping_patience = early_stopping_patience
        self.epochs_without_improvement = 0
        self.best_val_acc = -1.0

        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir

        print("=" * 70)
        print("TEMPORAL BRANCH (VideoMAE) - 150 SOURCE GROUP TEST")
        print("=" * 70)
        print(f"Device                : {self.device}")
        print(f"Batch Size            : {batch_size}")
        print(f"Gradient Accumulation : {accumulation_steps}")
        print(f"Effective Batch Size  : {batch_size * accumulation_steps}")
        print(f"Epochs                : {num_epochs}")
        print(f"Early Stopping        : patience={early_stopping_patience}")
        print(f"Train CSV             : {train_csv}")
        print(f"Validation CSV        : {val_csv}")
        print(f"Checkpoint Directory  : {checkpoint_dir}")
        print("=" * 70)

        # -------------------------------------------------
        # Dataset (FULL split -- no subsetting)
        # -------------------------------------------------

        self.train_dataset = TemporalDeepfakeDataset(
            metadata_csv=train_csv,
            dataset_root=dataset_root,
            split="train",
        )

        self.val_dataset = TemporalDeepfakeDataset(
            metadata_csv=val_csv,
            dataset_root=dataset_root,
            split="val",
        )

        print()
        print(f"Training samples      : {len(self.train_dataset):,}")
        print(f"Validation samples    : {len(self.val_dataset):,}")

        # -------------------------------------------------
        # DataLoader
        # -------------------------------------------------

        loader_kwargs = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 4

        if torch.cuda.is_available():
            loader_kwargs["pin_memory_device"] = "cuda"

        labels = self.train_dataset.get_labels()
        class_counts = np.bincount(labels, minlength=2)
        sample_weights = 1.0 / class_counts[labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(labels),
            replacement=True,
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            sampler=sampler,
            **loader_kwargs,
        )

        self.class_weights = torch.tensor(
            len(labels) / (2.0 * class_counts), dtype=torch.float32
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            persistent_workers=(num_workers > 0),
            prefetch_factor=4 if num_workers > 0 else None,
            **({"pin_memory_device": "cuda"} if torch.cuda.is_available() else {}),
        )

        # -------------------------------------------------
        # Model
        # -------------------------------------------------

        self.model = TemporalClassifier(
            num_classes=num_classes,
            freeze_backbone=freeze_backbone,
            gradient_checkpointing=gradient_checkpointing,
        ).to(self.device)

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in trainable_params)
        print(f"Trainable Parameters  : {n_trainable:,}")

        # -------------------------------------------------
        # Optimizer / AMP
        # -------------------------------------------------

        self.optimizer = torch.optim.AdamW(
            trainable_params, lr=lr, weight_decay=weight_decay,
        )

        self.scaler = GradScaler("cuda", enabled=(self.device == "cuda"))

    # =====================================================
    # GPU Temperature
    # =====================================================

    def get_gpu_temperature(self):
        if self.device != "cuda":
            return None
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                return None
            return int(result.stdout.strip().splitlines()[0])
        except Exception:
            return None

    def check_gpu_temperature(self, epoch, step):
        temperature = self.get_gpu_temperature()
        if temperature is None:
            return False

        if temperature >= 85:
            print()
            print("=" * 70)
            print(f"WARNING: GPU temperature = {temperature}°C")
            print("Temperature is too high.")
            print("Stopping training safely.")
            print("=" * 70)
            self.save_checkpoint(epoch=epoch, val_acc=self.best_val_acc, tag="temperature_stop")
            return True

        if temperature >= 80 and step % 100 == 0:
            print(f"WARNING: GPU temperature = {temperature}°C")

        return False

    # =====================================================
    # Forward
    # =====================================================

    def _forward_step(self, clip):
        clip = clip.to(self.device, non_blocking=True)
        return self.model(clip)

    # =====================================================
    # Train One Epoch
    # =====================================================

    def train_one_epoch(self, epoch):
        self.model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        self.optimizer.zero_grad(set_to_none=True)
        num_steps = len(self.train_loader)

        for step, batch in enumerate(self.train_loader):

            if (step + 1) % 100 == 0:
                if self.check_gpu_temperature(epoch, step + 1):
                    return (
                        running_loss / max(step, 1),
                        correct / total if total else 0.0,
                        True,
                    )

            clip = batch["clip"]
            label = batch["label"].to(self.device, non_blocking=True)

            with autocast(
                device_type=("cuda" if self.device == "cuda" else "cpu"),
                enabled=(self.device == "cuda"),
            ):
                outputs = self._forward_step(clip)
                evidence = outputs["evidence"]

                loss = compute_edl_loss(
                    evidence=evidence,
                    target=label,
                    epoch=epoch,
                    num_classes=self.num_classes,
                    annealing_step=self.annealing_step,
                    device=self.device,
                )

                loss_for_backward = loss / self.accumulation_steps

            self.scaler.scale(loss_for_backward).backward()

            should_step = (
                (step + 1) % self.accumulation_steps == 0
                or (step + 1) == num_steps
            )

            if should_step:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()

            dirichlet = compute_dirichlet(evidence)
            prediction = torch.argmax(dirichlet["probability"], dim=1)

            correct += (prediction == label).sum().item()
            total += label.size(0)

            if (step + 1) % 20 == 0 or (step + 1) == num_steps:
                print(f"Step {step + 1}/{num_steps}   Loss: {running_loss / (step + 1):.4f}")

        average_loss = running_loss / num_steps if num_steps > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        return average_loss, accuracy, False

    # =====================================================
    # Validation
    # =====================================================

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0
        total_uncertainty = 0.0

        for batch in self.val_loader:
            clip = batch["clip"]
            label = batch["label"].to(self.device, non_blocking=True)

            with autocast(
                device_type=("cuda" if self.device == "cuda" else "cpu"),
                enabled=(self.device == "cuda"),
            ):
                outputs = self._forward_step(clip)
                evidence = outputs["evidence"]

                loss = compute_edl_loss(
                    evidence=evidence,
                    target=label,
                    epoch=epoch,
                    num_classes=self.num_classes,
                    annealing_step=self.annealing_step,
                    device=self.device,
                )

            running_loss += loss.item()

            dirichlet = compute_dirichlet(evidence)
            prediction = torch.argmax(dirichlet["probability"], dim=1)

            correct += (prediction == label).sum().item()
            total += label.size(0)
            total_uncertainty += dirichlet["uncertainty"].sum().item()

        average_loss = running_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        average_uncertainty = total_uncertainty / total if total > 0 else 0.0

        return average_loss, accuracy, average_uncertainty

    # =====================================================
    # Training Loop
    # =====================================================

    def fit(self):
        print()
        print("=" * 70)
        print("STARTING TEMPORAL 150-SOURCE-GROUP TEST TRAINING")
        print("=" * 70)
        print()

        for epoch in range(1, self.num_epochs + 1):
            start_time = time.time()

            train_loss, train_acc, stopped = self.train_one_epoch(epoch)

            if stopped:
                print("Training stopped because of GPU temperature.")
                return

            val_loss, val_acc, val_uncertainty = self.validate(epoch)

            elapsed = time.time() - start_time

            print()
            print("=" * 70)
            print(f"Epoch {epoch}/{self.num_epochs}")
            print("=" * 70)
            print(f"Train Loss          : {train_loss:.4f}")
            print(f"Train Accuracy      : {train_acc:.4f}")
            print(f"Validation Loss     : {val_loss:.4f}")
            print(f"Validation Accuracy : {val_acc:.4f}")
            print(f"Avg Uncertainty     : {val_uncertainty:.4f}")
            print(f"Time                : {elapsed:.1f}s")

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.epochs_without_improvement = 0

                self.save_checkpoint(epoch=epoch, val_acc=val_acc, tag="best")
                print(f"New best validation accuracy: {val_acc:.4f}")
            else:
                self.epochs_without_improvement += 1
                print(
                    f"No validation improvement. "
                    f"Early stopping patience: "
                    f"{self.epochs_without_improvement}/{self.early_stopping_patience}"
                )

            self.save_checkpoint(epoch=epoch, val_acc=val_acc, tag="last")

            if self.epochs_without_improvement >= self.early_stopping_patience:
                print()
                print("=" * 70)
                print("EARLY STOPPING")
                print("=" * 70)
                print(
                    f"Validation accuracy did not improve for "
                    f"{self.early_stopping_patience} consecutive epochs."
                )
                print(f"Best Validation Accuracy : {self.best_val_acc:.4f}")
                print(f"Best checkpoint           : {os.path.join(self.checkpoint_dir, 'best.pt')}")
                print("=" * 70)
                return

        print()
        print("=" * 70)
        print("TEST TRAINING FINISHED")
        print("=" * 70)
        print(f"Best Validation Accuracy : {self.best_val_acc:.4f}")
        print(f"Best checkpoint           : {os.path.join(self.checkpoint_dir, 'best.pt')}")
        print("Next step                : evaluate using data/split/test_150/test.csv")
        print("=" * 70)

    # =====================================================
    # Save Checkpoint
    # =====================================================

    def save_checkpoint(self, epoch, val_acc, tag="last"):
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "best_val_acc": self.best_val_acc,
            "val_acc": val_acc,
        }

        path = os.path.join(self.checkpoint_dir, f"{tag}.pt")
        torch.save(checkpoint, path)
        print(f"Checkpoint saved -> {path}")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    trainer = TemporalTestTrainer(
        train_csv="data/split/test_150/train.csv",
        val_csv="data/split/test_150/val.csv",
        dataset_root="data/raw",
        checkpoint_dir="checkpoints/temporal_test_150",
        batch_size=8,
        accumulation_steps=2,
        lr=1e-4,
        weight_decay=1e-4,
        num_epochs=5,
        num_classes=2,
        annealing_step=10,
        freeze_backbone=True,
        gradient_checkpointing=True,
        num_workers=8,
        early_stopping_patience=2,
    )

    trainer.fit()