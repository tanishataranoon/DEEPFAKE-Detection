"""
=========================================================
Appearance Branch - 150 Source Group Test Training
=========================================================

Small end-to-end training experiment.

Dataset split:
    100 source groups -> TRAIN
     25 source groups -> VALIDATION
     25 source groups -> TEST

IMPORTANT:
- Test CSV is NOT used during training.
- Validation is used only during training for model selection.
- Final evaluation must use test.csv with the saved best.pt.
- Existing full/small checkpoints are NOT overwritten.

Checkpoint directory:
    checkpoints/appearance_test_150/

Designed for a single 8GB GPU.
=========================================================
"""

import os
import time
import subprocess
import sys
from pathlib import Path
import torch

# Fixes the 'all_tied_weights_keys' crash caused by newer Transformers versions
_orig_getattr = torch.nn.Module.__getattr__
def _patched_getattr(self, name):
    if name == "all_tied_weights_keys":
        return {}
    return _orig_getattr(self, name)
torch.nn.Module.__getattr__ = _patched_getattr

# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from sklearn.metrics import balanced_accuracy_score

from data.raw.appearanch.dataset import DeepfakeDataset
from models.appearence.appearance_classifier import AppearanceClassifier
from models.heads.evidence_head import compute_dirichlet
from losses.edl_loss import compute_edl_loss
import numpy as np
from torch.utils.data import WeightedRandomSampler
# =========================================================
# RTX 4080 PERFORMANCE SETTINGS
# =========================================================
if torch.cuda.is_available():
    # RTX 40-series GPUs benefit from TF32 for compatible matrix operations.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


class AppearanceTestTrainer:

    def __init__(
        self,
        train_csv="data/split/test_150/train.csv",
        val_csv="data/split/test_150/val.csv",
        dataset_root="data/raw",
        checkpoint_dir="checkpoints/appearance_test_150",
        batch_size=8,
        accumulation_steps=2,
        lr=1e-4,
        weight_decay=1e-4,
        device=None,
        num_epochs=10,
        num_classes=2,
        annealing_step=10,
        freeze_backbone=True,
        num_workers=8,
        early_stopping_patience=2,
    ):

        self.device = device or (
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps
        self.num_epochs = num_epochs
        self.num_classes = num_classes
        self.annealing_step = annealing_step

        # Early stopping
        self.early_stopping_patience = early_stopping_patience
        self.epochs_without_improvement = 0

        self.best_val_acc = -1.0

        os.makedirs(
            checkpoint_dir,
            exist_ok=True,
        )

        self.checkpoint_dir = checkpoint_dir

        print("=" * 70)
        print("APPEARANCE BRANCH - 150 SOURCE GROUP TEST")
        print("=" * 70)
        print(f"Device                : {self.device}")
        print(f"Batch Size            : {batch_size}")
        print(f"Gradient Accumulation : {accumulation_steps}")
        print(
            f"Effective Batch Size  : "
            f"{batch_size * accumulation_steps}"
        )
        print(f"Epochs                : {num_epochs}")
        print(f"Early Stopping        : patience={early_stopping_patience}")

        print(f"Train CSV             : {train_csv}")
        print(f"Validation CSV        : {val_csv}")
        print(f"Checkpoint Directory  : {checkpoint_dir}")
        print("=" * 70)

        # -------------------------------------------------
        # Dataset
        # -------------------------------------------------

        self.train_dataset = DeepfakeDataset(
            metadata_csv=train_csv,
            dataset_root=dataset_root,
            split="train",
        )

        self.val_dataset = DeepfakeDataset(
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

        # Keep workers alive between epochs and prefetch batches so the GPU
        # spends less time waiting for video/frame loading.
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 4

        if torch.cuda.is_available():
            loader_kwargs["pin_memory_device"] = "cuda"

        labels = self.train_dataset.get_labels()          # you'll need to add this method — see below
        class_counts = np.bincount(labels)
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
            len(labels) / (2.0 * class_counts), dtype=torch.float32)

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            persistent_workers=(num_workers > 0),
            prefetch_factor=4 if num_workers > 0 else None,
            **(
                {"pin_memory_device": "cuda"}
                if torch.cuda.is_available()
                else {}
            ),
        )

        # -------------------------------------------------
        # Model
        # -------------------------------------------------

        self.model = AppearanceClassifier(
            num_classes=num_classes,
            freeze_backbone=freeze_backbone,
        ).to(self.device)

        trainable_params = [
            p
            for p in self.model.parameters()
            if p.requires_grad
        ]

        n_trainable = sum(
            p.numel()
            for p in trainable_params
        )

        print(
            f"Trainable Parameters  : "
            f"{n_trainable:,}"
        )

        # -------------------------------------------------
        # Optimizer
        # -------------------------------------------------

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay,
        )

        # -------------------------------------------------
        # AMP
        # -------------------------------------------------

        self.scaler = GradScaler(
            "cuda",
            enabled=(self.device == "cuda"),
        )

    # =====================================================
    # GPU Temperature
    # =====================================================

    def get_gpu_temperature(self):

        if self.device != "cuda":
            return None

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode != 0:
                return None

            return int(
                result.stdout.strip().splitlines()[0]
            )

        except Exception:
            return None

    def check_gpu_temperature(self, epoch, step):

        temperature = self.get_gpu_temperature()

        if temperature is None:
            return False

        # Critical temperature: stop training
        if temperature >= 85:

            print()
            print("=" * 70)
            print(
                f"WARNING: GPU temperature = "
                f"{temperature}°C"
            )
            print("Temperature is too high.")
            print("Stopping training safely.")
            print("=" * 70)

            self.save_checkpoint(
                epoch=epoch,
                val_acc=self.best_val_acc,
                tag="temperature_stop",
            )

            return True

        # Warning temperature: print only every 100 steps
        if temperature >= 80 and step % 100 == 0:
            print(
                f"WARNING: GPU temperature = "
                f"{temperature}°C"
            )

        return False
    # =====================================================
    # Forward
    # =====================================================

    def _forward_step(self, clip):

        clip = clip.to(
            self.device,
            non_blocking=True,
        )

        return self.model(clip)

    # =====================================================
    # Train One Epoch
    # =====================================================

    def train_one_epoch(self, epoch):

        self.model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        self.optimizer.zero_grad(
            set_to_none=True
        )

        num_steps = len(self.train_loader)

        for step, batch in enumerate(
            self.train_loader
        ):

            # Check GPU temperature every 100 steps
            if (step + 1) % 100 == 0:
                if self.check_gpu_temperature(
                    epoch,
                    step + 1,
                ):
                    return (
                        running_loss / max(step, 1),
                        correct / total if total else 0.0,
                        True,
                    )

            clip = batch["clip"]

            label = batch["label"].to(
                self.device,
                non_blocking=True,
            )

            with autocast(
                device_type=(
                    "cuda"
                    if self.device == "cuda"
                    else "cpu"
                ),
                enabled=(self.device == "cuda"),
            ):

                outputs = self._forward_step(
                    clip
                )

                evidence = outputs["evidence"]

                loss = compute_edl_loss(
                    evidence=evidence,
                    target=label,
                    epoch=epoch,
                    num_classes=self.num_classes,
                    annealing_step=self.annealing_step,
                    device=self.device,
                )

                loss_for_backward = (
                    loss
                    / self.accumulation_steps
                )

            self.scaler.scale(
                loss_for_backward
            ).backward()

            should_step = (
                (step + 1) % self.accumulation_steps == 0
                or (step + 1) == num_steps
            )

            if should_step:

                self.scaler.step(
                    self.optimizer
                )

                self.scaler.update()

                self.optimizer.zero_grad(
                    set_to_none=True
                )

            running_loss += loss.item()

            dirichlet = compute_dirichlet(
                evidence
            )

            prediction = torch.argmax(
                dirichlet["probability"],
                dim=1,
            )

            correct += (
                prediction == label
            ).sum().item()

            total += label.size(0)

            if (
                (step + 1) % 20 == 0
                or (step + 1) == num_steps
            ):

                print(
                    f"Step {step + 1}/{num_steps}   "
                    f"Loss: "
                    f"{running_loss / (step + 1):.4f}"
                )

        average_loss = (
            running_loss / num_steps
            if num_steps > 0
            else 0.0
        )

        accuracy = (
            correct / total
            if total > 0
            else 0.0
        )

        return (
            average_loss,
            accuracy,
            False,
        )

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

            label = batch["label"].to(
                self.device,
                non_blocking=True,
            )

            with autocast(
                device_type=(
                    "cuda"
                    if self.device == "cuda"
                    else "cpu"
                ),
                enabled=(self.device == "cuda"),
            ):

                outputs = self._forward_step(
                    clip
                )

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

            dirichlet = compute_dirichlet(
                evidence
            )

            prediction = torch.argmax(
                dirichlet["probability"],
                dim=1,
            )

            correct += (
                prediction == label
            ).sum().item()

            total += label.size(0)

            total_uncertainty += (
                dirichlet["uncertainty"]
                .sum()
                .item()
            )

        average_loss = (
            running_loss / len(self.val_loader)
            if len(self.val_loader) > 0
            else 0.0
        )

        accuracy = (
            correct / total
            if total > 0
            else 0.0
        )

        average_uncertainty = (
            total_uncertainty / total
            if total > 0
            else 0.0
        )

        return (
            average_loss,
            accuracy,
            average_uncertainty,
        )

    # =====================================================
    # Training Loop
    # =====================================================

    def fit(self):

        print()
        print("=" * 70)
        print("STARTING 150-SOURCE-GROUP TEST TRAINING")
        print("=" * 70)
        print()

        for epoch in range(
            1,
            self.num_epochs + 1,
        ):

            start_time = time.time()

            (
                train_loss,
                train_acc,
                stopped,
            ) = self.train_one_epoch(
                epoch
            )

            if stopped:
                print(
                    "Training stopped because of GPU temperature."
                )
                return

            (
                val_loss,
                val_acc,
                val_uncertainty,
            ) = self.validate(
                epoch
            )

            elapsed = (
                time.time()
                - start_time
            )

            print()
            print("=" * 70)
            print(
                f"Epoch {epoch}/{self.num_epochs}"
            )
            print("=" * 70)

            print(
                f"Train Loss          : "
                f"{train_loss:.4f}"
            )

            print(
                f"Train Accuracy      : "
                f"{train_acc:.4f}"
            )

            print(
                f"Validation Loss     : "
                f"{val_loss:.4f}"
            )

            print(
                f"Validation Accuracy : "
                f"{val_acc:.4f}"
            )

            print(
                f"Avg Uncertainty     : "
                f"{val_uncertainty:.4f}"
            )

            print(
                f"Time                : "
                f"{elapsed:.1f}s"
            )

        # -------------------------------------------------
        # Best checkpoint + Early Stopping
        # -------------------------------------------------

        if val_acc > self.best_val_acc:

            # New best validation accuracy
            self.best_val_acc = val_acc
            self.epochs_without_improvement = 0

            self.save_checkpoint(
                epoch=epoch,
                val_acc=val_acc,
                tag="best",
            )

            print(
                f"New best validation accuracy: "
                f"{val_acc:.4f}"
            )

        else:

            # No improvement this epoch
            self.epochs_without_improvement += 1

            print(
                f"No validation improvement. "
                f"Early stopping patience: "
                f"{self.epochs_without_improvement}/"
                f"{self.early_stopping_patience}"
            )

        # -------------------------------------------------
        # Latest checkpoint
        # -------------------------------------------------

        self.save_checkpoint(
            epoch=epoch,
            val_acc=val_acc,
            tag="last",
        )

        # -------------------------------------------------
        # Early stopping check
        # -------------------------------------------------

        if (
            self.epochs_without_improvement
            >= self.early_stopping_patience
        ):

            print()
            print("=" * 70)
            print("EARLY STOPPING")
            print("=" * 70)
            print(
                f"Validation accuracy did not improve for "
                f"{self.early_stopping_patience} consecutive epochs."
            )
            print(
                f"Best Validation Accuracy : "
                f"{self.best_val_acc:.4f}"
            )
            print(
                f"Best checkpoint           : "
                f"{os.path.join(self.checkpoint_dir, 'best.pt')}"
            )
            print("=" * 70)

            return

        print()
        print("=" * 70)
        print("TEST TRAINING FINISHED")
        print("=" * 70)
        print(
            f"Best Validation Accuracy : "
            f"{self.best_val_acc:.4f}"
        )
        print(
            "Best checkpoint           : "
            f"{os.path.join(self.checkpoint_dir, 'best.pt')}"
        )
        print(
            "Next step                : "
            "evaluate using data/split/test_150/test.csv"
        )
        print("=" * 70)

    # =====================================================
    # Save Checkpoint
    # =====================================================

    def save_checkpoint(
        self,
        epoch,
        val_acc,
        tag="last",
    ):

        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "best_val_acc": self.best_val_acc,
            "val_acc": val_acc,
        }

        path = os.path.join(
            self.checkpoint_dir,
            f"{tag}.pt",
        )

        torch.save(
            checkpoint,
            path,
        )

        print(
            f"Checkpoint saved -> {path}"
        )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    trainer = AppearanceTestTrainer(
        train_csv="data/split/test_150/train.csv",
        val_csv="data/split/test_150/val.csv",
        dataset_root="data/raw",

        # IMPORTANT:
        # Separate from the normal appearance checkpoints.
        checkpoint_dir="checkpoints/appearance_test_150",

        batch_size=8,
        accumulation_steps=2,

        lr=1e-4,
        weight_decay=1e-4,

        # Small test run:
        num_epochs=10,
        num_classes=2,
        annealing_step=10,
        freeze_backbone=True,
        num_workers=8,
        early_stopping_patience=2,
    )

    trainer.fit()