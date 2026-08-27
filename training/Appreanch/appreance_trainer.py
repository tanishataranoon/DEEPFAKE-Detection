"""
=========================================================
Appearance Trainer

Trains the Appearance Branch using:

AppearanceClassifier
        ↓
Evidence
        ↓
EDL Loss

Supports

- Mixed Precision (AMP)
- Gradient Accumulation
- Checkpoint Saving
- Validation
- Resume Training

Designed for a single 8GB GPU.
=========================================================
"""

import os
import time
import subprocess
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from data.dataset.appearance.deepfake_dataset import DeepfakeDataset

from models.appearence.appearance_classifier import AppearanceClassifier

from models.heads.evidence_head import compute_dirichlet

from losses.edl_loss import compute_edl_loss


class AppearanceTrainer:

    """
    Trainer for the Appearance Branch.
    """

    def __init__(
        self,
        train_csv,
        val_csv,
        dataset_root,
        checkpoint_dir="checkpoints/appearance",
        batch_size=2,
        accumulation_steps=8,
        lr=1e-4,
        weight_decay=1e-4,
        num_epochs=1, #30
        num_classes=2,
        annealing_step=10,
        freeze_backbone=True,
        num_workers=4,
        device=None,
    ):

        # -------------------------------------------------
        # Device
        # -------------------------------------------------

        self.device = device or (
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        # -------------------------------------------------
        # Hyperparameters
        # -------------------------------------------------

        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps
        self.num_epochs = num_epochs
        self.num_classes = num_classes
        self.annealing_step = annealing_step

        self.best_val_acc = 0.0

        # -------------------------------------------------
        # Checkpoints
        # -------------------------------------------------

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(
            self.checkpoint_dir,
            exist_ok=True,
        )

        print("=" * 60)
        print("Appearance Trainer")
        print("=" * 60)
        print(f"Device               : {self.device}")
        print(f"Batch Size           : {batch_size}")
        print(f"Gradient Accumulation: {accumulation_steps}")
        print(
            f"Effective Batch Size : "
            f"{batch_size * accumulation_steps}"
        )
        print(f"Epochs               : {num_epochs}")
        print("=" * 60)

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

        # -------------------------------------------------
        # DataLoader
        # -------------------------------------------------

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
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
            f"Trainable Parameters : "
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
            "cuda",enabled=(self.device == "cuda"),
        )
    def get_gpu_temperature(self):
        """
        Returns current NVIDIA GPU temperature in Celsius.

        Returns None if temperature cannot be read.
        """

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

            temperature = int(
                result.stdout.strip().splitlines()[0]
            )

            return temperature

        except Exception:
            return None
    def check_gpu_temperature(self, epoch, step):
        """
        Checks GPU temperature.

        80C  -> warning
        85C  -> stop training safely
        """

        temperature = self.get_gpu_temperature()

        if temperature is None:
            return False

        if temperature >= 85:

            print()
            print("=" * 60)
            print(
                f"WARNING: GPU temperature = "
                f"{temperature}°C"
            )
            print(
                "GPU temperature is too high."
            )
            print(
                "Stopping training safely."
            )
            print("=" * 60)

            # Save emergency checkpoint
            self._save_checkpoint(
                epoch,
                0.0,
                tag="temperature_stop",
            )

            return True

        if temperature >= 80:

            print(
                f"WARNING: GPU temperature = "
                f"{temperature}°C"
            )

        return False
    # =====================================================
    # Forward Step
    # =====================================================

    def _forward_step(self, clip):

        clip = clip.to(
            self.device,
            non_blocking=True,
        )

        outputs = self.model(clip)

        return outputs
# =====================================================
# Train One Epoch
# =====================================================

    def train_one_epoch(
        self,
        epoch ):
        """
        Train the model for one epoch.

        Returns
        -------
        average_loss : float

        accuracy : float
        """

        self.model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader):

            # ---------------------------------------------
            # Load batch
            # ---------------------------------------------

            clip = batch["clip"]

            label = batch["label"].to(
                self.device,
                non_blocking=True,
            )

            # ---------------------------------------------
            # Forward
            # ---------------------------------------------

            with autocast(device_type="cuda" if self.device == "cuda" else "cpu",
                        enabled=(self.device == "cuda"),):

                outputs = self._forward_step(
                    clip,
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

                # Gradient accumulation
                loss = (
                    loss /
                    self.accumulation_steps
                )

            # ---------------------------------------------
            # Backward
            # ---------------------------------------------

            self.scaler.scale(
                loss
            ).backward()

            # ---------------------------------------------
            # Optimizer step
            # ---------------------------------------------

            if (
                (step + 1)
                %
                self.accumulation_steps
                == 0
            ):

                self.scaler.step(
                    self.optimizer
                )

                self.scaler.update()

                self.optimizer.zero_grad()

            # ---------------------------------------------
            # Statistics
            # ---------------------------------------------

            running_loss += (
                loss.item()
                *
                self.accumulation_steps
            )

            dirichlet = compute_dirichlet(
                evidence,
            )

            prediction = torch.argmax(
                dirichlet["probability"],
                dim=1,
            )

            correct += (
                prediction == label
            ).sum().item()

            total += label.size(0)

            # ---------------------------------------------
            # Progress
            # ---------------------------------------------

            if (
                (step + 1)
                %
                20
                == 0
            ):

                print(
                    f"Step "
                    f"{step + 1}/{len(self.train_loader)}   "
                    f"Loss: "
                    f"{running_loss / (step + 1):.4f}"
                )

        # ---------------------------------------------
        # Handle remaining gradients
        # ---------------------------------------------

        if (
            len(self.train_loader)
            %
            self.accumulation_steps
            != 0
        ):

            self.scaler.step(
                self.optimizer,
            )

            self.scaler.update()

            self.optimizer.zero_grad()

        average_loss = (
            running_loss /
            len(self.train_loader)
        )

        accuracy = (
            correct / total
            if total > 0
            else 0.0
        )

        return (
            average_loss,
            accuracy,
        )
    # =====================================================
    # Validation
    # =====================================================

    @torch.no_grad()
    def validate(
        self,
        epoch,
    ):
        """
        Evaluate the Appearance Branch on the validation set.

        Returns
        -------
        average_loss : float

        accuracy : float

        average_uncertainty : float
        """

        self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        total_uncertainty = 0.0

        for batch in self.val_loader:

            # -----------------------------------------
            # Load batch
            # -----------------------------------------

            clip = batch["clip"]

            label = batch["label"].to(
                self.device,
                non_blocking=True,
            )

            # -----------------------------------------
            # Forward
            # -----------------------------------------

            outputs = self._forward_step(
                clip,
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

            # -----------------------------------------
            # Dirichlet
            # -----------------------------------------

            dirichlet = compute_dirichlet(
                evidence,
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
            running_loss /
            len(self.val_loader)
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
        """
        Complete training loop.
        """

        print("\nStarting training...\n")

        for epoch in range(
            1,
            self.num_epochs + 1,
        ):

            start_time = time.time()

            train_loss, train_acc = self.train_one_epoch(
                epoch,
            )

            (
                val_loss,
                val_acc,
                val_uncertainty,
            ) = self.validate(
                epoch,
            )

            elapsed = (
                time.time()
                - start_time
            )

            print("=" * 70)
            print(
                f"Epoch "
                f"{epoch}/{self.num_epochs}"
            )
            print("=" * 70)

            print(
                f"Train Loss        : "
                f"{train_loss:.4f}"
            )

            print(
                f"Train Accuracy    : "
                f"{train_acc:.4f}"
            )

            print(
                f"Validation Loss   : "
                f"{val_loss:.4f}"
            )

            print(
                f"Validation Accuracy : "
                f"{val_acc:.4f}"
            )

            print(
                f"Avg Uncertainty   : "
                f"{val_uncertainty:.4f}"
            )

            print(
                f"Time              : "
                f"{elapsed:.1f}s"
            )

            # -----------------------------------------
            # Best checkpoint
            # -----------------------------------------

            if val_acc > self.best_val_acc:

                self.best_val_acc = val_acc

                self.save_checkpoint(
                    epoch=epoch,
                    val_acc=val_acc,
                    tag="best",
                )

            # -----------------------------------------
            # Latest checkpoint
            # -----------------------------------------

            self.save_checkpoint(
                epoch=epoch,
                val_acc=val_acc,
                tag="last",
            )

            print()
    # =====================================================
    # Save Checkpoint
    # =====================================================

    def save_checkpoint(
        self,
        epoch,
        val_acc,
        tag="last",
    ):
        """
        Save model checkpoint.
        """

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


    # =====================================================
    # Load Checkpoint
    # =====================================================

    def load_checkpoint(
        self,
        checkpoint_path,
    ):
        """
        Resume training from a checkpoint.
        """

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        self.model.load_state_dict(
            checkpoint["model_state"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer_state"]
        )

        self.scaler.load_state_dict(
            checkpoint["scaler_state"]
        )

        self.best_val_acc = checkpoint[
            "best_val_acc"
        ]

        print(
            f"Loaded checkpoint "
            f"(Epoch {checkpoint['epoch']})"
        )

        return checkpoint["epoch"]
if __name__ == "__main__":

    trainer = AppearanceTrainer(

        train_csv="data/split/small/train.csv",

        val_csv="data/split/small/val.csv",

        dataset_root="data/raw",

        checkpoint_dir="checkpoints/appearance",

        batch_size=2,

        accumulation_steps=8,

        lr=1e-4,
        num_epochs=25,
        # num_epochs=30,

        freeze_backbone=True,

    )

    trainer.fit()