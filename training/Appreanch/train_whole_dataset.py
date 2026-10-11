"""
=========================================================
Temporal Branch - Full Dataset Training
=========================================================

Uses the SAME cached face crops as the Appearance Branch.

Train:
    data/split/train.csv
    data/cache/train/

Validation:
    data/split/val.csv
    data/cache/val/

Test:
    data/split/test.csv is NOT used during training.

Model:
    TemporalClassifier
        └── VideoMAE backbone
              ├── first blocks frozen
              └── last 4 blocks fine-tuned
        └── Evidence Head

EDL:
    0 = Real
    1 = Fake

Uncertain:
    Not a learned class.
    It is derived from EDL uncertainty.

Main configuration:
    Batch size              = 16
    Gradient accumulation   = 1
    Effective batch size    = 16
    VideoMAE LR             = 1e-5
    Head LR                 = 5e-5
    Epochs                  = 20
    Fine-tune last 4 blocks
    Early stopping patience = 3

Output:
    checkpoints/temporal_full/
        best.pt
        last.pt
        temperature_stop.pt
=========================================================
"""

import os
import sys
import time
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler

from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
)


# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================
# PROJECT IMPORTS
# =========================================================

from data.raw.temporal.cached_dataset import CachedClipDataset

from models.temporal.temporal_classifier import TemporalClassifier
from models.heads.evidence_head import compute_dirichlet
from losses.edl_loss import compute_edl_loss


# =========================================================
# TRAINER
# =========================================================

class TemporalTrainer:

    def __init__(
        self,

        train_csv="data/split/train.csv",
        val_csv="data/split/val.csv",

        checkpoint_dir="checkpoints/temporal_full",

        # -------------------------------------------------
        # BATCH CONFIGURATION
        # -------------------------------------------------

        batch_size=16,
        accumulation_steps=1,

        # -------------------------------------------------
        # LEARNING RATES
        # -------------------------------------------------

        backbone_lr=1e-5,
        head_lr=5e-5,

        weight_decay=1e-4,

        # -------------------------------------------------
        # TRAINING
        # -------------------------------------------------

        num_epochs=20,
        num_classes=2,
        annealing_step=10,

        # Fine-tune last 4 VideoMAE blocks
        finetune_last_n_blocks=4,

        gradient_checkpointing=True,

        num_workers=8,

        early_stopping_patience=6,


        uncertainty_threshold=0.50,

        gradient_clip_norm=1.0,

        # -------------------------------------------------
        # LR SCHEDULER
        # -------------------------------------------------

        lr_scheduler_factor=0.5,
        lr_scheduler_patience=1,
        lr_scheduler_min_lr=1e-7,

        device=None,

        # -------------------------------------------------
        # CACHE
        # -------------------------------------------------

        train_cache="data/cache/train",
        val_cache="data/cache/val",
    ):

        # =================================================
        # DEVICE
        # =================================================

        if device is None:
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            self.device = device


        # =================================================
        # CONFIGURATION
        # =================================================

        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps

        self.backbone_lr = backbone_lr
        self.head_lr = head_lr
        self.weight_decay = weight_decay

        self.num_epochs = num_epochs
        self.num_classes = num_classes
        self.annealing_step = annealing_step

        self.finetune_last_n_blocks = finetune_last_n_blocks

        self.num_workers = num_workers

        self.early_stopping_patience = (
            early_stopping_patience
        )

        self.epochs_without_improvement = 0

        self.uncertainty_threshold = (
            uncertainty_threshold
        )

        self.gradient_clip_norm = gradient_clip_norm

        self.best_balanced_accuracy = -1.0
        self.best_epoch = 0


        # =================================================
        # CHECKPOINT DIRECTORY
        # =================================================

        self.checkpoint_dir = checkpoint_dir

        os.makedirs(
            self.checkpoint_dir,
            exist_ok=True
        )


        # =================================================
        # GPU SETTINGS
        # =================================================

        if torch.cuda.is_available():

            torch.backends.cuda.matmul.allow_tf32 = True

            torch.backends.cudnn.allow_tf32 = True

            torch.backends.cudnn.benchmark = True


        # =================================================
        # CONFIGURATION SUMMARY
        # =================================================

        print("=" * 78)
        print(
            "TEMPORAL BRANCH (VideoMAE) - "
            "FULL DATASET TRAINING"
        )
        print("=" * 78)

        print(f"Device                  : {self.device}")

        if self.device == "cuda":
            print(
                f"GPU                     : "
                f"{torch.cuda.get_device_name(0)}"
            )

        print(
            f"Batch size              : "
            f"{batch_size}"
        )

        print(
            f"Gradient accumulation   : "
            f"{accumulation_steps}"
        )

        print(
            f"Effective batch size    : "
            f"{batch_size * accumulation_steps}"
        )

        print(
            f"VideoMAE last N layers  : "
            f"{finetune_last_n_blocks}"
        )

        print(
            f"VideoMAE learning rate  : "
            f"{backbone_lr:.2e}"
        )

        print(
            f"Head learning rate      : "
            f"{head_lr:.2e}"
        )

        print(
            f"Weight decay            : "
            f"{weight_decay:.2e}"
        )

        print(
            f"Maximum epochs          : "
            f"{num_epochs}"
        )

        print(
            f"Early stopping patience : "
            f"{early_stopping_patience}"
        )

        print(
            f"Uncertainty threshold   : "
            f"{uncertainty_threshold:.2f}"
        )

        print(
            f"Gradient clip norm      : "
            f"{gradient_clip_norm:.2f}"
        )

        print(
            "Class balancing         : "
            "WeightedRandomSampler"
        )

        print(
            "EDL class weights       : "
            "Disabled"
        )

        print(
            f"Train cache             : "
            f"{train_cache}"
        )

        print(
            f"Validation cache        : "
            f"{val_cache}"
        )

        print(
            f"Checkpoint directory    : "
            f"{checkpoint_dir}"
        )

        print("=" * 78)


        # =================================================
        # CHECK CACHE
        # =================================================

        if not os.path.isdir(train_cache):
            raise FileNotFoundError(
                f"Training cache not found: {train_cache}\n"
                "Run build_all_cache.py first."
            )

        if not os.path.isdir(val_cache):
            raise FileNotFoundError(
                f"Validation cache not found: {val_cache}\n"
                "Run build_all_cache.py first."
            )


        # =================================================
        # DATASETS
        # =================================================

        print()
        print("Loading cached datasets...")

        self.train_dataset = CachedClipDataset(
            train_csv,
            train_cache
        )

        self.val_dataset = CachedClipDataset(
            val_csv,
            val_cache
        )

        print()
        print(
            f"Training samples        : "
            f"{len(self.train_dataset):,}"
        )

        print(
            f"Validation samples      : "
            f"{len(self.val_dataset):,}"
        )


        # =================================================
        # CLASS DISTRIBUTION
        # =================================================

        labels = self._get_labels(train_csv)

        if labels.size == 0:
            raise RuntimeError(
                "Training dataset contains no labels."
            )

        if np.any(labels < 0) or np.any(
            labels >= num_classes
        ):
            raise ValueError(
                f"Training labels must be in "
                f"[0, {num_classes - 1}]. "
                f"Found: {np.unique(labels)}"
            )

        class_counts = np.bincount(
            labels,
            minlength=num_classes
        ).astype(np.float64)

        print()
        print("Training class distribution:")
        print("-" * 78)

        for class_id, count in enumerate(
            class_counts
        ):

            class_name = (
                "Real"
                if class_id == 0
                else "Fake"
            )

            print(
                f"  Class {class_id} "
                f"({class_name}): "
                f"{int(count):,}"
            )

        if np.any(class_counts == 0):
            raise RuntimeError(
                "At least one class is absent "
                "from the training split."
            )


        # =================================================
        # WEIGHTED RANDOM SAMPLER
        # =================================================

        print()
        print(
            "Creating WeightedRandomSampler..."
        )

        sample_weights = (
            1.0 / class_counts[labels]
        )

        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(
                sample_weights,
                dtype=torch.double
            ),
            num_samples=len(labels),
            replacement=True,
        )

        print(
            "Class balancing will be handled "
            "by the sampler."
        )

        print(
            "EDL class weighting is disabled "
            "to avoid double correction."
        )


        # =================================================
        # DATALOADERS
        # =================================================

        train_loader_kwargs = dict(
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=(
                self.device == "cuda"
            ),
            drop_last=False,
        )

        val_loader_kwargs = dict(
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(
                self.device == "cuda"
            ),
            drop_last=False,
        )

        if num_workers > 0:

            train_loader_kwargs[
                "persistent_workers"
            ] = True

            train_loader_kwargs[
                "prefetch_factor"
            ] = 4

            val_loader_kwargs[
                "persistent_workers"
            ] = True

            val_loader_kwargs[
                "prefetch_factor"
            ] = 4


        self.train_loader = DataLoader(
            self.train_dataset,
            **train_loader_kwargs
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            **val_loader_kwargs
        )

        print()
        print(
            f"Training batches        : "
            f"{len(self.train_loader):,}"
        )

        print(
            f"Validation batches      : "
            f"{len(self.val_loader):,}"
        )


        # =================================================
        # MODEL
        # =================================================

        print()
        print("Loading TemporalClassifier...")

        self.model = TemporalClassifier(
            num_classes=num_classes,
            freeze_backbone=True,
            gradient_checkpointing=(
                gradient_checkpointing
            ),
        ).to(self.device)

        self.model.unfreeze_backbone(
            last_n_blocks=finetune_last_n_blocks
        )

        self._print_parameter_summary()


        # =================================================
        # OPTIMIZER
        # =================================================

        backbone_parameters = []
        head_parameters = []

        BACKBONE_PREFIX = (
            "backbone.backbone.backbone."
        )

        for name, parameter in (
            self.model.named_parameters()
        ):

            if not parameter.requires_grad:
                continue

            if name.startswith(
                BACKBONE_PREFIX
            ):
                backbone_parameters.append(
                    parameter
                )
            else:
                head_parameters.append(
                    parameter
                )

        if not backbone_parameters:
            raise RuntimeError(
                "No trainable VideoMAE backbone "
                "parameters were found."
            )

        if not head_parameters:
            raise RuntimeError(
                "No trainable task-specific "
                "head parameters were found."
            )


        self.optimizer = torch.optim.AdamW(
            [
                {
                    "params": backbone_parameters,
                    "lr": backbone_lr
                },
                {
                    "params": head_parameters,
                    "lr": head_lr
                },
            ],
            weight_decay=weight_decay,
        )


        # =================================================
        # LR SCHEDULER
        # =================================================

        self.scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=lr_scheduler_factor,
                patience=lr_scheduler_patience,
                min_lr=lr_scheduler_min_lr,
            )
        )


        # =================================================
        # MIXED PRECISION
        # =================================================

        self.scaler = GradScaler(
            "cuda",
            enabled=(
                self.device == "cuda"
            )
        )


    # =====================================================
    # GET LABELS
    # =====================================================

    @staticmethod
    def _get_labels(csv_file):

        df = pd.read_csv(csv_file)

        if "label" not in df.columns:
            raise ValueError(
                f"'label' column not found "
                f"in {csv_file}"
            )

        labels = (
            df["label"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({
                "real": 0,
                "fake": 1,
                "0": 0,
                "1": 1,
            })
        )

        if labels.isna().any():

            bad_labels = (
                df.loc[
                    labels.isna(),
                    "label"
                ].unique()
            )

            raise ValueError(
                f"Unknown labels found: "
                f"{bad_labels}"
            )

        return labels.to_numpy(
            dtype=np.int64
        )


    # =====================================================
    # PARAMETER SUMMARY
    # =====================================================

    def _print_parameter_summary(self):

        total = 0
        trainable = 0

        backbone_trainable = 0
        head_trainable = 0

        BACKBONE_PREFIX = (
            "backbone.backbone.backbone."
        )

        for name, parameter in (
            self.model.named_parameters()
        ):

            total += parameter.numel()

            if parameter.requires_grad:

                trainable += parameter.numel()

                if name.startswith(
                    BACKBONE_PREFIX
                ):
                    backbone_trainable += (
                        parameter.numel()
                    )
                else:
                    head_trainable += (
                        parameter.numel()
                    )

        print()
        print("Parameter summary")
        print("-" * 78)

        print(
            f"Total parameters        : "
            f"{total:,}"
        )

        print(
            f"Trainable parameters    : "
            f"{trainable:,}"
        )

        print(
            f"Frozen parameters       : "
            f"{total - trainable:,}"
        )

        print(
            f"Trainable VideoMAE      : "
            f"{backbone_trainable:,}"
        )

        print(
            f"Trainable head          : "
            f"{head_trainable:,}"
        )


    # =====================================================
    # CURRENT LEARNING RATES
    # =====================================================

    def _current_lrs(self):

        return [
            group["lr"]
            for group
            in self.optimizer.param_groups
        ]


    # =====================================================
    # GPU TEMPERATURE
    # =====================================================

    def get_gpu_temperature(self):

        if self.device != "cuda":
            return None

        try:

            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=temperature.gpu",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode != 0:
                return None

            output = (
                result.stdout
                .strip()
                .splitlines()
            )

            return (
                int(output[0])
                if output
                else None
            )

        except Exception:
            return None


    def check_gpu_temperature(
        self,
        epoch,
        step
    ):

        temperature = (
            self.get_gpu_temperature()
        )

        if temperature is None:
            return False

        if temperature >= 85:

            print()
            print("=" * 78)
            print(
                f"GPU temperature = "
                f"{temperature}°C"
            )
            print(
                "Stopping training safely."
            )
            print("=" * 78)

            self.save_checkpoint(
                epoch=epoch,
                balanced_accuracy=(
                    self.best_balanced_accuracy
                ),
                val_accuracy=None,
                tag="temperature_stop",
                stop_reason=(
                    f"GPU temperature "
                    f"reached {temperature}°C"
                ),
            )

            return True

        if (
            temperature >= 80
            and step % 100 == 0
        ):

            print(
                f"WARNING: GPU temperature "
                f"= {temperature}°C"
            )

        return False


    # =====================================================
    # FORWARD
    # =====================================================

    def _forward_step(self, clip):

        # CachedClipDataset returns:
        # [B, T, 3, 224, 224] uint8

        clip = clip.to(
            self.device,
            non_blocking=True
        )

        # Convert uint8 [0,255]
        # to float [0,1].
        clip = clip.float() / 255.0

        return self.model(clip)


    # =====================================================
    # EDL REPORTING
    # =====================================================

    def _edl_report(
        self,
        evidence,
        labels=None
    ):

        dirichlet = compute_dirichlet(
            evidence
        )

        probabilities = (
            dirichlet["probability"]
        )

        uncertainty = (
            dirichlet["uncertainty"]
            .view(-1)
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        )

        uncertain_mask = (
            uncertainty
            >= self.uncertainty_threshold
        )

        reported_class = (
            predicted_class.clone()
        )

        # 2 = Uncertain
        reported_class[
            uncertain_mask
        ] = 2

        return {
            "dirichlet": dirichlet,
            "predicted_class": predicted_class,
            "uncertainty": uncertainty,
            "uncertain_mask": uncertain_mask,
            "reported_class": reported_class,
        }


    # =====================================================
    # UNPACK CACHED BATCH
    # =====================================================

    @staticmethod
    def _unpack_batch(batch):

        # CachedClipDataset returns:
        # (clip, label)

        clip, label = batch

        return clip, label


    # =====================================================
    # TRAIN ONE EPOCH
    # =====================================================

    def train_one_epoch(
        self,
        epoch
    ):

        self.model.train()

        running_loss = 0.0

        correct = 0
        total = 0

        self.optimizer.zero_grad(
            set_to_none=True
        )

        num_steps = len(
            self.train_loader
        )

        for step, batch in enumerate(
            self.train_loader
        ):

            current_step = step + 1


            # ---------------------------------------------
            # GPU TEMPERATURE
            # ---------------------------------------------

            if current_step % 100 == 0:

                if self.check_gpu_temperature(
                    epoch,
                    current_step
                ):

                    average_loss = (
                        running_loss
                        / max(current_step, 1)
                    )

                    accuracy = (
                        correct / total
                        if total > 0
                        else 0.0
                    )

                    return (
                        average_loss,
                        accuracy,
                        True
                    )


            # ---------------------------------------------
            # GET BATCH
            # ---------------------------------------------

            clip, label = (
                self._unpack_batch(batch)
            )

            label = label.to(
                self.device,
                non_blocking=True
            )


            # ---------------------------------------------
            # FORWARD + LOSS
            # ---------------------------------------------

            with autocast(
                device_type="cuda",
                enabled=(
                    self.device == "cuda"
                )
            ):

                outputs = (
                    self._forward_step(
                        clip
                    )
                )

                evidence = (
                    outputs["evidence"]
                )

                loss = compute_edl_loss(
                    evidence=evidence,
                    target=label,
                    epoch=epoch,
                    num_classes=self.num_classes,
                    annealing_step=(
                        self.annealing_step
                    ),
                    class_weights=None,
                    device=self.device,
                )

                loss_for_backward = (
                    loss
                    / self.accumulation_steps
                )


            # ---------------------------------------------
            # BACKPROPAGATION
            # ---------------------------------------------

            self.scaler.scale(
                loss_for_backward
            ).backward()


            # ---------------------------------------------
            # UPDATE WEIGHTS
            # ---------------------------------------------

            should_step = (
                current_step
                % self.accumulation_steps
                == 0
                or current_step == num_steps
            )

            if should_step:

                self.scaler.unscale_(
                    self.optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_norm,
                )

                self.scaler.step(
                    self.optimizer
                )

                self.scaler.update()

                self.optimizer.zero_grad(
                    set_to_none=True
                )


            # ---------------------------------------------
            # LOSS
            # ---------------------------------------------

            running_loss += loss.item()


            # ---------------------------------------------
            # TRAINING ACCURACY
            # ---------------------------------------------

            with torch.no_grad():

                report = (
                    self._edl_report(
                        evidence,
                        label
                    )
                )

                prediction = (
                    report["predicted_class"]
                )

                correct += (
                    prediction == label
                ).sum().item()

                total += (
                    label.size(0)
                )


            # ---------------------------------------------
            # PRINT PROGRESS
            # ---------------------------------------------

            if (
                current_step % 20 == 0
                or current_step == num_steps
            ):

                print(
                    f"Step "
                    f"{current_step:,}/"
                    f"{num_steps:,}   "
                    f"Loss: "
                    f"{running_loss / current_step:.4f}"
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
            False
        )


    # =====================================================
    # VALIDATION
    # =====================================================

    @torch.no_grad()
    def validate(self, epoch):

        self.model.eval()

        running_loss = 0.0

        all_labels = []
        all_predictions = []
        all_uncertainties = []

        total_real = 0
        total_fake = 0
        total_uncertain = 0


        for batch in self.val_loader:

            clip, label = (
                self._unpack_batch(batch)
            )

            label = label.to(
                self.device,
                non_blocking=True
            )


            with autocast(
                device_type="cuda",
                enabled=(
                    self.device == "cuda"
                )
            ):

                outputs = (
                    self._forward_step(
                        clip
                    )
                )

                evidence = (
                    outputs["evidence"]
                )

                loss = compute_edl_loss(
                    evidence=evidence,
                    target=label,
                    epoch=epoch,
                    num_classes=self.num_classes,
                    annealing_step=(
                        self.annealing_step
                    ),
                    class_weights=None,
                    device=self.device,
                )


            running_loss += loss.item()


            report = (
                self._edl_report(
                    evidence,
                    label
                )
            )

            prediction = (
                report["predicted_class"]
            )

            uncertainty = (
                report["uncertainty"]
            )

            reported_class = (
                report["reported_class"]
            )


            all_labels.extend(
                label.cpu()
                .numpy()
                .tolist()
            )

            all_predictions.extend(
                prediction.cpu()
                .numpy()
                .tolist()
            )

            all_uncertainties.extend(
                uncertainty.cpu()
                .numpy()
                .tolist()
            )


            total_real += int(
                (reported_class == 0)
                .sum()
                .item()
            )

            total_fake += int(
                (reported_class == 1)
                .sum()
                .item()
            )

            total_uncertain += int(
                (reported_class == 2)
                .sum()
                .item()
            )


        if not all_labels:

            raise RuntimeError(
                "Validation loader produced "
                "no samples."
            )


        labels_np = np.asarray(
            all_labels,
            dtype=np.int64
        )

        predictions_np = np.asarray(
            all_predictions,
            dtype=np.int64
        )

        uncertainties_np = np.asarray(
            all_uncertainties,
            dtype=np.float64
        )


        # ---------------------------------------------
        # METRICS
        # ---------------------------------------------

        accuracy = float(
            (
                predictions_np
                == labels_np
            ).mean()
        )

        balanced_accuracy = float(
            balanced_accuracy_score(
                labels_np,
                predictions_np
            )
        )


        # ---------------------------------------------
        # CONFUSION MATRIX
        # ---------------------------------------------

        cm = confusion_matrix(
            labels_np,
            predictions_np,
            labels=[0, 1]
        )

        true_real = int(
            cm[0, 0]
        )

        false_fake = int(
            cm[0, 1]
        )

        false_real = int(
            cm[1, 0]
        )

        true_fake = int(
            cm[1, 1]
        )


        # ---------------------------------------------
        # REAL RECALL
        # ---------------------------------------------

        real_mask = (
            labels_np == 0
        )

        real_recall = (
            float(
                (
                    predictions_np[
                        real_mask
                    ] == 0
                ).mean()
            )
            if real_mask.any()
            else 0.0
        )


        # ---------------------------------------------
        # FAKE RECALL
        # ---------------------------------------------

        fake_mask = (
            labels_np == 1
        )

        fake_recall = (
            float(
                (
                    predictions_np[
                        fake_mask
                    ] == 1
                ).mean()
            )
            if fake_mask.any()
            else 0.0
        )


        # ---------------------------------------------
        # OTHER METRICS
        # ---------------------------------------------

        average_loss = (
            running_loss
            / len(self.val_loader)
            if len(self.val_loader) > 0
            else 0.0
        )

        average_uncertainty = float(
            uncertainties_np.mean()
        )

        uncertain_rate = float(
            (
                uncertainties_np
                >= self.uncertainty_threshold
            ).mean()
        )


        return {

            "loss":
                average_loss,

            "accuracy":
                accuracy,

            "balanced_accuracy":
                balanced_accuracy,

            "real_recall":
                real_recall,

            "fake_recall":
                fake_recall,

            "average_uncertainty":
                average_uncertainty,

            "uncertain_rate":
                uncertain_rate,

            "real_count":
                total_real,

            "fake_count":
                total_fake,

            "uncertain_count":
                total_uncertain,

            "true_real":
                true_real,

            "false_fake":
                false_fake,

            "false_real":
                false_real,

            "true_fake":
                true_fake,

            "total":
                len(labels_np),
        }


    # =====================================================
    # MAIN TRAINING LOOP
    # =====================================================

    def fit(self, start_epoch=1):

        print()
        print("=" * 78)
        print(
            "STARTING FULL-DATASET "
            "TEMPORAL TRAINING"
        )
        print("=" * 78)


        for epoch in range(
            start_epoch,
            self.num_epochs + 1
        ):

            start_time = time.time()


            # ---------------------------------------------
            # TRAIN
            # ---------------------------------------------

            (
                train_loss,
                train_acc,
                stopped
            ) = self.train_one_epoch(
                epoch
            )


            if stopped:

                print()
                print(
                    "Training stopped because "
                    "of GPU temperature."
                )

                return


            # ---------------------------------------------
            # VALIDATE
            # ---------------------------------------------

            val = self.validate(
                epoch
            )

            elapsed = (
                time.time()
                - start_time
            )


            # ---------------------------------------------
            # LR SCHEDULER
            # ---------------------------------------------

            lrs_before = (
                self._current_lrs()
            )

            self.scheduler.step(
                val["balanced_accuracy"]
            )

            lrs_after = (
                self._current_lrs()
            )


            # ---------------------------------------------
            # PRINT RESULTS
            # ---------------------------------------------

            print()
            print("=" * 78)

            print(
                f"Epoch "
                f"{epoch}/"
                f"{self.num_epochs}"
            )

            print("=" * 78)

            print(
                f"Train Loss             : "
                f"{train_loss:.4f}"
            )

            print(
                f"Train Accuracy         : "
                f"{train_acc:.4f}"
            )

            print(
                f"Validation Loss        : "
                f"{val['loss']:.4f}"
            )

            print(
                f"Validation Accuracy    : "
                f"{val['accuracy']:.4f}"
            )

            print(
                f"Validation Balanced Acc: "
                f"{val['balanced_accuracy']:.4f}"
            )

            print(
                f"Real Recall            : "
                f"{val['real_recall']:.4f}"
            )

            print(
                f"Fake Recall            : "
                f"{val['fake_recall']:.4f}"
            )

            print(
                f"Avg EDL Uncertainty    : "
                f"{val['average_uncertainty']:.4f}"
            )

            print(
                f"Uncertain Rate         : "
                f"{val['uncertain_rate']:.4f}"
            )

            print()

            print(
                f"Reported Real          : "
                f"{val['real_count']:,}"
            )

            print(
                f"Reported Fake          : "
                f"{val['fake_count']:,}"
            )

            print(
                f"Reported Uncertain     : "
                f"{val['uncertain_count']:,}"
            )

            print()

            print("Confusion Matrix")

            print(
                "                  "
                "Pred Real   Pred Fake"
            )

            print(
                f"Actual Real       "
                f"{val['true_real']:>9,}   "
                f"{val['false_fake']:>9,}"
            )

            print(
                f"Actual Fake       "
                f"{val['false_real']:>9,}   "
                f"{val['true_fake']:>9,}"
            )

            print()

            print(
                f"Time                   : "
                f"{elapsed:.1f}s"
            )


            if lrs_after != lrs_before:

                print()

                print(
                    f"LR reduced: "
                    f"backbone "
                    f"{lrs_before[0]:.2e} -> "
                    f"{lrs_after[0]:.2e}, "
                    f"head "
                    f"{lrs_before[1]:.2e} -> "
                    f"{lrs_after[1]:.2e}"
                )

            else:

                print(
                    f"Current LR: "
                    f"backbone "
                    f"{lrs_after[0]:.2e}, "
                    f"head "
                    f"{lrs_after[1]:.2e}"
                )


            # ---------------------------------------------
            # CHECK IMPROVEMENT
            # ---------------------------------------------

            improved = (
                val["balanced_accuracy"]
                > self.best_balanced_accuracy
            )


            if improved:

                self.best_balanced_accuracy = (
                    val["balanced_accuracy"]
                )

                self.best_epoch = epoch

                self.epochs_without_improvement = 0


                self.save_checkpoint(
                    epoch=epoch,
                    balanced_accuracy=(
                        val["balanced_accuracy"]
                    ),
                    val_accuracy=(
                        val["accuracy"]
                    ),
                    tag="best",
                    stop_reason=None,
                )

                print(
                    "New best balanced accuracy."
                )


            else:

                self.epochs_without_improvement += 1

                print(
                    "No balanced-accuracy "
                    "improvement: "
                    f"{self.epochs_without_improvement}/"
                    f"{self.early_stopping_patience}"
                )


            # ---------------------------------------------
            # SAVE LAST CHECKPOINT
            # ---------------------------------------------

            self.save_checkpoint(
                epoch=epoch,
                balanced_accuracy=(
                    val["balanced_accuracy"]
                ),
                val_accuracy=(
                    val["accuracy"]
                ),
                tag="last",
                stop_reason=None,
            )


            # ---------------------------------------------
            # EARLY STOPPING
            # ---------------------------------------------

            if (
                self.epochs_without_improvement
                >= self.early_stopping_patience
            ):

                print()
                print("=" * 78)
                print("EARLY STOPPING")
                print("=" * 78)

                print(
                    "Validation balanced "
                    "accuracy did not improve "
                    "for "
                    f"{self.early_stopping_patience} "
                    "consecutive epochs."
                )

                print(
                    f"Best balanced accuracy : "
                    f"{self.best_balanced_accuracy:.4f}"
                )

                print(
                    f"Best epoch             : "
                    f"{self.best_epoch}"
                )

                print(
                    "Best checkpoint        : "
                    f"{self.checkpoint_dir}/best.pt"
                )

                print("=" * 78)

                return


        # =================================================
        # FINISHED
        # =================================================

        print()
        print("=" * 78)
        print(
            "FULL-DATASET TRAINING FINISHED"
        )
        print("=" * 78)

        print(
            f"Best balanced accuracy : "
            f"{self.best_balanced_accuracy:.4f}"
        )

        print(
            f"Best epoch             : "
            f"{self.best_epoch}"
        )

        print(
            f"Best checkpoint        : "
            f"{self.checkpoint_dir}/best.pt"
        )

        print("=" * 78)


    # =====================================================
    # SAVE CHECKPOINT
    # =====================================================

    def save_checkpoint(
        self,
        epoch,
        balanced_accuracy,
        val_accuracy,
        tag="last",
        stop_reason=None,
    ):

        checkpoint = {

            "epoch":
                epoch,

            "model_state":
                self.model.state_dict(),

            "optimizer_state":
                self.optimizer.state_dict(),

            "scaler_state":
                self.scaler.state_dict(),

            "scheduler_state":
                self.scheduler.state_dict(),

            "best_balanced_accuracy":
                self.best_balanced_accuracy,

            "best_epoch":
                self.best_epoch,

            "val_balanced_accuracy":
                balanced_accuracy,

            "val_accuracy":
                val_accuracy,

            "finetune_last_n_blocks":
                self.finetune_last_n_blocks,

            "backbone_lr":
                self.backbone_lr,

            "head_lr":
                self.head_lr,

            "weight_decay":
                self.weight_decay,

            "uncertainty_threshold":
                self.uncertainty_threshold,

            "num_classes":
                self.num_classes,

            "annealing_step":
                self.annealing_step,

            "batch_size":
                self.batch_size,

            "accumulation_steps":
                self.accumulation_steps,

            "gradient_clip_norm":
                self.gradient_clip_norm,

            "stop_reason":
                stop_reason,
        }


        path = os.path.join(
            self.checkpoint_dir,
            f"{tag}.pt"
        )

        torch.save(
            checkpoint,
            path
        )

        print(
            f"Checkpoint saved -> {path}"
        )


    # =====================================================
    # RESUME TRAINING
    # =====================================================

    def load_checkpoint(
        self,
        checkpoint_path
    ):

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )


        self.model.load_state_dict(
            checkpoint["model_state"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer_state"]
        )


        scaler_state = (
            checkpoint.get(
                "scaler_state"
            )
        )

        if scaler_state:
            self.scaler.load_state_dict(
                scaler_state
            )


        scheduler_state = (
            checkpoint.get(
                "scheduler_state"
            )
        )

        if scheduler_state:
            self.scheduler.load_state_dict(
                scheduler_state
            )


        self.best_balanced_accuracy = (
            checkpoint.get(
                "best_balanced_accuracy",
                -1.0
            )
        )

        self.best_epoch = (
            checkpoint.get(
                "best_epoch",
                checkpoint.get(
                    "epoch",
                    0
                )
            )
        )

        self.epochs_without_improvement = 0

        loaded_epoch = (
            checkpoint.get(
                "epoch",
                0
            )
        )


        print()
        print("=" * 78)

        print(
            f"Loaded checkpoint "
            f"from epoch "
            f"{loaded_epoch}"
        )

        print(
            f"Best balanced accuracy : "
            f"{self.best_balanced_accuracy:.4f}"
        )

        print("=" * 78)


        return loaded_epoch


# =========================================================
# MAIN
# =========================================================

def main():

    checkpoint_dir = (
        "checkpoints/temporal_full"
    )

    resume_path = os.path.join(
        checkpoint_dir,
        "last.pt"
    )

    resume = os.path.exists(
        resume_path
    )


    trainer = TemporalTrainer(

        train_csv=
            "data/split/train.csv",

        val_csv=
            "data/split/val.csv",

        checkpoint_dir=
            checkpoint_dir,


        # =================================================
        # IMPORTANT:
        # Batch size = 16
        # No gradient accumulation
        # =================================================

        batch_size=16,

        accumulation_steps=1,


        # =================================================
        # LEARNING RATES
        # =================================================

        backbone_lr=1e-5,

        head_lr=5e-5,

        weight_decay=1e-4,


        # =================================================
        # TRAINING
        # =================================================

        num_epochs=20,

        num_classes=2,

        annealing_step=10,


        # =================================================
        # VIDEO MAE
        # =================================================

        finetune_last_n_blocks=4,

        gradient_checkpointing=True,


        # =================================================
        # DATA LOADING
        # =================================================

        num_workers=8,


        # =================================================
        # EARLY STOPPING
        # =================================================

        early_stopping_patience=3,


        # =================================================
        # EDL
        # =================================================

        uncertainty_threshold=0.50,


        # =================================================
        # STABILITY
        # =================================================

        gradient_clip_norm=1.0,


        # =================================================
        # LR SCHEDULER
        # =================================================

        lr_scheduler_factor=0.5,

        lr_scheduler_patience=1,

        lr_scheduler_min_lr=1e-7,


        # =================================================
        # CACHE
        # =================================================

        train_cache=
            "data/cache/train",

        val_cache=
            "data/cache/val",
    )


    # =====================================================
    # RESUME IF CHECKPOINT EXISTS
    # =====================================================

    start_epoch = 1

    if resume:

        print()
        print(
            f"Found existing checkpoint: "
            f"{resume_path}"
        )

        print(
            "Resuming training..."
        )

        start_epoch = (
            trainer.load_checkpoint(
                resume_path
            ) + 1
        )


    # =====================================================
    # START TRAINING
    # =====================================================

    trainer.fit(
        start_epoch=start_epoch
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()