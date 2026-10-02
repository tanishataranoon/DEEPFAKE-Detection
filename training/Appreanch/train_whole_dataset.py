
"""
=========================================================
Appearance Branch - Full Dataset Training
=========================================================

Train:
    data/split/train.csv

Validation:
    data/split/val.csv

Test:
    data/split/test.csv is NOT used here.

Model:
    AppearanceClassifier
        └── AIMv2-Large
              ├── first blocks frozen
              └── last 4 transformer blocks fine-tuned
        ├── Attention Pooling
        └── Evidence Head

EDL:
    0 = Real
    1 = Fake

"Uncertain" is NOT a learned class.
It is derived from EDL uncertainty for reporting.

Training:
    Batch size              : 8
    Gradient accumulation   : 2
    Effective batch size    : 16
    AIMv2 learning rate     : 1e-5
    Head/pooling LR         : 5e-5
    Weight decay            : 1e-4
    Maximum epochs          : 30
    Early stopping patience : 5

Class imbalance:
    WeightedRandomSampler is used.
    EDL class weights are NOT additionally applied.

Best checkpoint metric:
    Validation balanced accuracy

Output:
    checkpoints/appearance_full/
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

DATASET_ROOT = PROJECT_ROOT / "data" / "raw"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================
# PROJECT IMPORTS
# =========================================================

from data.raw.dataset import DeepfakeDataset
from data.raw.cached_dataset import CachedClipDataset
from torch.utils.data import Subset

from models.appearence.appearance_classifier import (
    AppearanceClassifier,
)

from models.heads.evidence_head import (
    compute_dirichlet,
)

from losses.edl_loss import compute_edl_loss


# =========================================================
# TRAINER
# =========================================================

class AppearanceTrainer:
    """
    Full-dataset trainer for the Appearance Branch.

    Architecture:

        Face clip
            ↓
        AIMv2
            ↓
        Attention Pooling
            ↓
        Evidence Head
            ↓
        EDL
            ↓
        Real / Fake
            ↓
        Uncertainty reporting

    AIMv2:
        Only the final N transformer blocks are fine-tuned.

    Class imbalance:
        WeightedRandomSampler is used.
        No additional EDL class weighting is used.
    """

    def __init__(
        self,

        train_csv="data/split/train.csv",
        val_csv="data/split/val.csv",
        dataset_root="data/raw",

        checkpoint_dir="checkpoints/appearance_full",

        batch_size=8,
        accumulation_steps=2,

        backbone_lr=1e-5,
        head_lr=5e-5,

        weight_decay=1e-4,

        num_epochs=30,
        num_classes=2,
        annealing_step=10,

        finetune_last_n_blocks=4,

        num_workers=4,

        early_stopping_patience=5,

        uncertainty_threshold=0.50,

        gradient_clip_norm=1.0,

        device=None,

        # --- speed-ups ---
        train_cache=None,          # e.g. "data/cache/train"
        val_cache=None,            # e.g. "data/cache/val"
        samples_per_epoch=None,    # None = len(train set)
        val_subset=None,           # None = full val set every epoch
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

        self.finetune_last_n_blocks = (
            finetune_last_n_blocks
        )

        self.num_workers = num_workers

        self.early_stopping_patience = (
            early_stopping_patience
        )

        self.epochs_without_improvement = 0

        self.uncertainty_threshold = (
            uncertainty_threshold
        )

        self.gradient_clip_norm = (
            gradient_clip_norm
        )

        self.best_balanced_accuracy = -1.0
        self.best_epoch = 0

        # =================================================
        # DATASET ROOT
        # =================================================

        self.dataset_root = Path(dataset_root)

        if not self.dataset_root.is_absolute():
            self.dataset_root = (
                PROJECT_ROOT / self.dataset_root
            )

        # =================================================
        # CHECKPOINT DIRECTORY
        # =================================================

        self.checkpoint_dir = checkpoint_dir

        os.makedirs(
            self.checkpoint_dir,
            exist_ok=True,
        )

        # =================================================
        # GPU PERFORMANCE SETTINGS
        # =================================================

        if torch.cuda.is_available():

            torch.backends.cuda.matmul.allow_tf32 = True

            torch.backends.cudnn.allow_tf32 = True

            torch.backends.cudnn.benchmark = True

        # =================================================
        # CONFIGURATION SUMMARY
        # =================================================

        print("=" * 78)
        print("APPEARANCE BRANCH - FULL DATASET TRAINING")
        print("=" * 78)

        print(
            f"Device                  : {self.device}"
        )

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
            f"AIMv2 last N blocks     : "
            f"{finetune_last_n_blocks}"
        )

        print(
            f"AIMv2 learning rate     : "
            f"{backbone_lr:.2e}"
        )

        print(
            f"Head/pooling LR         : "
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
            f"Train CSV               : "
            f"{train_csv}"
        )

        print(
            f"Validation CSV          : "
            f"{val_csv}"
        )

        print(
            f"Checkpoint directory    : "
            f"{checkpoint_dir}"
        )

        print("=" * 78)

        # =================================================
        # DATASETS
        # =================================================

        if train_cache and val_cache:
            self.train_dataset = CachedClipDataset(train_csv, train_cache)
            self.val_dataset = CachedClipDataset(val_csv, val_cache)
        else:
            self.train_dataset = DeepfakeDataset(
                metadata_csv=train_csv,
                video_root=self.dataset_root,
            )

            self.val_dataset = DeepfakeDataset(
                metadata_csv=val_csv,
                video_root=self.dataset_root,
            )

        # Fixed, stratified validation subset (same clips every epoch)
        if val_subset:
            vdf = pd.read_csv(val_csv)
            vlab = vdf["label"].astype(str).str.lower().str.strip().map(
                {"real": 0, "fake": 1, "0": 0, "1": 1}
            ).to_numpy()
            rng = np.random.default_rng(0)
            per_class = val_subset // 2
            chosen = []
            for c in (0, 1):
                idx = np.where(vlab == c)[0]
                chosen.append(rng.choice(idx, size=min(per_class, len(idx)), replace=False))
            chosen = np.sort(np.concatenate(chosen))
            self.val_dataset = Subset(self.val_dataset, chosen.tolist())
            print(f"Validation subset       : {len(chosen):,} clips (fixed, class-balanced)")

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

        labels = self._get_labels(
            train_csv
        )

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
            minlength=num_classes,
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
                f"  Class {class_id} ({class_name}): "
                f"{int(count):,}"
            )

        if np.any(class_counts == 0):

            raise RuntimeError(
                "At least one class is absent from "
                "the training split."
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
                dtype=torch.double,
            ),
            num_samples=samples_per_epoch or len(labels),
            replacement=True,
        )

        print(
            "Class balancing will be handled by "
            "the sampler."
        )

        print(
            "EDL class weighting is disabled to "
            "avoid double correction."
        )

        # =================================================
        # DATALOADERS
        # =================================================

        train_loader_kwargs = dict(
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=(self.device == "cuda"),
            drop_last=False,
        )

        val_loader_kwargs = dict(
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(self.device == "cuda"),
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

        # IMPORTANT:
        # Do NOT use pin_memory_device="cuda".
        # It is deprecated in the current environment.

        self.train_loader = DataLoader(
            self.train_dataset,
            **train_loader_kwargs,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            **val_loader_kwargs,
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

        self.model = AppearanceClassifier(
            num_classes=num_classes,
            freeze_backbone=True,
            uncertainty_threshold=uncertainty_threshold,
        ).to(self.device)

        # =================================================
        # PARTIAL AIMv2 FINE-TUNING
        # =================================================

        self.model.unfreeze_backbone(
            last_n_blocks=finetune_last_n_blocks,
        )

        # =================================================
        # PARAMETER SUMMARY
        # =================================================

        self._print_parameter_summary()

        # =================================================
        # OPTIMIZER
        # =================================================

        backbone_parameters = []
        head_parameters = []

        for name, parameter in (
            self.model.named_parameters()
        ):

            if not parameter.requires_grad:
                continue

            if name.startswith(
                "backbone.backbone.backbone."
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
                "No trainable AIMv2 backbone parameters "
                "were found."
            )

        if not head_parameters:

            raise RuntimeError(
                "No trainable task-specific parameters "
                "were found."
            )

        self.optimizer = torch.optim.AdamW(
            [
                {
                    "params": backbone_parameters,
                    "lr": backbone_lr,
                },
                {
                    "params": head_parameters,
                    "lr": head_lr,
                },
            ],
            weight_decay=weight_decay,
        )

        print()

        print(
            f"Optimizer AIMv2 params : "
            f"{sum(p.numel() for p in backbone_parameters):,}"
        )

        print(
            f"Optimizer head params  : "
            f"{sum(p.numel() for p in head_parameters):,}"
        )

        # =================================================
        # MIXED PRECISION
        # =================================================

        self.scaler = GradScaler(
            "cuda",
            enabled=(self.device == "cuda"),
        )

    # =====================================================
    # HELPERS
    # =====================================================

    @staticmethod
    def _get_labels(csv_file):
        """
        Read labels from the training CSV.

        Supported labels:

            real -> 0
            fake -> 1
            0    -> 0
            1    -> 1
        """

        df = pd.read_csv(csv_file)

        if "label" not in df.columns:
            raise ValueError(
                f"'label' column not found in {csv_file}. "
                f"Available columns: {list(df.columns)}"
            )

        labels = (
            df["label"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(
                {
                    "real": 0,
                    "fake": 1,
                    "0": 0,
                    "1": 1,
                }
            )
        )

        if labels.isna().any():

            bad_labels = (
                df.loc[
                    labels.isna(),
                    "label"
                ]
                .unique()
            )

            raise ValueError(
                f"Unknown labels found in {csv_file}: "
                f"{bad_labels}"
            )

        return labels.to_numpy(
            dtype=np.int64
        )

    def _print_parameter_summary(self):

        total = 0
        trainable = 0

        backbone_trainable = 0
        head_trainable = 0

        for name, parameter in (
            self.model.named_parameters()
        ):

            total += parameter.numel()

            if parameter.requires_grad:

                trainable += parameter.numel()

                if name.startswith(
                    "backbone.backbone.backbone."
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
            f"Trainable AIMv2         : "
            f"{backbone_trainable:,}"
        )

        print(
            f"Trainable head/pooling  : "
            f"{head_trainable:,}"
        )

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
                    "--format=csv,noheader,nounits",
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

            if not output:
                return None

            return int(output[0])

        except Exception:

            return None

    def check_gpu_temperature(
        self,
        epoch,
        step,
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
                    f"GPU temperature reached "
                    f"{temperature}°C"
                ),
            )

            return True

        if temperature >= 80 and step % 100 == 0:

            print(
                f"WARNING: GPU temperature = "
                f"{temperature}°C"
            )

        return False

    # =====================================================
    # FORWARD
    # =====================================================

    def _forward_step(self, clip):

        clip = clip.to(
            self.device,
            non_blocking=True,
        )

        # Cached dataset returns uint8; convert on the GPU.
        if clip.dtype == torch.uint8:
            clip = clip.float().div_(255.0)

        return self.model(clip)

    # =====================================================
    # EDL REPORTING
    # =====================================================

    def _edl_report(
        self,
        evidence,
        labels=None,
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
            dim=1,
        )

        uncertain_mask = (
            uncertainty
            >= self.uncertainty_threshold
        )

        # 0 = Real
        # 1 = Fake
        # 2 = Uncertain

        reported_class = (
            predicted_class.clone()
        )

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
    # TRAIN ONE EPOCH
    # =====================================================

    def train_one_epoch(self, epoch):

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
            # GPU TEMPERATURE CHECK
            # ---------------------------------------------

            if current_step % 100 == 0:

                if self.check_gpu_temperature(
                    epoch,
                    current_step,
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
                        True,
                    )

            # DeepfakeDataset returns:
            #
            #     (clip, label)
            #
            # NOT a dictionary.

            clip, label = batch

            label = label.to(
                self.device,
                non_blocking=True,
            )

            # ---------------------------------------------
            # FORWARD + LOSS
            # ---------------------------------------------

            with autocast(
                device_type="cuda",
                enabled=(
                    self.device == "cuda"
                ),
            ):

                outputs = (
                    self._forward_step(
                        clip
                    )
                )

                evidence = (
                    outputs["evidence"]
                )

                # IMPORTANT:
                # No class_weights here.
                #
                # WeightedRandomSampler already
                # balances the training distribution.

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
            # BACKWARD
            # ---------------------------------------------

            self.scaler.scale(
                loss_for_backward
            ).backward()

            # ---------------------------------------------
            # OPTIMIZER STEP
            # ---------------------------------------------

            should_step = (
                current_step
                % self.accumulation_steps
                == 0
                or
                current_step
                == num_steps
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
            # METRICS
            # ---------------------------------------------

            running_loss += (
                loss.item()
            )

            with torch.no_grad():

                report = self._edl_report(
                    evidence,
                    label,
                )

                prediction = (
                    report[
                        "predicted_class"
                    ]
                )

                correct += (
                    prediction == label
                ).sum().item()

                total += label.size(0)

            # ---------------------------------------------
            # PROGRESS
            # ---------------------------------------------

            if (
                current_step % 20 == 0
                or
                current_step == num_steps
            ):

                print(
                    f"Step {current_step:,}/"
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
            False,
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

            # DeepfakeDataset returns:
            #
            #     (clip, label)

            clip, label = batch

            label = label.to(
                self.device,
                non_blocking=True,
            )

            with autocast(
                device_type="cuda",
                enabled=(
                    self.device == "cuda"
                ),
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

            running_loss += (
                loss.item()
            )

            report = self._edl_report(
                evidence,
                label,
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
                label.detach()
                .cpu()
                .numpy()
                .tolist()
            )

            all_predictions.extend(
                prediction.detach()
                .cpu()
                .numpy()
                .tolist()
            )

            all_uncertainties.extend(
                uncertainty.detach()
                .cpu()
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

        # =================================================
        # VALIDATION RESULTS
        # =================================================

        if not all_labels:

            raise RuntimeError(
                "Validation loader produced "
                "no samples."
            )

        labels_np = np.asarray(
            all_labels,
            dtype=np.int64,
        )

        predictions_np = np.asarray(
            all_predictions,
            dtype=np.int64,
        )

        uncertainties_np = np.asarray(
            all_uncertainties,
            dtype=np.float64,
        )

        # =================================================
        # ACCURACY
        # =================================================

        accuracy = float(
            (
                predictions_np
                == labels_np
            ).mean()
        )

        # =================================================
        # BALANCED ACCURACY
        # =================================================

        balanced_accuracy = float(
            balanced_accuracy_score(
                labels_np,
                predictions_np,
            )
        )

        # =================================================
        # CONFUSION MATRIX
        #
        # rows    = actual
        # columns = predicted
        #
        #       Real Fake
        # Real
        # Fake
        # =================================================

        cm = confusion_matrix(
            labels_np,
            predictions_np,
            labels=[0, 1],
        )

        true_real = int(cm[0, 0])
        false_fake = int(cm[0, 1])
        false_real = int(cm[1, 0])
        true_fake = int(cm[1, 1])

        # =================================================
        # REAL RECALL
        # =================================================

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

        # =================================================
        # FAKE RECALL
        # =================================================

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

        # =================================================
        # LOSS
        # =================================================

        average_loss = (
            running_loss
            / len(self.val_loader)
            if len(self.val_loader) > 0
            else 0.0
        )

        # =================================================
        # EDL UNCERTAINTY
        # =================================================

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
            "loss": average_loss,

            "accuracy": accuracy,

            "balanced_accuracy": (
                balanced_accuracy
            ),

            "real_recall": real_recall,

            "fake_recall": fake_recall,

            "average_uncertainty": (
                average_uncertainty
            ),

            "uncertain_rate": uncertain_rate,

            "real_count": total_real,

            "fake_count": total_fake,

            "uncertain_count": (
                total_uncertain
            ),

            "true_real": true_real,

            "false_fake": false_fake,

            "false_real": false_real,

            "true_fake": true_fake,

            "total": len(labels_np),
        }

    # =====================================================
    # MAIN TRAINING LOOP
    # =====================================================

    def fit(self, start_epoch=1):

        print()
        print("=" * 78)
        print(
            "STARTING FULL-DATASET "
            "APPEARANCE TRAINING"
        )
        print("=" * 78)

        for epoch in range(
            start_epoch,
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

                print()
                print(
                    "Training stopped because "
                    "of GPU temperature."
                )

                return

            # =================================================
            # VALIDATION
            # =================================================

            val = self.validate(
                epoch
            )

            elapsed = (
                time.time()
                - start_time
            )

            # =================================================
            # EPOCH REPORT
            # =================================================

            print()
            print("=" * 78)

            print(
                f"Epoch "
                f"{epoch}/{self.num_epochs}"
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
                "                  Pred Real   Pred Fake"
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

            # =================================================
            # BEST CHECKPOINT
            # =================================================

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
                        val[
                            "balanced_accuracy"
                        ]
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

            # =================================================
            # ALWAYS SAVE LAST
            # =================================================

            self.save_checkpoint(
                epoch=epoch,
                balanced_accuracy=(
                    val[
                        "balanced_accuracy"
                    ]
                ),
                val_accuracy=(
                    val["accuracy"]
                ),
                tag="last",
                stop_reason=None,
            )

            # =================================================
            # EARLY STOPPING
            # =================================================

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
                    f"for "
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

                best_path = os.path.join(
                    self.checkpoint_dir,
                    "best.pt",
                )

                print(
                    f"Best checkpoint        : "
                    f"{best_path}"
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

        best_path = os.path.join(
            self.checkpoint_dir,
            "best.pt",
        )

        print(
            f"Best checkpoint        : "
            f"{best_path}"
        )

        print("=" * 78)

    # =====================================================
    # CHECKPOINT
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

            "epoch": epoch,

            "model_state":
                self.model.state_dict(),

            "optimizer_state":
                self.optimizer.state_dict(),

            "scaler_state":
                self.scaler.state_dict(),

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
    # RESUME TRAINING
    # =====================================================

    def load_checkpoint(
        self,
        checkpoint_path,
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

        scaler_state = checkpoint.get(
            "scaler_state"
        )

        if scaler_state:
            self.scaler.load_state_dict(
                scaler_state
            )

        self.best_balanced_accuracy = (
            checkpoint.get(
                "best_balanced_accuracy",
                -1.0,
            )
        )

        self.best_epoch = (
            checkpoint.get(
                "best_epoch",
                checkpoint.get(
                    "epoch",
                    0,
                ),
            )
        )

        self.epochs_without_improvement = 0

        loaded_epoch = checkpoint.get(
            "epoch",
            0,
        )

        print()
        print("=" * 78)

        print(
            f"Loaded checkpoint from epoch "
            f"{loaded_epoch}"
        )

        print(
            f"Best balanced accuracy : "
            f"{self.best_balanced_accuracy:.4f}"
        )

        print("=" * 78)

        return loaded_epoch


# =========================================================
# ENTRY POINT
# =========================================================

def main():

    trainer = AppearanceTrainer(

        train_csv="data/split/train.csv",

        val_csv="data/split/val.csv",

        dataset_root="data/raw",

        checkpoint_dir=(
            "checkpoints/appearance_full"
        ),

        # RTX 4090
        batch_size=16,

        # Effective batch size:
        # 16 × 1 = 16
        accumulation_steps=1,

        # Pretrained AIMv2:
        # smaller learning rate
        backbone_lr=1e-5,

        # New/task-specific layers:
        # larger learning rate
        head_lr=5e-5,

        weight_decay=1e-4,

        num_epochs=20,

        # 0 = Real
        # 1 = Fake
        num_classes=2,

        # EDL KL annealing
        annealing_step=4,

        # Fine-tune final 4 AIMv2 blocks
        finetune_last_n_blocks=4,

        # RTX 4090 / current environment
        num_workers=8,

        early_stopping_patience=6,

        # Reporting only.
        # NOT a third learned class.
        uncertainty_threshold=0.50,

        # Stability during fine-tuning.
        gradient_clip_norm=1.0,

        train_cache="data/cache/train",
        val_cache="data/cache/val",
        samples_per_epoch=None,
        val_subset=None,
    )

    trainer.fit()


if __name__ == "__main__":
    main()

