"""
=========================================================
Appearance Classifier
---------------------------------------------------------
Complete Appearance Branch for deepfake detection.

Pipeline

Face Image
      │
      ▼
Appearance Model (AIMv2)
      │
      ▼
1024-D Feature Vector
      │
      ▼
Evidence Head
      │
      ▼
Evidence (Real, Fake)

This module combines the AppearanceModel and the
EvidenceHead into a single network that can be trained
or evaluated as one model.

The trainer interacts only with this class instead of
managing the backbone and evidence head separately.

Output

(B,2) non-negative evidence

The evidence can later be converted into Dirichlet
parameters using:

    compute_dirichlet()

Reference
---------
Sensoy et al.
"Evidential Deep Learning"
NeurIPS 2018.
=========================================================
"""

import torch
import torch.nn as nn

from models.appearence.model import AppearanceModel
from models.heads.evidence_head import EvidenceHead


class AppearanceClassifier(nn.Module):
    """
    Appearance Branch for Evidential Deep Learning.

    Components
    ----------
    AppearanceModel
        AIMv2 backbone with attention pooling.

    EvidenceHead
        Maps pooled features to non-negative evidence.

    Input
    -----
    images

        Shape

            (B,3,224,224)

    Output
    ------
    evidence

        Shape

            (B,num_classes)

    attention_weights

        Attention weights produced by the
        AppearanceModel.
    """

    def __init__(
        self,
        num_classes=2,
        hidden_dim=256,
        dropout=0.3,
        freeze_backbone=True,
    ):
        super().__init__()

        # ---------------------------------------------
        # Appearance Feature Extractor
        # ---------------------------------------------

        self.backbone = AppearanceModel(
            freeze_backbone=freeze_backbone
        )

        # ---------------------------------------------
        # Evidential Classifier
        # ---------------------------------------------

        self.head = EvidenceHead(
            feature_dim=self.backbone.feature_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, clip):
        """
        Forward pass.

        Parameters
        ----------
        clip : Tensor

            Shape

                (B,3,224,224)

        Returns
        -------
        evidence : Tensor

            Shape

                (B,num_classes)

        attention_weights : Tensor

            Attention weights from the
            AppearanceModel.
        """

        outputs = self.backbone(clip)

        evidence = self.head(
            outputs["features"]
        )

        return {

            "evidence": evidence,

            "features": outputs["features"],

            "attention": outputs["attention"],

        }

    def freeze_backbone(self):
        """
        Freeze all backbone parameters.
        """

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self):
        """
        Unfreeze all backbone parameters.
        """

        for parameter in self.backbone.parameters():
            parameter.requires_grad = True

    @property
    def feature_dim(self):
        """
        Output feature dimension produced by the
        AppearanceModel.
        """

        return self.backbone.feature_dim