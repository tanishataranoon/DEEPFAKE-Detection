"""
=========================================================
Temporal Classifier
---------------------------------------------------------
Complete Temporal Branch. Mirrors AppearanceClassifier
exactly, so both branches' evidence is directly
comparable/fusable and the same EDL loss / Dirichlet code
works unmodified for both.
=========================================================
"""

import torch.nn as nn

from models.temporal.model import TemporalModel
from models.heads.evidence_head import EvidenceHead


class TemporalClassifier(nn.Module):

    def __init__(
        self,
        num_classes=2,
        hidden_dim=256,
        dropout=0.3,
        freeze_backbone=True,
        gradient_checkpointing=False,
    ):
        super().__init__()

        self.backbone = TemporalModel(
            freeze_backbone=freeze_backbone,
            gradient_checkpointing=gradient_checkpointing,
        )

        self.head = EvidenceHead(
            feature_dim=self.backbone.feature_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, clip):
        outputs = self.backbone(clip)
        evidence = self.head(outputs["features"])

        return {
            "evidence": evidence,
            "features": outputs["features"],
        }

    def freeze_backbone(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self, last_n_blocks=None):
        self.backbone.unfreeze_backbone(last_n_blocks=last_n_blocks)

    @property
    def feature_dim(self):
        return self.backbone.feature_dim