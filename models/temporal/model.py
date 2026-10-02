"""
=========================================================
Temporal Model
---------------------------------------------------------
Temporal Branch backbone for deepfake detection.

Pipeline
--------
16-frame clip (B,T,C,H,W)
        │
        ▼
VideoMAE Backbone
(processes the WHOLE clip's spatiotemporal patches at once,
 unlike AIMv2 which sees one frame at a time)
        │
        ▼
Patch Token Features (B,N,D)
        │
        ▼
Mean Pooling over tokens
        │
        ▼
Clip Feature (B,D)
        │
        ▼
Evidence Head (next module)

Stage 1: freeze VideoMAE, train only the Evidence Head.
Stage 2: unfreeze part/all of VideoMAE for fine-tuning.
=========================================================
"""

import torch
import torch.nn as nn

from transformers import VideoMAEModel


class TemporalBackbone(nn.Module):
    """
    Wraps Hugging Face's pretrained VideoMAE model.
    Receives the WHOLE 16-frame clip at once.

    Input  : (B,T,C,H,W)   e.g. (B,16,3,224,224)
    Output : (B,D)          D = 768 for videomae-base
    """

    def __init__(
        self,
        model_name="MCG-NJU/videomae-base",
        freeze=True,
        dropout=0.20,
        gradient_checkpointing=False,
    ):
        super().__init__()

        self.backbone = VideoMAEModel.from_pretrained(model_name)
        self.feature_dim = self.backbone.config.hidden_size

        self.freeze = freeze
        self.dropout = nn.Dropout(p=dropout)

        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

        if self.freeze:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            self.backbone.eval()

    def forward(self, clip):
        if self.freeze:
            with torch.no_grad():
                outputs = self.backbone(pixel_values=clip)
        else:
            outputs = self.backbone(pixel_values=clip)

        hidden = outputs.last_hidden_state          # (B, num_patches, D)
        features = hidden.mean(dim=1)                 # (B, D)
        features = self.dropout(features)
        return features

    def train(self, mode=True):
        """Keep the frozen backbone in eval() mode even if the parent
        model switches to train(), same reasoning as AppearanceBackbone."""
        super().train(mode)
        if self.freeze:
            self.backbone.eval()
        return self


class TemporalModel(nn.Module):
    """
    Complete Temporal Branch (backbone only, no evidence head).
    Output: {"features": (B,D)}
    """

    def __init__(
        self,
        model_name="MCG-NJU/videomae-base",
        freeze_backbone=True,
        gradient_checkpointing=False,
    ):
        super().__init__()

        self.backbone = TemporalBackbone(
            model_name=model_name,
            freeze=freeze_backbone,
            gradient_checkpointing=gradient_checkpointing,
        )

    def forward(self, clip):
        features = self.backbone(clip)
        return {"features": features}

    def unfreeze_backbone(self, last_n_blocks=None):
        """
        last_n_blocks=None -> unfreeze the entire backbone.
        last_n_blocks=N    -> only unfreeze the last N encoder layers.
        """
        self.backbone.freeze = False
        self.backbone.backbone.train()

        if last_n_blocks is None:
            for parameter in self.backbone.backbone.parameters():
                parameter.requires_grad = True
            return

        for parameter in self.backbone.backbone.parameters():
            parameter.requires_grad = False

        layers = self.backbone.backbone.encoder.layer
        for layer in layers[-last_n_blocks:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True

    @property
    def feature_dim(self):
        return self.backbone.feature_dim


if __name__ == "__main__":
    print("=" * 60)
    print("Temporal Model")
    print("=" * 60)

    model = TemporalModel(freeze_backbone=True)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Feature Dimension : {model.feature_dim}")
    print(f"Total Parameters  : {total_params:,}")
    print(f"Trainable         : {trainable_params:,}")
    print(f"Frozen            : {total_params - trainable_params:,}")

    dummy = torch.randn(1, 16, 3, 224, 224)
    print("\nDummy Input:", dummy.shape)

    model.eval()
    with torch.no_grad():
        output = model(dummy)

    features = output["features"]
    print("\nFeature Shape :", features.shape)
    print("Contains NaN  :", torch.isnan(features).any())
    print("\n" + "=" * 60)
    print("MODEL TEST PASSED")
    print("=" * 60)