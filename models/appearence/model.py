"""
=========================================================
Appearance Model
---------------------------------------------------------
Appearance Branch for Deepfake Detection

Pipeline
--------
16-frame clip
        │
        ▼
AIMv2 Backbone
(per-frame feature extraction)
        │
        ▼
Frame Features
(B,T,D)
        │
        ▼
Attention Pooling
        │
        ▼
Clip Feature
(B,D)
        │
        ▼
Evidence Head (next module)

Stage 1
--------
Freeze AIMv2.
Train only:
    • Attention Pooling
    • Evidence Head

Stage 2
--------
Unfreeze part (or all) of AIMv2 for fine-tuning.
=========================================================
"""

import torch
import torch.nn as nn

from transformers import AutoModel


# =========================================================
# Attention Pooling
# =========================================================

class AttentionPooling(nn.Module):
    """
    Learns which frames inside a clip contain the most useful
    appearance information.

    Input
    -----
    x
        Shape:
            (B, T, D)

        B = batch size

        T = number of frames
            (normally 16)

        D = AIMv2 feature dimension
            (1024 for aimv2-large)

    Output
    ------
    pooled
        Shape:
            (B, D)

    attention
        Shape:
            (B, T)

        Attention weights sum to 1 across all frames.
    """

    def __init__(
        self,
        feature_dim,
        hidden_dim=256,
    ):
        super().__init__()

        self.attention = nn.Sequential(

            nn.Linear(
                feature_dim,
                hidden_dim,
            ),

            nn.Tanh(),

            nn.Linear(
                hidden_dim,
                1,
            ),
        )

    def forward(self, x):

        # -----------------------------------------
        # x
        #
        # (B,T,D)
        # -----------------------------------------

        scores = self.attention(x)

        # (B,T,1)

        weights = torch.softmax(
            scores,
            dim=1,
        )

        pooled = (x * weights).sum(dim=1)

        # remove final singleton dimension

        weights = weights.squeeze(-1)

        return pooled, weights


# =========================================================
# AIMv2 Backbone
# =========================================================

class AppearanceBackbone(nn.Module):
    """
    Wraps Apple's pretrained AIMv2 model.

    Receives ONE frame at a time.

    Input

        (B,C,H,W)

    Output

        (B,D)

    where

        D = hidden_size
    """

    def __init__(
        self,
        model_name="apple/aimv2-large-patch14-224",
        freeze=True,
        dropout=0.20,
    ):
        super().__init__()

        # -----------------------------------------
        # Load pretrained backbone
        # -----------------------------------------

        self.backbone = AutoModel.from_pretrained(

            model_name,

            revision="ac764a25c832c7dc5e11871daa588e98e3cdbfb7",

            trust_remote_code=True,
        )

        # -----------------------------------------
        # Determine feature dimension
        # -----------------------------------------

        if hasattr(
            self.backbone.config,
            "hidden_size",
        ):

            self.feature_dim = (
                self.backbone.config.hidden_size
            )

        elif hasattr(
            self.backbone.config,
            "embed_dim",
        ):

            self.feature_dim = (
                self.backbone.config.embed_dim
            )

        else:

            raise RuntimeError(
                "Cannot determine AIMv2 feature dimension."
            )

        self.freeze = freeze

        self.dropout = nn.Dropout(
            p=dropout,
        )

        # -----------------------------------------
        # Stage 1
        #
        # Freeze pretrained backbone.
        # -----------------------------------------

        if self.freeze:

            for parameter in self.backbone.parameters():

                parameter.requires_grad = False

            self.backbone.eval()

    def forward(
        self,
        images,
    ):
        """
        Parameters
        ----------
        images

            Shape

                (B,C,H,W)

        Returns
        -------

        features

            Shape

                (B,D)
        """

        if self.freeze:

            with torch.no_grad():

                outputs = self.backbone(
                    pixel_values=images,
                )

        else:

            outputs = self.backbone(
                pixel_values=images,
            )

        # -----------------------------------------
        # AIMv2 returns

        # (B, Tokens, Hidden)

        # Example

        # (32,256,1024)

        # Mean pooling over tokens gives one
        # descriptor per image.

        # -----------------------------------------

        hidden = outputs.last_hidden_state

        features = hidden.mean(dim=1)

        features = self.dropout(features)

        return features

    def train(
        self,
        mode=True,
    ):
        """
        Override nn.Module.train()

        During Stage 1 the backbone always stays
        in evaluation mode even if the parent model
        switches to training.

        This prevents BatchNorm / Dropout behaviour
        from changing inside the frozen backbone.
        """

        super().train(mode)

        if self.freeze:

            self.backbone.eval()

        return self
# =========================================================
# Complete Appearance Model
# =========================================================

class AppearanceModel(nn.Module):
    """
    Complete Appearance Branch.

    Pipeline

        Input Clip
            (B,T,C,H,W)

                │

                ▼

        AIMv2
        (frame features)

                │

                ▼

        Attention Pooling

                │

                ▼

        Clip Feature
            (B,D)

    Output

        Dictionary

        {
            "features": (B,D),
            "attention": (B,T)
        }
    """

    def __init__(
        self,
        model_name="apple/aimv2-large-patch14-224",
        freeze_backbone=True,
    ):
        super().__init__()

        self.backbone = AppearanceBackbone(
            model_name=model_name,
            freeze=freeze_backbone,
        )

        self.pooling = AttentionPooling(
            feature_dim=self.backbone.feature_dim,
        )

    def forward(
        self,
        clip,
    ):
        """
        Parameters
        ----------
        clip

            Shape

                (B,T,C,H,W)

        Returns
        -------

        dict

            features
                (B,D)

            attention
                (B,T)
        """

        B, T, C, H, W = clip.shape

        # -----------------------------------------
        # Merge batch and temporal dimension
        #
        # (B,T,C,H,W)
        #
        # ->
        #
        # (B*T,C,H,W)
        # -----------------------------------------

        frames = clip.reshape(
            B * T,
            C,
            H,
            W,
        )

        # -----------------------------------------
        # Extract frame features
        # -----------------------------------------

        frame_features = self.backbone(frames)

        # -----------------------------------------
        # Restore temporal dimension
        #
        # (B*T,D)
        #
        # ->
        #
        # (B,T,D)
        # -----------------------------------------

        frame_features = frame_features.reshape(
            B,
            T,
            -1,
        )

        # -----------------------------------------
        # Learn which frames are most important
        # -----------------------------------------

        pooled_features, attention = self.pooling(
            frame_features,
        )

        return {

            "features": pooled_features,

            "attention": attention,

        }

    # =====================================================
    # Stage 2 Fine-tuning
    # =====================================================

    def unfreeze_backbone(
        self,
        last_n_blocks=None,
    ):
        """
        Enables fine-tuning.

        last_n_blocks=None

            Unfreeze the entire backbone.

        last_n_blocks=N

            Only unfreeze the last N transformer
            blocks.

        Useful when GPU memory is limited.
        """

        self.backbone.freeze = False

        self.backbone.backbone.train()

        # -----------------------------------------
        # Unfreeze everything
        # -----------------------------------------

        if last_n_blocks is None:

            for parameter in self.backbone.backbone.parameters():

                parameter.requires_grad = True

            return

        # -----------------------------------------
        # Freeze everything first
        # -----------------------------------------

        for parameter in self.backbone.backbone.parameters():

            parameter.requires_grad = False

        # -----------------------------------------
        # Locate transformer blocks
        # -----------------------------------------

        if hasattr(
            self.backbone.backbone,
            "trunk",
        ):

            blocks = self.backbone.backbone.trunk.blocks

        elif hasattr(
            self.backbone.backbone,
            "encoder",
        ):

            blocks = self.backbone.backbone.encoder.layers

        else:

            raise RuntimeError(
                "Unable to locate transformer blocks."
            )

        # -----------------------------------------
        # Unfreeze last N blocks
        # -----------------------------------------

        for block in blocks[-last_n_blocks:]:

            for parameter in block.parameters():

                parameter.requires_grad = True
    @property
    def feature_dim(self):
        """
        Dimension of the pooled appearance feature vector.
        """

        return self.backbone.feature_dim

# =========================================================
# Stand-alone Test
# =========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Appearance Model")
    print("=" * 60)

    model = AppearanceModel()

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Feature Dimension : {model.backbone.feature_dim}")
    print(f"Total Parameters  : {total_params:,}")
    print(f"Trainable         : {trainable_params:,}")
    print(f"Frozen            : {total_params-trainable_params:,}")

    print()

    dummy = torch.randn(
        2,
        16,
        3,
        224,
        224,
    )

    print("Dummy Input")

    print(dummy.shape)

    print()

    model.eval()

    with torch.no_grad():

        output = model(dummy)

    features = output["features"]

    attention = output["attention"]

    print("Output")

    print("Feature Shape :", features.shape)

    print("Attention Shape :", attention.shape)

    print()

    print("Attention Sum")

    print(attention.sum(dim=1))

    print()

    print("Contains NaN")

    print("Features :", torch.isnan(features).any())

    print("Attention:", torch.isnan(attention).any())

    print()

    print("First Sample Attention")

    print(attention[0])

    print()

    print("=" * 60)
    print("MODEL TEST PASSED")
    print("=" * 60)