import torch
import torch.nn as nn
from transformers import Aimv2VisionModel

class AttentionPooling(nn.Module):
    """Attention pooling over the T frames of a clip."""

    def __init__(self, feature_dim, hidden_dim=256):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        # x: (B,T,D)
        scores = self.attention(x)                 # (B,T,1)
        weights = torch.softmax(scores, dim=1)     # (B,T,1)
        pooled = (x * weights).sum(dim=1)          # (B,D)
        return pooled, weights.squeeze(-1)         # (B,T)


class AppearanceBackbone(nn.Module):
    """AIMv2 feature extractor for individual frames."""

    def __init__(
        self,
        model_name="apple/aimv2-large-patch14-224",
        freeze=True,
        dropout=0.20,
    ):
        super().__init__()

        self.backbone = Aimv2VisionModel.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        if hasattr(self.backbone.config, "hidden_size"):
            self.feature_dim = self.backbone.config.hidden_size
        elif hasattr(self.backbone.config, "embed_dim"):
            self.feature_dim = self.backbone.config.embed_dim
        else:
            raise RuntimeError("Cannot determine AIMv2 feature dimension.")

        self.freeze = bool(freeze)
        self.dropout = nn.Dropout(dropout)

        if self.freeze:
            self.freeze_all()

    def freeze_all(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.freeze = True
        self.backbone.eval()

    def unfreeze_all(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
        self.freeze = False
        self.backbone.train()

    def forward(self, images):
        # When fully frozen, avoid building a graph.
        if self.freeze:
            with torch.no_grad():
                outputs = self.backbone(pixel_values=images)
        else:
            outputs = self.backbone(pixel_values=images)

        features = outputs.last_hidden_state.mean(dim=1)
        return self.dropout(features)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.backbone.eval()
        return self


class AppearanceModel(nn.Module):
    """
    16-frame appearance model:
        frames -> AIMv2 -> attention pooling -> clip feature
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
            feature_dim=self.backbone.feature_dim
        )

    def forward(self, clip):
        # clip: (B,T,C,H,W)
        B, T, C, H, W = clip.shape
        frames = clip.reshape(B * T, C, H, W)

        frame_features = self.backbone(frames)
        frame_features = frame_features.reshape(B, T, -1)

        pooled, attention = self.pooling(frame_features)

        return {
            "features": pooled,
            "attention": attention,
        }

    def _get_blocks(self):
        model = self.backbone.backbone

        if hasattr(model, "trunk") and hasattr(model.trunk, "blocks"):
            return model.trunk.blocks

        if hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
            return model.encoder.layers

        raise RuntimeError(
            "Unable to locate AIMv2 transformer blocks. "
            "Expected model.trunk.blocks or model.encoder.layers."
        )

    def unfreeze_backbone(self, last_n_blocks=None):
        """
        last_n_blocks=None -> all AIMv2 parameters trainable.
        last_n_blocks=N -> only final N transformer blocks trainable.
        """
        if last_n_blocks is not None:
            if not isinstance(last_n_blocks, int) or last_n_blocks <= 0:
                raise ValueError("last_n_blocks must be a positive integer or None.")

            blocks = self._get_blocks()
            if last_n_blocks > len(blocks):
                raise ValueError(
                    f"Requested {last_n_blocks} blocks, but model has {len(blocks)}."
                )

            # Freeze every AIMv2 parameter first.
            for p in self.backbone.backbone.parameters():
                p.requires_grad = False

            # Train only the final N transformer blocks.
            for block in blocks[-last_n_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True

            self.backbone.freeze = False
            self.backbone.backbone.train()
            return

        self.backbone.unfreeze_all()

    def get_backbone_trainable_status(self):
        total = sum(p.numel() for p in self.backbone.backbone.parameters())
        trainable = sum(
            p.numel()
            for p in self.backbone.backbone.parameters()
            if p.requires_grad
        )
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
        }

    @property
    def feature_dim(self):
        return self.backbone.feature_dim


if __name__ == "__main__":
    model = AppearanceModel(freeze_backbone=True)
    print("Feature dimension:", model.feature_dim)
    print("Initially:", model.get_backbone_trainable_status())

    model.unfreeze_backbone(last_n_blocks=4)
    print("After last-4 fine-tuning:", model.get_backbone_trainable_status())

    x = torch.randn(1, 16, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        y = model(x)

    print("Feature shape:", y["features"].shape)
    print("Attention shape:", y["attention"].shape)
    print("Attention sum:", y["attention"].sum(dim=1))
    print("MODEL TEST PASSED")
