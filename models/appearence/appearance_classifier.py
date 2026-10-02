"""
=========================================================
Appearance Classifier
---------------------------------------------------------
Complete Appearance Branch for deepfake detection.

The network has TWO learned classes:

    0 = Real
    1 = Fake

Uncertainty is NOT a third learned class.

Instead, Evidential Deep Learning converts the two-class
evidence into a Dirichlet distribution and derives:

    Real probability
    Fake probability
    EDL uncertainty

For reporting, a sample can therefore be displayed as:

    REAL
    FAKE
    UNCERTAIN

without training a third "uncertain" class.
=========================================================
"""

import torch
import torch.nn as nn

from models.appearence.model import AppearanceModel
from models.heads.evidence_head import (
    EvidenceHead,
    compute_dirichlet,
)


class AppearanceClassifier(nn.Module):
    """
    Appearance branch:

        AIMv2
          ↓
        Attention Pooling
          ↓
        Evidence Head
          ↓
        Real/Fake evidence
          ↓
        Dirichlet statistics
    """

    def __init__(
        self,
        num_classes=2,
        hidden_dim=256,
        dropout=0.3,
        freeze_backbone=True,
        uncertainty_threshold=0.50,
    ):
        super().__init__()

        if num_classes != 2:
            raise ValueError(
                "This AppearanceClassifier is designed for "
                "binary Real/Fake EDL. Uncertainty is derived "
                "from the Dirichlet distribution, not learned "
                "as a third class."
            )

        self.num_classes = num_classes
        self.uncertainty_threshold = uncertainty_threshold

        self.backbone = AppearanceModel(
            freeze_backbone=freeze_backbone
        )

        self.head = EvidenceHead(
            feature_dim=self.backbone.feature_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, clip):
        """
        Input
        -----
        clip:
            (B,T,C,H,W)

        Output
        ------
        evidence:
            (B,2)

        features:
            (B,D)

        attention:
            (B,T)
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
        Freeze all AIMv2 parameters.
        """

        for parameter in self.backbone.backbone.parameters():
            parameter.requires_grad = False

        self.backbone.backbone.freeze = True

    def unfreeze_backbone(
        self,
        last_n_blocks=None,
    ):
        """
        Fine-tune AIMv2.

        last_n_blocks=None
            Unfreeze all AIMv2 parameters.

        last_n_blocks=4
            Freeze all AIMv2 parameters first, then unfreeze
            only the final four transformer blocks.
        """

        self.backbone.unfreeze_backbone(
            last_n_blocks=last_n_blocks
        )

    @torch.no_grad()
    def edl_output(
        self,
        evidence,
        uncertainty_threshold=None,
    ):
        """
        Convert two-class evidence into:

            probability
            uncertainty
            predicted class
            reported class

        Reported class:

            0 = Real
            1 = Fake
            2 = Uncertain
        """

        threshold = (
            self.uncertainty_threshold
            if uncertainty_threshold is None
            else uncertainty_threshold
        )

        dirichlet = compute_dirichlet(
            evidence
        )

        probability = dirichlet["probability"]

        uncertainty = (
            dirichlet["uncertainty"].view(-1)
        )

        predicted_class = torch.argmax(
            probability,
            dim=1,
        )

        reported_class = predicted_class.clone()

        reported_class[
            uncertainty >= threshold
        ] = 2

        return {
            "alpha": dirichlet["alpha"],
            "S": dirichlet["S"],
            "belief": dirichlet["belief"],
            "probability": probability,
            "uncertainty": uncertainty,
            "predicted_class": predicted_class,
            "reported_class": reported_class,
        }

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        clip,
        uncertainty_threshold=None,
    ):
        """
        Convenience inference method.

        Returns a dictionary containing:

            prediction:
                "Real", "Fake", or "Uncertain"

            probability_real
            probability_fake
            uncertainty
            evidence
        """

        outputs = self.forward(clip)

        edl = self.edl_output(
            outputs["evidence"],
            uncertainty_threshold=uncertainty_threshold,
        )

        reported = edl["reported_class"]

        names = (
            "Real",
            "Fake",
            "Uncertain",
        )

        predictions = [
            names[int(index)]
            for index in reported
        ]

        return {
            "prediction": predictions,
            "probability_real": (
                edl["probability"][:, 0]
            ),
            "probability_fake": (
                edl["probability"][:, 1]
            ),
            "uncertainty": edl["uncertainty"],
            "evidence": outputs["evidence"],
            "attention": outputs["attention"],
            "features": outputs["features"],
        }

    @property
    def feature_dim(self):
        return self.backbone.feature_dim


if __name__ == "__main__":

    print("=" * 70)
    print("APPEARANCE CLASSIFIER SANITY TEST")
    print("=" * 70)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    model = AppearanceClassifier(
        num_classes=2,
        freeze_backbone=True,
        uncertainty_threshold=0.50,
    ).to(device)

    # -----------------------------------------------------
    # Verify partial fine-tuning
    # -----------------------------------------------------

    model.unfreeze_backbone(
        last_n_blocks=4
    )

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Total parameters     : {total:,}"
    )

    print(
        f"Trainable parameters : {trainable:,}"
    )

    print(
        f"Frozen parameters    : "
        f"{total - trainable:,}"
    )

    # -----------------------------------------------------
    # Forward
    # -----------------------------------------------------

    dummy = torch.randn(
        1,
        16,
        3,
        224,
        224,
        device=device,
    )

    model.eval()

    with torch.no_grad():
        output = model(dummy)

    evidence = output["evidence"]

    print()
    print(
        f"Evidence shape       : "
        f"{tuple(evidence.shape)}"
    )

    print(
        f"Feature shape        : "
        f"{tuple(output['features'].shape)}"
    )

    print(
        f"Attention shape      : "
        f"{tuple(output['attention'].shape)}"
    )

    # -----------------------------------------------------
    # EDL output
    # -----------------------------------------------------

    edl = model.edl_output(
        evidence
    )

    print()
    print(
        f"Real probability     : "
        f"{edl['probability'][0, 0].item():.4f}"
    )

    print(
        f"Fake probability     : "
        f"{edl['probability'][0, 1].item():.4f}"
    )

    print(
        f"EDL uncertainty      : "
        f"{edl['uncertainty'][0].item():.4f}"
    )

    reported_names = (
        "Real",
        "Fake",
        "Uncertain",
    )

    print(
        f"Reported result      : "
        f"{reported_names[int(edl['reported_class'][0])]}"
    )

    print()
    print("=" * 70)
    print("CLASSIFIER SANITY TEST PASSED")
    print("=" * 70)
