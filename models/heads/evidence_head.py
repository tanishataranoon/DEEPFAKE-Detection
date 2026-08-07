"""
=========================================================
Evidence Head
---------------------------------------------------------
Maps a pooled feature vector into non-negative evidence
for each class using Evidential Deep Learning (EDL).

This class is used by both:

• Appearance Branch
• Temporal Branch

Each branch owns its own EvidenceHead instance with
independent learnable weights.

Unlike a conventional classifier that predicts class
probabilities with Softmax, this module predicts
non-negative evidence.

Evidence is converted into a Dirichlet distribution:

    alpha = evidence + 1

From the Dirichlet parameters we obtain

    • belief mass
    • expected probability
    • uncertainty

These outputs are later consumed by the
Dempster–Shafer Fusion module.

Reference:
Sensoy et al.
"Evidential Deep Learning"
NeurIPS 2018
=========================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidenceHead(nn.Module):
    """
    Shared Evidence Head for Appearance and Temporal models.

    Input
    -----
    features : Tensor (B, feature_dim)

        Pooled feature vectors produced by the backbone.

        Appearance branch:
            (B, 1024)

        Temporal branch:
            (B, feature_dim)

    Output
    ------
    evidence : Tensor (B, num_classes)

        Non-negative evidence for every class.

        Example (Binary Deepfake Detection):

            (B, 2)

        evidence[:,0] -> Real evidence
        evidence[:,1] -> Fake evidence
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int = 2,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.num_classes = num_classes

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features):
        """
        Parameters
        ----------
        features : Tensor
            Shape:
                (B, feature_dim)

        Returns
        -------
        evidence : Tensor
            Shape:
                (B, num_classes)

            Every element is >= 0.
        """

        logits = self.classifier(features)

        # Softplus guarantees positive evidence while
        # maintaining smooth gradients.
        evidence = F.softplus(logits)

        return evidence


def compute_dirichlet(evidence):
    """
    Convert evidence into Dirichlet statistics.

    Parameters
    ----------
    evidence : Tensor
        Shape:
            (B, K)

        Non-negative evidence predicted by the
        Evidence Head.

    Returns
    -------
    dict containing

    alpha : Tensor
        (B, K)

        Dirichlet parameters.

            alpha = evidence + 1

    S : Tensor
        (B, 1)

        Dirichlet strength.

            S = sum(alpha)

    belief : Tensor
        (B, K)

        Belief mass assigned to each class.

            belief = evidence / S

    uncertainty : Tensor
        (B, 1)

        Remaining uncertainty mass.

            uncertainty = K / S

    probability : Tensor
        (B, K)

        Expected categorical probability.

            probability = alpha / S
    """

    num_classes = evidence.size(-1)

    alpha = evidence + 1

    S = alpha.sum(dim=-1, keepdim=True)

    belief = evidence / S

    uncertainty = num_classes / S

    probability = alpha / S

    return {
        "alpha": alpha,
        "S": S,
        "belief": belief,
        "uncertainty": uncertainty,
        "probability": probability,
    }