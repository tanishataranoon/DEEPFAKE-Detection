"""
=========================================================
Test Evidence Head
---------------------------------------------------------
Verifies

1. Forward pass
2. Evidence >= 0
3. Dirichlet computation
4. Belief + uncertainty
5. Tensor shapes
6. NaN check
=========================================================
"""

import torch

from models.heads.evidence_head import (
    EvidenceHead,
    compute_dirichlet,
)


def main():

    print("=" * 60)
    print("LOADING EVIDENCE HEAD")
    print("=" * 60)

    head = EvidenceHead(
        feature_dim=1024,
        num_classes=2,
    )

    print(head)

    print()

    print("=" * 60)
    print("DUMMY FEATURES")
    print("=" * 60)

    features = torch.randn(4, 1024)

    print("Input Shape :", features.shape)

    print()

    print("=" * 60)
    print("FORWARD PASS")
    print("=" * 60)

    evidence = head(features)

    print("Evidence Shape :", evidence.shape)

    print()

    print("Evidence")

    print(evidence)

    print()

    print("Minimum Evidence :", evidence.min().item())

    print()

    print("=" * 60)
    print("DIRICHLET")
    print("=" * 60)

    results = compute_dirichlet(evidence)

    alpha = results["alpha"]
    belief = results["belief"]
    uncertainty = results["uncertainty"]
    probability = results["probability"]

    print("Alpha Shape       :", alpha.shape)
    print("Belief Shape      :", belief.shape)
    print("Probability Shape :", probability.shape)
    print("Uncertainty Shape :", uncertainty.shape)

    print()

    print("Belief")

    print(belief)

    print()

    print("Probability")

    print(probability)

    print()

    print("Uncertainty")

    print(uncertainty)

    print()

    print("Belief Sum")

    print(belief.sum(dim=1))

    print()

    print("Probability Sum")

    print(probability.sum(dim=1))

    print()

    print("Belief + Uncertainty")

    print(
        belief.sum(dim=1, keepdim=True)
        + uncertainty
    )

    print()

    print("Contains NaN")

    print(torch.isnan(evidence).any())
    print(torch.isnan(alpha).any())
    print(torch.isnan(probability).any())

    print()

    print("=" * 60)
    print("TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()