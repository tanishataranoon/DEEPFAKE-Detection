"""
Dempster-Shafer Evidence Fusion (3-way decision)
---------------------------------------------------------
Fuses the EDL/Dirichlet outputs of the Appearance and Temporal
branches with Dempster's rule of combination.

Final decision is one of:

    REAL       (prediction_id = 0)
    FAKE       (prediction_id = 1)
    UNCERTAIN  (prediction_id = 2)

A sample is UNCERTAIN if EITHER:
    1. fused uncertainty >= uncertainty_threshold
       (the branches together still "don't know"), OR
    2. conflict >= conflict_threshold
       (the branches strongly disagree; Dempster's rule would
       otherwise hide this by renormalizing and lowering the
       uncertainty).

Otherwise the decision is argmax of the final probability.

Probability: probability_i = belief_i + uncertainty / K

Reference:
    Han et al., "Trusted Multi-View Classification", ICLR 2021.
---------------------------------------------------------
"""

import torch

PREDICTION_NAMES = {0: "REAL", 1: "FAKE", 2: "UNCERTAIN"}


def compute_ds_combination(
    dirichlet_appearance,
    dirichlet_temporal,
    uncertainty_threshold=0.40,
    conflict_threshold=0.50,
):
    """
    Parameters
    ----------
    dirichlet_appearance, dirichlet_temporal : dict
        Required keys: "belief" (B, K), "uncertainty" (B, 1)
        Optional keys: "evidence", "alpha", "S"

    uncertainty_threshold : float
        fused uncertainty >= this  -> UNCERTAIN
    conflict_threshold : float
        conflict >= this           -> UNCERTAIN
        (set to None or > 1.0 to disable the conflict rule)

    Both thresholds should be chosen on the VALIDATION set,
    not the test set.

    Returns
    -------
    dict with: evidence, alpha, S, belief, uncertainty, probability,
    dirichlet_probability, conflict, prediction, prediction_id,
    uncertain_reason, uncertainty_threshold, conflict_threshold
    """

    # ---------------- 1. Read branch BBAs ----------------
    belief_a = dirichlet_appearance["belief"]
    uncertainty_a = dirichlet_appearance["uncertainty"]
    belief_t = dirichlet_temporal["belief"]
    uncertainty_t = dirichlet_temporal["uncertainty"]

    if belief_a.ndim != 2 or belief_t.ndim != 2:
        raise ValueError("Belief tensors must have shape (B, K).")
    if uncertainty_a.ndim != 2 or uncertainty_t.ndim != 2:
        raise ValueError("Uncertainty tensors must have shape (B, 1).")
    if belief_a.shape != belief_t.shape:
        raise ValueError("Belief tensors must have the same shape.")
    if uncertainty_a.shape != uncertainty_t.shape:
        raise ValueError("Uncertainty tensors must have the same shape.")
    if uncertainty_a.size(-1) != 1:
        raise ValueError("Uncertainty tensor must have shape (B, 1).")

    num_classes = belief_a.size(-1)
    if num_classes != 2:
        raise ValueError(
            f"Expected exactly 2 classes (Real/Fake), got K={num_classes}."
        )

    # ---------------- 2. Numerical safety ----------------
    eps = 1e-8
    belief_a = torch.clamp(belief_a, min=0.0)
    belief_t = torch.clamp(belief_t, min=0.0)
    uncertainty_a = torch.clamp(uncertainty_a, min=0.0, max=1.0)
    uncertainty_t = torch.clamp(uncertainty_t, min=0.0, max=1.0)

    # ---------------- 3. Normalize each BBA ----------------
    # belief_real + belief_fake + uncertainty = 1
    total_a = belief_a.sum(dim=-1, keepdim=True) + uncertainty_a
    total_t = belief_t.sum(dim=-1, keepdim=True) + uncertainty_t

    belief_a = belief_a / torch.clamp(total_a, min=eps)
    uncertainty_a = uncertainty_a / torch.clamp(total_a, min=eps)
    belief_t = belief_t / torch.clamp(total_t, min=eps)
    uncertainty_t = uncertainty_t / torch.clamp(total_t, min=eps)

    # ---------------- 4. Conflict ----------------
    # C = sum_{i != j} bA_i * bT_j
    agreement = (belief_a * belief_t).sum(dim=-1, keepdim=True)
    conflict = (
        belief_a.sum(dim=-1, keepdim=True) * belief_t.sum(dim=-1, keepdim=True)
        - agreement
    )
    conflict = torch.clamp(conflict, min=0.0, max=1.0)

    # ---------------- 5. Dempster normalization ----------------
    normalizer = torch.clamp(1.0 - conflict, min=eps)

    # ---------------- 6. Fused belief ----------------
    fused_belief = (
        belief_a * belief_t
        + belief_a * uncertainty_t
        + belief_t * uncertainty_a
    ) / normalizer

    # ---------------- 7. Fused uncertainty ----------------
    fused_uncertainty = (uncertainty_a * uncertainty_t) / normalizer

    # ---------------- 8. Final BBA normalization ----------------
    fused_total = fused_belief.sum(dim=-1, keepdim=True) + fused_uncertainty
    fused_belief = fused_belief / torch.clamp(fused_total, min=eps)
    fused_uncertainty = fused_uncertainty / torch.clamp(fused_total, min=eps)

    # ---------------- 9. Probability ----------------
    probability = fused_belief + fused_uncertainty / num_classes
    probability = probability / torch.clamp(
        probability.sum(dim=-1, keepdim=True), min=eps
    )

    # ---------------- 10. Fused evidence (for logging) ----------------
    safe_uncertainty = torch.clamp(fused_uncertainty, min=eps, max=1.0)
    fused_S = num_classes / safe_uncertainty
    fused_evidence = fused_belief * fused_S
    fused_alpha = fused_evidence + 1.0
    fused_S = fused_alpha.sum(dim=-1, keepdim=True)
    dirichlet_probability = fused_alpha / torch.clamp(fused_S, min=eps)

    # ---------------- 11. Decision: REAL / FAKE / UNCERTAIN ----------------
    prediction_id = torch.argmax(probability, dim=-1)

    high_uncertainty = fused_uncertainty.squeeze(-1) >= uncertainty_threshold

    if conflict_threshold is None:
        high_conflict = torch.zeros_like(high_uncertainty)
    else:
        high_conflict = conflict.squeeze(-1) >= conflict_threshold

    uncertain_mask = high_uncertainty | high_conflict

    prediction_id = torch.where(
        uncertain_mask,
        torch.full_like(prediction_id, 2),
        prediction_id,
    )

    # Why was a sample marked UNCERTAIN? (for debugging / analysis)
    uncertain_reason = []
    for hu, hc in zip(
        high_uncertainty.detach().cpu().tolist(),
        high_conflict.detach().cpu().tolist(),
    ):
        if hu and hc:
            uncertain_reason.append("uncertainty+conflict")
        elif hu:
            uncertain_reason.append("uncertainty")
        elif hc:
            uncertain_reason.append("conflict")
        else:
            uncertain_reason.append("")

    prediction = [
        PREDICTION_NAMES[p] for p in prediction_id.detach().cpu().tolist()
    ]

    # ---------------- 12. Return ----------------
    return {
        "appearance_evidence": dirichlet_appearance.get("evidence", None),
        "temporal_evidence": dirichlet_temporal.get("evidence", None),
        "evidence": fused_evidence,
        "alpha": fused_alpha,
        "S": fused_S,
        "belief": fused_belief,
        "uncertainty": fused_uncertainty,
        "probability": probability,
        "dirichlet_probability": dirichlet_probability,
        "conflict": conflict,
        "prediction": prediction,
        "prediction_id": prediction_id,
        "uncertain_reason": uncertain_reason,
        "uncertainty_threshold": uncertainty_threshold,
        "conflict_threshold": conflict_threshold,
    }


def format_ds_result(fused_output, index=0):
    """Format one fused sample for human-readable output."""
    evidence = fused_output["evidence"][index]
    probability = fused_output["probability"][index]
    uncertainty = fused_output["uncertainty"][index, 0].item()
    conflict = fused_output["conflict"][index, 0].item()
    prediction = fused_output["prediction"][index]
    reason = fused_output["uncertain_reason"][index]

    detection = prediction + (f"  (reason: {reason})" if reason else "")

    return (
        "\n"
        "==================================================\n"
        "Dempster-Shafer Fusion Result\n"
        "==================================================\n"
        "\n"
        "Evidence:\n"
        f"    Real = {evidence[0].item():.4f}\n"
        f"    Fake = {evidence[1].item():.4f}\n"
        "\n"
        "Output:\n"
        f"    Real probability = {probability[0].item():.4f}\n"
        f"    Fake probability = {probability[1].item():.4f}\n"
        f"    Uncertainty      = {uncertainty:.4f}\n"
        "\n"
        f"Conflict             = {conflict:.4f}\n"
        "\n"
        f"Detection : {detection}\n"
        "==================================================\n"
    )