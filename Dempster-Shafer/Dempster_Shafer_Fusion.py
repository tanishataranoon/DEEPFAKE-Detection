
"""
=========================================================
Dempster-Shafer Evidence Fusion
---------------------------------------------------------
Fuses the independent EDL/Dirichlet outputs of:

    1. Appearance Branch
    2. Temporal Branch

using Dempster's Rule of Combination.

Final decision:

    REAL
    FAKE
    UNCERTAIN

The module keeps the three concepts separate:

    Evidence
        Raw Dirichlet evidence produced by the branches.

    Belief
        Dempster-Shafer belief mass assigned to Real/Fake.

    Uncertainty
        Remaining "don't know" mass.

    Probability
        Final probability used for the Real/Fake decision.

Important:
    Evidence is NOT a probability.
    Evidence values do not have to sum to 1.

For K classes:

    alpha_i = evidence_i + 1

    S = sum(alpha)

    probability_i = alpha_i / S

For the fused Dempster-Shafer BBA, the final probability is
obtained using:

    probability_i = belief_i + uncertainty / K

This distributes the remaining unknown mass equally between
the classes.

Reference:
    Han et al., "Trusted Multi-View Classification", ICLR 2021.
=========================================================
"""

import torch


def compute_ds_combination(
    dirichlet_appearance,
    dirichlet_temporal,
    uncertainty_threshold=0.40,
):
    """
    Combine Appearance and Temporal EDL outputs using
    Dempster-Shafer Theory.

    Parameters
    ----------
    dirichlet_appearance : dict
        Output from compute_dirichlet() for Appearance.

        Required keys:
            "belief"       : (B, K)
            "uncertainty"  : (B, 1)

        Optional:
            "alpha"        : (B, K)
            "S"            : (B, 1)

    dirichlet_temporal : dict
        Output from compute_dirichlet() for Temporal.

        Required keys:
            "belief"       : (B, K)
            "uncertainty"  : (B, 1)

        Optional:
            "alpha"        : (B, K)
            "S"            : (B, 1)

    uncertainty_threshold : float
        If fused uncertainty is >= this value, the final
        detection is "UNCERTAIN".

        IMPORTANT:
        This threshold should ideally be selected using
        the validation set, NOT the test set.

    Returns
    -------
    dict
        Contains:

            appearance_evidence
            temporal_evidence
            fused_evidence

            alpha
            S

            belief
            uncertainty
            probability

            conflict

            prediction
            prediction_id

    Class order:
        0 = Real
        1 = Fake
    """

    # =====================================================
    # 1. Read branch BBAs
    # =====================================================

    belief_a = dirichlet_appearance["belief"]
    uncertainty_a = dirichlet_appearance["uncertainty"]

    belief_t = dirichlet_temporal["belief"]
    uncertainty_t = dirichlet_temporal["uncertainty"]

    # -----------------------------------------------------
    # Basic shape validation
    # -----------------------------------------------------

    if belief_a.ndim != 2 or belief_t.ndim != 2:
        raise ValueError(
            "Appearance and Temporal belief tensors must "
            "have shape (B, K)."
        )

    if uncertainty_a.ndim != 2 or uncertainty_t.ndim != 2:
        raise ValueError(
            "Appearance and Temporal uncertainty tensors "
            "must have shape (B, 1)."
        )

    if belief_a.shape != belief_t.shape:
        raise ValueError(
            "Appearance and Temporal belief tensors must "
            "have the same shape."
        )

    if uncertainty_a.shape != uncertainty_t.shape:
        raise ValueError(
            "Appearance and Temporal uncertainty tensors "
            "must have the same shape."
        )

    if uncertainty_a.size(-1) != 1:
        raise ValueError(
            "Uncertainty tensor must have shape (B, 1)."
        )

    num_classes = belief_a.size(-1)

    if num_classes != 2:
        raise ValueError(
            f"This implementation expects exactly 2 classes "
            f"(Real/Fake), but received K={num_classes}."
        )

    # =====================================================
    # 2. Numerical safety
    # =====================================================

    eps = 1e-8

    belief_a = torch.clamp(belief_a, min=0.0)
    belief_t = torch.clamp(belief_t, min=0.0)

    uncertainty_a = torch.clamp(
        uncertainty_a,
        min=0.0,
        max=1.0,
    )

    uncertainty_t = torch.clamp(
        uncertainty_t,
        min=0.0,
        max=1.0,
    )

    # =====================================================
    # 3. Normalize each branch's BBA
    #
    # A valid BBA must satisfy:
    #
    #   belief_real + belief_fake + uncertainty = 1
    #
    # Small floating-point errors are corrected here.
    # =====================================================

    total_a = belief_a.sum(dim=-1, keepdim=True) + uncertainty_a

    total_t = belief_t.sum(dim=-1, keepdim=True) + uncertainty_t

    belief_a = belief_a / torch.clamp(total_a, min=eps)
    uncertainty_a = uncertainty_a / torch.clamp(total_a, min=eps)

    belief_t = belief_t / torch.clamp(total_t, min=eps)
    uncertainty_t = uncertainty_t / torch.clamp(total_t, min=eps)

    # =====================================================
    # 4. Dempster-Shafer conflict
    #
    # Conflict occurs when:
    #
    # Appearance -> Real
    # Temporal   -> Fake
    #
    # OR
    #
    # Appearance -> Fake
    # Temporal   -> Real
    #
    # K = sum_i != j bA_i * bT_j
    # =====================================================

    agreement = (
        belief_a * belief_t
    ).sum(dim=-1, keepdim=True)

    belief_sum_a = belief_a.sum(
        dim=-1,
        keepdim=True,
    )

    belief_sum_t = belief_t.sum(
        dim=-1,
        keepdim=True,
    )

    conflict = (
        belief_sum_a * belief_sum_t
        - agreement
    )

    conflict = torch.clamp(
        conflict,
        min=0.0,
        max=1.0,
    )

    # =====================================================
    # 5. Dempster normalization
    #
    # 1 - conflict is the non-conflicting mass.
    #
    # If conflict approaches 1, ordinary Dempster
    # combination becomes numerically unstable.
    # =====================================================

    normalizer = torch.clamp(
        1.0 - conflict,
        min=eps,
    )

    # =====================================================
    # 6. Fused belief
    #
    # For each class i:
    #
    # b_i =
    #
    #     [bA_i*bT_i
    #      + bA_i*uT
    #      + bT_i*uA]
    #
    #     / (1-K)
    #
    # This means:
    #
    #   agreement      -> reinforces class
    #   one uncertain  -> other branch contributes
    #   disagreement   -> conflict
    # =====================================================

    fused_belief = (
        belief_a * belief_t
        + belief_a * uncertainty_t
        + belief_t * uncertainty_a
    ) / normalizer

    # =====================================================
    # 7. Fused uncertainty
    #
    # Only mass that was uncertain in BOTH branches remains
    # directly uncertain.
    # =====================================================

    fused_uncertainty = (
        uncertainty_a * uncertainty_t
    ) / normalizer

    # =====================================================
    # 8. Final BBA normalization
    #
    # Protect against tiny floating-point errors.
    # =====================================================

    fused_total = (
        fused_belief.sum(dim=-1, keepdim=True)
        + fused_uncertainty
    )

    fused_belief = fused_belief / torch.clamp(
        fused_total,
        min=eps,
    )

    fused_uncertainty = fused_uncertainty / torch.clamp(
        fused_total,
        min=eps,
    )

    # =====================================================
    # 9. Convert fused BBA into final probabilities
    #
    # The uncertainty is "don't know" mass.
    #
    # We distribute it equally across the two classes:
    #
    #   P(Real) = belief_real + uncertainty / 2
    #   P(Fake) = belief_fake + uncertainty / 2
    #
    # Therefore:
    #
    #   P(Real) + P(Fake) = 1
    #
    # This is the probability used for the class decision.
    # =====================================================

    probability = (
        fused_belief
        + fused_uncertainty / num_classes
    )

    probability = probability / torch.clamp(
        probability.sum(dim=-1, keepdim=True),
        min=eps,
    )

    # =====================================================
    # 10. Fused evidence
    #
    # The original branch evidence is useful for logging.
    #
    # We reconstruct a consistent evidence representation
    # from the fused BBA.
    #
    # For Dirichlet:
    #
    #   S = K / uncertainty
    #
    #   evidence_i = belief_i * S
    #
    #   alpha_i = evidence_i + 1
    #
    # This representation is used for reporting/debugging.
    # =====================================================

    safe_uncertainty = torch.clamp(
        fused_uncertainty,
        min=eps,
        max=1.0,
    )

    fused_S = num_classes / safe_uncertainty

    fused_evidence = fused_belief * fused_S

    fused_alpha = fused_evidence + 1.0

    # -----------------------------------------------------
    # IMPORTANT:
    #
    # For consistency, S is recalculated from alpha.
    #
    # This prevents alpha/S inconsistency caused by the
    # previous implementation.
    # -----------------------------------------------------

    fused_S = fused_alpha.sum(
        dim=-1,
        keepdim=True,
    )

    dirichlet_probability = (
        fused_alpha / torch.clamp(
            fused_S,
            min=eps,
        )
    )

    # =====================================================
    # 11. Detection decision
    #
    # Uncertainty has priority.
    #
    # Example:
    #
    #   Real probability = 0.31
    #   Fake probability = 0.29
    #   Uncertainty      = 0.40
    #
    # If threshold = 0.40:
    #
    #   detection = UNCERTAIN
    #
    # Even though Real has the slightly larger probability.
    # =====================================================

    class_prediction = torch.argmax(
        probability,
        dim=-1,
    )

    prediction_id = class_prediction.clone()

    uncertain_mask = (
        fused_uncertainty.squeeze(-1)
        >= uncertainty_threshold
    )

    prediction_id = torch.where(
        uncertain_mask,
        torch.full_like(prediction_id, 2),
        prediction_id,
    )

    # =====================================================
    # 12. Human-readable prediction
    #
    #   0 = REAL
    #   1 = FAKE
    #   2 = UNCERTAIN
    # =====================================================

    prediction = []

    for pred in prediction_id.detach().cpu().tolist():

        if pred == 0:
            prediction.append("REAL")

        elif pred == 1:
            prediction.append("FAKE")

        else:
            prediction.append("UNCERTAIN")

    # =====================================================
    # 13. Return everything
    # =====================================================

    return {
        # ---------------------------------------------
        # Original branch evidence
        # ---------------------------------------------
        "appearance_evidence": dirichlet_appearance.get(
            "evidence",
            None,
        ),

        "temporal_evidence": dirichlet_temporal.get(
            "evidence",
            None,
        ),

        # ---------------------------------------------
        # Fused Dirichlet representation
        # ---------------------------------------------
        "evidence": fused_evidence,

        "alpha": fused_alpha,

        "S": fused_S,

        # ---------------------------------------------
        # Dempster-Shafer BBA
        # ---------------------------------------------
        "belief": fused_belief,

        "uncertainty": fused_uncertainty,

        # ---------------------------------------------
        # Final probabilities
        # ---------------------------------------------
        "probability": probability,

        # Standard Dirichlet probability is also returned
        # for debugging/comparison.
        "dirichlet_probability": dirichlet_probability,

        # ---------------------------------------------
        # Fusion diagnostics
        # ---------------------------------------------
        "conflict": conflict,

        # ---------------------------------------------
        # Final prediction
        # ---------------------------------------------
        "prediction": prediction,

        "prediction_id": prediction_id,

        # ---------------------------------------------
        # Threshold used
        # ---------------------------------------------
        "uncertainty_threshold": uncertainty_threshold,
    }


def format_ds_result(
    fused_output,
    index=0,
):
    """
    Format one fused sample for human-readable output.

    Example:

        Evidence:
            Real = 0.8000
            Fake = 0.7000

        Output:
            Real probability = 0.3100
            Fake probability = 0.2900
            Uncertainty      = 0.4000

        Conflict:
            0.1200

        Detection:
            UNCERTAIN
    """

    # ---------------------------------------------
    # Evidence
    # ---------------------------------------------

    evidence = fused_output["evidence"][index]

    # ---------------------------------------------
    # Probability
    # ---------------------------------------------

    probability = fused_output["probability"][index]

    # ---------------------------------------------
    # Uncertainty
    # ---------------------------------------------

    uncertainty = fused_output["uncertainty"][index, 0]

    # ---------------------------------------------
    # Conflict
    # ---------------------------------------------

    conflict = fused_output["conflict"][index, 0]

    # ---------------------------------------------
    # Prediction
    # ---------------------------------------------

    prediction = fused_output["prediction"][index]

    real_evidence = evidence[0].item()
    fake_evidence = evidence[1].item()

    real_probability = probability[0].item()
    fake_probability = probability[1].item()

    uncertainty_value = uncertainty.item()
    conflict_value = conflict.item()

    return (
        "\n"
        "==================================================\n"
        "Dempster-Shafer Fusion Result\n"
        "==================================================\n"
        "\n"
        "Evidence:\n"
        f"    Real = {real_evidence:.4f}\n"
        f"    Fake = {fake_evidence:.4f}\n"
        "\n"
        "Output:\n"
        f"    Real probability = {real_probability:.4f}\n"
        f"    Fake probability = {fake_probability:.4f}\n"
        f"    Uncertainty      = {uncertainty_value:.4f}\n"
        "\n"
        f"Conflict             = {conflict_value:.4f}\n"
        "\n"
        f"Detection : {prediction}\n"
        "==================================================\n"
    )
