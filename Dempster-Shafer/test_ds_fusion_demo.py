"""
Demo: show the REAL Dempster combination using a MOCK temporal branch.

Put this file next to your original fusion file and set FUSION_MODULE
below to that file's name (without .py).

For each scenario it prints:
    - Appearance BBA, Temporal BBA
    - Conflict
    - Every Dempster term, calculated by hand
    - Fused result from compute_ds_combination()
    - A check that hand calculation == your function
"""

import importlib
import torch

FUSION_MODULE = "de-fusion-3class"  # <-- new file with the conflict rule

compute_ds_combination = importlib.import_module(
    FUSION_MODULE
).compute_ds_combination

K = 2  # 0 = Real, 1 = Fake


def make_dirichlet(evidence):
    """Mock EDL output from raw evidence (alpha = evidence + 1)."""
    evidence = torch.tensor(evidence, dtype=torch.float32)
    alpha = evidence + 1.0
    S = alpha.sum(-1, keepdim=True)
    return {
        "evidence": evidence,
        "alpha": alpha,
        "S": S,
        "belief": evidence / S,
        "uncertainty": K / S,
    }


# [Real evidence, Fake evidence] per sample
SCENARIOS = {
    "1. Both say FAKE (agreement)": ([[1.0, 9.0]], [[0.5, 7.0]]),
    "2. Appearance=REAL, Temporal=FAKE (conflict)": ([[9.0, 1.0]], [[1.0, 9.0]]),
    "3. Temporal unsure (near-zero evidence)": ([[1.0, 10.0]], [[0.1, 0.1]]),
    "4. Both unsure": ([[0.3, 0.2]], [[0.2, 0.4]]),
}


def hand_dempster(bA, uA, bT, uT):
    """Dempster's rule written out term by term (pure python)."""
    conflict = bA[0] * bT[1] + bA[1] * bT[0]
    norm = 1.0 - conflict
    terms = []
    fused_b = []
    for i in range(K):
        both = bA[i] * bT[i]
        a_only = bA[i] * uT
        t_only = bT[i] * uA
        terms.append((both, a_only, t_only))
        fused_b.append((both + a_only + t_only) / norm)
    fused_u = uA * uT / norm
    return conflict, norm, terms, fused_b, fused_u


def main():
    names = ["Real", "Fake"]
    all_ok = True

    for title, (ev_a, ev_t) in SCENARIOS.items():
        dA = make_dirichlet(ev_a)
        dT = make_dirichlet(ev_t)

        out = compute_ds_combination(
            dA, dT, uncertainty_threshold=0.40, conflict_threshold=0.50
        )

        bA = dA["belief"][0].tolist()
        uA = dA["uncertainty"][0, 0].item()
        bT = dT["belief"][0].tolist()
        uT = dT["uncertainty"][0, 0].item()

        conflict, norm, terms, hb, hu = hand_dempster(bA, uA, bT, uT)

        print("=" * 62)
        print(title)
        print("=" * 62)
        print(f"Appearance BBA : Real={bA[0]:.4f} Fake={bA[1]:.4f} u={uA:.4f}")
        print(f"Temporal   BBA : Real={bT[0]:.4f} Fake={bT[1]:.4f} u={uT:.4f}")
        print(f"\nConflict = bA_real*bT_fake + bA_fake*bT_real = {conflict:.4f}")
        print(f"Normalizer (1 - conflict)                     = {norm:.4f}\n")

        for i in range(K):
            both, a_only, t_only = terms[i]
            print(
                f"{names[i]:>4}: ({both:.4f} both + {a_only:.4f} A-with-uT "
                f"+ {t_only:.4f} T-with-uA) / {norm:.4f} = {hb[i]:.4f}"
            )
        print(f"  u : ({uA:.4f} * {uT:.4f}) / {norm:.4f} = {hu:.4f}")

        fb = out["belief"][0].tolist()
        fu = out["uncertainty"][0, 0].item()
        fc = out["conflict"][0, 0].item()
        prob = out["probability"][0].tolist()

        print("\nYour compute_ds_combination():")
        print(f"  Fused belief : Real={fb[0]:.4f} Fake={fb[1]:.4f}")
        print(f"  Fused u      : {fu:.4f}")
        print(f"  Conflict     : {fc:.4f}")
        print(f"  Probability  : Real={prob[0]:.4f} Fake={prob[1]:.4f}")
        reason = out["uncertain_reason"][0]
        print(
            f"  Detection    : {out['prediction'][0]}"
            + (f"  (reason: {reason})" if reason else "")
        )

        ok = (
            abs(fb[0] - hb[0]) < 1e-4
            and abs(fb[1] - hb[1]) < 1e-4
            and abs(fu - hu) < 1e-4
            and abs(fc - conflict) < 1e-4
        )
        all_ok &= ok

        # Did fusion actually change anything vs appearance alone?
        changed = abs(fu - uA) > 1e-4 or abs(fb[1] - bA[1]) > 1e-4
        print(f"\n  Matches hand calculation : {'YES' if ok else 'NO  <-- PROBLEM'}")
        print(f"  Fusion changed result    : {'YES' if changed else 'no'}")
        print(
            f"  Uncertainty  A={uA:.4f}  T={uT:.4f}  ->  fused={fu:.4f}\n"
        )

    print("ALL SCENARIOS MATCH" if all_ok else "SOME SCENARIOS MISMATCH")


if __name__ == "__main__":
    main()