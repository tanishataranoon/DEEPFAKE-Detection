"""
=========================================================
Evidential Deep Learning Loss
---------------------------------------------------------
Implements the loss from:
Sensoy, Kaplan, Kandemir (NeurIPS 2018),
"Evidential Deep Learning to Quantify Classification
Uncertainty."

Two parts:
  1. Expected MSE between the Dirichlet mean (alpha/S) and
     the one-hot target -- this is the "get the right answer"
     term.
  2. A KL-divergence regularizer that pushes evidence for the
     WRONG class toward zero (never penalizes evidence for the
     correct class). This term is annealed in gradually so the
     model isn't punished for having ANY evidence early in
     training, before it has learned to tell real from fake at
     all -- annealing too early can collapse training into
     always predicting "uncertain."
=========================================================
"""

import torch
import torch.nn.functional as F


def compute_kl_divergence(alpha, num_classes, device):
    """
    KL divergence between Dir(alpha) and the uniform Dirichlet
    Dir(1, ..., 1).
    """
    ones = torch.ones((1, num_classes), dtype=torch.float32, device=device)

    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_ones = torch.sum(ones, dim=1, keepdim=True)

    lgamma_alpha_sum = torch.lgamma(S_alpha)
    lgamma_alpha_each = torch.lgamma(alpha).sum(dim=1, keepdim=True)
    lgamma_ones_sum = torch.lgamma(S_ones)
    lgamma_ones_each = torch.lgamma(ones).sum(dim=1, keepdim=True)

    first_term = (lgamma_alpha_sum - lgamma_alpha_each) - (lgamma_ones_sum - lgamma_ones_each)

    digamma_term = torch.digamma(alpha) - torch.digamma(S_alpha)
    second_term = ((alpha - ones) * digamma_term).sum(dim=1, keepdim=True)

    kl = first_term + second_term
    return kl


def compute_edl_loss(evidence, target, epoch, num_classes, annealing_step,
                      class_weights=None, device=None):
    """
    evidence       : (B, K) non-negative evidence from EvidenceHead
    target         : (B,) integer class labels (0 = real, 1 = fake)
    epoch          : current training epoch (int) -- drives KL annealing
    num_classes    : K
    annealing_step : number of epochs over which the KL term ramps
                      from 0 -> 1 (e.g. 10 means full strength by epoch 10)
    """
    if device is None:
        device = evidence.device
    y = F.one_hot(target, num_classes=num_classes).float().to(device)

    alpha = evidence + 1
    S = torch.sum(alpha, dim=1, keepdim=True)

    if class_weights is not None:
        w = class_weights.to(device).unsqueeze(0)          # (1, K)
        err = torch.sum(w * (y - alpha / S) ** 2, dim=1, keepdim=True)
        var = torch.sum(w * alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    else:
        err = torch.sum((y - alpha / S) ** 2, dim=1, keepdim=True)
        var = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)

    loss_mse = err + var
    annealing_coef = torch.tensor(min(1.0, epoch / annealing_step), dtype=torch.float32, device=device)
    alpha_tilde = y + (1 - y) * alpha
    kl = compute_kl_divergence(alpha_tilde, num_classes, device)
    return (loss_mse + annealing_coef * kl).mean()