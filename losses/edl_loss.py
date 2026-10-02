import torch
import torch.nn.functional as F


def compute_kl_divergence(alpha, num_classes, device=None):
    """
    KL(Dir(alpha) || Dir(1,...,1)).
    """
    if device is None:
        device = alpha.device

    ones = torch.ones(
        (1, num_classes),
        dtype=alpha.dtype,
        device=device,
    )

    S_alpha = alpha.sum(dim=1, keepdim=True)
    S_ones = ones.sum(dim=1, keepdim=True)

    first_term = (
        torch.lgamma(S_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        - torch.lgamma(S_ones)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
    )

    digamma_term = torch.digamma(alpha) - torch.digamma(S_alpha)
    second_term = ((alpha - ones) * digamma_term).sum(
        dim=1, keepdim=True
    )

    return first_term + second_term


def compute_edl_loss(
    evidence,
    target,
    epoch,
    num_classes=2,
    annealing_step=10,
    class_weights=None,
    device=None,
):
    """
    Sensoy-style EDL loss.

    target:
        0 = Real
        1 = Fake

    class_weights:
        Optional tensor of shape (2). Use only for training imbalance.
    """
    if device is None:
        device = evidence.device

    target = target.long().to(device)
    y = F.one_hot(
        target,
        num_classes=num_classes
    ).to(dtype=evidence.dtype)

    alpha = evidence + 1.0
    S = alpha.sum(dim=1, keepdim=True)
    probs = alpha / S

    if class_weights is not None:
        w = class_weights.to(
            device=device,
            dtype=evidence.dtype,
        ).view(1, num_classes)

        err = (w * (y - probs).pow(2)).sum(dim=1, keepdim=True)

        var = (
            w
            * alpha
            * (S - alpha)
            / (S * S * (S + 1.0))
        ).sum(dim=1, keepdim=True)
    else:
        err = (y - probs).pow(2).sum(dim=1, keepdim=True)

        var = (
            alpha
            * (S - alpha)
            / (S * S * (S + 1.0))
        ).sum(dim=1, keepdim=True)

    loss_mse = err + var

    if annealing_step <= 0:
        annealing_coef = 1.0
    else:
        annealing_coef = min(
            1.0,
            float(epoch) / float(annealing_step),
        )

    # Remove target-class evidence from the KL regularizer.
    alpha_tilde = y + (1.0 - y) * alpha

    kl = compute_kl_divergence(
        alpha_tilde,
        num_classes,
        device,
    )

    return (loss_mse + annealing_coef * kl).mean()
