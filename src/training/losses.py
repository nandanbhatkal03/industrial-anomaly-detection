"""Training losses live next to each gradient-trained model (`model.loss`).
This module exposes simple helpers for experiments that need a standalone loss."""
import torch
import torch.nn.functional as F


def mse_reconstruction(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(recon, target)


def normalised_feature_distance(t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """0.5 * || norm(t) - norm(s) ||^2 summed over channels, averaged over pixels (student-teacher)."""
    return (0.5 * (F.normalize(t, dim=1) - F.normalize(s, dim=1)) ** 2).sum(1).mean()
