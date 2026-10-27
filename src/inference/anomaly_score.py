"""Anomaly-map utilities shared by every model, plus score normalisation."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def gaussian_kernel1d(sigma: float, device=None, dtype=torch.float32) -> torch.Tensor:
    radius = int(4 * sigma + 0.5)
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur on (B,1,H,W); stays on the model's device."""
    if sigma <= 0:
        return x
    k = gaussian_kernel1d(sigma, x.device, x.dtype)
    r = k.numel() // 2
    x = F.pad(x, (r, r, r, r), mode="replicate")
    x = F.conv2d(x, k.view(1, 1, 1, -1))
    return F.conv2d(x, k.view(1, 1, -1, 1))


def upsample_map(x: torch.Tensor, size) -> torch.Tensor:
    if tuple(x.shape[-2:]) == tuple(size):
        return x
    return F.interpolate(x, size=tuple(size), mode="bilinear", align_corners=False)


def image_score_from_map(maps: torch.Tensor) -> torch.Tensor:
    """Image-level score = maximum of the (smoothed) anomaly map. maps: (B,H,W)."""
    return maps.flatten(1).max(dim=1).values


def normalise_score(value, threshold: float):
    """Map a raw score to [0,1] so that the decision threshold sits at exactly 0.5."""
    import numpy as np

    return np.clip(np.asarray(value, dtype=np.float32) / (2.0 * max(float(threshold), 1e-12)), 0.0, 1.0)
