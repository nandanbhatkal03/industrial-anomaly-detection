from __future__ import annotations

import torch
import torch.nn as nn

from src.inference.anomaly_score import gaussian_blur, image_score_from_map, upsample_map


class BaseAnomalyModel(nn.Module):
    """Common interface: fit() (or gradient training) -> predict(images) -> (scores, maps)."""

    name = "base"
    needs_gradient_training = False

    def __init__(self, sigma: float = 4.0):
        super().__init__()
        self.sigma = float(sigma)

    # ----- to be implemented by subclasses -----
    def fit(self, loader, device, log=None) -> None:
        raise NotImplementedError

    def _raw_map(self, images: torch.Tensor) -> torch.Tensor:
        """Return a (B,1,h,w) anomaly map at any resolution (higher = more anomalous)."""
        raise NotImplementedError

    def get_artifacts(self) -> dict:
        """Fitted, non-parameter tensors that must be saved with the checkpoint."""
        return {}

    def set_artifacts(self, artifacts: dict) -> None:
        pass

    def loss(self, images: torch.Tensor) -> torch.Tensor:  # gradient-trained models only
        raise NotImplementedError

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    # ----- shared inference -----
    @torch.no_grad()
    def predict(self, images: torch.Tensor):
        self.eval()
        raw = self._raw_map(images)
        maps = gaussian_blur(upsample_map(raw, images.shape[-2:]), self.sigma).squeeze(1)
        return image_score_from_map(maps), maps

    # ----- bookkeeping for efficiency metrics -----
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def artifact_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.get_artifacts().values() if torch.is_tensor(t))

    @staticmethod
    def _to(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return t if t.device == ref.device else t.to(ref.device)
