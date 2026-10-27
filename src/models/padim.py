"""PaDiM-style baseline: per-position Gaussian over pretrained CNN features + Mahalanobis distance."""
from __future__ import annotations

import torch

from .base import BaseAnomalyModel
from .feature_extractor import FeatureExtractor, fuse_features


class PaDiM(BaseAnomalyModel):
    name = "padim"

    def __init__(self, backbone="resnet18", layers=None, pretrained=True, reduced_dim=100, eps=0.01,
                 local_kernel=1, sigma=4.0, seed=0, weights_path=None):
        super().__init__(sigma)
        self.extractor = FeatureExtractor(backbone, layers, pretrained, weights_path)
        total = sum(self.extractor.out_channels)
        self.reduced_dim = min(int(reduced_dim), total) if reduced_dim else total
        self.eps, self.local_kernel = float(eps), int(local_kernel)
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("channel_idx", torch.randperm(total, generator=g)[: self.reduced_dim].sort().values)
        self.mean = None      # (d, P)
        self.inv_cov = None   # (P, d, d)
        self.hw = None

    def _embed(self, images):
        e = fuse_features(self.extractor(images), self.local_kernel)
        return e[:, self.channel_idx]

    def fit(self, loader, device, log=None):
        self.to(device).eval()
        acc_dev = torch.device("cpu") if device.type == "mps" else device  # MPS has no float64
        n, s1, s2 = 0, None, None
        with torch.no_grad():
            for batch in loader:
                e = self._embed(batch["image"].to(device))
                B, d, h, w = e.shape
                e = e.flatten(2).to(acc_dev, torch.float64)  # (B,d,P)
                if s1 is None:
                    P = e.shape[-1]
                    s1 = torch.zeros(d, P, dtype=torch.float64, device=acc_dev)
                    s2 = torch.zeros(P, d, d, dtype=torch.float64, device=acc_dev)
                    self.hw = (h, w)
                s1 += e.sum(0)
                s2 += torch.einsum("bdp,bep->pde", e, e)
                n += B
        mean = s1 / n
        cov = (s2 - n * torch.einsum("dp,ep->pde", mean, mean)) / max(n - 1, 1)
        cov += self.eps * torch.eye(d, dtype=torch.float64, device=acc_dev)
        self.mean = mean.float().to(device)
        self.inv_cov = torch.linalg.inv(cov).float().to(device)
        if log:
            log(f"PaDiM fitted on {n} images: d={d}, positions={P}, cov memory={self.artifact_bytes() / 2**20:.1f} MB")

    def _raw_map(self, images):
        if self.mean is None:
            raise RuntimeError("Model is not fitted")
        self.mean, self.inv_cov = self._to(self.mean, images), self._to(self.inv_cov, images)
        e = self._embed(images)
        B, d, h, w = e.shape
        if (h, w) != tuple(self.hw):
            raise ValueError(f"Feature map {(h, w)} differs from fitted {tuple(self.hw)}: use the training image size")
        delta = (e.flatten(2) - self.mean.unsqueeze(0)).permute(2, 0, 1)  # (P,B,d)
        m = torch.bmm(delta, self.inv_cov)
        dist = (m * delta).sum(-1).clamp_min(0).sqrt()                     # (P,B)
        return dist.t().reshape(B, 1, h, w)

    def get_artifacts(self):
        return {} if self.mean is None else {"mean": self.mean.cpu(), "inv_cov": self.inv_cov.cpu(),
                                              "hw": torch.tensor(self.hw)}

    def set_artifacts(self, a):
        self.mean, self.inv_cov, self.hw = a["mean"], a["inv_cov"], tuple(int(v) for v in a["hw"])
