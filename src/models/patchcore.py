"""Memory-bank anomaly detection (PatchCore-style) with the knobs needed for the lightweight study.

PatchMemoryModel exposes: backbone/layers, local aggregation, optional channel standardisation
(feature_norm), optional Gaussian random projection of the STORED features (proj_dim),
coreset subsampling (greedy k-center or random) and low-precision storage (memory_dtype).
`PatchCore` below = the reference configuration; `LightPatch` (proposed_model.py) = the proposed one.
"""
from __future__ import annotations

import torch

from .base import BaseAnomalyModel
from .feature_extractor import FeatureExtractor, fuse_features

_DTYPES = {"float32": torch.float32, "float16": torch.float16}


def greedy_coreset(x: torch.Tensor, n_select: int, proj_dim: int = 128, seed: int = 0, device=None) -> torch.Tensor:
    """k-center greedy subsampling (the PatchCore coreset). Returns selected row indices."""
    n = x.shape[0]
    if n_select >= n:
        return torch.arange(n)
    device = device or x.device
    g = torch.Generator().manual_seed(seed)
    z = x.to(device)
    if z.shape[1] > proj_dim:  # cheap distance approximation (Johnson-Lindenstrauss)
        z = z @ (torch.randn(z.shape[1], proj_dim, generator=g) / proj_dim ** 0.5).to(device)
    first = int(torch.randint(n, (1,), generator=g))
    selected = [first]
    min_d = ((z - z[first]) ** 2).sum(1)
    for _ in range(n_select - 1):
        i = int(torch.argmax(min_d))
        selected.append(i)
        min_d = torch.minimum(min_d, ((z - z[i]) ** 2).sum(1))
    return torch.tensor(selected)


class PatchMemoryModel(BaseAnomalyModel):
    name = "patch_memory"

    def __init__(self, backbone="resnet18", layers=None, pretrained=True, local_kernel=3, proj_dim=None,
                 coreset_ratio=0.1, coreset_method="greedy", feature_norm=False, memory_dtype="float32",
                 k=1, sigma=4.0, seed=0, weights_path=None, query_chunk=1024):
        super().__init__(sigma)
        if coreset_method not in {"greedy", "random"}:
            raise ValueError("coreset_method must be 'greedy' or 'random'")
        self.extractor = FeatureExtractor(backbone, layers, pretrained, weights_path)
        self.local_kernel, self.proj_dim = int(local_kernel), (int(proj_dim) if proj_dim else None)
        self.coreset_ratio, self.coreset_method = float(coreset_ratio), coreset_method
        self.feature_norm, self.memory_dtype = bool(feature_norm), memory_dtype
        self.k, self.seed, self.query_chunk = int(k), int(seed), int(query_chunk)
        self.memory = None            # (M, D) stored (possibly fp16)
        self.proj = None              # (C, proj_dim)
        self.fn_mean = self.fn_std = None
        self._mem32 = None

    # ----- feature helpers -----
    def _patches(self, images):
        e = fuse_features(self.extractor(images), self.local_kernel)  # (B,C,h,w)
        B, C, h, w = e.shape
        return e.permute(0, 2, 3, 1).reshape(B * h * w, C), (B, h, w)

    def _transform(self, q):
        if self.feature_norm:
            q = (q - self._to(self.fn_mean, q)) / self._to(self.fn_std, q)
        if self.proj is not None:
            q = q @ self._to(self.proj, q)
        return q

    # ----- fitting -----
    def fit(self, loader, device, log=None):
        self.to(device).eval()
        chunks = []
        with torch.no_grad():
            for batch in loader:
                p, _ = self._patches(batch["image"].to(device))
                chunks.append(p.float().cpu())
        x = torch.cat(chunks)
        n_all, C = x.shape
        if self.feature_norm:  # cheap domain adaptation: standardise channels with NORMAL statistics
            self.fn_mean, self.fn_std = x.mean(0), x.std(0).clamp_min(1e-6)
            x = (x - self.fn_mean) / self.fn_std
        if self.proj_dim and self.proj_dim < C:
            g = torch.Generator().manual_seed(self.seed + 1)
            self.proj = torch.randn(C, self.proj_dim, generator=g) / self.proj_dim ** 0.5
            x = x @ self.proj
        n_sel = max(1, int(round(self.coreset_ratio * n_all)))
        if self.coreset_method == "greedy":
            idx = greedy_coreset(x, n_sel, seed=self.seed, device=device)
        else:
            idx = torch.randperm(n_all, generator=torch.Generator().manual_seed(self.seed))[:n_sel]
        self.memory = x[idx].to(_DTYPES[self.memory_dtype]).contiguous()
        self._mem32 = None
        if log:
            log(f"{self.name}: {n_all} patches (C={C}) -> memory {tuple(self.memory.shape)} "
                f"{self.memory_dtype} = {self.artifact_bytes() / 2**20:.2f} MB")

    # ----- inference -----
    def _raw_map(self, images):
        if self.memory is None:
            raise RuntimeError("Model is not fitted")
        q, (B, h, w) = self._patches(images)
        q = self._transform(q)
        if self._mem32 is None or self._mem32.device != q.device:
            self._mem32 = self.memory.to(q.device, torch.float32)
        out = []
        for c in q.split(self.query_chunk):
            d = torch.cdist(c, self._mem32)
            out.append(d.min(1).values if self.k == 1 else d.topk(self.k, largest=False).values.mean(1))
        return torch.cat(out).reshape(B, 1, h, w)

    def get_artifacts(self):
        a = {}
        if self.memory is not None:
            a["memory"] = self.memory.cpu()
        for key, val in (("proj", self.proj), ("fn_mean", self.fn_mean), ("fn_std", self.fn_std)):
            if val is not None:
                a[key] = val.cpu()
        return a

    def set_artifacts(self, a):
        self.memory, self.proj = a["memory"], a.get("proj")
        self.fn_mean, self.fn_std = a.get("fn_mean"), a.get("fn_std")
        self._mem32 = None


class PatchCore(PatchMemoryModel):
    """Reference baseline: WideResNet50 mid-level features, local aggregation, 10% greedy coreset."""
    name = "patchcore"

    def __init__(self, backbone="wide_resnet50_2", **kw):
        super().__init__(backbone=backbone, **kw)
