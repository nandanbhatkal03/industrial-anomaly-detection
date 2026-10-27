"""EfficientAD-INSPIRED model (not the official implementation).

Keeps the two ideas that make EfficientAD fast: (1) a tiny fully-convolutional Patch Description Network
(PDN, receptive field 33) used for both teacher and student, and (2) hard-feature-mining loss for the student.
Differences from the paper (so do not report it as "EfficientAD"): teacher is distilled from ResNet18 mid-level
features (not WideResNet-101), no autoencoder branch for logical anomalies, no ImageNet penalty, far shorter
training. For the official baseline use the `anomalib` library.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAnomalyModel
from .feature_extractor import FeatureExtractor, fuse_features


def make_pdn(out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(3, 64, 4, padding=3), nn.ReLU(True), nn.AvgPool2d(2, 2, padding=1),
        nn.Conv2d(64, 128, 4, padding=3), nn.ReLU(True), nn.AvgPool2d(2, 2, padding=1),
        nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(True), nn.Conv2d(128, out_ch, 4))


class EfficientADLite(BaseAnomalyModel):
    name = "efficientad_lite"

    def __init__(self, out_channels=128, teacher_epochs=30, student_epochs=60, lr=1e-3, hard_quantile=0.999,
                 distill_backbone="resnet18", sigma=4.0, seed=0, pretrained=True):
        super().__init__(sigma)
        self.out_ch, self.t_ep, self.s_ep, self.lr = int(out_channels), int(teacher_epochs), int(student_epochs), float(lr)
        self.hard_q, self.distill_backbone, self.seed, self._pretrained = float(hard_quantile), distill_backbone, int(seed), pretrained
        self.teacher, self.student = make_pdn(self.out_ch), make_pdn(self.out_ch)
        self.t_mean = self.t_std = self.q_a = self.q_b = None

    @staticmethod
    def _target(extractor, proj, x, size):
        f = fuse_features(extractor(x), 3)
        f = F.interpolate(f, size=size, mode="bilinear", align_corners=False)
        f = (f - f.mean((0, 2, 3), keepdim=True)) / (f.std((0, 2, 3), keepdim=True) + 1e-6)
        return torch.einsum("bchw,cd->bdhw", f, proj)

    @staticmethod
    def _hard(e, q):
        k = max(1, int(math.ceil((1 - q) * e.numel())))
        return torch.topk(e.flatten(), k).values.mean()

    def fit(self, loader, device, log=None):
        torch.manual_seed(self.seed)
        self.to(device)
        ext = FeatureExtractor(self.distill_backbone, None, self._pretrained).to(device).eval()
        g = torch.Generator().manual_seed(self.seed)
        proj = (torch.randn(sum(ext.out_channels), self.out_ch, generator=g) / self.out_ch ** 0.5).to(device)
        opt = torch.optim.Adam(self.teacher.parameters(), lr=self.lr)
        self.teacher.train()
        for ep in range(self.t_ep):  # stage 1: distil teacher
            tot = 0.0
            for b in loader:
                x = b["image"].to(device)
                out = self.teacher(x)
                with torch.no_grad():
                    tgt = self._target(ext, proj, x, out.shape[-2:])
                loss = F.mse_loss(out, tgt)
                opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
            if log and (ep == 0 or (ep + 1) % 10 == 0):
                log(f"[efficientad_lite] teacher epoch {ep + 1}/{self.t_ep} loss={tot / len(loader):.4f}")
        self.teacher.eval()
        with torch.no_grad():
            outs = torch.cat([self.teacher(b["image"].to(device)) for b in loader])
            self.t_mean, self.t_std = outs.mean((0, 2, 3), keepdim=True), outs.std((0, 2, 3), keepdim=True) + 1e-6
        opt = torch.optim.Adam(self.student.parameters(), lr=self.lr)
        self.student.train()
        for ep in range(self.s_ep):  # stage 2: student imitates teacher on NORMAL images, hard-feature loss
            tot = 0.0
            for b in loader:
                x = b["image"].to(device)
                with torch.no_grad():
                    t = (self.teacher(x) - self.t_mean) / self.t_std
                loss = self._hard((t - self.student(x)) ** 2, self.hard_q)
                opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
            if log and (ep == 0 or (ep + 1) % 10 == 0):
                log(f"[efficientad_lite] student epoch {ep + 1}/{self.s_ep} loss={tot / len(loader):.4f}")
        self.student.eval()
        with torch.no_grad():  # score normalisation from normal training maps
            m = torch.cat([self._diff(b["image"].to(device)).flatten() for b in loader])
            n = min(m.numel(), 1_000_000)
            m = m[torch.randperm(m.numel(), device=m.device)[:n]]
            self.q_a, self.q_b = torch.quantile(m, 0.9).reshape(1), torch.quantile(m, 0.995).reshape(1)
        if log:
            log(f"efficientad_lite fitted: params={self.num_parameters() / 1e6:.2f}M")

    def _diff(self, x):
        t = (self.teacher(x) - self._to(self.t_mean, x)) / self._to(self.t_std, x)
        return ((t - self.student(x)) ** 2).mean(1, keepdim=True)

    def _raw_map(self, images):
        if self.q_a is None:
            raise RuntimeError("Model is not fitted")
        qa, qb = self._to(self.q_a, images), self._to(self.q_b, images)
        return (self._diff(images) - qa) / (qb - qa).clamp_min(1e-8)

    def get_artifacts(self):
        return {} if self.q_a is None else {"t_mean": self.t_mean.cpu(), "t_std": self.t_std.cpu(),
                                              "q_a": self.q_a.cpu(), "q_b": self.q_b.cpu()}

    def set_artifacts(self, a):
        self.t_mean, self.t_std, self.q_a, self.q_b = a["t_mean"], a["t_std"], a["q_a"], a["q_b"]
