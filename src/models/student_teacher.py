"""Student-teacher feature regression baseline (STPM-style).

A frozen pretrained teacher and a randomly initialised student (same architecture) are compared at
several depths. The student is trained ONLY on normal images, so it imitates the teacher well only
on normal appearance; the discrepancy map highlights anomalies. (For the official EfficientAD
baseline use a reference implementation such as the Apache-2.0 'anomalib' library and cite the paper.)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import BaseAnomalyModel
from .feature_extractor import FeatureExtractor


class StudentTeacher(BaseAnomalyModel):
    name = "student_teacher"
    needs_gradient_training = True

    def __init__(self, backbone="resnet18", layers=("layer1", "layer2", "layer3"), pretrained=True,
                 sigma=4.0, weights_path=None):
        super().__init__(sigma)
        layers = list(layers)
        self.teacher = FeatureExtractor(backbone, layers, pretrained, weights_path, trainable=False)
        self.student = FeatureExtractor(backbone, layers, pretrained=False, trainable=True)

    def trainable_parameters(self):
        return list(self.student.parameters())

    def _diffs(self, images):
        with torch.no_grad():
            t = self.teacher(images)
        s = self.student(images)
        return [0.5 * ((F.normalize(a, dim=1) - F.normalize(b, dim=1)) ** 2).sum(1, keepdim=True)
                for a, b in zip(t, s)]

    def loss(self, images):
        return sum(d.mean() for d in self._diffs(images))

    def fit(self, loader, device, log=None):
        raise RuntimeError("Gradient-trained model: use src.training.trainer.train_model")

    def _raw_map(self, images):
        diffs = self._diffs(images)
        top = diffs[0].shape[-2:]
        return torch.stack([F.interpolate(d, size=top, mode="bilinear", align_corners=False) for d in diffs]).mean(0)
