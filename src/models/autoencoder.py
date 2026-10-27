"""Convolutional autoencoder - reconstruction baseline (error map = anomaly map)."""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseAnomalyModel


def _down(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 4, 2, 1, bias=False), nn.BatchNorm2d(o), nn.LeakyReLU(0.2, True))


def _up(i, o):
    return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))


class ConvAutoencoder(BaseAnomalyModel):
    name = "autoencoder"
    needs_gradient_training = True

    def __init__(self, latent_channels=64, sigma=4.0, pretrained=True):  # `pretrained` unused (API symmetry)
        super().__init__(sigma)
        ch = [3, 32, 64, 128, 256, 256]  # 5 stride-2 stages: input must be divisible by 32
        self.encoder = nn.Sequential(*[_down(ch[i], ch[i + 1]) for i in range(5)],
                                     nn.Conv2d(256, latent_channels, 1))
        self.decoder = nn.Sequential(nn.Conv2d(latent_channels, 256, 1),
                                     _up(256, 256), _up(256, 128), _up(128, 64), _up(64, 32),
                                     nn.ConvTranspose2d(32, 3, 4, 2, 1))

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def loss(self, images):
        return torch.mean((self(images) - images) ** 2)

    def fit(self, loader, device, log=None):
        raise RuntimeError("Gradient-trained model: use src.training.trainer.train_model")

    def _raw_map(self, images):
        return ((self(images) - images) ** 2).mean(1, keepdim=True)
