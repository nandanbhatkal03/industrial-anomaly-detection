"""Lightweight CNN feature extractors (spec section 6.1-C).

Backbones are selected by name and compared experimentally, never assumed.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor

# Mid-level (stride 8 and 16) nodes per backbone. Mid-level features transfer best to
# industrial textures; the deepest (stride 32) features are too ImageNet-specific.
DEFAULT_LAYERS = {
    "resnet18": ["layer2", "layer3"],
    "resnet34": ["layer2", "layer3"],
    "resnet50": ["layer2", "layer3"],
    "wide_resnet50_2": ["layer2", "layer3"],
    "mobilenet_v3_large": ["features.6", "features.12"],
    "mobilenet_v3_small": ["features.3", "features.8"],
    "mobilenet_v2": ["features.6", "features.13"],
    "efficientnet_b0": ["features.3", "features.5"],
}


def build_backbone(name: str, pretrained: bool = True, weights_path: str | None = None) -> nn.Module:
    if name not in DEFAULT_LAYERS:
        raise ValueError(f"Unsupported backbone '{name}'. Choose from {sorted(DEFAULT_LAYERS)}")
    ctor = getattr(torchvision.models, name)
    if weights_path:  # offline use: a state_dict you downloaded yourself
        m = ctor(weights=None)
        m.load_state_dict(torch.load(weights_path, map_location="cpu"))
        return m
    return ctor(weights="DEFAULT" if pretrained else None)


class FeatureExtractor(nn.Module):
    """Returns a list of intermediate feature maps ordered from shallow to deep."""

    def __init__(self, backbone: str = "resnet18", layers: list[str] | None = None, pretrained: bool = True,
                 weights_path: str | None = None, trainable: bool = False):
        super().__init__()
        if backbone not in DEFAULT_LAYERS:
            raise ValueError(f"Unsupported backbone '{backbone}'. Choose from {sorted(DEFAULT_LAYERS)}")
        self.backbone_name = backbone
        self.layers = list(layers) if layers else DEFAULT_LAYERS[backbone]
        self.trainable = trainable
        net = build_backbone(backbone, pretrained, weights_path)
        self.body = create_feature_extractor(net, return_nodes={l: l for l in self.layers})
        for p in self.body.parameters():
            p.requires_grad = trainable
        self.body.eval()
        with torch.no_grad():
            outs = self.body(torch.zeros(1, 3, 128, 128))
        self.out_channels = [outs[l].shape[1] for l in self.layers]
        self.strides = [128 // outs[l].shape[-1] for l in self.layers]

    def train(self, mode: bool = True):
        # A frozen extractor must stay in eval mode (BatchNorm statistics!) even if the parent trains.
        return super().train(mode and self.trainable)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        out = self.body(x)
        return [out[l] for l in self.layers]


def fuse_features(feats: list[torch.Tensor], local_kernel: int = 1) -> torch.Tensor:
    """Local neighbourhood aggregation (PatchCore-style), resize to the shallowest map, concat."""
    size = feats[0].shape[-2:]
    fused = []
    for f in feats:
        if local_kernel > 1:
            f = F.avg_pool2d(f, local_kernel, stride=1, padding=local_kernel // 2)
        if f.shape[-2:] != size:
            f = F.interpolate(f, size=size, mode="bilinear", align_corners=False)
        fused.append(f)
    return torch.cat(fused, dim=1)
