"""Optional augmentation for NORMAL training images.

Deliberately conservative: geometric transforms can turn a *normal* object into an unseen
(and therefore 'anomalous') appearance - e.g. a flipped transistor or rotated metal_nut -
so only mild photometric jitter is enabled and flips are opt-in per category.
"""
from __future__ import annotations

import random

from PIL import Image
from torchvision import transforms as T


def build_train_augmentation(aug_cfg: dict | None):
    if not aug_cfg or not aug_cfg.get("enabled", False):
        return None
    ops = []
    jitter = float(aug_cfg.get("color_jitter", 0.0))
    if jitter > 0:
        ops.append(T.ColorJitter(brightness=jitter, contrast=jitter))
    hflip = bool(aug_cfg.get("hflip", False))
    composed = T.Compose(ops) if ops else None

    def _apply(img: Image.Image) -> Image.Image:
        if composed is not None:
            img = composed(img)
        if hflip and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    return _apply
