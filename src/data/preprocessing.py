"""Single preprocessing implementation shared by training, evaluation and the app.

Using ONE class everywhere guarantees train/inference consistency (spec section 6.1-B).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DENOISE_MODES = ("none", "gaussian", "bilateral", "nlm")


class Preprocessor:
    def __init__(self, image_size: int = 256, center_crop: int | None = None, denoise: str = "none",
                 mean=IMAGENET_MEAN, std=IMAGENET_STD):
        if denoise not in DENOISE_MODES:
            raise ValueError(f"denoise must be one of {DENOISE_MODES}")
        self.image_size = int(image_size)
        self.center_crop = int(center_crop) if center_crop else None
        self.denoise = denoise
        self.mean = tuple(mean)
        self.std = tuple(std)

    # ---- (de)serialisation, stored inside checkpoints ----
    def to_dict(self) -> dict:
        return {"image_size": self.image_size, "center_crop": self.center_crop, "denoise": self.denoise,
                "mean": list(self.mean), "std": list(self.std)}

    @classmethod
    def from_dict(cls, d: dict) -> "Preprocessor":
        return cls(**d)

    @classmethod
    def from_config(cls, data_cfg: dict) -> "Preprocessor":
        return cls(data_cfg.get("image_size", 256), data_cfg.get("center_crop"), data_cfg.get("denoise", "none"))

    @property
    def out_size(self) -> int:
        return self.center_crop or self.image_size

    # ---- loading ----
    @staticmethod
    def to_pil(img) -> Image.Image:
        if isinstance(img, (str, Path)):
            img = Image.open(img)
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype(np.uint8))
        if not isinstance(img, Image.Image):
            raise TypeError(f"Unsupported image type: {type(img)}")
        return img.convert("RGB")  # grayscale MVTec categories are replicated to 3 channels

    def _geometry(self, img: Image.Image, resample) -> Image.Image:
        img = TF.resize(img, [self.image_size, self.image_size], interpolation=resample)
        if self.center_crop:
            img = TF.center_crop(img, [self.center_crop, self.center_crop])
        return img

    def _denoise(self, arr: np.ndarray) -> np.ndarray:
        if self.denoise == "gaussian":
            return cv2.GaussianBlur(arr, (3, 3), 0)
        if self.denoise == "bilateral":
            return cv2.bilateralFilter(arr, 9, 50, 50)
        if self.denoise == "nlm":
            return cv2.fastNlMeansDenoisingColored(arr, None, 5, 5, 7, 21)
        return arr

    # ---- main API ----
    def display_image(self, img) -> np.ndarray:
        """uint8 HxWx3 image after geometry (and denoise) - what the model 'sees', for visualisation."""
        pil = self._geometry(self.to_pil(img), TF.InterpolationMode.BILINEAR)
        arr = np.asarray(pil)
        return self._denoise(arr) if self.denoise != "none" else arr

    def image_to_tensor(self, img) -> torch.Tensor:
        arr = self.display_image(img)
        t = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).float() / 255.0
        return TF.normalize(t, list(self.mean), list(self.std))

    def mask_to_tensor(self, mask) -> torch.Tensor:
        pil = self._geometry(Image.open(mask).convert("L") if isinstance(mask, (str, Path)) else mask,
                             TF.InterpolationMode.NEAREST)
        return (torch.from_numpy(np.asarray(pil).copy()).float() > 127).float().unsqueeze(0)

    def empty_mask(self) -> torch.Tensor:
        return torch.zeros(1, self.out_size, self.out_size)

    def denormalize(self, t: torch.Tensor) -> np.ndarray:
        mean = torch.tensor(self.mean, device=t.device).view(3, 1, 1)
        std = torch.tensor(self.std, device=t.device).view(3, 1, 1)
        arr = (t * std + mean).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        return (arr * 255).round().astype(np.uint8)
