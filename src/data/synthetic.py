"""Tiny synthetic MVTec-format dataset for smoke tests and CI (no download needed).

NOT a substitute for MVTec AD - real results must be produced on the real benchmark.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _texture(rng: np.random.RandomState, size: int) -> np.ndarray:
    base = cv2.resize(rng.rand(8, 8).astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC)
    fine = cv2.GaussianBlur(rng.rand(size, size).astype(np.float32), (0, 0), 1.0)
    g = 0.55 + 0.15 * (base - 0.5) + 0.10 * (fine - 0.5)
    img = np.stack([g * 0.95, g, g * 1.05], -1)
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def _add_defect(img: np.ndarray, rng: np.random.RandomState, kind: str):
    size = img.shape[0]
    mask = np.zeros((size, size), np.uint8)
    cx, cy = rng.randint(size // 5, 4 * size // 5, 2)
    if kind == "blob":
        cv2.ellipse(mask, (int(cx), int(cy)), (int(size * rng.uniform(.05, .1)), int(size * rng.uniform(.04, .08))),
                    float(rng.randint(0, 180)), 0, 360, 255, -1)
        img[mask > 0] = (img[mask > 0] * 0.35).astype(np.uint8)
    else:  # scratch
        ang = rng.uniform(0, np.pi)
        dx, dy = int(np.cos(ang) * size * .25), int(np.sin(ang) * size * .25)
        cv2.line(mask, (int(cx - dx), int(cy - dy)), (int(cx + dx), int(cy + dy)), 255, max(2, size // 40))
        img[mask > 0] = np.clip(img[mask > 0].astype(int) + 90, 0, 255).astype(np.uint8)
    return img, mask


def generate_synthetic_mvtec(root, category: str = "synthetic", n_train: int = 24, n_test_good: int = 6,
                             n_test_bad: int = 12, size: int = 128, seed: int = 0) -> Path:
    rng = np.random.RandomState(seed)
    cat = Path(root) / category
    for sub in ("train/good", "test/good", "test/blob", "test/scratch", "ground_truth/blob", "ground_truth/scratch"):
        (cat / sub).mkdir(parents=True, exist_ok=True)

    def save(path, arr):
        cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if arr.ndim == 3 else arr)

    for i in range(n_train):
        save(cat / "train/good" / f"{i:03d}.png", _texture(rng, size))
    for i in range(n_test_good):
        save(cat / "test/good" / f"{i:03d}.png", _texture(rng, size))
    for i in range(n_test_bad):
        kind = "blob" if i % 2 == 0 else "scratch"
        img, mask = _add_defect(_texture(rng, size), rng, kind)
        save(cat / "test" / kind / f"{i:03d}.png", img)
        save(cat / "ground_truth" / kind / f"{i:03d}_mask.png", mask)
    return cat
