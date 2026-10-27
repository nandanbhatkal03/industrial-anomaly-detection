"""Threshold calibration.

DEPLOYABLE thresholds use only held-out NORMAL validation data (no anomalous sample is available
in a real line). 'Oracle' best-F1 thresholds use test labels and are reported only as an upper bound.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve


def calibrate(val_scores: np.ndarray, val_maps: np.ndarray, cfg: dict) -> dict:
    s = np.asarray(val_scores, dtype=np.float64)
    method = cfg.get("image_method", "mean_std")
    if method == "mean_std":
        img_thr = s.mean() + cfg.get("image_k", 3.0) * s.std()
    elif method == "max":
        img_thr = s.max() * cfg.get("image_margin", 1.1)
    elif method == "percentile":
        img_thr = np.percentile(s, cfg.get("image_percentile", 99.0))
    else:
        raise ValueError(f"Unknown image_method '{method}'")
    flat = np.asarray(val_maps, dtype=np.float32).ravel()
    if flat.size > 5_000_000:
        flat = np.random.RandomState(0).choice(flat, 5_000_000, replace=False)
    pix_thr = np.percentile(flat, cfg.get("pixel_percentile", 99.9))
    return {"image_threshold": float(img_thr), "pixel_threshold": float(pix_thr),
            "val_score_mean": float(s.mean()), "val_score_std": float(s.std()),
            "val_score_max": float(s.max()), "n_val": int(s.size), "image_method": method}


def best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Oracle threshold maximising F1 (uses labels -> reporting only)."""
    p, r, t = precision_recall_curve(labels, scores)
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    i = int(np.argmax(f1))
    return float(t[i]), float(f1[i])


def apply_threshold(amap: np.ndarray, threshold: float) -> np.ndarray:
    return np.asarray(amap) > threshold
