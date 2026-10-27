from __future__ import annotations

import cv2
import numpy as np


def overlay_heatmap(image: np.ndarray, norm_map: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """image: uint8 RGB (H,W,3); norm_map in [0,1] (0.5 = decision boundary). Returns uint8 RGB."""
    heat = cv2.applyColorMap((np.clip(norm_map, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image, 1 - alpha, heat, alpha, 0)


def overlay_mask(image: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha: float = 0.45) -> np.ndarray:
    out = image.copy()
    layer = np.zeros_like(out)
    layer[:] = color
    m = mask.astype(bool)
    out[m] = cv2.addWeighted(out, 1 - alpha, layer, alpha, 0)[m]
    contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.drawContours(out, contours, -1, color, 1)


def draw_regions(image: np.ndarray, regions: list[dict], color=(255, 255, 0)) -> np.ndarray:
    out = image.copy()
    for r in regions:
        x, y, w, h = r["bbox"]
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cv2.putText(out, f"#{r['id']}", (x, max(y - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out
