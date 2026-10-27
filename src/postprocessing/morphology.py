from __future__ import annotations

import cv2
import numpy as np


def clean_mask(mask: np.ndarray, open_kernel: int = 3, close_kernel: int = 5, min_area: int = 30) -> np.ndarray:
    """Open (remove speckle) -> close (fill holes) -> drop connected components below min_area."""
    m = mask.astype(np.uint8)
    if open_kernel > 1:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((open_kernel, open_kernel), np.uint8))
    if close_kernel > 1:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((close_kernel, close_kernel), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[lab == i] = 1
    return out.astype(bool)
