from __future__ import annotations

import cv2
import numpy as np


def extract_regions(mask: np.ndarray, amap: np.ndarray) -> list[dict]:
    """Connected defect regions with bounding box, area and score statistics (8-connectivity)."""
    n, lab, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    total = mask.shape[0] * mask.shape[1]
    regions = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        vals = amap[lab == i]
        regions.append({"id": len(regions) + 1, "bbox": [x, y, w, h], "area_px": area,
                        "area_ratio": area / total, "max_score": float(vals.max()),
                        "mean_score": float(vals.mean()),
                        "centroid": [float(cents[i][0]), float(cents[i][1])]})
    return sorted(regions, key=lambda r: -r["max_score"])
