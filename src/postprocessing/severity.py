"""Heuristic severity index (Objective 4: "where feasible").

NOTE: this index is NOT validated against human severity grades - report it as an
exploratory, interpretable measure, not as a ground-truth quality score.
"""
from __future__ import annotations

import numpy as np


def severity_index(max_map: float, pixel_threshold: float, area_ratio: float, extent_ref: float = 0.05) -> float:
    intensity = float(np.clip((max_map - pixel_threshold) / max(pixel_threshold, 1e-12), 0.0, 1.0))
    extent = float(np.clip(area_ratio / max(extent_ref, 1e-12), 0.0, 1.0))
    return 0.6 * intensity + 0.4 * extent


def severity_level(index: float, anomalous: bool) -> str:
    if not anomalous:
        return "none"
    return "low" if index < 0.33 else "medium" if index < 0.66 else "high"
