from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.integrate import trapezoid
from sklearn.metrics import roc_auc_score


def pixel_auroc(masks: np.ndarray, maps: np.ndarray) -> float:
    m = masks.astype(bool).ravel()
    return float(roc_auc_score(m, maps.ravel())) if 0 < m.sum() < m.size else float("nan")


def aupro(masks: np.ndarray, maps: np.ndarray, fpr_limit: float = 0.3, num_thresholds: int = 300,
          max_normal_pixels: int = 2_000_000, seed: int = 0) -> float:
    """Area under the per-region-overlap (PRO) curve for FPR in [0, fpr_limit], normalised to [0,1].

    Thresholds are sampled on the FPR axis (via quantiles of normal-pixel scores), so the
    low-FPR region that matters is densely covered. Connected components use 8-connectivity,
    as in the MVTec AD evaluation protocol.
    """
    masks = masks.astype(bool)
    if not masks.any():
        return float("nan")
    normal = maps[~masks].astype(np.float64)
    if normal.size == 0:
        return float("nan")
    if normal.size > max_normal_pixels:
        normal = np.random.RandomState(seed).choice(normal, max_normal_pixels, replace=False)
    normal.sort()
    comps = []
    for m, a in zip(masks, maps):
        if m.any():
            lab, n = ndimage.label(m, structure=np.ones((3, 3), int))
            comps += [np.sort(a[lab == i]) for i in range(1, n + 1)]
    targets = np.concatenate([[0.0], np.geomspace(1e-5, fpr_limit, num_thresholds)])
    thr = np.quantile(normal, 1.0 - targets)
    thr[0] = np.nextafter(normal[-1], np.inf)
    fpr = (len(normal) - np.searchsorted(normal, thr, side="left")) / len(normal)
    pro = np.mean([(len(c) - np.searchsorted(c, thr, side="left")) / len(c) for c in comps], axis=0)
    if fpr[-1] > fpr_limit and len(fpr) > 1:  # trim the tail exactly at the limit
        keep = fpr <= fpr_limit
        y_lim = np.interp(fpr_limit, fpr, pro)
        fpr, pro = np.append(fpr[keep], fpr_limit), np.append(pro[keep], y_lim)
    return float(trapezoid(pro, fpr) / fpr_limit)


def segmentation_metrics(masks: np.ndarray, preds: np.ndarray) -> dict:
    """Micro-averaged over ALL test pixels (normal images contribute false positives)."""
    m, p = masks.astype(bool), preds.astype(bool)
    tp, fp, fn = int((m & p).sum()), int((~m & p).sum()), int((m & ~p).sum())
    return {"iou": tp / max(tp + fp + fn, 1), "dice": 2 * tp / max(2 * tp + fp + fn, 1),
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1)}


def best_pixel_f1(masks: np.ndarray, maps: np.ndarray, max_pixels: int = 2_000_000, seed: int = 0) -> tuple[float, float]:
    """Oracle pixel threshold maximising F1 on (a subsample of) test pixels - upper bound only."""
    from src.postprocessing.threshold import best_f1_threshold

    m, a = masks.astype(bool).ravel(), maps.ravel()
    if m.size > max_pixels:
        idx = np.random.RandomState(seed).choice(m.size, max_pixels, replace=False)
        m, a = m[idx], a[idx]
    if not m.any():
        return float("nan"), float("nan")
    return best_f1_threshold(m, a)
