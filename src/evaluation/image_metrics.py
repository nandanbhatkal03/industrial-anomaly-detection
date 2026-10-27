from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def image_auroc(labels, scores) -> float:
    labels = np.asarray(labels)
    return float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else float("nan")


def image_ap(labels, scores) -> float:
    labels = np.asarray(labels)
    return float(average_precision_score(labels, scores)) if len(np.unique(labels)) == 2 else float("nan")


def threshold_metrics(labels, scores, threshold: float) -> dict:
    y = np.asarray(labels).astype(bool)
    p = np.asarray(scores) > threshold
    tp, fp, fn, tn = int((y & p).sum()), int((~y & p).sum()), int((y & ~p).sum()), int((~y & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "accuracy": (tp + tn) / max(len(y), 1),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}
