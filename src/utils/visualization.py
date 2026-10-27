from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.inference.heatmap import overlay_heatmap


def save_prediction_grid(path, image, gt_mask, norm_map, pred_mask, title: str = "") -> None:
    """image | ground truth | heatmap overlay | predicted mask."""
    fig, ax = plt.subplots(1, 4, figsize=(12, 3.3))
    ax[0].imshow(image); ax[0].set_title("input")
    ax[1].imshow(gt_mask, cmap="gray", vmin=0, vmax=1); ax[1].set_title("ground truth")
    ax[2].imshow(overlay_heatmap(image, norm_map)); ax[2].set_title("anomaly heatmap")
    ax[3].imshow(pred_mask, cmap="gray", vmin=0, vmax=1); ax[3].set_title("predicted mask")
    for a in ax:
        a.axis("off")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_tradeoff(rows: list[dict], path, x="latency_ms_mean", y="image_auroc", size="deployed_size_mb") -> None:
    """Accuracy-vs-latency scatter (Objective 5): one point per (method, tag), averaged over categories."""
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(f"{r['method']}:{r['tag']}", []).append(r)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, rs in groups.items():
        xs, ys, ss = (float(np.nanmean([float(r[k]) if r.get(k) is not None else np.nan for r in rs])) for k in (x, y, size))
        ax.scatter(xs, ys, s=40 + 4 * ss, alpha=0.7)
        ax.annotate(name, (xs, ys), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel(x.replace("_", " ") + " (lower is better)")
    ax.set_ylabel(y.replace("_", " ") + " (higher is better)")
    ax.set_title("Accuracy-efficiency trade-off (marker area ~ deployed size)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
