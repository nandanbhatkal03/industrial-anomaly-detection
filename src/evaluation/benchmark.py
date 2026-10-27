"""End-to-end experiment runner: train -> calibrate (normal-only) -> save checkpoint ->
evaluate the SAVED checkpoint (image + pixel + efficiency metrics) -> write JSON/CSV/figures."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataloader import build_dataloaders, build_datasets
from src.data.preprocessing import Preprocessor
from src.evaluation import efficiency_metrics as eff
from src.evaluation.image_metrics import image_ap, image_auroc, threshold_metrics
from src.evaluation.pixel_metrics import aupro, best_pixel_f1, pixel_auroc, segmentation_metrics
from src.inference.anomaly_score import normalise_score
from src.models import build_model
from src.postprocessing.threshold import best_f1_threshold, calibrate
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.trainer import train_model
from src.utils.config import deep_update, resolve_device, resolve_path
from src.utils.logger import Tracker, get_logger
from src.utils.seed import set_seed
from src.utils.visualization import plot_tradeoff, save_prediction_grid

log = get_logger()


def make_tag(method: str, model_cfg: dict, tag: str | None = None) -> str:
    return tag or str(model_cfg.get("backbone", "default"))


def run_name(method: str, tag: str, category: str) -> str:
    return f"{method}_{tag}_{category}".replace("/", "-")


def checkpoint_path(cfg: dict, method: str, tag: str, category: str) -> Path:
    return resolve_path(cfg["paths"]["checkpoints"]) / f"{run_name(method, tag, category)}.pt"


@torch.no_grad()
def collect_predictions(model, loader: DataLoader, device) -> dict:
    model.eval().to(device)
    scores, maps, masks, labels, paths, types = [], [], [], [], [], []
    for b in loader:
        s, m = model.predict(b["image"].to(device))
        scores.append(s.float().cpu().numpy())
        maps.append(m.float().cpu().numpy())
        masks.append(b["mask"].squeeze(1).numpy() > 0.5)
        labels.append(b["label"].numpy())
        paths += list(b["path"])
        types += list(b["defect_type"])
    return {"scores": np.concatenate(scores), "maps": np.concatenate(maps), "masks": np.concatenate(masks),
            "labels": np.concatenate(labels), "paths": paths, "types": types}


# --------------------------------------------------------------------------- training
def train_and_calibrate(cfg: dict, method: str, category: str, tag: str | None = None,
                        model_overrides: dict | None = None, device=None) -> Path:
    device = device or resolve_device(cfg.get("device", "auto"))
    model_cfg = deep_update(cfg["models"][method], model_overrides or {})
    tag = make_tag(method, model_cfg, tag)
    set_seed(cfg.get("seed", 42))
    pre = Preprocessor.from_config(cfg["data"])
    loaders = build_dataloaders(cfg, category, pre)
    model = build_model(method, model_cfg, pretrained=cfg.get("pretrained", True))
    log.info(f"== {run_name(method, tag, category)} on {device} | train={len(loaders['train'].dataset)} "
             f"val={len(loaders['val'].dataset) if 'val' in loaders else 0}")
    info = train_model(model, method, cfg, loaders["train"], loaders.get("val"), device)

    cal_loader = loaders.get("val")
    on_train = cal_loader is None
    if on_train:
        log.warning("No validation split (val_fraction=0 or too few images): calibrating on TRAIN images - optimistic!")
        cal_loader = DataLoader(loaders["train"].dataset, batch_size=cfg["data"].get("batch_size", 16))
    cal_pred = collect_predictions(model, cal_loader, device)
    calibration = calibrate(cal_pred["scores"], cal_pred["maps"], cfg["calibration"])
    calibration["calibrated_on_train"] = on_train
    meta = {"category": category, "tag": tag, "train_time_s": info["train_time_s"], "seed": cfg.get("seed"),
            "torch": str(torch.__version__), "device": str(device), "n_train": len(loaders["train"].dataset),
            "history": info["history"][-1:] if info["history"] else []}
    path = save_checkpoint(checkpoint_path(cfg, method, tag, category), model, method, model_cfg, pre, calibration, meta)
    log.info(f"saved {path.name}: image_thr={calibration['image_threshold']:.4f} pixel_thr={calibration['pixel_threshold']:.4f}")
    return path


# --------------------------------------------------------------------------- evaluation
def evaluate_checkpoint(cfg: dict, ckpt_path, category: str, device=None, visualize: bool = True) -> dict:
    device = device or resolve_device(cfg.get("device", "auto"))
    ckpt_path = Path(ckpt_path)
    model, pre, cal, meta = load_checkpoint(ckpt_path, device)
    method, tag = meta["method"], meta.get("tag", "default")
    ev = cfg["evaluation"]
    test_ds = build_datasets(cfg, category, pre)["test"]
    pred = collect_predictions(model, DataLoader(test_ds, batch_size=cfg["data"].get("batch_size", 16)), device)
    y, s, m, a = pred["labels"], pred["scores"], pred["masks"], pred["maps"]

    row = {"category": category, "method": method, "tag": tag, "backbone": meta["model_cfg"].get("backbone", "-"),
           "run": run_name(method, tag, category), "n_test": int(len(y)), "n_anomalous": int(y.sum()),
           "image_auroc": image_auroc(y, s), "image_ap": image_ap(y, s),
           "pixel_auroc": pixel_auroc(m, a),
           "aupro": aupro(m, a, ev["aupro_fpr_limit"], ev["aupro_thresholds"]),
           "image_threshold": cal["image_threshold"], "pixel_threshold": cal["pixel_threshold"]}
    for k, v in threshold_metrics(y, s, cal["image_threshold"]).items():
        row[f"image_{k}"] = v
    for k, v in segmentation_metrics(m, a > cal["pixel_threshold"]).items():
        row[f"pixel_{k}"] = v
    row["oracle_image_f1"] = best_f1_threshold(y, s)[1] if len(np.unique(y)) == 2 else float("nan")
    row["oracle_pixel_f1"] = best_pixel_f1(m, a, ev["max_pixels_for_oracle"])[1]

    b = cfg["benchmark"]
    row.update(eff.footprint(model, ckpt_path))
    row.update(eff.benchmark_latency(model, device, pre.out_size, b["warmup"], b["iters"], b["batch_size"]))
    row["gflops"] = eff.count_flops(model, pre.out_size, device)
    row["train_time_s"] = meta.get("train_time_s")

    out_dir = resolve_path(cfg["paths"]["results"])
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics" / f"{row['run']}.json").write_text(json.dumps({"row": row, "calibration": cal}, indent=1))
    if ev.get("save_anomaly_maps"):
        (out_dir / "anomaly_maps").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / "anomaly_maps" / f"{row['run']}.npz", maps=a.astype(np.float16), scores=s, labels=y)
    if visualize:
        _save_visualizations(out_dir / "visualizations" / row["run"], pre, pred, cal, ev["num_visualizations"])
    log.info(f"{row['run']}: imgAUROC={row['image_auroc']:.4f} pxAUROC={row['pixel_auroc']:.4f} "
             f"AUPRO={row['aupro']:.4f} F1={row['image_f1']:.3f} {row['latency_ms_mean']:.1f}ms "
             f"size={row['deployed_size_mb']:.1f}MB")
    return row


def _save_visualizations(out_dir: Path, pre: Preprocessor, pred: dict, cal: dict, n: int) -> None:
    from PIL import Image
    y = pred["labels"]
    bad, good = np.where(y == 1)[0], np.where(y == 0)[0]
    pick = list(bad[np.linspace(0, len(bad) - 1, min(n // 2 + n % 2, len(bad))).astype(int)]) if len(bad) else []
    pick += list(good[np.linspace(0, len(good) - 1, min(n // 2, len(good))).astype(int)]) if len(good) else []
    for i in pick:
        img = pre.display_image(Image.open(pred["paths"][i]))
        nm = normalise_score(pred["maps"][i], cal["pixel_threshold"])
        save_prediction_grid(out_dir / f"{Path(pred['paths'][i]).parent.name}_{Path(pred['paths'][i]).stem}.png", img,
                             pred["masks"][i], nm, pred["maps"][i] > cal["pixel_threshold"],
                             f"{pred['types'][i]} | score={pred['scores'][i]:.3f} (thr {cal['image_threshold']:.3f})")


def run_experiment(cfg: dict, method: str, category: str, tag: str | None = None, model_overrides: dict | None = None,
                   device=None, skip_existing: bool = False, visualize: bool = True) -> dict:
    model_cfg = deep_update(cfg["models"][method], model_overrides or {})
    tag = make_tag(method, model_cfg, tag)
    name = run_name(method, tag, category)
    done = resolve_path(cfg["paths"]["results"]) / "metrics" / f"{name}.json"
    if skip_existing and done.exists():
        log.info(f"skip {name} (exists)")
        return json.loads(done.read_text())["row"]
    tr = cfg.get("tracking", {})
    tracker = Tracker(tr.get("mlflow", False), tr.get("experiment_name", "iad"), name)
    tracker.log_params({"method": method, "category": category, "tag": tag, **{f"model.{k}": v for k, v in model_cfg.items()}})
    t0 = time.perf_counter()
    ckpt = train_and_calibrate(cfg, method, category, tag, model_overrides, device)
    row = evaluate_checkpoint(cfg, ckpt, category, device, visualize)
    row["wall_time_s"] = time.perf_counter() - t0
    tracker.log_metrics({k: v for k, v in row.items() if isinstance(v, (int, float))})
    tracker.end()
    return row


# --------------------------------------------------------------------------- reporting
SUMMARY_COLS = ["image_auroc", "pixel_auroc", "aupro", "image_f1", "pixel_iou", "latency_ms_mean", "fps",
                "params_m", "deployed_size_mb"]


def _f(v) -> float:
    return float("nan") if v is None else float(v)


def load_rows(results_dir) -> list[dict]:
    return [json.loads(p.read_text())["row"] for p in sorted((Path(results_dir) / "metrics").glob("*.json"))]


def write_summary(rows: list[dict], results_dir) -> None:
    if not rows:
        return
    tdir = Path(results_dir) / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(tdir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    # per (method, tag) mean over categories -> the table used in the thesis
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["method"], r["tag"]), []).append(r)
    lines = ["| method | tag | #cat | " + " | ".join(SUMMARY_COLS) + " |", "|" + "---|" * (len(SUMMARY_COLS) + 3)]
    for (me, tg), rs in sorted(groups.items()):
        vals = [np.nanmean([_f(r.get(c)) for r in rs]) for c in SUMMARY_COLS]
        lines.append(f"| {me} | {tg} | {len(rs)} | " + " | ".join(f"{v:.4g}" for v in vals) + " |")
    (tdir / "summary.md").write_text("\n".join(lines) + "\n")
    plot_tradeoff(rows, Path(results_dir) / "visualizations" / "tradeoff.png")
