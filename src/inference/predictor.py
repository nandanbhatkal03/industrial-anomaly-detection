"""Inference pipeline used by scripts/infer.py, the evaluation code and the Streamlit app:
preprocess -> model -> normalised score/map -> thresholds -> morphology -> regions -> severity."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from src.postprocessing.localization import extract_regions
from src.postprocessing.morphology import clean_mask
from src.postprocessing.severity import severity_index, severity_level
from src.training.checkpoint import load_checkpoint
from src.utils.config import load_config, resolve_device
from .anomaly_score import normalise_score


@dataclass
class InspectionResult:
    image: np.ndarray                  # uint8 HxWx3, as seen by the model
    score: float
    norm_score: float                  # 0..1, decision boundary at 0.5
    is_anomalous: bool
    image_threshold: float
    pixel_threshold: float
    anomaly_map: np.ndarray            # raw float HxW
    norm_map: np.ndarray               # 0..1, pixel threshold at 0.5
    mask: np.ndarray                   # bool HxW (cleaned)
    regions: list = field(default_factory=list)
    severity_index: float = 0.0
    severity: str = "none"
    timing_ms: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"score": self.score, "norm_score": float(self.norm_score), "is_anomalous": bool(self.is_anomalous),
                "image_threshold": self.image_threshold, "pixel_threshold": self.pixel_threshold,
                "n_regions": len(self.regions), "regions": self.regions,
                "severity_index": self.severity_index, "severity": self.severity, "timing_ms": self.timing_ms}


class Predictor:
    def __init__(self, checkpoint_path, device: str = "auto", cfg: dict | None = None):
        self.cfg = cfg or load_config()
        self.device = resolve_device(device)
        self.model, self.pre, self.calibration, self.meta = load_checkpoint(checkpoint_path, self.device)
        self.checkpoint_path = str(checkpoint_path)

    @torch.no_grad()
    def predict(self, image, image_threshold: float | None = None, pixel_threshold: float | None = None,
                postprocess: bool = True) -> InspectionResult:
        pc = self.cfg["postprocess"]
        img_thr = float(image_threshold if image_threshold is not None else self.calibration["image_threshold"])
        pix_thr = float(pixel_threshold if pixel_threshold is not None else self.calibration["pixel_threshold"])
        t0 = time.perf_counter()
        display = self.pre.display_image(image)
        x = self.pre.image_to_tensor(image).unsqueeze(0).to(self.device)
        t1 = time.perf_counter()
        scores, maps = self.model.predict(x)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        score, amap = float(scores[0]), maps[0].float().cpu().numpy()
        anomalous = score > img_thr
        raw_mask = amap > pix_thr
        if postprocess:
            raw_mask = clean_mask(raw_mask, pc["open_kernel"], pc["close_kernel"], pc["min_region_area"])
        report = anomalous or not pc.get("localize_only_if_anomalous", True)
        mask = raw_mask if report else np.zeros_like(raw_mask)
        regions = extract_regions(mask, amap) if mask.any() else []
        sev = severity_index(float(amap.max()), pix_thr, float(mask.mean()), pc["severity_extent_ref"]) if anomalous else 0.0
        t3 = time.perf_counter()
        return InspectionResult(
            image=display, score=score, norm_score=float(normalise_score(score, img_thr)), is_anomalous=anomalous,
            image_threshold=img_thr, pixel_threshold=pix_thr, anomaly_map=amap,
            norm_map=normalise_score(amap, pix_thr), mask=mask, regions=regions, severity_index=sev,
            severity=severity_level(sev, anomalous),
            timing_ms={"preprocess": (t1 - t0) * 1e3, "model": (t2 - t1) * 1e3, "postprocess": (t3 - t2) * 1e3,
                       "total": (t3 - t0) * 1e3})
