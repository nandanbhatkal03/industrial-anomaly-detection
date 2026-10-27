from __future__ import annotations

from pathlib import Path

import torch

from src.data.preprocessing import Preprocessor
from src.models import build_model

CKPT_VERSION = 1


def save_checkpoint(path, model, method: str, model_cfg: dict, preprocessor: Preprocessor,
                    calibration: dict, meta: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "version": CKPT_VERSION, "method": method, "model_cfg": dict(model_cfg),
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "artifacts": model.get_artifacts(), "preprocessor": preprocessor.to_dict(),
        "calibration": calibration, "meta": meta or {},
    }, path)
    return path


def load_checkpoint(path, device="cpu"):
    """Returns (model, preprocessor, calibration, meta). Backbone weights come from the checkpoint,
    so no internet access is needed at inference time."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(ck["method"], ck["model_cfg"], pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.set_artifacts(ck["artifacts"])
    model.to(device).eval()
    meta = {**ck.get("meta", {}), "method": ck["method"], "model_cfg": ck["model_cfg"]}
    return model, Preprocessor.from_dict(ck["preprocessor"]), ck["calibration"], meta
