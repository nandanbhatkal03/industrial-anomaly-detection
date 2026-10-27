"""YAML config loading with deep-merge and command-line overrides."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIGS = ["configs/default.yaml", "configs/mvtec.yaml", "configs/model.yaml", "configs/training.yaml"]


def deep_update(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(paths: list[str] | None = None, overrides: list[str] | None = None) -> dict:
    cfg: dict = {}
    for p in paths or DEFAULT_CONFIGS:
        p = Path(p)
        if not p.is_absolute():
            p = ROOT / p
        with open(p, "r", encoding="utf-8") as f:
            cfg = deep_update(cfg, yaml.safe_load(f) or {})
    return apply_overrides(cfg, overrides or [])


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply 'a.b.c=value' overrides; the value is parsed as YAML (so 0.1, null, [1,2] work)."""
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got: {item}")
        key, raw = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def resolve_path(cfg_path: str | Path) -> Path:
    p = Path(cfg_path)
    return p if p.is_absolute() else ROOT / p


def resolve_device(name: str = "auto"):
    import torch

    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)
