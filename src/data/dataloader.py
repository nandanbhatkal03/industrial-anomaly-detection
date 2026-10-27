from __future__ import annotations

from torch.utils.data import DataLoader

from src.utils.config import resolve_path
from .dataset import MVTecDataset, save_split_manifest
from .preprocessing import Preprocessor


def build_datasets(cfg: dict, category: str, preprocessor: Preprocessor | None = None) -> dict:
    d = cfg["data"]
    pre = preprocessor or Preprocessor.from_config(d)
    root = resolve_path(cfg["paths"]["data_root"])
    common = dict(root=root, category=category, preprocessor=pre,
                  val_fraction=d.get("val_fraction", 0.1), seed=cfg.get("seed", 42))
    ds = {
        "train": MVTecDataset(split="train", augment_cfg=d.get("augment"), **common),
        "val": MVTecDataset(split="val", **common),
        "test": MVTecDataset(split="test", **common),
    }
    save_split_manifest(root, category, common["val_fraction"], common["seed"], resolve_path(cfg["paths"]["splits"]))
    return ds


def build_dataloaders(cfg: dict, category: str, preprocessor: Preprocessor | None = None) -> dict:
    d = cfg["data"]
    ds = build_datasets(cfg, category, preprocessor)
    bs, nw = d.get("batch_size", 16), d.get("num_workers", 0)
    return {k: DataLoader(v, batch_size=bs, shuffle=(k == "train"), num_workers=nw, drop_last=False)
            for k, v in ds.items() if len(v) > 0}
