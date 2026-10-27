"""MVTec-AD-format dataset (also works for any dataset arranged the same way, e.g. the
synthetic data from src/data/synthetic.py).

Layout:  <root>/<category>/train/good/*.png
         <root>/<category>/test/<good|defect_type>/*.png
         <root>/<category>/ground_truth/<defect_type>/<name>_mask.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentation import build_train_augmentation
from .preprocessing import Preprocessor

MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
    "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT) if folder.is_dir() else []


def split_train_val(paths: list[Path], val_fraction: float, seed: int) -> tuple[list[Path], list[Path]]:
    """Deterministic split of defect-free images into fit / calibration sets."""
    paths = sorted(paths)
    if val_fraction <= 0 or len(paths) < 5:
        return paths, []
    n_val = max(1, int(round(len(paths) * val_fraction)))
    perm = np.random.RandomState(seed).permutation(len(paths))
    val_idx = set(perm[:n_val].tolist())
    train = [p for i, p in enumerate(paths) if i not in val_idx]
    val = [p for i, p in enumerate(paths) if i in val_idx]
    return train, val


class MVTecDataset(Dataset):
    def __init__(self, root, category: str, split: str, preprocessor: Preprocessor,
                 val_fraction: float = 0.1, seed: int = 42, augment_cfg: dict | None = None):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train | val | test")
        self.root = Path(root)
        self.category = category
        self.split = split
        self.pre = preprocessor
        self.augment = build_train_augmentation(augment_cfg) if split == "train" else None
        cat_dir = self.root / category
        if not cat_dir.is_dir():
            raise FileNotFoundError(
                f"Category folder not found: {cat_dir}\n"
                "Download MVTec AD (see data/README.md) or create synthetic data with "
                "`python scripts/download_dataset.py --synthetic`.")
        self.samples: list[dict] = []
        if split in {"train", "val"}:
            train_paths, val_paths = split_train_val(_images(cat_dir / "train" / "good"), val_fraction, seed)
            for p in (train_paths if split == "train" else val_paths):
                self.samples.append({"path": p, "mask": None, "label": 0, "defect_type": "good"})
        else:
            test_dir = cat_dir / "test"
            for sub in sorted(d for d in test_dir.iterdir() if d.is_dir()):
                for p in _images(sub):
                    if sub.name == "good":
                        self.samples.append({"path": p, "mask": None, "label": 0, "defect_type": "good"})
                    else:
                        m = cat_dir / "ground_truth" / sub.name / f"{p.stem}_mask.png"
                        self.samples.append({"path": p, "mask": m if m.exists() else None,
                                             "label": 1, "defect_type": sub.name})
        if not self.samples and split != "val":
            raise RuntimeError(f"No images found for {category}/{split} under {cat_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict:
        s = self.samples[i]
        img = Preprocessor.to_pil(s["path"])
        if self.augment is not None:
            img = self.augment(img)
        mask = self.pre.mask_to_tensor(s["mask"]) if s["mask"] is not None else self.pre.empty_mask()
        return {"image": self.pre.image_to_tensor(img), "mask": mask, "label": s["label"],
                "path": str(s["path"]), "defect_type": s["defect_type"]}


def save_split_manifest(root, category: str, val_fraction: float, seed: int, out_dir) -> Path:
    """Write the exact train/val file lists to data/splits for reproducibility."""
    train, val = split_train_val(_images(Path(root) / category / "train" / "good"), val_fraction, seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{category}_seed{seed}_val{val_fraction}.json"
    path.write_text(json.dumps({"train": [p.name for p in train], "val": [p.name for p in val]}, indent=1))
    return path
