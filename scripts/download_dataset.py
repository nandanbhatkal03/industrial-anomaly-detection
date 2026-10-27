"""Prepare MVTec AD (manual licensed download) or generate synthetic smoke-test data.

MVTec AD is licensed CC BY-NC-SA 4.0 (non-commercial research use) and must be downloaded by you from
https://www.mvtec.com/company/research/datasets/mvtec-ad  - this script never redistributes it.
"""
import tarfile
from pathlib import Path

from _common import base_parser, get_cfg

from src.data.dataset import MVTEC_CATEGORIES
from src.data.synthetic import generate_synthetic_mvtec
from src.utils.config import resolve_path


def check(root: Path, categories) -> bool:
    ok = True
    for c in categories:
        tr = len(list((root / c / "train" / "good").glob("*.*")))
        te = len(list((root / c / "test").rglob("*.*"))) if (root / c / "test").exists() else 0
        gt = len(list((root / c / "ground_truth").rglob("*.*"))) if (root / c / "ground_truth").exists() else 0
        status = "OK " if tr and te else "MISSING"
        ok &= bool(tr and te)
        print(f"[{status}] {c:<12} train/good={tr:<4} test={te:<4} gt_masks={gt}")
    return ok


def main():
    p = base_parser(__doc__)
    p.add_argument("--extract", help="path to mvtec_anomaly_detection.tar.xz you downloaded")
    p.add_argument("--synthetic", action="store_true", help="generate a tiny synthetic dataset (smoke tests only)")
    p.add_argument("--check", action="store_true", help="verify the folder structure")
    a = p.parse_args()
    cfg = get_cfg(a)
    root = resolve_path(cfg["paths"]["data_root"])
    if a.extract:
        root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(a.extract) as tf:
            tf.extractall(root)
        print(f"extracted to {root}")
    if a.synthetic:
        cat = generate_synthetic_mvtec(root, "synthetic", size=cfg["data"]["image_size"] // 2)
        print(f"synthetic data written to {cat}\nUse: --set data.categories=[synthetic] data.image_size=128")
    if a.check or not (a.extract or a.synthetic):
        cats = cfg["data"]["categories"] if root.exists() else MVTEC_CATEGORIES
        print(f"data_root = {root}")
        raise SystemExit(0 if check(root, cats) else 1)


if __name__ == "__main__":
    main()
