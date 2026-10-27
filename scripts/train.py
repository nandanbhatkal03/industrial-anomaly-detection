"""Fit a model on normal images, calibrate thresholds on held-out normal images, save a checkpoint."""
from _common import base_parser, get_cfg

from src.evaluation.benchmark import train_and_calibrate
from src.utils.config import apply_overrides


def main():
    p = base_parser(__doc__)
    p.add_argument("--method", required=True, choices=["autoencoder", "padim", "patchcore", "student_teacher", "efficientad_lite", "lightpatch"])
    p.add_argument("--category", required=True)
    p.add_argument("--backbone", default=None)
    p.add_argument("--tag", default=None)
    a = p.parse_args()
    cfg = get_cfg(a)
    over = {"backbone": a.backbone} if a.backbone else {}
    if over and a.method == "lightpatch":
        over["layers"] = None  # use that backbone's default mid-level layers
    print(train_and_calibrate(cfg, a.method, a.category, a.tag, over))


if __name__ == "__main__":
    main()
