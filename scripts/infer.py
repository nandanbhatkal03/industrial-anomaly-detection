"""Run the full inspection pipeline on one image and save a visual report."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _common import base_parser, get_cfg

from src.inference.heatmap import draw_regions, overlay_heatmap, overlay_mask
from src.inference.predictor import Predictor


def main():
    p = base_parser(__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--out", default="results/visualizations/infer")
    a = p.parse_args()
    cfg = get_cfg(a)
    pred = Predictor(a.checkpoint, cfg.get("device", "auto"), cfg)
    r = pred.predict(a.image)
    print(json.dumps(r.to_dict(), indent=2))
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.8))
    ax[0].imshow(r.image); ax[0].set_title("input")
    ax[1].imshow(overlay_heatmap(r.image, r.norm_map)); ax[1].set_title(f"heatmap (score {r.score:.3f})")
    ax[2].imshow(draw_regions(overlay_mask(r.image, r.mask), r.regions))
    ax[2].set_title(f"{'ANOMALY' if r.is_anomalous else 'normal'} | severity: {r.severity}")
    for x in ax:
        x.axis("off")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{Path(a.image).stem}_report.png", dpi=120, bbox_inches="tight")
    print("saved", out / f"{Path(a.image).stem}_report.png")


if __name__ == "__main__":
    main()
