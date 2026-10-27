"""Evaluate a saved checkpoint on the MVTec test split (image, pixel and efficiency metrics)."""
from _common import base_parser, get_cfg

from src.evaluation.benchmark import evaluate_checkpoint, load_rows, write_summary
from src.utils.config import resolve_path


def main():
    p = base_parser(__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--category", required=True)
    p.add_argument("--no-vis", action="store_true")
    a = p.parse_args()
    cfg = get_cfg(a)
    evaluate_checkpoint(cfg, a.checkpoint, a.category, visualize=not a.no_vis)
    res = resolve_path(cfg["paths"]["results"])
    write_summary(load_rows(res), res)


if __name__ == "__main__":
    main()
