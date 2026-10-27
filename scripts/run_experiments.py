"""Run baselines / the proposed model / ablations over categories and build the summary tables.

Examples
  python scripts/run_experiments.py --methods padim patchcore lightpatch --categories bottle carpet
  python scripts/run_experiments.py --ablation backbone --categories bottle
  python scripts/run_experiments.py --summary-only
"""
from _common import base_parser, get_cfg

from src.evaluation.benchmark import load_rows, run_experiment, write_summary
from src.utils.config import resolve_path
from src.utils.logger import get_logger

log = get_logger()
DEFAULT_METHODS = ["autoencoder", "padim", "patchcore", "student_teacher", "efficientad_lite", "lightpatch"]


def main():
    p = base_parser(__doc__)
    p.add_argument("--methods", nargs="*", default=None)
    p.add_argument("--categories", nargs="*", default=None)
    p.add_argument("--ablation", default=None, help="name of a group in configs/model.yaml -> ablation")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--summary-only", action="store_true")
    a = p.parse_args()
    cfg = get_cfg(a)
    res = resolve_path(cfg["paths"]["results"])
    cats = a.categories or cfg["data"]["categories"]
    if not a.summary_only:
        jobs = []
        if a.ablation:
            g = cfg["ablation"][a.ablation]
            for v in g["values"]:
                over = {g["param"]: v}
                if g["param"] == "backbone":
                    over["layers"] = None
                jobs += [(g["method"], c, f"abl-{g['param']}-{v}", over) for c in cats]
        else:
            jobs = [(m, c, None, None) for m in (a.methods or DEFAULT_METHODS) for c in cats]
        for i, (m, c, tag, over) in enumerate(jobs, 1):
            log.info(f"##### job {i}/{len(jobs)}: {m} / {c} / {tag}")
            try:
                run_experiment(cfg, m, c, tag, over, skip_existing=a.skip_existing)
            except Exception as e:  # keep the sweep alive; report at the end
                log.error(f"FAILED {m}/{c}/{tag}: {e}")
    rows = load_rows(res)
    write_summary(rows, res)
    log.info(f"summary written to {res / 'tables'} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
