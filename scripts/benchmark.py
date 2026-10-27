"""Latency / FPS / size / memory of a checkpoint. Run in a FRESH process so peak-RSS is meaningful."""
import json

from _common import base_parser, get_cfg

from src.evaluation import efficiency_metrics as eff
from src.training.checkpoint import load_checkpoint
from src.utils.config import resolve_device


def main():
    p = base_parser(__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--batch-size", type=int, default=None)
    a = p.parse_args()
    cfg = get_cfg(a)
    device = resolve_device(cfg.get("device", "auto"))
    model, pre, _, _ = load_checkpoint(a.checkpoint, device)
    b = cfg["benchmark"]
    out = {**eff.footprint(model, a.checkpoint),
           **eff.benchmark_latency(model, device, pre.out_size, b["warmup"], b["iters"], a.batch_size or b["batch_size"]),
           "gflops": eff.count_flops(model, pre.out_size, device)}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
