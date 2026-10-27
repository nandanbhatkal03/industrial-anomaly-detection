"""Proposed lightweight pipeline: LightPatch.

Design (each element maps to a research gap in docs/research_gap.md and is ablated by
`scripts/run_experiments.py --ablation <name>`):

  1. lightweight backbone, mid-level multi-scale features, local aggregation        (Gap 1, Gap 2)
  2. channel standardisation with NORMAL-only statistics (`feature_norm`)             (Gap 4)
  3. Gaussian random projection of the stored features (`proj_dim`)                   (Gap 1, Gap 5)
  4. small greedy coreset (`coreset_ratio`) stored in fp16 (`memory_dtype`)           (Gap 1, Gap 5)
  5. kNN scoring + Gaussian smoothing -> anomaly map; normal-only threshold calibration (Gap 3, Gap 6)

HONESTY NOTE: none of these is individually novel. The research contribution has to be the
*measured* accuracy / localisation / latency / size trade-off. If an ablation shows an element does
not help, report that - a clean negative result is a valid thesis finding.
"""
from __future__ import annotations

from .patchcore import PatchMemoryModel


class LightPatch(PatchMemoryModel):
    name = "lightpatch"

    def __init__(self, backbone="resnet18", layers=None, proj_dim=128, coreset_ratio=0.05, feature_norm=True,
                 memory_dtype="float16", **kw):
        super().__init__(backbone=backbone, layers=layers, proj_dim=proj_dim, coreset_ratio=coreset_ratio,
                         feature_norm=feature_norm, memory_dtype=memory_dtype, **kw)
