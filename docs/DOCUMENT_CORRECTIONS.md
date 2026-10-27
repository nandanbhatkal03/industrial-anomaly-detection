# Corrections to `Industrial_Anomaly_Detection_Project_Phase_I.md`

Every item below is something in the specification that was wrong, inconsistent, or would have caused a
flawed experiment. Items 1-4 are factual/formatting errors; 5-12 are methodological issues that would
weaken the thesis if left as written. Where a fix touched the spec text it is already applied in `docs/*.md`.

| # | Where | Problem | Correction |
|---|---|---|---|
| 1 | §10.1 | Stray tool-output tokens (`citeturn0search0turn0search12`, `citeturn0search1`) pasted into the text. | Removed. |
| 2 | §10 / §10.3 | Claims "pure open-source ... freely available public datasets" but **MVTec AD is CC BY-NC-SA 4.0** (non-commercial, share-alike) - not an open-source licence, and LOCO AD has the same licence. Verified on MVTec's own page. | Wording changed to "open-source *software* + a freely available *non-commercial research* dataset". Dataset is never committed or redistributed (`.gitignore`, `data/README.md`); results must carry the licence/attribution. |
| 3 | §11 item 11 | MVTec AD URL contains markdown escapes (`\_--\_`) so the link is broken. | Fixed in `literature_review.md`. |
| 4 | §3.5 | PatchCore described as using "locally **aggregated**" features; the paper's term is "locally **aware**" patch features (neighbourhood aggregation). | Wording corrected. |
| 5 | §3.2 / §11 | Only the 2019 CVPR MVTec paper is cited; the extended journal version (IJCV 2021, DOI 10.1007/s11263-020-01400-4) is the one MVTec asks users to cite as well. | Added. |
| 6 | §6.1-A, §7 | "Anomalous images reserved for validation/testing" - ambiguous. MVTec has **no anomalous validation set**; tuning thresholds/hyper-parameters on the test anomalies leaks labels and inflates F1/IoU. | Protocol fixed: deployable thresholds are calibrated **only on held-out normal validation images** (`calibration` in `configs/default.yaml`). Best-F1 "oracle" numbers are also reported, but labelled as upper bounds. |
| 7 | §7 | Model size / "parameters" listed as efficiency metrics. For PaDiM (per-position covariances) and PatchCore (memory bank) the network parameters hide most of the real footprint. | `deployed_size_mb` = parameters + fitted artefacts (memory bank / covariance / projection); reported next to params. Runtime memory and fp16-storage effects are measured separately. |
| 8 | §6.1-B / §9 | Data augmentation is listed as a standard preprocessing step (and Albumentations as a dependency). PaDiM / PatchCore use none, and geometric augmentation can make a *normal* object look anomalous (rotated/flipped screw, transistor, metal_nut). | Augmentation is **off by default**, photometric only; flips are opt-in. Albumentations dropped (torchvision is enough). |
| 9 | §9 vs §10.1 | Two overlapping technology tables that disagree (Albumentations, SciPy, JupyterLab appear in only one). | Merged into `requirements.txt` (single source of truth). |
| 10 | §7 | Latency metrics undefined - end-to-end vs model-only, batch size, warm-up, device. | Defined in `efficiency_metrics.py`: batch 1, warm-up, mean/p50/p95, GPU-synchronised, CPU thread count recorded; pipeline (pre/post) timing is reported separately by `Predictor`. |
| 11 | §7 | AUPRO / IoU / Dice are listed without definitions. | AUPRO: FPR limit 0.3, 8-connected regions (MVTec protocol). IoU/Dice: micro-averaged over all test pixels at the calibrated pixel threshold. All implemented in `pixel_metrics.py` with unit tests. |
| 12 | §6.1-D, §13 | The "proposed model" is never actually defined ("one of four candidates"), so Objective 3 cannot be implemented as written. §13 correctly warns against claiming novelty from a lightweight backbone. | `LightPatch` defined in `src/models/proposed_model.py` as an explicit, ablatable design; docs state that its value must be proven by measurement, and a negative result is acceptable. |

## Not changed, but you must act on it
* **IEEE references 5-10 (2023-2026) were not independently verified** - see the table in `literature_review.md`. A search for paper 10's exact title found nothing. Open each link and confirm before citing.
* The spec places "baseline experimentation" in Phase-I but lists implementation objectives (3-6) without a phase split - see the phase mapping in `objectives.md`; confirm with your guide.
* "Ubuntu Linux" in §10.1 is a recommendation, not a requirement; everything here also runs on Windows/macOS (CPU/CUDA/MPS).
* An official **EfficientAD** baseline is not included. `efficientad_lite` is an EfficientAD-*inspired* model (tiny PDN, hard-feature loss; no autoencoder branch, ResNet18-distilled teacher). For the official one use the Apache-2.0 `anomalib` library and cite the paper; `student_teacher` is STPM-style.
