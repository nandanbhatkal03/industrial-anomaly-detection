# Experimental Protocol

## Benchmark
MVTec AD, categories: `bottle, carpet, metal_nut, screw, transistor` (1 texture + 4 objects; edit in `configs/mvtec.yaml`).
Image size 256 (no crop). Seed 42. Training on `train/good` minus a 10% normal validation split.

## Models compared (same data, same protocol)
Autoencoder | PaDiM (resnet18) | PatchCore (wide_resnet50_2) | Student-teacher (resnet18) | **LightPatch** (proposed).
Optional external reference: EfficientAD via `anomalib`.

## Metrics
Image: AUROC, AP, precision, recall, F1, accuracy (at the calibrated threshold) + oracle best-F1.
Pixel: AUROC, AUPRO (FPR<=0.3), IoU, Dice, precision, recall + oracle pixel F1.
Efficiency: params, deployed size (params+artefacts), checkpoint size, latency mean/p50/p95 (batch 1), FPS, GFLOPs, peak RAM / GPU memory, train time.

## Rules that keep the study honest
1. Thresholds come from normal validation images only. Test labels are never used to tune anything deployable.
2. Hyper-parameters are chosen on a validation criterion, not on test AUROC; report the grid, not just the winner.
3. Same preprocessing, resolution and hardware for all models; record device and threads (done automatically).
4. Run `scripts/benchmark.py` in a fresh process for peak-memory numbers; close other heavy applications.
5. Report mean over categories **and** per-category tables; report per-defect-type failures for small defects.
6. One seed is not enough for the stochastic parts (AE, student-teacher, random coreset): repeat with 3+ seeds (`--set seed=1`) before claiming differences of <1 point.

## Commands
```bash
python scripts/run_experiments.py --methods padim patchcore lightpatch --categories bottle carpet
python scripts/run_experiments.py --ablation backbone      # also proj_dim, coreset_ratio, feature_norm, coreset_method
python scripts/run_experiments.py --summary-only           # rebuild tables from results/metrics/*.json
```
Outputs: `results/metrics/*.json`, `results/tables/summary.{csv,md}`, `results/visualizations/tradeoff.png`, per-run prediction grids.
