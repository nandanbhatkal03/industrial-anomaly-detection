# Lightweight Deep Learning-Based Visual Anomaly Detection for Industrial Defect Inspection

Detects and localises defects using models trained on **defect-free images only**,
and studies the **accuracy - localisation - efficiency** trade-off. Built from `Industrial_Anomaly_Detection_Project_Phase_I.md`;
**read `docs/DOCUMENT_CORRECTIONS.md` first** - it lists what was wrong in the specification and how this repo handles it.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                                                 # works offline (random weights, synthetic data)

# 1) data (licensed separately - see data/README.md)
python scripts/download_dataset.py --extract mvtec_anomaly_detection.tar.xz
python scripts/download_dataset.py --check

# 2) one model, one category (downloads ImageNet backbone weights on first use)
python scripts/train.py    --method lightpatch --category bottle
python scripts/evaluate.py --checkpoint models/checkpoints/lightpatch_resnet18_bottle.pt --category bottle
python scripts/infer.py    --checkpoint models/checkpoints/lightpatch_resnet18_bottle.pt --image path/to/img.png

# 3) full study (baselines + proposed) and ablations -> results/tables/summary.md, tradeoff.png
python scripts/run_experiments.py --methods padim patchcore lightpatch
python scripts/run_experiments.py --ablation backbone

# 4) dashboard / optional REST API
streamlit run app/app.py
uvicorn app.api:app --port 8000      # POST /predict?checkpoint=<name> (file upload)
```

Offline plumbing check (random weights, fake data) - add `--no-pretrained` to any script. No data yet? `python scripts/download_dataset.py --synthetic` then add
`--set data.categories=[synthetic] data.image_size=128` to any command (plumbing check only - not a result).

## Methods
| Key | What | Role |
|---|---|---|
| `autoencoder` | conv autoencoder, reconstruction error | reconstruction baseline |
| `padim` | per-position Gaussian + Mahalanobis (resnet18) | distribution baseline |
| `patchcore` | memory bank + greedy coreset (wide_resnet50_2) | accuracy reference |
| `student_teacher` | frozen teacher / trained student, feature discrepancy | student-teacher baseline |
| `efficientad_lite` | tiny PDN student-teacher with hard-feature loss (EfficientAD-**inspired**, not official) | lightweight neural baseline |
| `lightpatch` | **proposed**: light backbone + feature standardisation + random projection + small coreset + fp16 | research contribution (to be *proved* by ablation) |

Override anything: `--set models.lightpatch.backbone=mobilenet_v3_large models.lightpatch.layers=null models.lightpatch.proj_dim=64`.

## Layout
`configs/` YAML · `src/` library · `scripts/` CLIs · `app/` Streamlit · `tests/` pytest · `docs/` SRS/HLD/LLD/literature/corrections · `results/` outputs · `models/checkpoints/` weights.

## Honest limitations
* Numbers must be produced by **you** on real MVTec AD; nothing in this repo reports benchmark results.
* On CPU, `patchcore` (wide_resnet50_2, greedy coreset) is slow to fit; use a GPU or `coreset_method=random` for quick runs.
* The severity index is a heuristic, not validated against human grading.
* `efficientad_lite` is EfficientAD-*inspired* (see its docstring for the differences); for the official baseline use `anomalib`. `student_teacher` is STPM-style.
* Calibration from very few validation images is noisy (MVTec gives ~20 per category at val_fraction 0.1); report this and consider a larger split.
* Notebooks in `notebooks/` are thin wrappers over `src/` and have not been executed with real data.
