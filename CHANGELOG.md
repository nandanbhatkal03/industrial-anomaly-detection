# Changelog
## 0.2.0
* `efficientad_lite` model, REST API (`app/api.py`), 7 notebooks, `--no-pretrained` offline switch
* Verified: 33 pytest tests, full CLI sweep (6 methods + backbone ablation) on synthetic data, headless dashboard load

## 0.1.0
* Data pipeline (MVTec format, deterministic normal validation split, synthetic smoke data)
* Models: autoencoder, PaDiM, PatchCore, student-teacher, LightPatch (proposed)
* Normal-only threshold calibration; mask/region/severity post-processing
* Metrics: image, pixel (AUROC, AUPRO, IoU, Dice), efficiency (params, deployed size, latency, FPS, memory, GFLOPs)
* Experiment runner + ablations + summary tables + trade-off plot
* Streamlit dashboard, CLI scripts, pytest suite
