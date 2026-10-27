# Completion status (what is proven, what is not)

| Objective | Code | Verified here | Still requires YOU |
|---|---|---|---|
| 1 Study methods | 6 methods, `literature_review.md` | all train/predict/checkpoint (pytest) | verify IEEE refs 5-10 |
| 2 Gap analysis | `research_gap.md`, comparison table generator | table generated from measured runs | run on real MVTec AD, then fill the survey matrix with measured numbers |
| 3 Lightweight method | `LightPatch` + 5 ablation groups | end-to-end CLI sweep + backbone ablation on synthetic | ablations on real data; pick the final configuration on validation data |
| 4 Localise + quantify | heatmap, mask, regions, severity | pytest, CLI infer, dashboard | judge the severity index on real defects |
| 5 Accuracy-efficiency | all metrics + trade-off plot | AUPRO/metrics unit tests, benchmark CLI | real numbers; fresh-process benchmark on your target hardware |
| 6 Prototype | Streamlit + REST API + log | headless dashboard load, API test | try with real uploads; GPU/CPU timing |

**Not verifiable in the build sandbox:** MVTec AD (licensed download) and pretrained ImageNet weights are unreachable
there, so every number produced during development came from random weights on fake textures and means nothing.
