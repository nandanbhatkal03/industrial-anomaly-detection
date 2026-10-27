# Software Requirements Specification (SRS) - concise

## 1. Purpose and scope
A research prototype that detects and localises defects in industrial images using models trained on
defect-free images only, evaluates accuracy against efficiency, and exposes the best model through a local
inspection dashboard. Out of scope: PLC/robot integration, multi-camera sync, cloud deployment, real-time video.

## 2. Functional requirements
| ID | Requirement | Objective |
|---|---|---|
| FR-1 | Load MVTec-AD-format datasets; split `train/good` into fit and normal-validation sets deterministically. | 3 |
| FR-2 | Train/fit AE, PaDiM, PatchCore, student-teacher and LightPatch using normal images only. | 1, 3 |
| FR-3 | Produce an image-level score and a pixel-level anomaly map for any input image. | 3, 4 |
| FR-4 | Calibrate decision thresholds from normal validation images only and store them in the checkpoint. | 4 |
| FR-5 | Produce thresholded mask, cleaned regions (bbox, area, score) and a severity index. | 4 |
| FR-6 | Evaluate image AUROC/AP/P/R/F1/Acc and pixel AUROC/AUPRO/IoU/Dice/P/R. | 5 |
| FR-7 | Measure params, deployed size, latency (mean/p50/p95), FPS, peak memory, GFLOPs. | 5 |
| FR-8 | Run baseline and ablation sweeps and write CSV/Markdown tables and a trade-off plot. | 2, 5 |
| FR-9 | Dashboard: upload image(s), choose checkpoint, adjust sensitivity, show decision, score, heatmap, regions, severity; append to an inspection log. | 6 |
| FR-10 | CLI for train / evaluate / infer / benchmark. | 6 |

## 3. Non-functional requirements
| ID | Requirement |
|---|---|
| NFR-1 | Open-source software only (see `requirements.txt`); no paid APIs. Dataset is non-commercial licensed and not redistributed. |
| NFR-2 | Runs on CPU; GPU/MPS optional. |
| NFR-3 | Reproducible: fixed seeds, saved split manifests, config files, environment files, metrics JSON per run. |
| NFR-4 | Train/inference preprocessing identical (single `Preprocessor`, serialised into the checkpoint). |
| NFR-5 | No internet needed at inference (backbone weights stored in the checkpoint). |
| NFR-6 | Unit-tested core (`pytest`), including metric correctness. |

## 4. Data requirements
MVTec AD (primary; 3-5 categories first). Optional later: VisA, MVTec LOCO AD (same licence caveat).
Training: normal images only. Test: normal + anomalous with pixel masks.

## 5. Acceptance criteria
* All tests pass; end-to-end synthetic run completes.
* On MVTec AD (selected categories) a table with every metric in FR-6/FR-7 for every baseline and LightPatch.
* Dashboard returns a decision + heatmap in a time consistent with the measured latency.
