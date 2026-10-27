# Objectives and Traceability


## Objective 1 --- Study Existing Industrial Visual Anomaly Detection Methods

To conduct a comprehensive study of traditional image-processing,
machine-learning and deep-learning approaches used for industrial visual
inspection and anomaly detection, including feature-based methods,
autoencoders, CNN feature extractors, memory-bank methods,
student-teacher models and recent lightweight architectures.

## Objective 2 --- Identify the Research Gap in Accurate and Computationally Efficient Anomaly Detection

To compare existing approaches using functional and performance
parameters such as anomaly-detection accuracy, localization quality,
training-data requirements, inference latency, model size, memory
consumption and suitability for resource-constrained deployment, and
identify limitations that can be addressed through a lightweight
architecture.

## Objective 3 --- Develop a Lightweight Visual Anomaly Detection Method

To design and implement a lightweight deep-learning-based anomaly
detection pipeline that learns representative features from normal
industrial images and detects deviations without requiring extensive
labelled defect examples.

## Objective 4 --- Localize and Quantify Defective Regions

To generate anomaly maps and localize defective regions at pixel/region
level, followed where feasible by an anomaly score or severity measure,
so that the system provides more useful inspection information than a
simple normal/defective classification.

## Objective 5 --- Evaluate Accuracy--Efficiency Trade-offs

To experimentally compare the proposed approach with suitable baseline
methods using image-level and pixel-level metrics such as AUROC,
F1-score, precision, recall, pixel-level AUROC/AUPRO/IoU where
applicable, together with inference latency, FPS, parameter count, model
size and memory consumption.

## Objective 6 --- Develop and Validate an Automated Inspection Prototype

To integrate the trained model into an open-source software prototype
that accepts industrial images, performs preprocessing and anomaly
inference, displays anomaly scores and heatmaps, reports detected
regions and records evaluation results, followed by systematic testing
and result analysis.

------------------------------------------------------------------------

## Traceability matrix (objective -> implementation -> how it is verified)

| Obj | Implemented in | Verified by |
|---|---|---|
| 1 Study methods | `docs/literature_review.md`; baselines: `autoencoder.py`, `padim.py`, `patchcore.py`, `student_teacher.py` | survey matrix; baselines run on the same protocol |
| 2 Research gap | `docs/research_gap.md`; comparison table `results/tables/summary.md` (accuracy, localisation, data need, latency, size, memory) | table filled with *measured* numbers, not literature claims |
| 3 Lightweight method | `src/models/proposed_model.py` (`LightPatch`) trained on normal images only | `tests/test_model.py`; ablations |
| 4 Localise + quantify | `heatmap`, `postprocessing/*` (threshold, morphology, regions, severity index) | `tests/test_inference.py`; pixel AUROC / AUPRO / IoU / Dice |
| 5 Accuracy-efficiency | `evaluation/*`, `scripts/run_experiments.py`, `visualization.plot_tradeoff` | `summary.csv/md`, `tradeoff.png` |
| 6 Prototype | `app/app.py` (+ `scripts/infer.py`), inspection log CSV | `tests/test_inference.py::test_end_to_end_pipeline`, manual UI test |

### Phase mapping
The specification lists baseline experimentation under Phase-I (MCSP34) but Objectives 3-6 are implementation
objectives that belong mainly to Phase-II (MCSP41). This repository is organised so that Phase-I = literature,
design docs, data pipeline, autoencoder / PaDiM / PatchCore baselines on 3-5 categories; Phase-II = LightPatch,
ablations, full evaluation, Streamlit prototype, research paper. Confirm the split with your guide.
