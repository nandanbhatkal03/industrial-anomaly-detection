# Research Gap and Contribution

## Research gaps

Based on the literature, the following research gaps can be
investigated.

## Gap 1 --- Accuracy vs Computational Cost

High-performing industrial anomaly-detection methods can require
substantial feature-extraction and memory resources. Lightweight models
reduce computational cost but may lose fine-grained feature
representation, particularly for small or subtle defects.

**Potential research direction:**\
Develop or adapt a lightweight feature-extraction/anomaly-detection
pipeline and quantify the trade-off between accuracy and computational
cost.

------------------------------------------------------------------------

## Gap 2 --- Small and Subtle Defect Localization

Industrial defects may occupy only a small portion of an image. A model
can therefore achieve good image-level classification while producing
poor pixel-level localization.

**Potential research direction:**\
Improve multi-scale feature handling or anomaly-map generation to
preserve small-defect information.

------------------------------------------------------------------------

## Gap 3 --- Limited Availability of Defect Samples

Real production environments may contain many normal images but
relatively few examples of each possible defect. Supervised
classification therefore becomes difficult when new defect types appear.

**Potential research direction:**\
Use a normal-only or predominantly normal training strategy and evaluate
performance on previously unseen anomaly types.

------------------------------------------------------------------------

## Gap 4 --- Domain Shift Between Pretrained Features and Industrial Images

Many methods use feature extractors pretrained on natural-image
datasets. Industrial surfaces and textures can differ considerably from
natural-image distributions.

**Potential research direction:**\
Investigate lightweight feature adaptation, feature normalization or
domain-specific feature selection.

------------------------------------------------------------------------

## Gap 5 --- Deployment-Oriented Evaluation

Many research studies emphasize anomaly-detection accuracy but do not
give equal importance to model size, memory usage, CPU/GPU latency and
practical deployment constraints.

**Potential research direction:**\
Include both **quality metrics** and **deployment metrics** as
first-class evaluation criteria.

------------------------------------------------------------------------

## Gap 6 --- Unified Inspection Workflow

Anomaly score generation alone does not necessarily provide a usable
inspection system. A practical solution should combine preprocessing,
detection, localization, visualization, thresholding and reporting.

**Potential research direction:**\
Develop an end-to-end open-source inspection prototype that connects the
research model to an operator-facing interface.

------------------------------------------------------------------------

## Proposed research contribution

The project should **not** claim novelty merely because a lightweight
backbone is used. A defensible contribution should be established
experimentally.

A suitable contribution direction is:

> **An experimentally validated lightweight industrial visual
> anomaly-detection pipeline that jointly considers anomaly detection
> accuracy, defect localization quality and deployment efficiency, with
> systematic comparison against established baselines.**

Potential contributions may include:

1.  Lightweight feature extraction for industrial anomaly detection.
2.  Efficient multi-scale feature selection.
3.  Reduced memory requirement for anomaly representation.
4.  Improved small-defect localization.
5.  Threshold calibration for practical inspection.
6.  Comprehensive accuracy-versus-latency evaluation.
7.  An open-source inspection prototype.

The exact final contribution should be determined after the Phase-I
literature survey and baseline experiments rather than being assumed
before experimentation.

------------------------------------------------------------------------

## How this repository addresses each gap

| Gap | Code | Experiment that evidences it |
|---|---|---|
| 1 Accuracy vs cost | `src/models/proposed_model.py` (LightPatch), `src/evaluation/efficiency_metrics.py` | `run_experiments.py --ablation backbone / proj_dim / coreset_ratio` + trade-off plot |
| 2 Small defects | multi-scale fusion in `feature_extractor.fuse_features`; AUPRO in `pixel_metrics.py` | per-defect-type results; AUPRO / pixel AUROC vs image AUROC |
| 3 Few defect samples | normal-only training + normal-only calibration (`threshold.calibrate`) | all models trained without anomalies |
| 4 Domain shift | `feature_norm` (normal-statistics standardisation) | `--ablation feature_norm` |
| 5 Deployment metrics | params, deployed size (incl. memory bank), latency p50/p95, FPS, peak RAM/GPU | `results/tables/summary.*` |
| 6 Unified workflow | `inference/predictor.py`, `app/app.py` | Streamlit demo, `tests/test_inference.py` |
