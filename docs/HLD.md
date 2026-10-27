# High-Level Design

```text
 images (normal only) ──► Preprocessor ──► Backbone (frozen, mid-level layers)
                                              │ fuse: local aggregation + upsample + concat
                         ┌────────────────────┴──────────────────────┐
                         ▼                                           ▼
        Normal model fit (one pass or SGD)                    (inference) same embedding
   PaDiM: mean/cov  | PatchCore/LightPatch: memory bank       │
   AE / student-teacher: trained weights                      ▼
                         │                         score map (kNN / Mahalanobis / error)
                         ▼                                      │ upsample + Gaussian blur
              Checkpoint (weights + artefacts +                 ▼
              preprocessor + calibration)         image score = max(map)  +  anomaly map
                                                                │
        normal-validation scores ──► Calibration ──►  thresholds │
                                                                ▼
                                        mask → morphology → regions → severity → UI / report
```

## Design decisions
1. **One `Preprocessor`** for training, evaluation and the app; stored in the checkpoint.
2. **Common model interface** (`fit` / `predict -> (scores, maps)`), so any model plugs into the same evaluation and app code.
3. **Checkpoints are self-contained** (weights, fitted statistics, preprocessing, calibration) - evaluation always loads the *saved* file, so what is measured is what ships.
4. **Calibration uses normal data only**; oracle thresholds are reported separately.
5. **Efficiency is measured on the deployed artefact** (size includes memory bank / covariances).
6. **Proposed model (LightPatch)** = lightweight backbone + local aggregation + normal-statistics standardisation + Gaussian random projection + small coreset + fp16 storage; each element is an ablation switch.

## Technology
Python, PyTorch/torchvision, OpenCV, NumPy/SciPy, scikit-learn, Matplotlib, Streamlit, MLflow (optional), pytest.
