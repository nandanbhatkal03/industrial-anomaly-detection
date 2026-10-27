# Low-Level Design

| Module | Key API | Notes |
|---|---|---|
| `data/preprocessing.py` | `Preprocessor(image_size, center_crop, denoise)`; `image_to_tensor`, `mask_to_tensor`, `display_image`, `to_dict/from_dict` | resize (bilinear / nearest for masks) -> optional denoise -> ImageNet normalise |
| `data/dataset.py` | `MVTecDataset(root, category, split)`; `split_train_val` | split train/val/test; deterministic seeded val split |
| `models/feature_extractor.py` | `FeatureExtractor(backbone, layers)`, `fuse_features` | torchvision `create_feature_extractor`; resnet18/34/50, wide_resnet50_2, mobilenet_v2/v3, efficientnet_b0 |
| `models/padim.py` | `PaDiM.fit/predict` | streaming float64 mean/cov per position; Mahalanobis via `bmm` |
| `models/patchcore.py` | `PatchMemoryModel`, `PatchCore`, `greedy_coreset` | optional feature_norm, random projection, coreset (greedy/random), fp16 memory |
| `models/proposed_model.py` | `LightPatch` | proposed configuration |
| `models/student_teacher.py` | `StudentTeacher` | frozen teacher, trainable student, normalised feature discrepancy |
| `models/autoencoder.py` | `ConvAutoencoder` | 5 stride-2 stages, input divisible by 32 |
| `inference/anomaly_score.py` | `gaussian_blur`, `upsample_map`, `normalise_score` | normalised so threshold = 0.5 |
| `postprocessing/*` | `calibrate`, `clean_mask`, `extract_regions`, `severity_index` | |
| `evaluation/*` | `image_*`, `pixel_*` (incl. `aupro`), `efficiency_metrics`, `benchmark.run_experiment` | |
| `training/*` | `train_model`, `save_checkpoint`, `load_checkpoint` | |
| `inference/predictor.py` | `Predictor.predict(image) -> InspectionResult` | used by CLI and app |

## Checkpoint format (`.pt`, loaded with `weights_only=True`)
`{version, method, model_cfg, state_dict, artifacts, preprocessor, calibration, meta}`

## Anomaly score definition
`map = Gaussian_sigma( upsample( raw patch score ) )`, `score = max(map)`.
Decision: `score > image_threshold`. Mask: `map > pixel_threshold` -> open/close -> remove small components.
Severity index = `0.6 * clip((max_map - pixel_thr)/pixel_thr) + 0.4 * clip(area_ratio / extent_ref)` (heuristic, unvalidated).
