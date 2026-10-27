# Notebooks

The experiments are script-driven (`scripts/`) so they are reproducible and testable. Planned exploratory
notebooks (create them when needed; import from `src/`):

1. `01_dataset_exploration.ipynb` - class balance, sample images, defect-size distribution
2. `02_preprocessing.ipynb` - effect of resize / denoise on small defects
3. `03_baseline_autoencoder.ipynb`, `04_padim_experiment.ipynb`, `05_patchcore_experiment.ipynb`
4. `06_lightweight_model.ipynb` - LightPatch variants
5. `07_result_analysis.ipynb` - read `results/tables/summary.csv`, per-defect-type analysis
