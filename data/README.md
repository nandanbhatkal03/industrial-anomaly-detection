# Data

**MVTec AD** is licensed **CC BY-NC-SA 4.0** (non-commercial research use, attribution, share-alike).
Download it yourself from <https://www.mvtec.com/company/research/datasets/mvtec-ad> and cite both papers
(CVPR 2019, DOI 10.1109/CVPR.2019.00982; IJCV 2021, DOI 10.1007/s11263-020-01400-4).
Do **not** commit it to Git or redistribute it.

```bash
python scripts/download_dataset.py --extract /path/to/mvtec_anomaly_detection.tar.xz   # -> data/raw/mvtec_ad
python scripts/download_dataset.py --check
```

Expected layout: `data/raw/mvtec_ad/<category>/{train/good, test/<type>, ground_truth/<type>}`.

Smoke-test without the real data: `python scripts/download_dataset.py --synthetic` (tiny fake textures; plumbing check only).

`data/splits/` stores the exact train/validation file lists used for each run.
