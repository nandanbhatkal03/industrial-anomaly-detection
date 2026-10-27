import numpy as np

from src.evaluation.image_metrics import image_auroc, threshold_metrics
from src.evaluation.pixel_metrics import aupro, pixel_auroc, segmentation_metrics
from src.postprocessing.threshold import best_f1_threshold, calibrate


def _toy(n=10, size=32, seed=0):
    rng = np.random.RandomState(seed)
    masks = np.zeros((n, size, size), bool)
    for i in range(n // 2):
        masks[i, 8:14, 8:14] = True
    return masks, rng


def test_image_metrics():
    y = np.array([0, 0, 1, 1]); s = np.array([.1, .2, .8, .9])
    assert image_auroc(y, s) == 1.0
    m = threshold_metrics(y, s, 0.5)
    assert m["f1"] == 1.0 and m["tp"] == 2 and m["tn"] == 2
    assert np.isnan(image_auroc([0, 0], [.1, .2]))


def test_pixel_auroc_and_aupro_perfect():
    masks, rng = _toy()
    perfect = masks.astype(float) + 0.01 * rng.rand(*masks.shape)
    assert pixel_auroc(masks, perfect) > 0.999
    assert aupro(masks, perfect) > 0.99


def test_aupro_random_is_low():
    masks, rng = _toy(n=20, size=64)
    v = aupro(masks, rng.rand(*masks.shape))
    assert 0.05 < v < 0.30   # analytic expectation for a random map = fpr_limit / 2 = 0.15


def test_segmentation_metrics():
    m = np.zeros((2, 10, 10), bool); m[0, :5, :5] = True
    p = m.copy()
    s = segmentation_metrics(m, p)
    assert s["iou"] == 1 and s["dice"] == 1
    p2 = np.zeros_like(m)
    assert segmentation_metrics(m, p2)["iou"] == 0


def test_calibration_uses_only_normal_scores_and_oracle_is_upper_bound():
    val_scores = np.array([1.0, 1.1, 0.9, 1.05, 0.95])
    val_maps = np.random.RandomState(0).rand(5, 8, 8)
    cal = calibrate(val_scores, val_maps, {"image_method": "mean_std", "image_k": 3, "pixel_percentile": 99.9})
    assert cal["image_threshold"] > val_scores.max() * 0.99
    y = np.array([0, 0, 0, 1, 1]); s = np.array([.1, .2, .3, .25, .9])
    thr, f1 = best_f1_threshold(y, s)
    assert f1 >= threshold_metrics(y, s, 0.5)["f1"]
