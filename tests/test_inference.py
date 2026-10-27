import json

import numpy as np

from src.evaluation.benchmark import load_rows, run_experiment, write_summary
from src.inference.predictor import Predictor
from src.postprocessing.localization import extract_regions
from src.postprocessing.morphology import clean_mask
from src.postprocessing.severity import severity_index, severity_level


def test_morphology_and_regions():
    m = np.zeros((40, 40), bool); m[5:15, 5:15] = True; m[30, 30] = True  # one blob + one speck
    c = clean_mask(m, 3, 3, min_area=10)
    assert c[10, 10] and not c[30, 30]
    amap = np.random.RandomState(0).rand(40, 40)
    regs = extract_regions(c, amap)
    assert len(regs) == 1 and regs[0]["bbox"][:2] == [5, 5] and regs[0]["area_px"] > 50


def test_severity_monotonic():
    lo = severity_index(1.1, 1.0, 0.001); hi = severity_index(2.0, 1.0, 0.1)
    assert 0 <= lo < hi <= 1
    assert severity_level(hi, True) == "high" and severity_level(hi, False) == "none"


def test_end_to_end_pipeline(cfg):
    """train -> calibrate -> save -> evaluate -> predictor -> summary, on synthetic data (random weights)."""
    cfg["models"]["lightpatch"].update(backbone="resnet18", coreset_ratio=0.1, coreset_method="random")
    cfg["pretrained"] = False  # no internet in CI
    row = run_experiment(cfg, "lightpatch", "synthetic", tag="t")
    import src.evaluation.benchmark as bm
    for k in ("image_auroc", "pixel_auroc", "aupro", "image_f1", "pixel_iou", "latency_ms_mean", "deployed_size_mb"):
        assert k in row and row[k] == row[k]  # present and not NaN
    assert row["deployed_size_mb"] >= row["artifact_mb"] > 0
    ck = next((bm.resolve_path(cfg["paths"]["checkpoints"])).glob("*.pt"))
    pred = Predictor(ck, "cpu", cfg)
    img = next((bm.resolve_path(cfg["paths"]["data_root"]) / "synthetic" / "test" / "blob").glob("*.png"))
    r = pred.predict(img)
    assert r.anomaly_map.shape == r.image.shape[:2] and 0 <= r.norm_score <= 1
    assert json.dumps(r.to_dict())  # JSON serialisable
    # stricter thresholds must never create more alarms
    assert pred.predict(img, image_threshold=1e9).is_anomalous is False
    res = bm.resolve_path(cfg["paths"]["results"])
    write_summary(load_rows(res), res)
    assert (res / "tables" / "summary.csv").exists() and (res / "visualizations" / "tradeoff.png").exists()


def test_rest_api(cfg, tmp_path, monkeypatch):
    """API loads a checkpoint, scores an uploaded image, rejects unknown/traversal names."""
    import io
    from fastapi.testclient import TestClient
    import app.api as api
    from src.evaluation.benchmark import train_and_calibrate
    cfg["pretrained"] = False
    cfg["models"]["lightpatch"].update(coreset_method="random", coreset_ratio=0.1)
    ck = train_and_calibrate(cfg, "lightpatch", "synthetic", "t")
    monkeypatch.setattr(api, "CK_DIR", ck.parent)
    monkeypatch.setattr(api, "cfg", cfg)
    api._predictor.cache_clear()
    c = TestClient(api.app)
    assert c.get("/health").json() == {"status": "ok"} and ck.stem in c.get("/models").json()["checkpoints"]
    img = next((api.Path(cfg["paths"]["data_root"]) / "synthetic" / "test" / "blob").glob("*.png"))
    r = c.post("/predict", params={"checkpoint": ck.stem}, files={"file": ("a.png", img.read_bytes(), "image/png")})
    assert r.status_code == 200 and "score" in r.json()
    assert c.post("/predict", params={"checkpoint": "../x"}, files={"file": ("a.png", b"x")}).status_code == 404
    assert c.post("/predict", params={"checkpoint": ck.stem}, files={"file": ("a.png", b"notanimage")}).status_code == 400
