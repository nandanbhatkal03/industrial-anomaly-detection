from src.data.dataloader import build_datasets
from src.data.dataset import split_train_val


def test_splits_disjoint_deterministic(cfg):
    a, b = build_datasets(cfg, "synthetic"), build_datasets(cfg, "synthetic")
    tr = {s["path"].name for s in a["train"].samples}
    va = {s["path"].name for s in a["val"].samples}
    assert tr and va and not (tr & va)
    assert tr == {s["path"].name for s in b["train"].samples}
    assert len(a["train"]) + len(a["val"]) == 20


def test_test_split_labels_and_masks(cfg):
    ds = build_datasets(cfg, "synthetic")["test"]
    assert len(ds) == 16
    labels = [ds[i]["label"] for i in range(len(ds))]
    assert sum(labels) == 10
    for i in range(len(ds)):
        s = ds[i]
        assert s["image"].shape == (3, 64, 64) and s["mask"].shape == (1, 64, 64)
        assert set(s["mask"].unique().tolist()) <= {0.0, 1.0}
        assert (s["mask"].sum() > 0) == bool(s["label"])


def test_no_anomalous_in_training(cfg):
    ds = build_datasets(cfg, "synthetic")
    assert all(s["label"] == 0 for s in ds["train"].samples + ds["val"].samples)
    assert not split_train_val([], 0.1, 0)[1]
