import pytest
import torch
from torch.utils.data import DataLoader

from src.data.dataloader import build_datasets
from src.models import build_model
from src.models.feature_extractor import DEFAULT_LAYERS, FeatureExtractor
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.trainer import train_model

DEV = torch.device("cpu")


@pytest.fixture()
def loaders(cfg):
    ds = build_datasets(cfg, "synthetic")
    return {k: DataLoader(v, batch_size=8) for k, v in ds.items()}


@pytest.mark.parametrize("name", sorted(DEFAULT_LAYERS))
def test_backbone_feature_shapes(name):
    fe = FeatureExtractor(name, pretrained=False)
    outs = fe(torch.zeros(1, 3, 128, 128))
    assert len(outs) == 2 and outs[0].shape[-1] == 2 * outs[1].shape[-1]  # stride 8 then 16
    assert fe.strides == [8, 16]


def _cfg_for(cfg, method, **kw):
    m = dict(cfg["models"][method]); m.update(kw)
    if "backbone" in m:
        m["backbone"] = "resnet18"
    return m


@pytest.mark.parametrize("method,kw", [
    ("padim", {"reduced_dim": 20}),
    ("patchcore", {"coreset_ratio": 0.05}),
    ("lightpatch", {"coreset_ratio": 0.05, "proj_dim": 32}),
    ("lightpatch", {"coreset_method": "random", "feature_norm": False, "proj_dim": None}),
])
def test_one_pass_models_fit_predict_and_checkpoint(cfg, loaders, tmp_path, method, kw):
    mcfg = _cfg_for(cfg, method, **kw)
    model = build_model(method, mcfg, pretrained=False)
    train_model(model, method, cfg, loaders["train"], None, DEV)
    x = next(iter(loaders["test"]))["image"]
    s, m = model.predict(x)
    assert s.shape == (x.shape[0],) and m.shape == (x.shape[0], 64, 64)
    assert torch.isfinite(s).all() and (m >= 0).all()
    assert model.artifact_bytes() > 0
    p = save_checkpoint(tmp_path / "m.pt", model, method, mcfg, build_datasets(cfg, "synthetic")["test"].pre, {"image_threshold": 1.0, "pixel_threshold": 1.0})
    model2, _, _, _ = load_checkpoint(p, DEV)
    s2, _ = model2.predict(x)
    assert torch.allclose(s, s2, atol=1e-4)


def test_coreset_ratio_and_fp16_size(cfg, loaders):
    big = build_model("lightpatch", _cfg_for(cfg, "lightpatch", coreset_ratio=0.2, coreset_method="random"), pretrained=False)
    small = build_model("lightpatch", _cfg_for(cfg, "lightpatch", coreset_ratio=0.02, coreset_method="random"), pretrained=False)
    for m in (big, small):
        m.fit(loaders["train"], DEV)
    assert big.memory.shape[0] > 5 * small.memory.shape[0]
    assert big.memory.dtype == torch.float16
    assert big.memory.shape[1] == 32 or big.memory.shape[1] == 128


@pytest.mark.parametrize("method", ["autoencoder", "student_teacher"])
def test_gradient_models_train_and_loss_decreases(cfg, loaders, method):
    cfg["training"][method]["epochs"] = 3
    mcfg = _cfg_for(cfg, method)
    model = build_model(method, mcfg, pretrained=False)
    info = train_model(model, method, cfg, loaders["train"], loaders["val"], DEV)
    h = info["history"]
    assert len(h) == 3 and h[-1]["train_loss"] <= h[0]["train_loss"] * 1.05
    s, m = model.predict(next(iter(loaders["test"]))["image"])
    assert m.shape[-2:] == (64, 64) and torch.isfinite(s).all()


def test_unknown_method_and_backbone():
    with pytest.raises(ValueError):
        build_model("nope")
    with pytest.raises(ValueError):
        FeatureExtractor("vgg_unknown", pretrained=False)


def test_efficientad_lite_fit_predict_checkpoint(cfg, loaders, tmp_path):
    mcfg = {"out_channels": 16, "teacher_epochs": 2, "student_epochs": 2, "lr": 1e-3}
    model = build_model("efficientad_lite", mcfg, pretrained=False)
    train_model(model, "efficientad_lite", cfg, loaders["train"], None, DEV)
    x = next(iter(loaders["test"]))["image"]
    s, m = model.predict(x)
    assert m.shape == (x.shape[0], 64, 64) and torch.isfinite(s).all()
    pre = build_datasets(cfg, "synthetic")["test"].pre
    p = save_checkpoint(tmp_path / "e.pt", model, "efficientad_lite", mcfg, pre, {"image_threshold": 1.0, "pixel_threshold": 1.0})
    m2, _, _, _ = load_checkpoint(p, DEV)
    assert torch.allclose(s, m2.predict(x)[0], atol=1e-4)
