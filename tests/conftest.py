import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.synthetic import generate_synthetic_mvtec  # noqa: E402
from src.utils.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def data_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("mvtec_like")
    generate_synthetic_mvtec(root, "synthetic", n_train=20, n_test_good=6, n_test_bad=10, size=64, seed=1)
    return root


@pytest.fixture()
def cfg(data_root, tmp_path):
    c = load_config()
    c["paths"].update(data_root=str(data_root), checkpoints=str(tmp_path / "ckpt"),
                      results=str(tmp_path / "results"), splits=str(tmp_path / "splits"))
    c["data"].update(image_size=64, batch_size=8, num_workers=0, categories=["synthetic"])
    c["device"] = "cpu"
    c["benchmark"].update(warmup=1, iters=2)
    c["evaluation"].update(num_visualizations=2)
    c["postprocess"]["min_region_area"] = 5
    return c
