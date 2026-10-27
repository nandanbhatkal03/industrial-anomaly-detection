import numpy as np
import torch
from PIL import Image

from src.data.preprocessing import Preprocessor


def test_shapes_and_roundtrip():
    pre = Preprocessor(image_size=48)
    arr = (np.random.RandomState(0).rand(30, 40, 3) * 255).astype(np.uint8)
    t = pre.image_to_tensor(arr)
    assert t.shape == (3, 48, 48)
    disp = pre.display_image(arr)
    assert disp.shape == (48, 48, 3)
    assert np.abs(pre.denormalize(t).astype(int) - disp.astype(int)).max() <= 1


def test_grayscale_and_center_crop():
    pre = Preprocessor(image_size=64, center_crop=32)
    g = Image.fromarray(np.zeros((20, 20), np.uint8))
    assert pre.image_to_tensor(g).shape == (3, 32, 32)
    assert pre.empty_mask().shape == (1, 32, 32)


def test_mask_stays_binary():
    pre = Preprocessor(image_size=50)
    m = np.zeros((100, 100), np.uint8); m[20:40, 20:40] = 255
    t = pre.mask_to_tensor(Image.fromarray(m))
    assert set(t.unique().tolist()) <= {0.0, 1.0} and t.sum() > 0


def test_serialisation_and_denoise():
    pre = Preprocessor(64, None, "gaussian")
    pre2 = Preprocessor.from_dict(pre.to_dict())
    x = np.random.RandomState(1).randint(0, 255, (64, 64, 3), dtype=np.uint8)
    assert torch.allclose(pre.image_to_tensor(x), pre2.image_to_tensor(x))
