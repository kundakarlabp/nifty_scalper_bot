import numpy as np
from PIL import Image

from src.utils import image_ops


def test_load_image_without_torchvision(tmp_path):
    img = Image.new("RGB", (4, 4), color=(10, 20, 30))
    path = tmp_path / "test.png"
    img.save(path)

    arr = image_ops.load_image(path)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (4, 4, 3)
    assert arr[0, 0, 0] == 10
    assert arr[0, 0, 1] == 20
    assert arr[0, 0, 2] == 30
