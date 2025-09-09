"""Lightweight image operations with optional torch support.

This module avoids hard dependency on torch/torchvision. If those
libraries are unavailable, Pillow/NumPy operations are used instead.
"""

from __future__ import annotations

import logging
import platform
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)
_machine_logged = False

_TORCH_AVAILABLE = False
_read_image = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import torch  # noqa: F401
    from torchvision.io import read_image  # type: ignore

    _read_image = read_image
    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - we handle absence gracefully
    log.info("PyTorch not installed. Using CPU operations for image processing.")


def load_image(path: str | Path) -> np.ndarray:
    """Load image from *path* into an RGB NumPy array.

    Uses ``torchvision.io.read_image`` when available for performance; otherwise
    falls back to Pillow. Errors from the torch path are caught and the Pillow
    path is used instead so the caller does not need to care about optional
    dependencies.
    """
    global _machine_logged
    if not _machine_logged:
        log.info("machine: %s", platform.platform())
        _machine_logged = True

    p = Path(path)
    if _TORCH_AVAILABLE and _read_image is not None:
        try:
            return _read_image(str(p)).permute(1, 2, 0).numpy()
        except Exception:
            log.exception("Torch read_image failed. Falling back to Pillow.")
    with Image.open(p) as img:
        return np.asarray(img.convert("RGB"))


def save_image(array: np.ndarray, path: str | Path) -> None:
    """Save an RGB NumPy array to disk as PNG."""
    Image.fromarray(array.astype("uint8"), "RGB").save(path)
