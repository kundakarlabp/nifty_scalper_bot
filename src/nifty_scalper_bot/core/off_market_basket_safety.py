"""Off-market active-basket mutation guards."""

from __future__ import annotations

from typing import Any


def apply_patches() -> None:
    """Install universe-controller guards. Implemented by the production fix."""


def apply_app_patch(app_module: Any) -> None:
    """Install app basket-commit guards. Implemented by the production fix."""


__all__ = ["apply_app_patch", "apply_patches"]
