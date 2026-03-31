"""Plotly offline stub used during tests."""

from __future__ import annotations

from typing import Any


def plot(fig: Any, include_plotlyjs: str = "cdn", output_type: str = "div") -> str:
    """Return placeholder HTML for Plotly figures."""

    return "<div class='plotly-stub'></div>"


__all__ = ["plot"]
