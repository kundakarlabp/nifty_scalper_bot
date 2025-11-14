"""Minimal Plotly stub for unit tests."""

from __future__ import annotations

from typing import Any, Dict, List


class _Scatter:
    """Test stub emulating plotly.graph_objects.Scatter."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs: Dict[str, Any] = dict(kwargs)


class _Figure:
    """Test stub emulating plotly.graph_objects.Figure."""

    def __init__(self) -> None:
        self.traces: List[Any] = []
        self.layout: Dict[str, Any] = {}

    def add_trace(self, trace: Any) -> None:
        self.traces.append(trace)

    def update_layout(self, **kwargs: Any) -> None:
        self.layout.update(kwargs)


class _GraphObjects:
    """Container providing Figure and Scatter constructors."""

    Figure = _Figure
    Scatter = _Scatter


graph_objects = _GraphObjects()

__all__ = ["graph_objects"]
