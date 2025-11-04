"""Prometheus client stub used during unit tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

CONTENT_TYPE_LATEST = "text/plain; version=0.0.4"


class CollectorRegistry:
    """Test stub emulating prometheus_client.CollectorRegistry."""

    def __init__(self) -> None:
        self._metrics: dict[str, Any] = {}


def generate_latest(_registry: CollectorRegistry | None = None) -> bytes:
    """Return an empty Prometheus payload."""

    return b""


class _MultiProcessCollector:
    """Stub for multiprocess registry collector."""

    def __init__(self, _registry: CollectorRegistry) -> None:
        return None


multiprocess = SimpleNamespace(MultiProcessCollector=_MultiProcessCollector)

__all__ = [
    "CONTENT_TYPE_LATEST",
    "CollectorRegistry",
    "generate_latest",
    "multiprocess",
]
