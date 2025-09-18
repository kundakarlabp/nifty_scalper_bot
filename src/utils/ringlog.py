"""A lightweight append-only ring buffer for structured log records."""

from __future__ import annotations

from collections import deque
from typing import Any, Deque

from src.config import settings


def _bool_env(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "on"}


def _capacity() -> int:
    size = getattr(settings, "DIAG_RING_SIZE", None) or getattr(
        settings, "diag_ring_size", None
    )
    try:
        return max(int(size), 1) if size is not None else 512
    except (TypeError, ValueError):
        return 512


_BUFFER: Deque[dict[str, Any]] = deque(maxlen=_capacity())


def enabled() -> bool:
    """Return ``True`` when the diagnostics ring buffer should capture logs."""

    flag = getattr(settings, "LOG_RING_ENABLED", None)
    if flag is None:
        flag = getattr(settings, "log_ring_enabled", False)
    return _bool_env(flag)


def append(record: dict[str, Any]) -> None:
    """Append a structured record to the diagnostics ring buffer."""

    if not enabled():
        return
    _BUFFER.append(dict(record))


def tail(limit: int | None = None) -> list[dict[str, Any]]:
    """Return the newest ``limit`` records (or all records when ``None``)."""

    data = list(_BUFFER)
    if limit is None or limit >= len(data):
        return data
    if limit <= 0:
        return []
    return data[-limit:]


def snapshot() -> list[dict[str, Any]]:
    """Return a copy of the entire ring buffer."""

    return tail(None)


def clear() -> None:
    """Remove all records from the ring buffer."""

    _BUFFER.clear()


def capacity() -> int:
    """Return the maximum number of records stored in the buffer."""

    return _BUFFER.maxlen or _capacity()


__all__ = ["append", "tail", "snapshot", "clear", "capacity", "enabled"]

