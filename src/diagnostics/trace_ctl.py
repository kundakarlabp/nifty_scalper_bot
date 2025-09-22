"""Runtime control helpers for time-limited trace logging."""
from __future__ import annotations

import os
import time
from typing import Optional

_TRACE_ON = False
_TRACE_UNTIL = 0.0


def _resolve_ttl(ttl_sec: Optional[int]) -> int:
    """Return a sane TTL using ``ttl_sec`` or environment defaults."""

    if ttl_sec is not None:
        try:
            return max(1, int(ttl_sec))
        except (TypeError, ValueError):
            return 600

    raw_env = os.getenv("TRACE_TTL_SEC")
    if raw_env is not None:
        try:
            return max(1, int(raw_env))
        except (TypeError, ValueError):
            return 600

    return 600


def enable(ttl_sec: Optional[int] = None) -> None:
    """Enable trace output until ``ttl_sec`` seconds from now."""

    global _TRACE_ON, _TRACE_UNTIL

    ttl = _resolve_ttl(ttl_sec)
    _TRACE_ON = True
    _TRACE_UNTIL = time.time() + ttl


def disable() -> None:
    """Disable trace output immediately."""

    global _TRACE_ON, _TRACE_UNTIL

    _TRACE_ON = False
    _TRACE_UNTIL = 0.0


def active() -> bool:
    """Return ``True`` if tracing is currently enabled and not expired."""

    if not _TRACE_ON:
        return False

    if time.time() <= _TRACE_UNTIL:
        return True

    disable()
    return False


def remaining() -> float:
    """Return seconds remaining for the current trace window (<= 0 if inactive)."""

    if not _TRACE_ON:
        return 0.0
    return max(0.0, _TRACE_UNTIL - time.time())
