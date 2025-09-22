"""Runtime control helpers for time-limited trace logging."""
from __future__ import annotations

import os
import time
from typing import Optional

_TRACE_ON = False
_TRACE_UNTIL = 0.0


def enable(ttl_sec: Optional[int] = None) -> None:
    """Enable trace output until ``ttl_sec`` seconds from now.

    If ``ttl_sec`` is omitted, fall back to ``TRACE_TTL_SEC`` env var (default 600).
    """

    global _TRACE_ON, _TRACE_UNTIL

    ttl = int(ttl_sec or int(os.getenv("TRACE_TTL_SEC", "600")))
    _TRACE_ON = True
    _TRACE_UNTIL = time.time() + ttl


def disable() -> None:
    """Disable trace output immediately."""

    global _TRACE_ON, _TRACE_UNTIL

    _TRACE_ON = False
    _TRACE_UNTIL = 0.0


def active() -> bool:
    """Return ``True`` if tracing is currently enabled and not expired."""

    global _TRACE_ON

    if not _TRACE_ON:
        return False

    if time.time() <= _TRACE_UNTIL:
        return True

    # Auto-expire once the time limit has passed.
    disable()
    return False
