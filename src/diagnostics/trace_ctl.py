"""Helpers to toggle diagnostic trace capture with an auto-expiring window."""

from __future__ import annotations

import os
import threading
import time
from typing import Optional

def _default_ttl() -> int:
    raw = os.getenv("TRACE_TTL_SEC")
    if raw is None:
        return 600
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 600
    return max(value, 0)


TRACE_TTL_SEC = _default_ttl()

_lock = threading.Lock()
_expires_at: float = 0.0
_timer: Optional[threading.Timer] = None


def _cancel_timer() -> None:
    global _timer
    timer = _timer
    if timer is not None:
        timer.cancel()
        _timer = None


def _schedule_disable(delay: float) -> None:
    global _timer
    _cancel_timer()
    if delay <= 0:
        return
    timer = threading.Timer(delay, disable)
    timer.daemon = True
    timer.start()
    _timer = timer


def enable(ttl: Optional[int] = None) -> float:
    """Enable trace capture for ``ttl`` seconds (defaults to ``TRACE_TTL_SEC``)."""

    if ttl is None:
        seconds = TRACE_TTL_SEC
    else:
        try:
            seconds = max(int(ttl), 0)
        except (TypeError, ValueError):
            seconds = TRACE_TTL_SEC
    now = time.time()
    expires = now + seconds if seconds > 0 else 0.0
    with _lock:
        global _expires_at
        _expires_at = expires
        _schedule_disable(seconds)
    return expires


def disable() -> None:
    """Disable trace capture immediately."""

    with _lock:
        global _expires_at
        _expires_at = 0.0
        _cancel_timer()


def active() -> bool:
    """Return ``True`` when the trace capture window is currently active."""

    expires = remaining()
    return expires > 0


def remaining() -> float:
    """Return remaining window seconds (or ``0`` when inactive)."""

    with _lock:
        expires_at = _expires_at
    if expires_at <= 0:
        return 0.0
    delta = expires_at - time.time()
    if delta <= 0:
        disable()
        return 0.0
    return delta


__all__ = ["TRACE_TTL_SEC", "enable", "disable", "active", "remaining"]
