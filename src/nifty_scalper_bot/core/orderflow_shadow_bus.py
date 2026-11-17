"""Purpose: Share bounded OrderFlow observations with closed-trade research.

Responsibilities: Cache previously computed three-second OFI evidence from
the existing strategy evaluation without retrieving new market or broker data.
Safety: This channel has no entry/exit voting, broker order or risk authority.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Any

_LOCK = threading.Lock()
_OBSERVATIONS: dict[str, deque[dict[str, Any]]] = {}


def publish_orderflow_research(
    symbol: str,
    *,
    ofi_3s_norm: Any,
    quote_age_seconds: Any,
    ofi_updates_3s: Any,
    ofi_span_seconds: Any,
    sample_time: float | None = None,
) -> None:
    """Publish a previously computed, causally timestamped signal observation."""
    try:
        score = float(ofi_3s_norm)
        age = float(quote_age_seconds)
        updates = int(ofi_updates_3s)
        span = float(ofi_span_seconds)
        stamp = float(sample_time if sample_time is not None else time.time())
        if not all(math.isfinite(value) for value in (score, age, span, stamp)):
            return
        if stamp <= 0:
            return
    except (TypeError, ValueError, OverflowError):
        return
    event = {
        "ts": stamp,
        "ofi_3s_norm": score,
        "fresh": bool(0 <= age <= 2.0 and updates >= 3 and span >= 2.5),
        "source": "orderflow_existing_context_3s",
        "quote_age_seconds": age,
        "ofi_updates_3s": updates,
        "ofi_span_seconds": span,
    }
    key = str(symbol or "").upper().strip()
    if not key:
        return
    with _LOCK:
        if key not in _OBSERVATIONS and len(_OBSERVATIONS) >= 40:
            # Prevent indefinite growth from option rotation; no trade effects.
            oldest = min(
                _OBSERVATIONS,
                key=lambda value: (
                    _OBSERVATIONS[value][-1]["ts"] if _OBSERVATIONS[value] else 0.0
                ),
            )
            _OBSERVATIONS.pop(oldest, None)
        _OBSERVATIONS.setdefault(key, deque(maxlen=256)).append(event)


def read_orderflow_research(
    symbol: str, *, since: float, until: float
) -> list[dict[str, Any]]:
    """Read a trade-bounded copy without consuming or altering any signal."""
    key = str(symbol or "").upper().strip()
    with _LOCK:
        values = list(_OBSERVATIONS.get(key, ()))
    return [dict(value) for value in values if since <= float(value["ts"]) <= until]
