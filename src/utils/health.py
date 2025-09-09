"""Shared in-memory health state."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _State:
    """Mutable container for runtime health metrics."""

    last_tick_ts: float = 0.0


STATE = _State()
