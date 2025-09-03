from __future__ import annotations

"""Utilities for measuring data freshness."""

from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class Freshness:
    """Lag information for tick and bar data."""

    tick_lag_s: float | None
    bar_lag_s: float | None
    ok: bool


def compute(
    *,
    now: datetime,
    last_tick_ts: datetime | None,
    last_bar_open_ts: datetime | None,
    tf_seconds: int,
    max_tick_lag_s: int,
    max_bar_lag_s: int,
) -> Freshness:
    """Return freshness metrics given last tick/bar timestamps."""

    tick_lag = (
        None if last_tick_ts is None else max(0.0, (now - last_tick_ts).total_seconds())
    )
    bar_close = (
        None if last_bar_open_ts is None else last_bar_open_ts + timedelta(seconds=tf_seconds)
    )
    bar_lag = None if bar_close is None else max(0.0, (now - bar_close).total_seconds())
    ok = (
        (tick_lag is None or tick_lag <= max_tick_lag_s)
        and (bar_lag is None or bar_lag <= max_bar_lag_s)
    )
    return Freshness(tick_lag_s=tick_lag, bar_lag_s=bar_lag, ok=ok)
