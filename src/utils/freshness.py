from __future__ import annotations

"""Utilities for measuring data freshness."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import overload


@dataclass(frozen=True)
class Freshness:
    """Lag information for tick and bar data."""

    tick_lag_s: float | None
    bar_lag_s: float | None
    ok: bool


@overload
def _to_aware_utc(dt: None) -> None: ...


@overload
def _to_aware_utc(dt: datetime) -> datetime: ...


@overload
def _to_aware_utc(dt: str) -> datetime | None: ...


def _to_aware_utc(dt: datetime | str | None) -> datetime | None:
    """Return tz-aware UTC; assume naive inputs are UTC."""
    if dt is None:
        return None
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt)
        except ValueError:
            return None
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


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

    now_utc = _to_aware_utc(now)
    tick_ts = _to_aware_utc(last_tick_ts)
    bar_open = _to_aware_utc(last_bar_open_ts)
    bar_close = None if bar_open is None else bar_open + timedelta(seconds=tf_seconds)
    tick_lag = (
        None if tick_ts is None else max(0.0, (now_utc - tick_ts).total_seconds())
    )
    bar_lag = (
        None if bar_close is None else max(0.0, (now_utc - bar_close).total_seconds())
    )
    ok = (tick_lag is None or tick_lag <= max_tick_lag_s) and (
        bar_lag is None or bar_lag <= max_bar_lag_s
    )
    return Freshness(tick_lag_s=tick_lag, bar_lag_s=bar_lag, ok=ok)
