"""Canonical market timestamp normalization for NSE/NIFTY data.

Market-data timestamps are normalized to Asia/Kolkata because exchange
sessions, candle bucketing, and operator diagnostics are NSE-local. Internal
runtime/lifecycle code may still use UTC; this module is only for market ticks
and OHLC candles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import pandas as pd

IST = ZoneInfo("Asia/Kolkata")
UTC_EPOCH_SECONDS_FLOOR = 946684800  # 2000-01-01T00:00:00Z

_EXCHANGE_TIMESTAMP_FIELDS: tuple[str, ...] = (
    "exchange_timestamp",
    "last_trade_time",
    "last_traded_time",
    "last_trade_timestamp",
    "exchange_update_time",
    "last_price_time",
)
_BROKER_TIMESTAMP_FIELDS: tuple[str, ...] = ("timestamp", "ts", "date", "datetime")
_RECEIVED_AT_FIELDS: tuple[str, ...] = ("received_at", "received_ts", "received_time")


@dataclass(frozen=True, slots=True)
class MarketTimestamp:
    """Normalized market timestamp plus provenance for forensic logging."""

    timestamp: pd.Timestamp
    source: str
    raw_value: Any

    @property
    def isoformat(self) -> str:
        return self.timestamp.isoformat()


def coerce_market_timestamp(value: Any, *, naive_policy: str = "ist") -> pd.Timestamp:
    """Return an IST-aware timestamp.

    Numeric values are interpreted as UTC epoch seconds/milliseconds. Naive
    broker/exchange datetimes are interpreted as IST because NSE market feeds
    commonly send local exchange timestamps without a timezone. Timezone-aware
    values are converted to IST.
    """

    try:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            raw = float(value)
            if raw > 1e12:
                ts = pd.to_datetime(raw, unit="ms", utc=True, errors="coerce")
            elif raw > UTC_EPOCH_SECONDS_FLOOR:
                ts = pd.to_datetime(raw, unit="s", utc=True, errors="coerce")
            else:
                ts = pd.NaT
        else:
            ts = pd.Timestamp(value)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"unparseable timestamp: {value!r}") from exc

    if pd.isna(ts):
        raise ValueError(f"unparseable timestamp: {value!r}")

    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        if naive_policy != "ist":
            raise ValueError(f"naive timestamp rejected: {value!r}")
        return ts.tz_localize(IST)
    return ts.tz_convert(IST)


def normalize_market_tick_timestamp(payload: Mapping[str, Any]) -> MarketTimestamp:
    """Normalize a live market tick timestamp with explicit source priority.

    Priority is exchange/broker event time first. ``received_at`` is allowed
    only as an explicit fallback and is labelled as such; this prevents a
    generated wall-clock timestamp from silently masquerading as exchange time.
    A malformed higher-priority field does not hide a later valid timestamp.
    """

    present = False
    last_error: Exception | None = None
    for fields in (
        _EXCHANGE_TIMESTAMP_FIELDS,
        _BROKER_TIMESTAMP_FIELDS,
        _RECEIVED_AT_FIELDS,
    ):
        for field in fields:
            value = payload.get(field)
            if value is None or value == "":
                continue
            present = True
            try:
                timestamp = coerce_market_timestamp(value)
            except (TypeError, ValueError, OverflowError) as exc:
                last_error = exc
                continue
            return MarketTimestamp(timestamp=timestamp, source=field, raw_value=value)

    if not present:
        raise ValueError("missing market timestamp")
    raise ValueError("all present market timestamps are invalid") from last_error


def future_delta_seconds(ts: pd.Timestamp, *, now: pd.Timestamp) -> float:
    """Positive seconds when ``ts`` is ahead of ``now``; otherwise 0."""

    return max((ts - now).total_seconds(), 0.0)


def is_future_market_timestamp(
    ts: pd.Timestamp, *, now: pd.Timestamp, grace_seconds: float
) -> bool:
    return bool(ts > now + pd.Timedelta(seconds=max(float(grace_seconds), 0.0)))


def normalized_symbol(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text:
        raise ValueError("missing symbol")
    return text
