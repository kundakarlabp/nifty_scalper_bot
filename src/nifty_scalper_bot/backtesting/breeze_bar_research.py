"""Research-only completed-bar aggregation for identified Breeze one-minute candles.

Responsibilities: Create causal 5m/15m OHLCV bars without filling missing minutes.
Ownership: Offline historical research data normalization, not production DataHub.
Safety: No inferred bid/ask, expiry, historical lot size or executable fills.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any, Sequence
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


@dataclass(frozen=True, slots=True)
class ResearchBar:
    start: datetime
    available_at: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    open_interest: int | None
    source_minutes: int
    execution_eligible: bool = False


@dataclass(frozen=True, slots=True)
class ResampleAudit:
    bars: tuple[ResearchBar, ...]
    missing_windows: tuple[str, ...]
    out_of_session_minutes: tuple[str, ...] = ()
    source_time_convention_verified: bool = False
    historical_quotes_verified: bool = False


def aggregate_breeze_session(
    candles: Sequence[Sequence[Any]],
    *,
    day: date,
    interval_minutes: int,
    session_start: time,
    session_end: time,
) -> ResampleAudit:
    """Aggregate only fully observed time-grid windows in explicit IST session.

    Source stamps are assumed bar-start (as archived); their broker semantics
    must be verified independently before claiming replay parity.
    Caller supplies real session hours (including Muhurat/other specials).
    Any duplicate, invalid, nonminute or out-of-session bar fails closed.
    A missing source minute causes the WHOLE 5m/15m bar to be unavailable.
    """
    if (
        type(interval_minutes) is not int
        or interval_minutes not in {1, 5, 15}
        or session_start >= session_end
        or not isinstance(day, date)
    ):
        raise ValueError("invalid_session_or_interval")
    # Published 2025 Muhurat session is not a normal 375-minute trading day.
    if day == date(2025, 10, 21) and (
        session_start != time(13, 45) or session_end != time(14, 45)
    ):
        raise ValueError("invalid_or_conflicting_candle")
    first = datetime.combine(day, session_start, tzinfo=IST)
    stop = datetime.combine(day, session_end, tzinfo=IST)
    if (stop - first) < timedelta(minutes=interval_minutes):
        raise ValueError("session_too_short")
    observed: dict[datetime, tuple[float, float, float, float, int, int | None]] = {}
    for values in candles:
        if len(values) != 7:
            raise ValueError("invalid_candle_width")
        try:
            stamp = datetime.fromisoformat(str(values[0]))
            prices = (
                float(values[1]),
                float(values[2]),
                float(values[3]),
                float(values[4]),
            )
            vol = float(values[5])
            oi = None if values[6] is None else float(values[6])
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError("invalid_candle_data") from exc
        if (
            stamp.tzinfo is None
            or stamp.utcoffset() != timedelta(hours=5, minutes=30)
            or stamp.second
            or stamp.microsecond
            or not first <= stamp < stop
            or stamp in observed
            or not all(math.isfinite(x) and x > 0 for x in prices)
            or prices[1] < max(prices[0], prices[3])
            or prices[2] > min(prices[0], prices[3])
            or not math.isfinite(vol)
            or vol < 0
            or not vol.is_integer()
            or (
                oi is not None
                and (not math.isfinite(oi) or oi < 0 or not oi.is_integer())
            )
        ):
            raise ValueError("invalid_or_conflicting_candle")
        observed[stamp] = (*prices, int(vol), None if oi is None else int(oi))
    out: list[ResearchBar] = []
    missing: list[str] = []
    window = timedelta(minutes=interval_minutes)
    cursor = first
    while cursor + window <= stop:
        stamps = tuple(cursor + timedelta(minutes=j) for j in range(interval_minutes))
        rows = [observed.get(ts) for ts in stamps]
        if any(row is None for row in rows):
            missing.append(cursor.isoformat())
        else:
            complete = [row for row in rows if row is not None]
            out.append(
                ResearchBar(
                    start=cursor,
                    available_at=cursor + window,
                    open=complete[0][0],
                    high=max(row[1] for row in complete),
                    low=min(row[2] for row in complete),
                    close=complete[-1][3],
                    volume=sum(row[4] for row in complete),
                    open_interest=complete[-1][5],
                    source_minutes=interval_minutes,
                )
            )
        cursor += window
    return ResampleAudit(tuple(out), tuple(missing))
