"""Pure NSE session-aware historical OHLC quality checks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Callable, Mapping, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

from nifty_scalper_bot.data.time_contract import coerce_market_timestamp

_LOWER_BOUND_UTC = datetime(2000, 1, 1, tzinfo=timezone.utc)
_FUTURE_GRACE = timedelta(days=366)
_DEFAULT_OPEN = time(9, 15)
_DEFAULT_CLOSE = time(15, 30)
_STALE_BLOCKERS = {
    "spot": "spot_history_stale",
    "spot_context": "spot_history_stale",
    "futures_context": "futures_context_history_stale",
    "selected_ce": "selected_ce_history_stale",
    "selected_pe": "selected_pe_history_stale",
}
_GAP_BLOCKERS = {
    "spot": "spot_history_gap_detected",
    "spot_context": "spot_history_gap_detected",
    "futures_context": "futures_context_history_gap_detected",
    "selected_ce": "selected_ce_history_gap_detected",
    "selected_pe": "selected_pe_history_gap_detected",
}


@dataclass(frozen=True, slots=True)
class HistoryQualityResult:
    symbol: str
    role: str
    required_bars: int
    available_bars: int
    first_bar_ts: datetime | None
    last_bar_ts: datetime | None
    expected_latest_closed_ts: datetime | None
    latest_bar_age_seconds: float | None
    latest_bar_fresh: bool
    recent_window_contiguous: bool
    missing_expected_minutes: tuple[datetime, ...]
    missing_expected_minute_count: int
    largest_intraday_gap_minutes: int
    provider_error: str | None
    blocker_reasons: tuple[str, ...]


def coerce_history_timestamp(value: Any) -> datetime | None:
    """Coerce supported history timestamp shapes to UTC-aware datetimes."""
    if value is None or isinstance(value, bool):
        return None
    try:
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        if isinstance(value, (int, float)):
            raw = float(value)
            abs_raw = abs(raw)
            if abs_raw < 1e11:
                ts = pd.to_datetime(raw, unit="s", utc=True, errors="coerce")
            elif abs_raw < 1e14:
                ts = pd.to_datetime(raw, unit="ms", utc=True, errors="coerce")
            elif abs_raw < 1e17:
                ts = pd.to_datetime(raw, unit="us", utc=True, errors="coerce")
            else:
                ts = pd.to_datetime(raw, unit="ns", utc=True, errors="coerce")
        else:
            ts = coerce_market_timestamp(value).tz_convert("UTC")
    except Exception:
        return None
    if pd.isna(ts):
        return None
    out = pd.Timestamp(ts).to_pydatetime().astimezone(timezone.utc)
    now = datetime.now(timezone.utc)
    if out < _LOWER_BOUND_UTC or out > now + _FUTURE_GRACE:
        return None
    return out


def _bar_ts(row: Mapping[str, Any] | Any) -> datetime | None:
    if isinstance(row, Mapping):
        value = (
            row.get("timestamp")
            or row.get("start")
            or row.get("date")
            or row.get("time")
        )
    else:
        value = getattr(row, "timestamp", None) or getattr(row, "start", None)
    return coerce_history_timestamp(value)


def _prev_trading_day(day: date, resolver: Callable[[date], bool]) -> date:
    cur = day
    for _ in range(370):
        if resolver(cur):
            return cur
        cur -= timedelta(days=1)
    return day


def expected_latest_closed_market_minute(
    *,
    now: datetime,
    market_timezone: ZoneInfo,
    market_open: time,
    market_close: time,
    trading_day_resolver: Callable[[date], bool],
    publication_grace_seconds: float,
) -> datetime | None:
    local_now = (
        now.astimezone(market_timezone)
        if now.tzinfo
        else now.replace(tzinfo=market_timezone)
    )
    today = local_now.date()
    grace = timedelta(seconds=max(float(publication_grace_seconds), 0.0))
    if not trading_day_resolver(today):
        day = _prev_trading_day(today - timedelta(days=1), trading_day_resolver)
        return datetime.combine(day, market_close, market_timezone).astimezone(
            timezone.utc
        )
    session_open = datetime.combine(today, market_open, market_timezone)
    session_close = datetime.combine(today, market_close, market_timezone)
    if local_now < session_open + timedelta(minutes=1) + grace:
        day = _prev_trading_day(today - timedelta(days=1), trading_day_resolver)
        return datetime.combine(day, market_close, market_timezone).astimezone(
            timezone.utc
        )
    if local_now > session_close + grace:
        return session_close.astimezone(timezone.utc)
    candidate = (local_now - grace).replace(second=0, microsecond=0) - timedelta(
        minutes=1
    )
    if candidate < session_open:
        return None
    if candidate > session_close:
        candidate = session_close
    return candidate.astimezone(timezone.utc)


def _same_session_gap_minutes(
    a: datetime, b: datetime, tz: ZoneInfo, resolver: Callable[[date], bool]
) -> list[datetime]:
    al = a.astimezone(tz).replace(second=0, microsecond=0)
    bl = b.astimezone(tz).replace(second=0, microsecond=0)
    if al.date() != bl.date() or not resolver(al.date()):
        return []
    if not (
        _DEFAULT_OPEN <= al.time() <= _DEFAULT_CLOSE
        and _DEFAULT_OPEN <= bl.time() <= _DEFAULT_CLOSE
    ):
        return []
    out: list[datetime] = []
    cur = al + timedelta(minutes=1)
    while cur < bl:
        if _DEFAULT_OPEN <= cur.time() <= _DEFAULT_CLOSE:
            out.append(cur.astimezone(timezone.utc))
        cur += timedelta(minutes=1)
    return out


def evaluate_history_quality(
    *,
    symbol: str,
    role: str,
    bars: Sequence[Mapping[str, Any] | Any],
    required_bars: int,
    now: datetime,
    market_timezone: ZoneInfo,
    trading_day_resolver: Callable[[date], bool],
    publication_grace_seconds: float,
    allowed_recent_missing_minutes: int,
    continuity_window_bars: int | None = None,
    provider_error: str | None = None,
) -> HistoryQualityResult:
    parsed = [_bar_ts(row) for row in bars or ()]
    invalid_seen = any(ts is None for ts in parsed) and bool(bars)
    timestamps = [
        ts.replace(second=0, microsecond=0) for ts in parsed if ts is not None
    ]
    blockers: list[str] = []
    if invalid_seen:
        blockers.append("history_timestamp_invalid")
    if provider_error:
        blockers.append("history_provider_error")
    ordered = sorted(set(timestamps))
    available = len(ordered)
    first = ordered[0] if ordered else None
    last = ordered[-1] if ordered else None
    expected = expected_latest_closed_market_minute(
        now=now,
        market_timezone=market_timezone,
        market_open=_DEFAULT_OPEN,
        market_close=_DEFAULT_CLOSE,
        trading_day_resolver=trading_day_resolver,
        publication_grace_seconds=publication_grace_seconds,
    )
    fresh = bool(last is not None and (expected is None or last >= expected))
    age = (now.astimezone(timezone.utc) - last).total_seconds() if last else None
    if not fresh:
        blockers.append(_STALE_BLOCKERS.get(role, f"{role}_history_stale"))
    window_size = max(int(required_bars or 0), int(continuity_window_bars or 0))
    window = ordered[-window_size:] if window_size > 0 else ordered
    missing: list[datetime] = []
    largest_gap = 0
    for a, b in zip(window, window[1:]):
        gap = _same_session_gap_minutes(a, b, market_timezone, trading_day_resolver)
        if gap:
            missing.extend(gap)
            largest_gap = max(largest_gap, len(gap) + 1)
    contiguous = len(missing) <= max(0, int(allowed_recent_missing_minutes or 0))
    if not contiguous:
        blockers.append(_GAP_BLOCKERS.get(role, f"{role}_history_gap_detected"))
    return HistoryQualityResult(
        symbol=str(symbol),
        role=str(role),
        required_bars=max(0, int(required_bars or 0)),
        available_bars=available,
        first_bar_ts=first,
        last_bar_ts=last,
        expected_latest_closed_ts=expected,
        latest_bar_age_seconds=age,
        latest_bar_fresh=fresh,
        recent_window_contiguous=contiguous,
        missing_expected_minutes=tuple(missing),
        missing_expected_minute_count=len(missing),
        largest_intraday_gap_minutes=largest_gap,
        provider_error=provider_error,
        blocker_reasons=tuple(dict.fromkeys(blockers)),
    )
