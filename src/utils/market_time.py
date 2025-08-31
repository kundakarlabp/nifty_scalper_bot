from __future__ import annotations

"""Helper utilities for Indian market session times."""

from datetime import datetime, time, timedelta, date as ddate
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
MKT_START = time(9, 15)
MKT_END = time(15, 30)


def is_market_open(now: datetime) -> bool:
    """Return ``True`` if ``now`` falls within market hours (Mon–Fri)."""
    now = now.astimezone(IST)
    if now.weekday() >= 5:  # Sat/Sun
        return False
    return MKT_START <= now.time() <= MKT_END


def prev_weekday(day: ddate) -> ddate:
    """Step backwards to the most recent weekday."""
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def session_bounds(day: ddate) -> tuple[datetime, datetime]:
    """Return the start/end datetimes for the trading session of ``day``."""
    day = prev_weekday(day)
    start = datetime(day.year, day.month, day.day, MKT_START.hour, MKT_START.minute, tzinfo=IST)
    end = datetime(day.year, day.month, day.day, MKT_END.hour, MKT_END.minute, tzinfo=IST)
    return start, end


def prev_session_bounds(now: datetime) -> tuple[datetime, datetime]:
    """Return the bounds of the last completed session relative to ``now``."""
    now = now.astimezone(IST)
    day = now.date()
    if now.weekday() >= 5 or now.time() < MKT_START:
        day = prev_weekday(day - timedelta(days=1))
    else:
        day = prev_weekday(day)
    return session_bounds(day)


def prev_session_last_20m(now: datetime) -> tuple[datetime, datetime]:
    """Return the last 20 minutes window of the previous session."""
    start, end = prev_session_bounds(now)
    return end - timedelta(minutes=20), end


__all__ = [
    "IST",
    "MKT_START",
    "MKT_END",
    "is_market_open",
    "prev_weekday",
    "session_bounds",
    "prev_session_bounds",
    "prev_session_last_20m",
]
