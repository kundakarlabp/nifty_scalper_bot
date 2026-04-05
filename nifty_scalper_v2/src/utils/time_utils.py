"""IST-aware time utilities for NiftyScalperBot v2.

All market logic operates in IST (Asia/Kolkata, UTC+5:30).
Timestamps are stored in UTC internally and converted for display.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from src.config.constants import SESSION_END, SESSION_START

IST = ZoneInfo("Asia/Kolkata")
UTC = timezone.utc


def now_ist() -> datetime:
    """Return current datetime in IST."""
    return datetime.now(IST)


def now_utc() -> datetime:
    """Return current datetime in UTC."""
    return datetime.now(UTC)


def to_ist(dt: datetime) -> datetime:
    """Convert any timezone-aware datetime to IST."""
    return dt.astimezone(IST)


def to_utc(dt: datetime) -> datetime:
    """Convert any timezone-aware datetime to UTC."""
    return dt.astimezone(UTC)


def ist_time_now() -> time:
    """Return current time in IST (no date)."""
    return now_ist().time()


def is_trading_hours(at: datetime | None = None) -> bool:
    """Return True if the given time (or now) is within the trading session.

    Session: Monday–Friday, 09:20–15:15 IST.
    """
    if at is None:
        at = now_ist()
    else:
        at = to_ist(at)

    # Weekday check: Mon=0 … Fri=4
    if at.weekday() >= 5:
        return False

    t = at.time()
    return SESSION_START <= t <= SESSION_END


def is_weekday(at: datetime | None = None) -> bool:
    """Return True if at (or now) is a weekday."""
    if at is None:
        at = now_ist()
    return to_ist(at).weekday() < 5


def market_open(at: datetime | None = None) -> datetime:
    """Return the market open time (SESSION_START) for today in IST."""
    if at is None:
        at = now_ist()
    ist_at = to_ist(at)
    return ist_at.replace(
        hour=SESSION_START.hour,
        minute=SESSION_START.minute,
        second=0,
        microsecond=0,
        tzinfo=IST,
    )


def market_close(at: datetime | None = None) -> datetime:
    """Return the market close time (SESSION_END) for today in IST."""
    if at is None:
        at = now_ist()
    ist_at = to_ist(at)
    return ist_at.replace(
        hour=SESSION_END.hour,
        minute=SESSION_END.minute,
        second=0,
        microsecond=0,
        tzinfo=IST,
    )


def time_to_open() -> timedelta:
    """Time remaining until market open. Negative if already open."""
    return market_open() - now_ist()


def time_to_close() -> timedelta:
    """Time remaining until market close. Negative if already closed."""
    return market_close() - now_ist()


def ist_timestamp(dt: datetime) -> str:
    """Format datetime as IST string for logging: '2026-04-02 09:20:00 IST'."""
    return to_ist(dt).strftime("%Y-%m-%d %H:%M:%S IST")
