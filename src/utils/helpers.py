"""General-purpose helper utilities used across the scalper bot."""

from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo


IST = ZoneInfo("Asia/Kolkata")


def get_weekly_expiry(
    now: datetime | None = None, *, use_same_day_before: str = "15:20"
) -> str:
    """Return the upcoming weekly expiry (Thursday) formatted as ``YYMMDD``.

    When today is Thursday the helper returns the same-day expiry until the
    cut-off specified by ``use_same_day_before``.  After the cut-off the expiry
    rolls forward to the following week to avoid stale contracts.  Datetimes are
    normalised to IST to match the NSE trading calendar.
    """

    current = now or datetime.now(IST)
    if current.tzinfo is None:
        current = current.replace(tzinfo=IST)
    else:
        current = current.astimezone(IST)

    cutoff = time.fromisoformat(use_same_day_before)
    days_ahead = (3 - current.weekday()) % 7
    if days_ahead == 0 and current.time() > cutoff:
        days_ahead = 7
    expiry = current + timedelta(days=days_ahead)
    return expiry.strftime("%y%m%d")


def get_next_thursday(now: datetime | None = None) -> str:
    """Backward-compatible alias for :func:`get_weekly_expiry`."""

    return get_weekly_expiry(now=now)


__all__ = ["get_weekly_expiry", "get_next_thursday"]

