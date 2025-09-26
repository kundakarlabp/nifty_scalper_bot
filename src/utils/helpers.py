"""General-purpose helper utilities used across the scalper bot."""

from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo


IST = ZoneInfo("Asia/Kolkata")


def get_weekly_expiry(
    now: datetime | None = None,
    *,
    cutoff: time = time(15, 20),
) -> str:
    """Return the upcoming weekly expiry (Thursday) formatted as ``YYMMDD``.

    The calculation is anchored to Indian Standard Time.  When today is
    Thursday the helper returns the same-day expiry until ``cutoff``; after the
    cut-off it rolls to the following week's contract.
    """

    current = now or datetime.now(tz=IST)
    if current.tzinfo is None:
        current = current.replace(tzinfo=IST)
    else:
        current = current.astimezone(IST)

    days_ahead = (3 - current.weekday()) % 7
    if days_ahead == 0 and current.time() > cutoff:
        days_ahead = 7
    expiry = current + timedelta(days=days_ahead)
    return expiry.strftime("%y%m%d")


def get_next_thursday(now: datetime | None = None) -> str:
    """Backward-compatible alias for :func:`get_weekly_expiry`."""

    return get_weekly_expiry(now=now)


__all__ = ["IST", "get_weekly_expiry", "get_next_thursday"]

