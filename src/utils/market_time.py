from __future__ import annotations

"""Market time helpers for IST trading sessions."""

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


def is_market_open(now: datetime) -> bool:
    """Return True if *now* (aware) falls within a trading session."""
    now = now.astimezone(IST)
    if now.weekday() >= 5:
        return False
    start, end = time(9, 15), time(15, 30)
    return start <= now.time() <= end


def last_session_window(now: datetime) -> tuple[datetime, datetime]:
    """Return start/end datetimes for the most recent trading session."""
    now = now.astimezone(IST)
    d = now.date()

    def weekday_back(d0):
        from datetime import timedelta as TD

        while d0.weekday() >= 5:
            d0 = d0 - TD(days=1)
        return d0

    if now.time() < time(9, 15):
        d = weekday_back(d - timedelta(days=1))
    elif now.time() > time(15, 30):
        d = weekday_back(d)
    else:
        d = weekday_back(d)

    start = datetime(d.year, d.month, d.day, 9, 15, tzinfo=IST)
    end = datetime(d.year, d.month, d.day, 15, 30, tzinfo=IST)
    return start, end
