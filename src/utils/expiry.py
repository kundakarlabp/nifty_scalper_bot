from __future__ import annotations

"""Expiry date utilities for NSE options."""

from datetime import datetime

import pandas as pd

IST = "Asia/Kolkata"


def _to_ist(ts: pd.Timestamp) -> pd.Timestamp:
    """Convert a timestamp to IST, localizing if naive."""
    return ts.tz_localize(IST) if ts.tzinfo is None else ts.tz_convert(IST)


def next_tuesday_expiry(now: datetime | None = None) -> pd.Timestamp:
    """Return the next Tuesday expiry at 15:30 IST.

    If ``now`` falls on Tuesday at or after 15:30, the expiry rolls to the
    following week.
    """
    ts = pd.Timestamp.now(tz=IST) if now is None else _to_ist(pd.Timestamp(now))
    if ts.weekday() == 1 and ts.time() >= pd.Timestamp("15:30", tz=IST).time():
        target = ts + pd.offsets.Week(weekday=1, n=1)
    else:
        target = ts + pd.offsets.Week(weekday=1)
    return target.normalize() + pd.Timedelta(hours=15, minutes=30)


def last_tuesday_of_month(now: datetime | None = None) -> pd.Timestamp:
    """Return the last Tuesday of ``now``'s month at 15:30 IST."""
    ts = pd.Timestamp.now(tz=IST) if now is None else _to_ist(pd.Timestamp(now))
    end = (ts + pd.offsets.MonthEnd(0)).normalize() + pd.Timedelta(hours=15, minutes=30)
    while end.weekday() != 1:
        end -= pd.Timedelta(days=1)
    return end
