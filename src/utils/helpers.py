"""General-purpose helper utilities used across the scalper bot."""

from __future__ import annotations

from datetime import datetime, timedelta


def get_next_thursday(now: datetime | None = None) -> str:
    """Return the next weekly expiry date formatted as ``YYMMDD``.

    The function mirrors the Zerodha/NSE weekly option expiry convention
    (Thursday).  When ``now`` already falls on a Thursday the expiry is rolled
    forward by one week to avoid returning an already-expired contract.

    Parameters
    ----------
    now:
        Optional override for the current timestamp.  Supplying the value makes
        the helper deterministic in tests while defaulting to
        :func:`datetime.now` during live trading.
    """

    current = now or datetime.now()
    # Python's ``weekday`` returns Monday=0 ... Sunday=6.  We want the upcoming
    # Thursday (index 3) and roll a full week ahead when today already is
    # Thursday so that the trading symbol is never stale.
    days_ahead = (3 - current.weekday() + 7) % 7
    if days_ahead == 0:
        days_ahead = 7
    expiry = current + timedelta(days=days_ahead)
    return expiry.strftime("%y%m%d")


__all__ = ["get_next_thursday"]

