"""IST clock helpers."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

IST_NAME = "Asia/Kolkata"
IST_LABEL = "IST"
IST = ZoneInfo(IST_NAME)


def now() -> datetime:
    return datetime.now(IST)


def timestamp(value: Any, *, errors: str = "raise") -> pd.Timestamp:
    try:
        ts = pd.Timestamp(value)
    except Exception:
        if errors == "coerce":
            return pd.NaT
        raise
    if pd.isna(ts):
        return pd.NaT
    if ts.tzinfo is None:
        return ts.tz_localize(IST)
    return ts.tz_convert(IST)


def minute(value: Any, *, errors: str = "raise") -> pd.Timestamp:
    ts = timestamp(value, errors=errors)
    if pd.isna(ts):
        return pd.NaT
    return ts.floor("1min")


__all__ = ["IST", "IST_LABEL", "IST_NAME", "minute", "now", "timestamp"]
