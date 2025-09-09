from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

TZ = ZoneInfo("Asia/Kolkata")


def last_closed_minute(now: datetime) -> datetime:
    """Return the timestamp of the last fully closed minute.

    ``now`` is converted to IST before flooring so callers may pass naive or
    foreign‑timezone datetimes.
    """
    if now.tzinfo is None:
        now = now.replace(tzinfo=TZ)
    else:
        now = now.astimezone(TZ)
    n = now.replace(second=0, microsecond=0)
    return n - timedelta(minutes=1)


def build_window(now: datetime, lookback_min: int) -> Tuple[datetime, datetime]:
    """Return start/end timestamps for a lookback window ending at the last closed minute."""
    end = last_closed_minute(now)
    start = end - timedelta(minutes=int(lookback_min))
    return start, end


def prepare_ohlc(df: pd.DataFrame, now: datetime) -> pd.DataFrame:
    """Normalize an OHLC frame for 1m analysis.

    - Convert the index to IST (assuming naive timestamps are already IST).
    - Floor to the minute.
    - Drop duplicate indices keeping the last occurrence.
    - Exclude the bar currently forming by dropping anything newer than the
      last fully closed minute.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    idx = pd.to_datetime(out.index)
    idx = idx.tz_localize(TZ) if idx.tz is None else idx.tz_convert(TZ)
    idx = idx.floor("1min")
    out.index = idx
    out = out[~out.index.duplicated(keep="last")].sort_index()

    now_ist = now.astimezone(TZ) if now.tzinfo else now.replace(tzinfo=TZ)
    cutoff = (now_ist - pd.Timedelta(seconds=5)).replace(second=0, microsecond=0)
    return out.loc[out.index <= cutoff]


def calc_bar_lag_s(ohlc: Optional[pd.DataFrame], now: datetime) -> Optional[int]:
    """Return age in seconds of the last bar in ``ohlc`` relative to ``now``.

    The function is tolerant to mismatched timezone awareness between ``now`` and
    the timestamp in ``ohlc``. If one datetime is timezone aware and the other is
    naive, the naive one is assumed to be in the other's timezone so that a
    meaningful subtraction can occur.

    If ``ohlc`` is ``None`` or empty, ``None`` is returned.
    """
    if ohlc is None or len(ohlc) == 0:
        return None
    try:
        ts = ohlc.index[-1].to_pydatetime()
    except Exception:
        return None

    now_dt = now
    if ts.tzinfo is None and now_dt.tzinfo is not None:
        ts = ts.replace(tzinfo=now_dt.tzinfo)
    elif ts.tzinfo is not None and now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=ts.tzinfo)
    elif ts.tzinfo is not None and now_dt.tzinfo is not None:
        ts = ts.astimezone(now_dt.tzinfo)

    return max(0, int((now_dt - ts).total_seconds()))
