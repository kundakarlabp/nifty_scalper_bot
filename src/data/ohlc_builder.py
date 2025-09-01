from __future__ import annotations

from datetime import datetime, timedelta
from typing import Tuple, Optional

import pandas as pd


def last_closed_minute(now: datetime) -> datetime:
    """Return the timestamp of the last fully closed minute."""
    n = now.replace(second=0, microsecond=0)
    return n - timedelta(minutes=1)


def build_window(now: datetime, lookback_min: int) -> Tuple[datetime, datetime]:
    """Return start/end timestamps for a lookback window ending at the last closed minute."""
    end = last_closed_minute(now)
    start = end - timedelta(minutes=int(lookback_min))
    return start, end


def calc_bar_lag_s(ohlc: Optional[pd.DataFrame], now: datetime) -> Optional[int]:
    """Return age in seconds of the last bar in ``ohlc`` relative to ``now``.

    If ``ohlc`` is ``None`` or empty, ``None`` is returned.
    """
    if ohlc is None or len(ohlc) == 0:
        return None
    try:
        ts = ohlc.index[-1].to_pydatetime()
    except Exception:
        return None
    return max(0, int((now - ts).total_seconds()))
