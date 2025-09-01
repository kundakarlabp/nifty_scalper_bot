"""Feature health checks to validate indicator inputs."""

from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
from typing import List, Optional

import pandas as pd


@dataclass
class FeatureHealth:
    """Represents the health of feature inputs."""

    bars_ok: bool
    atr_ok: bool
    fresh_ok: bool
    reasons: List[str]


def check(
    ohlc: Optional[pd.DataFrame],
    last_bar_ts: Optional[dt.datetime],
    *,
    atr_period: int = 14,
    max_age_s: int = 90,
) -> FeatureHealth:
    """Validate OHLC data for feature computation.

    Parameters
    ----------
    ohlc:
        DataFrame of 1-minute OHLC data.
    last_bar_ts:
        Timestamp of the last bar in ``ohlc``.
    atr_period:
        Period required for ATR calculation. ``bars_ok`` requires
        ``atr_period + 1`` bars to allow a rolling window.
    max_age_s:
        Maximum allowed age (in seconds) of ``last_bar_ts``.
    """

    reasons: List[str] = []
    bars_ok = isinstance(ohlc, pd.DataFrame) and len(ohlc) >= atr_period + 1
    if not bars_ok:
        reasons.append("bars_short")

    tz = last_bar_ts.tzinfo if last_bar_ts and last_bar_ts.tzinfo else dt.timezone.utc
    if last_bar_ts and last_bar_ts.tzinfo is None:
        last_bar_ts = last_bar_ts.replace(tzinfo=tz)
    now = dt.datetime.now(tz)
    age_ok = (now - last_bar_ts).total_seconds() <= max_age_s if last_bar_ts else False
    if not age_ok:
        reasons.append("data_stale")

    atr_ok = bars_ok  # ATR needs sufficient bars; NaNs handled by indicator

    return FeatureHealth(bars_ok=bars_ok, atr_ok=atr_ok, fresh_ok=age_ok, reasons=reasons)

