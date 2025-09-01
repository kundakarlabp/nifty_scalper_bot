"""Feature indicator helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def atr_pct(ohlc: pd.DataFrame, period: int = 14) -> float | None:
    """Return ATR as a percentage of closing price.

    ``None`` is returned if there are insufficient bars or if the result is not
    finite.
    """

    if ohlc is None or len(ohlc) < period + 1:
        return None

    # Ensure numeric float series; some backfill sources return object dtype
    h = pd.to_numeric(ohlc["high"], errors="coerce").astype(float)
    l = pd.to_numeric(ohlc["low"], errors="coerce").astype(float)
    c = pd.to_numeric(ohlc["close"], errors="coerce").astype(float)

    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=period).mean().iloc[-1]

    last_close = float(c.iloc[-1])
    if not np.isfinite(atr) or not np.isfinite(last_close) or last_close <= 0:
        return None
    return float(atr / last_close * 100.0)

