# src/utils/atr_helper.py
"""
Average True Range (ATR) utilities for real-time options/indices trading.

Features
- compute_atr_df: Vectorized ATR (SMA/EMA/RMA) from a pandas DataFrame of OHLCV
- compute_atr_arrays: Same as above, but accepts Python lists/arrays for speed
- incremental_wilder_update: O(1) ATR update (Wilder/RMA) from the last bar and previous ATR
- get_realtime_atr: Convenience wrapper to pull candles via KiteConnect and return latest ATR

Design notes
- Default method is Wilder's RMA (classic ATR), period=14.
- NaNs and short histories are handled gracefully (returns None if insufficient).
- No external TA libs required.
"""
from __future__ import annotations

from typing import Optional, Iterable, Any, Literal
import logging
from datetime import datetime, timedelta

import pandas as pd

logger = logging.getLogger(__name__)

ATRMethod = Literal["rma", "ema", "sma"]


# ---------- Core math helpers ----------

def _true_range_series(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Vectorized True Range."""
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def _ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False, min_periods=period).mean()


def _rma_wilder(s: pd.Series, period: int) -> pd.Series:
    """Wilder's RMA (a.k.a. smoothed moving average) commonly used for ATR."""
    sma = s.rolling(window=period, min_periods=period).mean()
    rma = sma.copy()
    alpha = 1.0 / period
    for i in range(period, len(s)):
        if pd.isna(rma.iat[i - 1]):
            rma.iat[i] = sma.iat[i]
        else:
            rma.iat[i] = rma.iat[i - 1] + alpha * (s.iat[i] - rma.iat[i - 1])
    return rma


def _sma(s: pd.Series, period: int) -> pd.Series:
    return s.rolling(window=period, min_periods=period).mean()


# ---------- Public APIs ----------

def compute_atr_df(
    df: pd.DataFrame,
    period: int = 14,
    method: ATRMethod = "rma",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pd.Series:
    """Compute ATR for an OHLC DataFrame. Returns a pd.Series aligned to df.index."""
    if df is None or df.empty:
        return pd.Series(dtype=float)

    for c in (high_col, low_col, close_col):
        if c not in df.columns:
            logger.warning("compute_atr_df: missing column '%s'", c)
            return pd.Series(dtype=float)

    high = pd.to_numeric(df[high_col], errors="coerce")
    low = pd.to_numeric(df[low_col], errors="coerce")
    close = pd.to_numeric(df[close_col], errors="coerce")

    tr = _true_range_series(high, low, close)

    m = (method or "rma").lower()
    if m == "ema":
        atr = _ema(tr, period)
    elif m == "sma":
        atr = _sma(tr, period)
    else:
        atr = _rma_wilder(tr, period)  # default: Wilder

    return atr


def compute_atr_arrays(
    highs: Iterable[float],
    lows: Iterable[float],
    closes: Iterable[float],
    period: int = 14,
    method: ATRMethod = "rma",
) -> Optional[float]:
    """Fast path returning ONLY the latest ATR value from arrays/lists."""
    try:
        df = pd.DataFrame({"high": list(highs), "low": list(lows), "close": list(closes)})
        atr = compute_atr_df(df, period=period, method=method)
        if atr.empty:
            return None
        val = atr.iloc[-1]
        return float(val) if pd.notna(val) else None
    except Exception as e:
        logger.debug("compute_atr_arrays error: %s", e)
        return None


def incremental_wilder_update(
    prev_atr: float,
    prev_close: float,
    high: float,
    low: float,
    close: float,
    period: int = 14,
) -> Optional[float]:
    """
    O(1) update for ATR (Wilder/RMA) using the most recent completed bar.

    newTR  = max(high - low, abs(high - prev_close), abs(low - prev_close))
    newATR = prev_atr + (newTR - prev_atr) / period
    """
    try:
        if period <= 0:
            return None
        if any(v is None for v in (prev_atr, prev_close, high, low, close)):
            return None
        tr1 = abs(high - low)
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        new_tr = max(tr1, tr2, tr3)
        return float(prev_atr + (new_tr - prev_atr) / float(period))
    except Exception as e:
        logger.debug("incremental_wilder_update error: %s", e)
        return None


def latest_atr_value(
    df: pd.DataFrame,
    period: int = 14,
    method: ATRMethod = "rma",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> Optional[float]:
    """Convenience: compute ATR series and return the latest non-NaN value."""
    s = compute_atr_df(
        df,
        period=period,
        method=method,
        high_col=high_col,
        low_col=low_col,
        close_col=close_col,
    )
    if s.empty:
        return None
    v = s.iloc[-1]
    return float(v) if pd.notna(v) else None


# ---------- Kite convenience wrapper (optional) ----------

def get_realtime_atr(
    kite: Any,
    instrument_token: int,
    period: int = 14,
    bars: int = 100,
    timeframe: str = "minute",
    include_partial: bool = False,
) -> Optional[float]:
    """
    Pull recent candles and return the latest ATR value.

    Args:
      kite: KiteConnect instance
      instrument_token: token for the instrument
      period: ATR period (default 14)
      bars: number of recent bars to fetch (>= period+1 recommended)
      timeframe: 'minute'|'3minute'|'5minute'|'15minute'|'day' (as supported by Kite)
      include_partial: if True, includes the last (possibly forming) candle
    """
    try:
        if not kite or not instrument_token:
            return None

        end = datetime.now()
        if timeframe == "minute":
            start = end - timedelta(minutes=max(int(bars * 1.1), period * 2))
        elif timeframe.endswith("minute"):
            mult = int(timeframe.replace("minute", ""))  # e.g., '3minute' -> 3
            start = end - timedelta(minutes=max(int(bars * mult * 1.1), period * mult * 2))
        elif timeframe == "day":
            start = end - timedelta(days=max(int(bars * 1.1), period * 2))
        else:
            start = end - timedelta(minutes=max(bars, period) * 2)

        candles = kite.historical_data(instrument_token, start, end, timeframe, oi=False) or []
        if not candles:
            return None

        df = pd.DataFrame(candles)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)

        # ensure required columns exist
        if not set(("high", "low", "close")).issubset(df.columns):
            return None

        if not include_partial and len(df) >= 2:
            df = df.iloc[:-1]

        return latest_atr_value(df, period=period, method="rma")
    except Exception as e:
        logger.debug("get_realtime_atr error: %s", e, exc_info=True)
        return None