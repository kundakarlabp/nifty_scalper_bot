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

from typing import Iterable, Optional, Literal, Any, Sequence
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ATRMethod = Literal["rma", "ema", "sma"]

__all__ = [
    "ATRMethod",
    "compute_atr_df",
    "compute_atr_arrays",
    "incremental_wilder_update",
    "latest_atr_value",
    "get_realtime_atr",
]


# ---------- Core math helpers (vectorized) ----------

def _true_range_series(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Vectorized True Range."""
    high = pd.to_numeric(high, errors="coerce").astype(float)
    low = pd.to_numeric(low, errors="coerce").astype(float)
    close = pd.to_numeric(close, errors="coerce").astype(float)

    prev_close = close.shift(1)
    # Use NumPy for max of three arrays
    a = (high - low).abs().to_numpy()
    b = (high - prev_close).abs().to_numpy()
    c = (low - prev_close).abs().to_numpy()
    tr = np.maximum.reduce([a, b, c])

    out = pd.Series(tr, index=high.index, dtype=float)
    return out


def _ema(s: pd.Series, period: int) -> pd.Series:
    # EMA of TR (not seeded to SMA; this is standard EMA behavior)
    return s.ewm(span=period, adjust=False, min_periods=period).mean()


def _rma_wilder_1d(tr: np.ndarray, period: int) -> np.ndarray:
    """
    Wilder's RMA computed on a 1-D numpy array, seeded with SMA(period).
    Returns an array aligned to tr (NaN before seed).
    """
    n = tr.shape[0]
    out = np.full(n, np.nan, dtype=float)
    if period <= 0 or n == 0:
        return out
    if n < period:
        return out
    # Seed with SMA at index period-1
    sma = np.nanmean(tr[:period])
    out[period - 1] = sma
    alpha = 1.0 / period
    prev = sma
    for i in range(period, n):
        x = tr[i]
        # If x is NaN, propagate previous (keeps series stable over missing bars)
        if np.isnan(x):
            out[i] = prev
            continue
        val = prev + alpha * (x - prev)
        out[i] = val
        prev = val
    return out


def _rma_wilder(s: pd.Series, period: int) -> pd.Series:
    tr = pd.to_numeric(s, errors="coerce").astype(float).to_numpy()
    out = _rma_wilder_1d(tr, period)
    return pd.Series(out, index=s.index, dtype=float)


def _sma(s: pd.Series, period: int) -> pd.Series:
    return s.rolling(window=period, min_periods=period).mean().astype(float)


# ---------- Public APIs ----------

def compute_atr_df(
    df: pd.DataFrame,
    period: int = 14,
    method: ATRMethod = "rma",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pd.Series:
    """
    Compute ATR for an OHLC DataFrame. Returns a pd.Series aligned to df.index.
    NaN for bars before the seed (per method/min_periods).
    """
    if df is None or df.empty or period <= 0:
        return pd.Series(dtype=float)

    for c in (high_col, low_col, close_col):
        if c not in df.columns:
            logger.warning("compute_atr_df: missing column '%s'", c)
            return pd.Series(dtype=float)

    high = pd.to_numeric(df[high_col], errors="coerce").astype(float)
    low = pd.to_numeric(df[low_col], errors="coerce").astype(float)
    close = pd.to_numeric(df[close_col], errors="coerce").astype(float)

    tr = _true_range_series(high, low, close)

    m = (method or "rma").lower()
    if m == "ema":
        atr = _ema(tr, period)
    elif m == "sma":
        atr = _sma(tr, period)
    else:
        atr = _rma_wilder(tr, period)  # default: Wilder

    return atr.astype(float)


def compute_atr_arrays(
    highs: Iterable[float],
    lows: Iterable[float],
    closes: Iterable[float],
    period: int = 14,
    method: ATRMethod = "rma",
) -> Optional[float]:
    """
    Fast path returning ONLY the latest ATR value from arrays/lists.
    Avoids DataFrame construction where possible.
    """
    try:
        h = np.asarray(list(highs), dtype=float)
        l = np.asarray(list(lows), dtype=float)
        c = np.asarray(list(closes), dtype=float)
        n = min(h.size, l.size, c.size)
        if n == 0 or period <= 0:
            return None
        h = h[-n:]; l = l[-n:]; c = c[-n:]
        # True range
        prev_c = np.roll(c, 1)
        prev_c[0] = np.nan
        tr = np.maximum.reduce([np.abs(h - l), np.abs(h - prev_c), np.abs(l - prev_c)])
        m = (method or "rma").lower()
        if m == "ema":
            # Fallback to pandas for EMA to honor min_periods logic
            df = pd.DataFrame({"tr": tr})
            atr = df["tr"].ewm(span=period, adjust=False, min_periods=period).mean().to_numpy()
        elif m == "sma":
            if n < period:
                return None
            atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().to_numpy()
        else:
            atr = _rma_wilder_1d(tr, period)
        if atr.size == 0 or np.isnan(atr[-1]):
            return None
        return float(atr[-1])
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

    Returns newATR or None if inputs invalid.
    """
    try:
        if period <= 0:
            return None
        vals = (prev_atr, prev_close, high, low, close)
        if any(v is None for v in vals):
            return None
        prev_atr = float(prev_atr); prev_close = float(prev_close)
        high = float(high); low = float(low); close = float(close)

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
    """
    Convenience: compute ATR series and return the latest non-NaN value.
    """
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

    Returns:
      float or None
    """
    try:
        if not kite or not instrument_token or period <= 0:
            return None

        end = datetime.now()
        # lookback computation with buffer for gaps
        if timeframe == "minute":
            start = end - timedelta(minutes=max(int(bars * 1.1), period * 2))
        elif timeframe.endswith("minute"):
            try:
                mult = int(timeframe.replace("minute", "")) or 1
            except Exception:
                mult = 1
            start = end - timedelta(minutes=max(int(bars * mult * 1.1), period * mult * 2))
        elif timeframe == "day":
            start = end - timedelta(days=max(int(bars * 1.1), period * 2))
        else:
            # fallback
            start = end - timedelta(minutes=max(bars, period) * 2)

        candles = kite.historical_data(
            instrument_token, start, end, timeframe, oi=False
        ) or []
        if not candles:
            return None

        df = pd.DataFrame(candles)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            # Make timezone-naive for consistency
            try:
                if getattr(df["date"].dt, "tz", None) is not None:
                    df["date"] = df["date"].dt.tz_convert(None)
            except Exception:
                # if tz-aware but tz_convert not available (already naive)
                pass
            df["date"] = df["date"].dt.tz_localize(None)
            df.set_index("date", inplace=True)

        cols = [c for c in ("open", "high", "low", "close") if c in df.columns]
        if len(cols) < 3:
            return None

        # Optionally drop the last partial bar
        if not include_partial and len(df) >= 2:
            df = df.iloc[:-1]

        return latest_atr_value(df, period=period, method="rma")
    except Exception as e:
        logger.debug("get_realtime_atr error: %s", e, exc_info=True)
        return None
