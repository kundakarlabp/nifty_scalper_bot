# src/utils/atr_helper.py
from __future__ import annotations

from typing import Optional
import pandas as pd


def _true_range(df: pd.DataFrame) -> pd.Series:
    """
    True Range (TR) per bar:
      max(high - low, abs(high - prev_close), abs(low - prev_close))
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    tr.name = "tr"
    return tr


def compute_atr_df(df: pd.DataFrame, period: int = 14, method: str = "rma") -> pd.Series:
    """
    Compute ATR series aligned to df.index.

    method:
      - "sma": simple moving average of TR
      - "ema": exponential moving average of TR (span=period)
      - "rma": Wilder's smoothing (EMA with alpha=1/period)
    """
    if df is None or df.empty or not {"high", "low", "close"}.issubset(df.columns):
        return pd.Series(dtype=float, name="atr")

    period = max(1, int(period))
    tr = _true_range(df)

    m = method.lower()
    if m == "sma":
        atr = tr.rolling(window=period, min_periods=period).mean()
    elif m == "ema":
        atr = tr.ewm(span=period, adjust=False).mean()
    else:  # "rma" (Wilder)
        atr = tr.ewm(alpha=1.0 / float(period), adjust=False).mean()

    atr.name = "atr"
    return atr


def latest_atr_value(
    df: pd.DataFrame, period: int = 14, method: str = "rma"
) -> Optional[float]:
    s = compute_atr_df(df, period=period, method=method)
    if s.empty:
        return None
    v = s.iloc[-1]
    if pd.isna(v):
        return None
    try:
        return float(v)
    except Exception:
        return None
