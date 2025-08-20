from __future__ import annotations

from typing import Optional

import pandas as pd


def _true_range(df: pd.DataFrame) -> pd.Series:
    """
    True Range (TR) per bar: max(high - low, abs(high - prev_close), abs(low - prev_close))
    """
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
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


def compute_atr(df: pd.DataFrame, period: int = 14, method: str = "rma") -> pd.Series:
    """
    Compute ATR. Requires columns: high, low, close.

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
        atr = tr.rolling(window=period, min_periods=1).mean()
    elif m == "ema":
        atr = tr.ewm(span=period, adjust=False).mean()
    else:  # Wilder's (RMA): alpha=1/period
        alpha = 1 / float(period)
        atr = tr.ewm(alpha=alpha, adjust=False).mean()

    atr.name = "atr"
    return atr


def atr_sl_tp_points(
    *,
    base_sl_points: float,
    base_tp_points: float,
    atr_value: Optional[float],
    sl_mult: float,
    tp_mult: float,
    confidence: float,
    sl_conf_adj: float = 0.0,
    tp_conf_adj: float = 0.0,
) -> tuple[float, float]:
    """
    Combine base SL/TP with ATR-based dynamic scaling and confidence nudges.
    """
    atr_part = float(atr_value or 0.0)
    sl = max(0.01, base_sl_points + atr_part * sl_mult - confidence * sl_conf_adj)
    tp = max(0.01, base_tp_points + atr_part * tp_mult + confidence * tp_conf_adj)
    return sl, tp