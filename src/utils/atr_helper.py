# src/utils/atr_helper.py
from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd


def _true_range(df: pd.DataFrame) -> pd.Series:
    """
    True Range (TR) per bar:
      max(high - low, abs(high - prev_close), abs(low - prev_close))
    Requires columns: high, low, close.
    """
    if df is None or df.empty or not {"high", "low", "close"}.issubset(df.columns):
        return pd.Series(dtype=float, name="tr", index=getattr(df, "index", None))

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
    Compute Average True Range (ATR). Requires columns: high, low, close.

    method:
      - "sma": simple moving average of TR
      - "ema": exponential moving average of TR (span=period)
      - "rma": Wilder's smoothing (EMA with alpha=1/period)  ← default
    """
    if df is None or df.empty or not {"high", "low", "close"}.issubset(df.columns):
        return pd.Series(dtype=float, name="atr", index=getattr(df, "index", None))

    period = max(1, int(period))
    tr = _true_range(df)

    m = (method or "rma").lower()
    if m == "sma":
        atr = tr.rolling(window=period, min_periods=period).mean()
    elif m == "ema":
        atr = tr.ewm(span=period, adjust=False, min_periods=period).mean()
    else:  # Wilder's (RMA): alpha=1/period
        alpha = 1.0 / float(period)
        atr = tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    atr.name = "atr"
    return atr


def compute_atr_df(df: pd.DataFrame, period: int = 14, method: str = "rma") -> pd.DataFrame:
    """
    Convenience: return a copy of df with an 'atr' column appended.
    Safe on empty/invalid input (returns a copy with possibly empty 'atr').
    """
    out = df.copy()
    out["atr"] = compute_atr(out, period=period, method=method)
    return out


def atr_sl_tp_points(
    *,
    base_sl_points: float,
    base_tp_points: float,
    atr_value: Optional[float],
    sl_mult: float,
    tp_mult: float,
    confidence: float,
    sl_conf_adj: float = 0.2,
    tp_conf_adj: float = 0.3,
) -> Tuple[float, float]:
    """
    Combine base SL/TP with ATR-based scaling and confidence nudges.

    - SL starts at (base_sl_points + ATR * sl_mult) and is **tightened** as confidence↑
      (i.e., smaller SL when confidence is high).
    - TP starts at (base_tp_points + ATR * tp_mult) and is **widened** as confidence↑.

    confidence is clamped to [0..10].
    Returns: (sl_points, tp_points), both >= 0.01
    """
    conf = max(0.0, min(10.0, float(confidence)))
    atr_part = float(atr_value or 0.0)

    # Base (ATR-scaled)
    sl_base = float(base_sl_points) + atr_part * float(sl_mult)
    tp_base = float(base_tp_points) + atr_part * float(tp_mult)

    # Confidence nudges (tighten SL, widen TP)
    sl_nudge = (conf / 10.0) * float(sl_conf_adj)
    tp_nudge = (conf / 10.0) * float(tp_conf_adj)

    sl_pts = max(0.01, sl_base * (1.0 - sl_nudge))
    tp_pts = max(0.01, tp_base * (1.0 + tp_nudge))
    return sl_pts, tp_pts
