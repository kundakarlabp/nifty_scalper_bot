from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd


# ---------------------------- Core ATR pieces ----------------------------

def _true_range(df: pd.DataFrame) -> pd.Series:
    """
    True Range (TR) per bar:
      TR = max( high-low, |high-prev_close|, |low-prev_close| )

    Safe on empty/invalid input (returns empty Series with matching index).
    Requires columns: high, low, close.
    """
    if df is None or df.empty or not {"high", "low", "close"}.issubset(df.columns):
        return pd.Series(dtype=float, name="tr", index=getattr(df, "index", None))

    high = pd.to_numeric(df["high"], errors="coerce").astype(float)
    low = pd.to_numeric(df["low"], errors="coerce").astype(float)
    close = pd.to_numeric(df["close"], errors="coerce").astype(float)
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
    Compute Average True Range (ATR) over `period`.

    method:
      - "sma": simple moving average of TR
      - "ema": exponential moving average of TR (span=period)
      - "rma": Wilder's smoothing (EMA with alpha=1/period)  ← default

    Returns a Series named 'atr' aligned to df's index. Initial `period-1`
    values will be NaN (by design); use `latest_atr_value(...)` when you need
    a single numeric value safely.
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
    else:  # Wilder (RMA): alpha = 1/period
        alpha = 1.0 / float(period)
        atr = tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    atr.name = "atr"
    return atr


def compute_atr_df(df: pd.DataFrame, period: int = 14, method: str = "rma") -> pd.DataFrame:
    """Return a copy of `df` with an 'atr' column appended. Safe on empty/invalid input."""
    out = df.copy()
    out.loc[:, "atr"] = compute_atr(out, period=period, method=method)
    return out


def latest_atr_value(atr_series: Optional[pd.Series], default: float = 0.0) -> float:
    """
    Return the most recent non-NaN ATR value, or `default` if unavailable.
    Helpful because ATR has NaNs in the warmup window.
    """
    if atr_series is None or len(atr_series) == 0:
        return float(default)
    last_valid = atr_series.dropna()
    if len(last_valid) == 0:
        return float(default)
    try:
        return float(last_valid.iloc[-1])
    except Exception:
        return float(default)


# ---------------------------- SL/TP shaping ----------------------------

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
    Build SL/TP distances (in *points*) from ATR and confidence.

    - Start with base_sl_points/base_tp_points (already ATR-scaled by caller).
    - Add ATR components (atr_value * multipliers).
    - Apply confidence nudges: higher confidence → tighter SL, wider TP.

    Returns: (sl_points, tp_points) both >= 0.01
    """
    # Sanity clamps
    base_sl = max(0.0, float(base_sl_points))
    base_tp = max(0.0, float(base_tp_points))
    atr_part = max(0.0, float(atr_value or 0.0))

    # Base + ATR portion
    sl_base = base_sl + atr_part * float(sl_mult)
    tp_base = base_tp + atr_part * float(tp_mult)

    # Confidence nudges (0..10 → 0..1)
    conf = max(0.0, min(10.0, float(confidence)))

    # Bound the effect so auto-adjust can’t go wild (±20% by default)
    sl_conf_adj = max(-0.2, min(0.2, float(sl_conf_adj)))
    tp_conf_adj = max(-0.2, min(0.2, float(tp_conf_adj)))

    sl_nudge = (conf / 10.0) * sl_conf_adj    # tighten SL
    tp_nudge = (conf / 10.0) * tp_conf_adj    # widen TP

    sl_pts = max(0.01, sl_base * (1.0 - sl_nudge))
    tp_pts = max(0.01, tp_base * (1.0 + tp_nudge))
    return sl_pts, tp_pts
