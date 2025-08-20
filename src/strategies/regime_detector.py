# src/signals/regime_detector.py
from __future__ import annotations

from typing import Literal, Tuple
import pandas as pd

Regime = Literal["trend_up", "trend_down", "range", "unknown"]

__all__ = ["detect_market_regime"]


def _align_last_common(
    a: pd.Series, b: pd.Series, c: pd.Series
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Align three series on the intersection of their indices and drop NaNs.
    Falls back to truncating to the minimum common length.
    """
    try:
        idx = a.index.intersection(b.index).intersection(c.index)
        if len(idx) > 0:
            a2 = a.reindex(idx)
            b2 = b.reindex(idx)
            c2 = c.reindex(idx)
            df = pd.concat([a2, b2, c2], axis=1).dropna()
            if not df.empty:
                return df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2]
    except Exception:
        pass
    m = min(len(a), len(b), len(c))
    return a.tail(m), b.tail(m), c.tail(m)


def detect_market_regime(
    df: pd.DataFrame,          # reserved for potential vol-aware tweaks
    adx: pd.Series,
    di_plus: pd.Series,
    di_minus: pd.Series,
    adx_trend_strength: int,
) -> Regime:
    """
    Rough regime classification for nudging the score.

    Heuristics:
      - Smooth over the last 3 values to avoid flapping.
      - If ADX >= adx_trend_strength and |DI+ - DI-| > threshold → 'trend_*'
      - Direction by DI+ vs DI-; else 'range'.
      - Returns 'trend_up' | 'trend_down' | 'range' | 'unknown'
    """
    if adx is None or di_plus is None or di_minus is None:
        return "unknown"

    adx, di_plus, di_minus = _align_last_common(adx, di_plus, di_minus)
    if len(adx) < 1:
        return "unknown"

    # Light smoothing to reduce bar-to-bar flip
    win = min(3, len(adx))
    adx_val = adx.tail(win).mean(skipna=True)
    dplus_val = di_plus.tail(win).mean(skipna=True)
    dminus_val = di_minus.tail(win).mean(skipna=True)

    if pd.isna(adx_val) or pd.isna(dplus_val) or pd.isna(dminus_val):
        return "unknown"

    di_gap = abs(dplus_val - dminus_val)
    di_gap_threshold = 10.0  # keep prior behavior, applied to smoothed values

    if adx_val >= float(adx_trend_strength) and di_gap > di_gap_threshold:
        return "trend_up" if dplus_val > dminus_val else "trend_down"

    return "range"
