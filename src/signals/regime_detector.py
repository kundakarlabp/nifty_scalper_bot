#src/signals/regime_detector.py
from __future__ import annotations
import pandas as pd

def detect_market_regime(
    df: pd.DataFrame,
    adx: pd.Series,
    di_plus: pd.Series,
    di_minus: pd.Series,
    adx_trend_strength: int,
) -> str:
    """
    Rough regime classification for nudging the score.
    Returns 'trend_up', 'trend_down', 'range', or 'unknown'.
    """
    if len(adx) < 2 or len(di_plus) < 2 or len(di_minus) < 2:
        return "unknown"

    current_adx = adx.iloc[-1]
    current_di_plus = di_plus.iloc[-1]
    current_di_minus = di_minus.iloc[-1]

    if pd.isna(current_adx) or pd.isna(current_di_plus) or pd.isna(current_di_minus):
        return "unknown"

    if current_adx > adx_trend_strength and abs(current_di_plus - current_di_minus) > 10:
        return "trend_up" if current_di_plus > current_di_minus else "trend_down"

    return "range"
