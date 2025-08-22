"""
src/signals/regime_detector.py

Market regime detection (trend vs range) using ADX and DI values.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def detect_market_regime(
    df: pd.DataFrame,
    adx: Optional[pd.Series] = None,
    di_plus: Optional[pd.Series] = None,
    di_minus: Optional[pd.Series] = None,
    adx_trend_strength: int = 20,
) -> str:
    """
    Detects market regime from spot OHLC with ADX/DI columns.

    Args:
        df: spot OHLC dataframe
        adx: ADX series (if None, will try to pull from df)
        di_plus: +DI series (if None, will try to pull from df)
        di_minus: -DI series (if None, will try to pull from df)
        adx_trend_strength: threshold for "trend" classification

    Returns:
        str: one of "trend_up", "trend_down", "range", or "unknown"
    """
    if df is None or df.empty:
        return "unknown"

    try:
        adx_series = adx if adx is not None else df[[c for c in df.columns if c.startswith("adx_")]].iloc[:, -1]
        di_plus_series = di_plus if di_plus is not None else df[[c for c in df.columns if c.startswith("di_plus_")]].iloc[:, -1]
        di_minus_series = di_minus if di_minus is not None else df[[c for c in df.columns if c.startswith("di_minus_")]].iloc[:, -1]
    except Exception:
        return "unknown"

    try:
        adx_val = float(adx_series.iloc[-1])
        di_p = float(di_plus_series.iloc[-1])
        di_m = float(di_minus_series.iloc[-1])
    except Exception:
        return "unknown"

    if adx_val >= adx_trend_strength:
        if di_p > di_m:
            return "trend_up"
        elif di_m > di_p:
            return "trend_down"
    return "range"


# Alias for backwards compatibility
def determine_regime(df: pd.DataFrame, **kwargs) -> str:
    """Legacy alias for detect_market_regime."""
    return detect_market_regime(df, **kwargs)