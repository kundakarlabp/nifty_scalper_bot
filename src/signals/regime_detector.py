# src/signals/regime_detector.py
from __future__ import annotations

from typing import Optional

import pandas as pd

from src.config import settings


def detect_market_regime(
    *,
    df: pd.DataFrame,
    adx: Optional[pd.Series] = None,
    di_plus: Optional[pd.Series] = None,
    di_minus: Optional[pd.Series] = None,
    adx_trend_strength: int = 20,
) -> str:
    """
    Return 'trend_up' | 'trend_down' | 'range'

    If ADX/DI series are not provided, tries to pick them from df columns:
      - adx, di_plus, di_minus OR suffixed forms adx_14, di_plus_14, di_minus_14
    Thresholds:
      - ADX >= adx_trend_strength => trending
      - else range
      - Direction from DI+ vs DI-; tie => range
    """
    if df is None or df.empty:
        return "range"

    # Pull from df if absent
    if adx is None:
        adx = df.get("adx")
        if adx is None:
            adx_cols = sorted([c for c in df.columns if c.startswith("adx_")])
            if adx_cols:
                adx = df[adx_cols[-1]]
    if di_plus is None:
        di_plus = df.get("di_plus")
        if di_plus is None:
            dip_cols = sorted([c for c in df.columns if c.startswith("di_plus_")])
            if dip_cols:
                di_plus = df[dip_cols[-1]]
    if di_minus is None:
        di_minus = df.get("di_minus")
        if di_minus is None:
            dim_cols = sorted([c for c in df.columns if c.startswith("di_minus_")])
            if dim_cols:
                di_minus = df[dim_cols[-1]]

    try:
        adx_val = float((adx.iloc[-1] if adx is not None and len(adx) else 0.0) or 0.0)
        dip = float((di_plus.iloc[-1] if di_plus is not None and len(di_plus) else 0.0) or 0.0)
        dim = float((di_minus.iloc[-1] if di_minus is not None and len(di_minus) else 0.0) or 0.0)
    except Exception:
        return "range"

    # Optional DI difference threshold from settings
    di_diff_threshold = float(getattr(getattr(settings, "strategy", object()), "di_diff_threshold", 10.0))

    if adx_val >= float(adx_trend_strength):
        if (dip - dim) >= di_diff_threshold:
            return "trend_up"
        if (dim - dip) >= di_diff_threshold:
            return "trend_down"
        # Trending but DI unclear
        return "range"

    return "range"
