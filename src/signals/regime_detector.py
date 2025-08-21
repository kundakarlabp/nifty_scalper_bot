from __future__ import annotations

from typing import Literal, Optional, Tuple

import pandas as pd

Regime = Literal["trend_up", "trend_down", "range"]


def detect_market_regime(
    df: pd.DataFrame,
    *,
    adx: Optional[pd.Series] = None,
    di_plus: Optional[pd.Series] = None,
    di_minus: Optional[pd.Series] = None,
    adx_trend_strength: float = 20.0,
    di_diff_threshold: float = 10.0,
) -> Tuple[Regime, float]:
    """
    Return (regime, strength) where strength is ~0..100.
    Heuristics: ADX high + DI spread => trend; else range.
    """
    if df is None or df.empty:
        return "range", 0.0

    adx_val = float(adx.iloc[-1]) if adx is not None and len(adx) else 0.0
    dip = float(di_plus.iloc[-1]) if di_plus is not None and len(di_plus) else 0.0
    dim = float(di_minus.iloc[-1]) if di_minus is not None and len(di_minus) else 0.0
    di_spread = abs(dip - dim)

    if adx_val >= adx_trend_strength and di_spread >= di_diff_threshold:
        if dip > dim:
            return "trend_up", min(100.0, adx_val + di_spread)
        else:
            return "trend_down", min(100.0, adx_val + di_spread)
    return "range", max(0.0, adx_val - adx_trend_strength)