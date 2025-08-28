# src/signals/regime_detector.py
from __future__ import annotations

from typing import Optional

import pandas as pd


def _pick_col(df: pd.DataFrame, base: str) -> Optional[pd.Series]:
    """Return column if exists, else try suffixed variants (e.g. adx_14)."""
    col = df.get(base)
    if col is not None:
        return col
    cand = sorted([c for c in df.columns if c.startswith(f"{base}_")])
    return df[cand[-1]] if cand else None


def detect_market_regime(
    *,
    df: pd.DataFrame,
    adx: Optional[pd.Series] = None,
    di_plus: Optional[pd.Series] = None,
    di_minus: Optional[pd.Series] = None,
    bb_width: Optional[pd.Series] = None,
) -> str:
    """
    Regime classification with hard no‑trade state.

    Returns 'trend', 'range', or 'no_trade'.
    """

    if df is None or df.empty:
        return "no_trade"

    # Pull series from df if not explicitly provided
    adx = adx if adx is not None else _pick_col(df, "adx")
    di_plus = di_plus if di_plus is not None else _pick_col(df, "di_plus")
    di_minus = di_minus if di_minus is not None else _pick_col(df, "di_minus")

    if bb_width is None:
        try:
            close = df["close"]
            mid = close.rolling(20).mean()
            std = close.rolling(20).std()
            upper = mid + 2 * std
            lower = mid - 2 * std
            bb_width = ((upper - lower) / mid) * 100.0
        except Exception:
            bb_width = None

    try:
        adx_val = float((adx.iloc[-1] if adx is not None and len(adx) else 0.0) or 0.0)
        dip = float((di_plus.iloc[-1] if di_plus is not None and len(di_plus) else 0.0) or 0.0)
        dim = float((di_minus.iloc[-1] if di_minus is not None and len(di_minus) else 0.0) or 0.0)
        bb_width_val = float((bb_width.iloc[-1] if bb_width is not None and len(bb_width) else 0.0) or 0.0)
    except Exception:
        return "no_trade"

    di_delta = abs(dip - dim)

    if adx_val >= 18 and di_delta >= 8 and bb_width_val >= 3.0:
        return "trend"
    if adx_val < 18 or bb_width_val < 2.0 or di_delta < 6:
        return "range"
    return "no_trade"
