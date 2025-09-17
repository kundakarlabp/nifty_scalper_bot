# src/signals/regime_detector.py
"""Market regime detection helpers.

This module classifies the current market regime into one of three buckets:
"TREND", "RANGE", or "NO_TRADE".  The thresholds are deterministic and can be
overridden by wiring a custom configuration, but defaults are provided for
common usage.  The implementation intentionally has no external dependencies
besides ``pandas`` so it can be unit tested easily.

The main entry point :func:`detect_market_regime` returns a :class:`RegimeResult`
dataclass with rich diagnostics which upstream components can use for scoring
and logging purposes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import pandas as pd

from src.config import RegimeSettings, settings


@dataclass
class RegimeResult:
    """Outcome of market regime detection.

    Parameters
    ----------
    regime:
        One of ``"TREND"``, ``"RANGE"`` or ``"NO_TRADE"``.
    adx, di_plus, di_minus, bb_width_pct:
        The numeric inputs used for the decision; included for transparency in
        diagnostics and logging.
    reason:
        Short human readable string explaining why the regime was classified as
        such.  This is especially useful when the result is ``"NO_TRADE"``.
    """

    regime: Literal["TREND", "RANGE", "NO_TRADE"]
    adx: float
    di_plus: float
    di_minus: float
    bb_width_pct: float
    reason: str


def _pick_col(df: pd.DataFrame, base: str) -> Optional[pd.Series]:
    """Return column ``base`` from ``df`` if present.

    The helper is tolerant to columns having a numeric suffix such as ``adx_14``
    and will return the last matching column when multiple variants are
    present.  ``None`` is returned when no appropriate column exists.
    """

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
    adx_trend_threshold: float | None = None,
    di_delta_trend_threshold: float | None = None,
    bb_width_trend_threshold: float | None = None,
    adx_range_threshold: float | None = None,
    di_delta_range_threshold: float | None = None,
    bb_width_range_threshold: float | None = None,
) -> RegimeResult:
    """Classify the market regime using ADX/DI and Bollinger width.

    Parameters are flexible; callers may pre‑compute indicator series and pass
    them explicitly, otherwise they will be sourced from ``df`` if available.

    Returns
    -------
    RegimeResult
        Dataclass containing the detected regime and diagnostic values.
    """

    if df is None or df.empty:
        return RegimeResult("NO_TRADE", 0.0, 0.0, 0.0, 0.0, "empty_df")

    # Pull series from df if not explicitly provided
    adx = adx if adx is not None else _pick_col(df, "adx")
    di_plus = di_plus if di_plus is not None else _pick_col(df, "di_plus")
    di_minus = di_minus if di_minus is not None else _pick_col(df, "di_minus")

    if bb_width is None:
        # Compute Bollinger band width percentage if not supplied.  Any failure
        # to compute is treated as a no‑trade regime to stay conservative.
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
        dip = float(
            (di_plus.iloc[-1] if di_plus is not None and len(di_plus) else 0.0) or 0.0
        )
        dim = float(
            (di_minus.iloc[-1] if di_minus is not None and len(di_minus) else 0.0)
            or 0.0
        )
        bb_width_val = float(
            (bb_width.iloc[-1] if bb_width is not None and len(bb_width) else 0.0)
            or 0.0
        )
    except Exception:
        return RegimeResult("NO_TRADE", 0.0, 0.0, 0.0, 0.0, "bad_inputs")

    di_delta = abs(dip - dim)

    try:
        regime_cfg = settings.regime
    except AttributeError:  # pragma: no cover - defensive fallback
        regime_cfg = RegimeSettings()

    adx_trend_threshold = float(
        adx_trend_threshold
        if adx_trend_threshold is not None
        else regime_cfg.adx_trend
    )
    di_delta_trend_threshold = float(
        di_delta_trend_threshold
        if di_delta_trend_threshold is not None
        else regime_cfg.di_delta_trend
    )
    bb_width_trend_threshold = float(
        bb_width_trend_threshold
        if bb_width_trend_threshold is not None
        else regime_cfg.bb_width_trend
    )
    adx_range_threshold = float(
        adx_range_threshold
        if adx_range_threshold is not None
        else regime_cfg.adx_range
    )
    di_delta_range_threshold = float(
        di_delta_range_threshold
        if di_delta_range_threshold is not None
        else regime_cfg.di_delta_range
    )
    bb_width_range_threshold = float(
        bb_width_range_threshold
        if bb_width_range_threshold is not None
        else regime_cfg.bb_width_range
    )

    if (
        adx_val >= adx_trend_threshold
        and di_delta >= di_delta_trend_threshold
        and bb_width_val >= bb_width_trend_threshold
    ):
        return RegimeResult(
            "TREND", adx_val, dip, dim, bb_width_val, "trend_conditions"
        )

    if (
        adx_val < adx_range_threshold
        or bb_width_val < bb_width_range_threshold
        or di_delta < di_delta_range_threshold
    ):
        return RegimeResult(
            "RANGE", adx_val, dip, dim, bb_width_val, "range_conditions"
        )

    return RegimeResult("NO_TRADE", adx_val, dip, dim, bb_width_val, "indecisive")


__all__ = ["RegimeResult", "detect_market_regime"]
