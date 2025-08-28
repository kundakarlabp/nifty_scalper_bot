"""Tests for the market regime detector."""

from __future__ import annotations

import pandas as pd

from src.signals.regime_detector import detect_market_regime, RegimeResult


def _make_series(val: float) -> pd.Series:
    return pd.Series([val])


def test_strong_trend_classification() -> None:
    df = pd.DataFrame({"close": [100.0]})
    res: RegimeResult = detect_market_regime(
        df=df,
        adx=_make_series(25.0),
        di_plus=_make_series(30.0),
        di_minus=_make_series(10.0),
        bb_width=_make_series(3.5),
    )
    assert res.regime == "TREND"


def test_indecisive_returns_no_trade() -> None:
    df = pd.DataFrame({"close": [100.0]})
    res = detect_market_regime(
        df=df,
        adx=_make_series(18.0),
        di_plus=_make_series(25.0),
        di_minus=_make_series(18.0),
        bb_width=_make_series(2.5),
    )
    assert res.regime == "NO_TRADE"

