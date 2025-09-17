"""Tests for the market regime detector."""

from __future__ import annotations

import pandas as pd

from src.config import settings
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


def test_settings_defaults_are_respected(monkeypatch) -> None:
    df = pd.DataFrame({"close": [100.0]})
    monkeypatch.setattr(settings.regime, "adx_trend", 5.0)
    monkeypatch.setattr(settings.regime, "di_delta_trend", 3.0)
    monkeypatch.setattr(settings.regime, "bb_width_trend", 0.5)

    res = detect_market_regime(
        df=df,
        adx=_make_series(10.0),
        di_plus=_make_series(14.0),
        di_minus=_make_series(9.0),
        bb_width=_make_series(0.8),
    )
    assert res.regime == "TREND"


def test_threshold_overrides_parameter() -> None:
    df = pd.DataFrame({"close": [100.0]})
    res = detect_market_regime(
        df=df,
        adx=_make_series(10.0),
        di_plus=_make_series(20.0),
        di_minus=_make_series(9.0),
        bb_width=_make_series(0.8),
        adx_trend_threshold=5.0,
        di_delta_trend_threshold=5.0,
        bb_width_trend_threshold=0.5,
    )
    assert res.regime == "TREND"

