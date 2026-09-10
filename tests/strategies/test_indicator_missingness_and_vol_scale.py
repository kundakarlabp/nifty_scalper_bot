"""Indicator missingness and realised-vol scaling.

Current VWAP evidence must never be a previous session's value. Slope must
not look like a flat market when the series is simply unavailable. Minute
returns must not be annualised as if they were daily.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from nifty_scalper_bot.strategies.indicators import (
    IndicatorEngine,
    realised_vol_annualisation_factor,
)

_IST = timezone(timedelta(hours=5, minutes=30))
_SYMBOL = "NFO:NIFTY2680424400CE"


def _session_stamp(minute: int) -> datetime:
    return datetime(2026, 8, 3, 9, 15, tzinfo=_IST) + timedelta(minutes=minute)


def test_rolling_vwap_does_not_present_a_stale_value_as_current() -> None:
    engine = IndicatorEngine()
    for minute in range(20):
        engine.update_price(
            _SYMBOL,
            {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0 + minute},
            volume=100,
            timestamp=_session_stamp(minute),
        )

    live = engine.get_vwap(_SYMBOL)
    assert live is not None

    for minute in range(20, 40):
        engine.update_price(
            _SYMBOL,
            {"open": 200.0, "high": 200.0, "low": 200.0, "close": 200.0},
            volume=0,
            timestamp=_session_stamp(minute),
        )

    assert engine.get_vwap(_SYMBOL) is None
    diagnostic = engine.get_stale_vwap_diagnostic(_SYMBOL)
    assert diagnostic is not None
    assert diagnostic["stale"] is True
    assert diagnostic["value"] == live


def test_slope_is_none_when_history_is_missing() -> None:
    engine = IndicatorEngine()

    assert engine.calculate_slope(_SYMBOL) is None
    assert engine.calculate_slope(_SYMBOL, indicator_name="volume") is None


def test_slope_is_none_for_an_unsupported_field_even_with_history() -> None:
    engine = IndicatorEngine()
    engine.update_price(
        _SYMBOL,
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0},
        volume=10,
        timestamp=_session_stamp(0),
    )
    for minute in range(1, 6):
        engine.update_price(
            _SYMBOL,
            {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0 + minute},
            volume=10,
            timestamp=_session_stamp(minute),
        )

    assert engine.calculate_slope(_SYMBOL, indicator_name="open") is None
    assert engine.calculate_slope(_SYMBOL, indicator_name="close") is not None


def test_minute_realised_vol_is_not_scaled_as_daily() -> None:
    """sqrt(252) would understate 1-minute realised vol by about 19x."""
    engine = IndicatorEngine()
    price = 100.0
    for minute in range(25):
        price *= 1.002 if minute % 3 else 0.997
        engine.update_price(
            _SYMBOL,
            {"open": price, "high": price, "low": price, "close": price},
            volume=100,
            timestamp=_session_stamp(minute),
        )

    value = engine.get_volatility_index(_SYMBOL, window=20)
    assert value is not None

    history = engine._histories[_SYMBOL]
    closes = history.get_closes(21)
    returns = [(closes[idx] / closes[idx - 1]) - 1.0 for idx in range(1, len(closes))]
    recent = returns[-20:]
    mean_return = sum(recent) / 20
    stdev = (sum((item - mean_return) ** 2 for item in recent) / 20) ** 0.5
    wrong = stdev * (252.0**0.5) * 100.0
    expected = stdev * realised_vol_annualisation_factor(60.0) * 100.0

    assert value == pytest.approx(expected)
    assert value / wrong == pytest.approx(
        realised_vol_annualisation_factor(60.0) / (252.0**0.5)
    )
    assert realised_vol_annualisation_factor(60.0) == pytest.approx(
        (252.0 * 375.0) ** 0.5
    )
