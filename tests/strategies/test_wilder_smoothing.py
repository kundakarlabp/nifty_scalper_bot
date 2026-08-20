"""ATR/RSI smoothing must match the platform-standard Wilder method (P1)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from nifty_scalper_bot.strategies.indicators import IndicatorEngine


def _wilder_reference(true_ranges: list[float], period: int) -> float:
    atr = sum(true_ranges[:period]) / period
    for value in true_ranges[period:]:
        atr += (value - atr) / period
    return atr


def _bars(ranges: list[float]) -> tuple[list[float], list[float], list[float]]:
    """Build OHLC where each bar's true range equals the requested range."""
    highs, lows, closes = [], [], []
    base = 100.0
    closes.append(base)
    highs.append(base)
    lows.append(base)
    for width in ranges:
        highs.append(base + width)
        lows.append(base)
        closes.append(base)
    return highs, lows, closes


def test_atr_uses_wilder_smoothing_by_default() -> None:
    engine = IndicatorEngine()
    ranges = [1.0] * 13 + [20.0] + [1.0] * 5
    highs, lows, closes = _bars(ranges)

    result = engine._calculate_atr(highs, lows, closes, 14)

    assert result == pytest.approx(_wilder_reference(ranges, 14))


def test_wilder_atr_retains_the_spike_a_rolling_mean_discards(
    monkeypatch,
) -> None:
    """A wide bar leaving a 14-bar window must not step ATR down on its own."""
    engine = IndicatorEngine()
    ranges = [1.0] * 13 + [20.0] + [1.0] * 14
    highs, lows, closes = _bars(ranges)

    wilder = engine._calculate_atr(highs, lows, closes, 14)

    monkeypatch.setenv("INDICATORS__WILDER_ATR", "false")
    rolling = engine._calculate_atr(highs, lows, closes, 14)

    # The spike has dropped out of the rolling window entirely.
    assert rolling == pytest.approx(1.0)
    assert wilder > rolling


def test_atr_still_raises_on_insufficient_history() -> None:
    engine = IndicatorEngine()
    highs, lows, closes = _bars([1.0] * 3)

    with pytest.raises(ValueError):
        engine._calculate_atr(highs, lows, closes, 14)


def test_rsi_wilder_mode_is_opt_in(monkeypatch) -> None:
    engine = IndicatorEngine()
    prices = [100.0 + i for i in range(10)] + [110.0 - i for i in range(15)]

    default_rsi = engine._calculate_rsi(prices, 14)
    monkeypatch.setenv("INDICATORS__WILDER_RSI", "true")
    wilder_rsi = engine._calculate_rsi(prices, 14)

    assert 0.0 <= default_rsi <= 100.0
    assert 0.0 <= wilder_rsi <= 100.0
    assert default_rsi != pytest.approx(wilder_rsi)


def test_rsi_edge_cases_are_preserved(monkeypatch) -> None:
    engine = IndicatorEngine()
    monkeypatch.setenv("INDICATORS__WILDER_RSI", "true")

    flat = [100.0] * 20
    assert engine._calculate_rsi(flat, 14) == pytest.approx(50.0)

    rising = [100.0 + i for i in range(20)]
    assert engine._calculate_rsi(rising, 14) == pytest.approx(100.0)


def test_adx_uses_wilder_directional_movement() -> None:
    engine = IndicatorEngine()
    closes = [100.0 + index for index in range(28)]
    highs = [value + 1.0 for value in closes]
    lows = [value - 1.0 for value in closes]

    assert engine._calculate_adx(highs, lows, closes, 14) == pytest.approx(100.0)


def test_adx_requires_two_wilder_windows() -> None:
    engine = IndicatorEngine()
    closes = [100.0 + index for index in range(15)]
    highs = [value + 1.0 for value in closes]
    lows = [value - 1.0 for value in closes]

    with pytest.raises(ValueError):
        engine._calculate_adx(highs, lows, closes, 14)


def test_adx_is_exposed_by_canonical_indicator_snapshots() -> None:
    engine = IndicatorEngine()
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for index in range(31):
        close = 100.0 + index
        engine.update_price(
            "NIFTY",
            {
                "open": close - 0.5,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
            },
            timestamp=start + timedelta(minutes=index),
        )

    assert engine.get_adx("NIFTY") == pytest.approx(100.0)
    assert engine.get_indicators("NIFTY")["adx"] == pytest.approx(100.0)
    assert engine.get_latest("NIFTY")["adx"] == pytest.approx(100.0)
