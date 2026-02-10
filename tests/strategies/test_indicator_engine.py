"""Unit tests for :mod:`nifty_scalper_bot.strategies.indicators`."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging

import numpy as np
import pytest

from nifty_scalper_bot.strategies.indicators import IndicatorEngine, PriceHistory


@pytest.fixture()
def engine() -> IndicatorEngine:
    return IndicatorEngine()


def _time_series(count: int) -> list[datetime]:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return [base + timedelta(minutes=index) for index in range(count)]


def test_price_history_capacity() -> None:
    history = PriceHistory(max_length=5)
    timestamps = _time_series(10)
    for idx, ts in enumerate(timestamps):
        history.add_tick(float(idx), volume=idx + 1, timestamp=ts)
    assert len(history) == 5
    assert history.get_closes() == [5.0, 6.0, 7.0, 8.0, 9.0]


def test_rsi_matches_reference(engine: IndicatorEngine) -> None:
    closes = [
        44.34,
        44.09,
        44.15,
        43.61,
        44.33,
        44.83,
        45.10,
        45.42,
        45.84,
        46.08,
        45.89,
        46.03,
        45.61,
        46.28,
        46.28,
    ]
    for ts, close in zip(_time_series(len(closes)), closes, strict=True):
        engine.update_price("RSI", close, volume=100, timestamp=ts)
    rsi = engine.get_rsi("RSI", period=14)
    assert rsi is not None
    assert rsi == pytest.approx(70.464, rel=1e-3)


def _ema_reference(prices: list[float], period: int) -> float:
    alpha = 2.0 / (period + 1.0)
    ema = float(np.mean(prices[:period]))
    for price in prices[period:]:
        ema += alpha * (price - ema)
    return ema


def test_ema_matches_reference(engine: IndicatorEngine) -> None:
    closes = [float(value) for value in range(1, 41)]
    for ts, close in zip(_time_series(len(closes)), closes, strict=True):
        engine.update_price("EMA", close, volume=100, timestamp=ts)
    ema = engine.get_ema("EMA", period=10)
    assert ema is not None
    assert ema == pytest.approx(_ema_reference(closes, 10), rel=1e-12)


def test_sma_matches_reference(engine: IndicatorEngine) -> None:
    closes = [float(value) for value in range(1, 26)]
    for ts, close in zip(_time_series(len(closes)), closes, strict=True):
        engine.update_price("SMA", close, volume=50, timestamp=ts)
    sma = engine.get_sma("SMA", period=5)
    assert sma is not None
    assert sma == pytest.approx(sum(closes[-5:]) / 5)


def test_macd_matches_reference(engine: IndicatorEngine) -> None:
    closes = [float(value) for value in range(1, 80)]
    for ts, close in zip(_time_series(len(closes)), closes, strict=True):
        engine.update_price("MACD", close, volume=10, timestamp=ts)
    macd = engine.get_macd("MACD")
    assert macd is not None
    fast_series = _ema_series(np.asarray(closes), 12)
    slow_series = _ema_series(np.asarray(closes), 26)
    macd_series = fast_series - slow_series
    valid_macd = macd_series[~np.isnan(macd_series)]
    signal_series = _ema_series(valid_macd, 9)
    expected_macd = float(valid_macd[-1])
    expected_signal = float(signal_series[~np.isnan(signal_series)][-1])
    expected_hist = expected_macd - expected_signal
    assert macd[0] == pytest.approx(expected_macd, rel=1e-12)
    assert macd[1] == pytest.approx(expected_signal, rel=1e-12)
    assert macd[2] == pytest.approx(expected_hist, rel=1e-12)


def _ema_series(values: np.ndarray, period: int) -> np.ndarray:
    result = np.empty_like(values, dtype=float)
    result[:] = np.nan
    if values.size < period:
        return result
    alpha = 2.0 / (period + 1.0)
    ema = float(np.mean(values[:period]))
    result[period - 1] = ema
    for idx in range(period, values.size):
        ema += alpha * (values[idx] - ema)
        result[idx] = ema
    return result


def test_bollinger_bands(engine: IndicatorEngine) -> None:
    closes = [float(value) for value in range(10, 50)]
    for ts, close in zip(_time_series(len(closes)), closes, strict=True):
        engine.update_price("BOLL", close, volume=1, timestamp=ts)
    bands = engine.get_bollinger_bands("BOLL", period=20, std_dev=2.0)
    assert bands is not None
    upper, middle, lower = bands
    window = np.asarray(closes[-20:], dtype=float)
    expected_middle = float(np.mean(window))
    expected_std = float(np.std(window, ddof=0))
    assert upper == pytest.approx(expected_middle + 2.0 * expected_std)
    assert middle == pytest.approx(expected_middle)
    assert lower == pytest.approx(expected_middle - 2.0 * expected_std)


def test_get_indicators_accepts_none_names(engine: IndicatorEngine) -> None:
    """Verify indicator guard. Args: engine. Returns: None. Raises: Exception."""
    logger = logging.getLogger(__name__)
    try:
        engine.update_price(
            'SNAPSHOT',
            {'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.5},
            volume=10,
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        result = engine.get_indicators('SNAPSHOT', None)
        assert 'vwap' in result
        result = engine.get_indicators('SNAPSHOT', [None, 'vwap'])
        assert 'vwap' in result
    except Exception as e:
        logger.error('Failure in test_get_indicators_accepts_none_names: %s', e)
        raise


def test_atr_matches_reference(engine: IndicatorEngine) -> None:
    highs = [
        48.70,
        48.72,
        48.90,
        48.87,
        48.82,
        49.05,
        49.20,
        49.35,
        49.92,
        50.19,
        50.12,
        49.66,
        49.88,
        50.19,
        50.36,
    ]
    lows = [
        47.79,
        48.14,
        48.39,
        48.37,
        48.24,
        48.64,
        48.94,
        48.86,
        49.50,
        49.87,
        49.20,
        48.90,
        49.43,
        49.73,
        49.26,
    ]
    closes = [
        48.16,
        48.61,
        48.75,
        48.63,
        48.74,
        49.03,
        49.07,
        49.32,
        49.91,
        50.13,
        49.53,
        49.50,
        49.75,
        50.03,
        49.50,
    ]
    timestamps = _time_series(len(closes))
    for idx, ts in enumerate(timestamps):
        engine.update_price(
            "ATR",
            {
                "open": closes[idx],
                "high": highs[idx],
                "low": lows[idx],
                "close": closes[idx],
            },
            volume=100,
            timestamp=ts,
        )
    atr = engine.get_atr("ATR", period=14)
    assert atr is not None
    expected_atr = _atr_reference(highs, lows, closes, 14)
    assert atr == pytest.approx(expected_atr, rel=1e-12)


def test_ensure_min_bars_uses_hydrator(engine: IndicatorEngine) -> None:
    payload = [
        {
            'open': 100.0,
            'high': 101.0,
            'low': 99.0,
            'close': 100.5,
            'volume': 1,
            'timestamp': datetime(2024, 1, 1, tzinfo=timezone.utc),
        },
        {
            'open': 100.5,
            'high': 102.0,
            'low': 100.0,
            'close': 101.0,
            'volume': 1,
            'timestamp': datetime(2024, 1, 1, 0, 1, tzinfo=timezone.utc),
        },
    ]

    ready = engine.ensure_min_bars('TEST', 2, hydrate=lambda _s, _m: payload)

    assert ready is True
    assert engine.has_min_bars('TEST', 2) is True


def _atr_reference(
    highs: list[float], lows: list[float], closes: list[float], period: int
) -> float:
    true_ranges = []
    for idx in range(1, len(highs)):
        true_ranges.append(
            max(
                highs[idx] - lows[idx],
                abs(highs[idx] - closes[idx - 1]),
                abs(lows[idx] - closes[idx - 1]),
            )
        )
    window = true_ranges[-period:]
    return float(sum(window) / len(window))


def test_vwap_matches_reference(engine: IndicatorEngine) -> None:
    closes = [101.0, 102.0, 103.5, 104.0, 105.0]
    volumes = [100, 150, 120, 80, 200]
    timestamps = _time_series(len(closes))
    for close, volume, ts in zip(closes, volumes, timestamps, strict=True):
        engine.update_price("VWAP", close, volume=volume, timestamp=ts)
    vwap = engine.get_vwap("VWAP", period=5)
    assert vwap is not None
    expected = float(np.dot(closes, volumes) / sum(volumes))
    assert vwap == pytest.approx(expected)


def test_get_indicators_returns_all_keys(engine: IndicatorEngine) -> None:
    closes = [float(value) for value in range(1, 60)]
    timestamps = _time_series(len(closes))
    for close, ts in zip(closes, timestamps, strict=True):
        engine.update_price("ALL", close, volume=10, timestamp=ts)
    indicators = engine.get_indicators("ALL")
    expected_keys = {
        "rsi",
        "ema",
        "sma",
        "macd",
        "macd_signal",
        "macd_histogram",
        "bollinger_upper",
        "bollinger_middle",
        "bollinger_lower",
        "atr",
        "vwap",
    }
    assert expected_keys.issubset(indicators.keys())


def test_get_indicators_accepts_none_names(engine: IndicatorEngine) -> None:
    ts = _time_series(1)[0]
    engine.update_price("NIFTY", 100.0, volume=10, timestamp=ts)

    indicators = engine.get_indicators("NIFTY")

    assert "close" in indicators


def test_is_ready(engine: IndicatorEngine) -> None:
    timestamps = _time_series(5)
    for ts in timestamps:
        engine.update_price("READY", 100.0, volume=1, timestamp=ts)
    assert not engine.is_ready("READY", min_bars=10)
    for extra in range(10):
        engine.update_price(
            "READY",
            100.0 + extra,
            volume=1,
            timestamp=timestamps[-1] + timedelta(minutes=extra + 1),
        )
    assert engine.is_ready("READY", min_bars=10)
