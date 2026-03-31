from __future__ import annotations

import pytest

from nifty_scalper_bot.strategies.vwap_mean_reversion import (
    VWAPMeanReversionStrategy,
    VWAPSignal,
)


def test_vwap_strategy_generates_sell_signal() -> None:
    strategy = VWAPMeanReversionStrategy(lookback=3, threshold_pct=0.001)
    market_data = {
        "NIFTY": {"ltp": 101.2},
        "NIFTY_bars": [
            {"close": 90.0, "volume": 100.0},
        ],
        "NIFTYFUT_bars": [
            {"close": 100.0, "volume": 50.0},
            {"close": 100.5, "volume": 60.0},
            {"close": 100.2, "volume": 40.0},
        ],
    }
    signal = strategy.generate_signal(market_data)
    assert isinstance(signal, VWAPSignal)
    assert signal.action == "SELL"
    assert signal.reason == "spot_above_vwap"
    assert signal.metadata["total_volume"] == pytest.approx(150.0)
    assert signal.metadata["vwap"] == pytest.approx(100.231, rel=1e-3)
    assert signal.metadata["threshold_pct"] >= 0.001
    assert signal.metadata["volatility_pct"] >= 0.0


def test_vwap_strategy_generates_buy_signal() -> None:
    strategy = VWAPMeanReversionStrategy(lookback=2, threshold_pct=0.001)
    market_data = {
        "NIFTY": {"ltp": 98.0},
        "NIFTYFUT_bars": [
            {"close": 100.0, "volume": 70.0},
            {"close": 99.5, "volume": 80.0},
        ],
    }
    signal = strategy.generate_signal(market_data)
    assert signal.action == "BUY"
    assert signal.reason == "spot_below_vwap"
    assert signal.metadata["bars_used"] == 2
    assert signal.metadata["threshold_pct"] >= 0.001
    assert signal.metadata["volatility_pct"] >= 0.0


def test_vwap_strategy_returns_hold_when_volume_missing() -> None:
    strategy = VWAPMeanReversionStrategy()
    market_data = {"NIFTY": {"ltp": 100.0}}
    signal = strategy.generate_signal(market_data)
    assert signal.action == "HOLD"
    assert signal.reason == "volume_data_missing"
