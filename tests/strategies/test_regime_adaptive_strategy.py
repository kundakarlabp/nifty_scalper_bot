from __future__ import annotations

from typing import Any, Dict

from nifty_scalper_bot.strategies.regime_adaptive import RegimeAdaptiveStrategy


class _DummyMarketDataManager:
    def __init__(self, tick: Dict[str, Any]) -> None:
        self._tick = tick

    def set_tick(self, tick: Dict[str, Any]) -> None:
        self._tick = tick

    def get_latest_tick(self, symbol: str) -> Dict[str, Any] | None:
        return dict(self._tick)


def _base_indicators() -> Dict[str, float]:
    return {
        "atr": 12.0,
        "atr_14": 12.0,
        "volume": 1500.0,
        "avg_volume": 800.0,
        "ema_fast": 105.0,
        "ema_slow": 100.0,
        "trend_score": 0.8,
        "iv_rank": 50.0,
    }


def test_regime_adaptive_trend_signal() -> None:
    tick = {
        "open_interest": 8_000,
        "implied_volatility": 0.45,
        "atr": 15.0,
        "volume": 1800.0,
        "avg_volume": 900.0,
    }
    mdm = _DummyMarketDataManager(tick)
    strategy = RegimeAdaptiveStrategy(market_data_manager=mdm)
    indicators = _base_indicators()
    signal = strategy.generate_signal(
        "NIFTY",
        indicators,
        current_price=100.0,
        position=None,
    )
    assert signal is not None
    assert signal.quantity >= 1
    assert signal.metadata["regime"] == "trend"


def test_regime_adaptive_range_filters_block() -> None:
    tick = {
        "open_interest": 2_000,
        "implied_volatility": 0.2,
        "atr": 3.0,
        "volume": 500.0,
        "avg_volume": 800.0,
    }
    mdm = _DummyMarketDataManager(tick)
    strategy = RegimeAdaptiveStrategy(market_data_manager=mdm)
    indicators = _base_indicators()
    indicators["ema_fast"] = 99.0
    indicators["ema_slow"] = 100.0
    indicators["trend_score"] = 0.1
    indicators["volume"] = 400.0
    indicators["avg_volume"] = 800.0
    indicators["atr"] = 3.0
    signal = strategy.generate_signal(
        "NIFTY",
        indicators,
        current_price=100.0,
        position=None,
    )
    assert signal is None
