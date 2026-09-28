import logging
from types import SimpleNamespace

from nifty_scalper_bot.config.regime_ontology import MarketRegime
from nifty_scalper_bot.core.market_regime import (
    MarketRegimeDetector,
    RegimeSnapshot,
    classify_runner_regime,
)
from nifty_scalper_bot.core.market_regime_manager import MarketRegimeManager
from nifty_scalper_bot.strategies.runner import StrategyRunner


def test_runner_regime_trend_classification() -> None:
    regime = classify_runner_regime(
        {
            "adx": 28.0,
            "atr": 10.0,
            "atr_average": 8.0,
            "vwap_slope": 0.2,
            "volume_expansion": 1.2,
        }
    )
    assert regime is MarketRegime.TREND


def test_runner_regime_volatile_classification() -> None:
    regime = classify_runner_regime(
        {
            "adx": 18.0,
            "atr": 20.0,
            "atr_average": 10.0,
            "vwap_slope": 0.0,
            "volume_expansion": 1.0,
        }
    )
    assert regime is MarketRegime.VOLATILE


def test_runner_regime_range_classification() -> None:
    regime = classify_runner_regime(
        {
            "adx": 10.0,
            "atr": 8.0,
            "atr_average": 8.5,
            "vwap_slope": 0.001,
            "volume_expansion": 1.0,
        }
    )
    assert regime is MarketRegime.RANGE


def test_runner_regime_low_activity_classification() -> None:
    regime = classify_runner_regime(
        {
            "adx": 20.0,
            "atr": 7.0,
            "atr_average": 7.5,
            "vwap_slope": 0.001,
            "volume_expansion": 0.4,
        }
    )
    assert regime is MarketRegime.LOW_ACTIVITY


def test_runner_regime_missing_adx_is_unknown_not_range() -> None:
    regime = classify_runner_regime(
        {
            "adx": None,
            "atr": 8.0,
            "atr_average": 8.5,
            "vwap_slope": 0.001,
            "volume_expansion": 1.0,
        }
    )
    assert regime is MarketRegime.UNKNOWN


def test_runner_uses_central_stable_regime_manager() -> None:
    manager = MarketRegimeManager(
        MarketRegimeDetector(),
        transition_confirmations=2,
    )
    manager.ingest_snapshot(RegimeSnapshot("NIFTY", "trend", 0.80, "baseline", 1.0, {}))
    manager.ingest_snapshot(RegimeSnapshot("NIFTY", "range", 0.70, "pending", 2.0, {}))

    runner = StrategyRunner.__new__(StrategyRunner)
    runner._strategy_manager = SimpleNamespace(_regime_manager=manager)
    runner._last_regime_inputs_by_symbol = {}
    runner._last_regime_by_symbol = {}
    runner._logger = logging.getLogger("test.runner.regime")

    regime = runner._compute_regime_snapshot("NSE:NIFTY")

    assert regime is MarketRegime.TREND
    inputs = runner._last_regime_inputs_by_symbol["NSE:NIFTY"]
    assert inputs["source"] == "market_regime_manager"
    assert inputs["stable_regime"] == "trend"
    assert inputs["raw_regime"] == "range"


def test_runner_without_central_regime_manager_fails_closed_unknown() -> None:
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._strategy_manager = SimpleNamespace(_regime_manager=None)
    runner._last_regime_inputs_by_symbol = {}
    runner._last_regime_by_symbol = {}

    assert runner._compute_regime_snapshot("NSE:NIFTY") is MarketRegime.UNKNOWN
