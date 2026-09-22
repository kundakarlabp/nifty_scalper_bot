from nifty_scalper_bot.config.regime_ontology import MarketRegime
from nifty_scalper_bot.core.market_regime import classify_runner_regime
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


def test_runner_uses_canonical_regime_classifier() -> None:
    globals_map = StrategyRunner._compute_regime_snapshot.__globals__
    assert globals_map["classify_runner_regime"] is classify_runner_regime
