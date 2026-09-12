from __future__ import annotations

import pytest

from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    EliteStrategiesSettings,
    GammaScalpingStrategyConfig,
    OIMaxPainStrategyConfig,
    OrderFlowStrategyConfig,
)
from nifty_scalper_bot.strategies.elite_strategies.gamma_scalping import (
    GammaScalpingStrategy,
)
from nifty_scalper_bot.strategies.elite_strategies.oi_max_pain import OIMaxPainStrategy
from nifty_scalper_bot.strategies.elite_validator import (
    EliteConfigError,
    validate_elite_config,
)


def test_oi_max_pain_uses_spot_not_option_premium() -> None:
    strategy = OIMaxPainStrategy(
        OIMaxPainStrategyConfig(min_confidence=0.0, min_deviation_pct=0.5),
        indicator_engine=None,
    )

    signal = strategy.generate_signal(
        "NFO:NIFTY2691524000CE",
        {
            "spot_price": 24_000.0,
            "max_pain": 24_200.0,
            "call_oi_wall": 24_500.0,
            "put_oi_wall": 23_800.0,
            "direction_bias": "CE",
        },
        100.0,
    )

    assert signal is not None
    assert signal.metadata["role"] == "context"
    assert signal.metadata["can_trigger"] is False
    assert signal.metadata["underlying_reference_price"] == 24_000.0
    assert signal.metadata["max_pain_deviation_pct"] == pytest.approx(0.8333, abs=1e-4)


def test_oi_max_pain_option_requires_underlying_spot() -> None:
    strategy = OIMaxPainStrategy(
        OIMaxPainStrategyConfig(min_confidence=0.0), indicator_engine=None
    )

    signal = strategy.generate_signal(
        "NFO:NIFTY2691524000CE",
        {"max_pain": 24_200.0, "direction_bias": "CE"},
        100.0,
    )

    assert signal is None


def test_gamma_pe_buy_keeps_premium_stop_below_entry(monkeypatch) -> None:
    monkeypatch.setenv("STRATEGY_MODE", "expiry_gamma")
    monkeypatch.setenv("ALLOW_EXPIRY_GAMMA_STRATEGIES", "true")
    strategy = GammaScalpingStrategy(
        GammaScalpingStrategyConfig(min_confidence=0.0, min_gamma=0.0005),
        indicator_engine=None,
    )

    signal = strategy.generate_signal(
        "NFO:NIFTY2691524000PE",
        {
            "gamma": 0.002,
            "theta": -2.0,
            "atr": 5.0,
            "direction_bias": "PE",
            "volatility_expansion_confirmed": True,
            "days_to_expiry": 0,
        },
        100.0,
    )

    assert signal is not None
    assert signal.stop_loss is not None and signal.stop_loss < 100.0
    assert signal.take_profit is not None and signal.take_profit > 100.0
    assert signal.metadata["side"] == "PE"
    assert signal.metadata["expiry_day"] is True


def test_gamma_rejects_non_expiry_session(monkeypatch) -> None:
    monkeypatch.setenv("STRATEGY_MODE", "expiry_gamma")
    monkeypatch.setenv("ALLOW_EXPIRY_GAMMA_STRATEGIES", "true")
    strategy = GammaScalpingStrategy(
        GammaScalpingStrategyConfig(min_confidence=0.0), indicator_engine=None
    )

    signal = strategy.generate_signal(
        "NFO:NIFTY2691624000CE",
        {
            "gamma": 0.002,
            "theta": -2.0,
            "atr": 5.0,
            "direction_bias": "CE",
            "volatility_expansion_confirmed": True,
            "days_to_expiry": 1,
        },
        100.0,
    )

    assert signal is None


def test_validator_checks_current_orderflow_fields() -> None:
    elite = EliteStrategiesSettings(
        order_flow=OrderFlowStrategyConfig(imbalance_ratio_min=0.5)
    )

    with pytest.raises(EliteConfigError, match="imbalance_ratio_min"):
        validate_elite_config(elite)
