from __future__ import annotations

from nifty_scalper_bot.strategies.elite_strategies.bb_squeeze import BBSqueezeStrategy
from nifty_scalper_bot.strategies.elite_strategies.builder import (
    build_elite_strategies,
    build_strategy_module,
    strategy_module_names,
)
from nifty_scalper_bot.strategies.elite_strategies.config import ELITE_STRATEGY_MODULES
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    BBSqueezeStrategyConfig,
    CPRBreakoutStrategyConfig,
    EliteStrategiesSettings,
    GammaScalpingStrategyConfig,
    RSIDivergenceStrategyConfig,
)
from nifty_scalper_bot.strategies.elite_strategies.cpr_breakout import CPRBreakoutStrategy
from nifty_scalper_bot.strategies.elite_strategies.gamma_scalping import GammaScalpingStrategy
from nifty_scalper_bot.strategies.elite_strategies.rsi_divergence import (
    RSIDivergenceStrategy,
)
from nifty_scalper_bot.strategies.signal_quality import build_trade_quality_evidence


def test_module_compatibility_list_is_derived_from_canonical_catalog() -> None:
    assert ELITE_STRATEGY_MODULES == strategy_module_names()
    assert "straddle_theta" not in ELITE_STRATEGY_MODULES


def test_compatibility_builder_uses_canonical_strategy_class() -> None:
    strategy = build_strategy_module("rsi_divergence", indicator_engine=None)
    assert isinstance(strategy, RSIDivergenceStrategy)


def test_theta_mode_does_not_restore_retired_single_leg_straddle(monkeypatch) -> None:
    monkeypatch.setenv("STRATEGY_MODE", "theta")
    strategies = build_elite_strategies(
        EliteStrategiesSettings(),
        indicator_engine=None,
    )
    assert "StraddleTheta" not in {strategy.name for strategy in strategies}


def test_secondary_context_strategies_require_observed_atr() -> None:
    bb = BBSqueezeStrategy(BBSqueezeStrategyConfig(min_confidence=0.0), None)
    cpr = CPRBreakoutStrategy(CPRBreakoutStrategyConfig(min_confidence=0.0), None)
    rsi = RSIDivergenceStrategy(RSIDivergenceStrategyConfig(min_confidence=0.0), None)

    assert bb._evaluate_signal("NSE:NIFTY", {"atr": 0.0}, 24000.0) is None
    assert cpr._evaluate_signal("NSE:NIFTY", {"atr": 0.0}, 24000.0) is None
    assert rsi._evaluate_signal("NSE:NIFTY", {"atr": 0.0}, 24000.0) is None


def test_expiry_gamma_requires_observed_atr(monkeypatch) -> None:
    monkeypatch.setenv("STRATEGY_MODE", "expiry_gamma")
    monkeypatch.setenv("ALLOW_EXPIRY_GAMMA_STRATEGIES", "true")
    strategy = GammaScalpingStrategy(
        GammaScalpingStrategyConfig(min_confidence=0.0),
        indicator_engine=None,
    )
    signal = strategy._evaluate_signal(
        "NFO:NIFTY2691523400CE",
        {
            "days_to_expiry": 0,
            "gamma": 0.002,
            "theta": -2.0,
            "direction_bias": "CE",
            "atr": 0.0,
        },
        100.0,
    )
    assert signal is None


def test_unknown_spread_is_non_blocking_but_not_passed_evidence(monkeypatch) -> None:
    monkeypatch.setenv("ORDER_MAX_SPREAD_PCT", "1.0")
    evidence = build_trade_quality_evidence(
        {
            "direction_bias": "CE",
            "quote_depth_valid": True,
            "tradable_quote": True,
            "regime": "TREND_UP",
        },
        side="CE",
    )

    assert evidence["quality_spread_observed"] is False
    assert evidence["quality_spread_pass"] is None
    assert evidence["quality_spread_status"] == "unknown"
    assert evidence["quality_spread_pct"] is None
    assert evidence["liquidity_score"] == 0.5
