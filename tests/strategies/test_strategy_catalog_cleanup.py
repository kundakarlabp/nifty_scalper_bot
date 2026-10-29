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
from nifty_scalper_bot.strategies.elite_strategies.cpr_breakout import (
    CPRBreakoutStrategy,
)
from nifty_scalper_bot.strategies.elite_strategies.gamma_scalping import (
    GammaScalpingStrategy,
)
from nifty_scalper_bot.strategies.elite_strategies.rsi_divergence import (
    RSIDivergenceStrategy,
)
from nifty_scalper_bot.strategies.entry_evidence import build_execution_evidence


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
    bb = BBSqueezeStrategy(BBSqueezeStrategyConfig(), None)
    cpr = CPRBreakoutStrategy(CPRBreakoutStrategyConfig(), None)
    rsi = RSIDivergenceStrategy(RSIDivergenceStrategyConfig(), None)

    assert bb._evaluate_signal("NSE:NIFTY", {"atr": 0.0}, 24000.0) is None
    assert cpr._evaluate_signal("NSE:NIFTY", {"atr": 0.0}, 24000.0) is None
    assert rsi._evaluate_signal("NSE:NIFTY", {"atr": 0.0}, 24000.0) is None


def test_expiry_gamma_requires_observed_atr(monkeypatch) -> None:
    monkeypatch.setenv("STRATEGY_MODE", "expiry_gamma")
    monkeypatch.setenv("ALLOW_EXPIRY_GAMMA_STRATEGIES", "true")
    strategy = GammaScalpingStrategy(
        GammaScalpingStrategyConfig(),
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
    evidence = build_execution_evidence(
        {
            "direction_bias": "CE",
            "quote_depth_valid": True,
            "tradable_quote": True,
            "regime": "TREND_UP",
        },
        side="CE",
    )

    assert evidence["spread_observed"] is False
    assert evidence["spread_ok"] is None
    assert evidence["spread_pct"] is None
    # A producer's provisional flag is not an executable bid/ask proof.
    assert evidence["execution_evidence_pass"] is False


def test_execution_evidence_rejects_supplied_understated_spread(monkeypatch) -> None:
    monkeypatch.setenv("ORDER_MAX_SPREAD_PCT", "1.0")
    quote = {
        "underlying_direction_bias": "CE",
        "required_data_present": True,
        "bid": 95.0,
        "ask": 105.0,
        "spread_pct": 0.1,  # stale/mis-scaled producer value
    }
    evidence = build_execution_evidence(quote, side="CE")

    assert evidence["bid_ask_valid"] is True
    assert evidence["spread_observed"] is True
    assert evidence["spread_pct"] == 10.0
    assert evidence["spread_ok"] is False
    assert evidence["execution_evidence_pass"] is False


def test_execution_evidence_preserves_stricter_spread(monkeypatch) -> None:
    monkeypatch.setenv("ORDER_MAX_SPREAD_PCT", "1.0")
    evidence = build_execution_evidence(
        {
            "bid": 99.9,
            "ask": 100.1,
            "spread_pct": 2.0,  # do not discard stricter producer evidence
        },
        side="CE",
    )

    assert evidence["spread_pct"] == 2.0
    assert evidence["spread_ok"] is False
    assert evidence["execution_evidence_pass"] is False


def test_proven_quote_evidence_requires_finite_bid_ask_and_spread(monkeypatch) -> None:
    monkeypatch.setenv("ORDER_MAX_SPREAD_PCT", "1.0")
    baseline = {
        "underlying_direction_bias": "CE",
        "required_data_present": True,
        "tradable_quote": True,
        "bid": 99.9,
        "ask": 100.1,
    }
    assert (
        build_execution_evidence(baseline, side="CE")["execution_evidence_pass"] is True
    )
    for invalid in (
        {"bid": float("nan")},
        {"ask": float("inf")},
        {"bid": 101.0, "ask": 100.0},
        {"stale_data_used": True},
        {"required_data_present": False},
        {"bid": None, "ask": None},
    ):
        assert (
            build_execution_evidence({**baseline, **invalid}, side="CE")[
                "execution_evidence_pass"
            ]
            is False
        )
