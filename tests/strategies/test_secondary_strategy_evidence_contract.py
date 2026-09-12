from __future__ import annotations

from nifty_scalper_bot.strategies.elite_strategies.bb_squeeze import BBSqueezeStrategy
from nifty_scalper_bot.strategies.elite_strategies.builder import _strategy_runtime_role
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    BBSqueezeStrategyConfig,
    CPRBreakoutStrategyConfig,
    RSIDivergenceStrategyConfig,
)
from nifty_scalper_bot.strategies.elite_strategies.cpr_breakout import CPRBreakoutStrategy
from nifty_scalper_bot.strategies.elite_strategies.rsi_divergence import RSIDivergenceStrategy
from nifty_scalper_bot.strategies.signal_quality import score_signal_quality


def test_bb_squeeze_respects_configured_threshold_without_hidden_floor() -> None:
    strategy = BBSqueezeStrategy(BBSqueezeStrategyConfig(squeeze_threshold_pct=0.5), object())
    signal = strategy._evaluate_signal(
        "NSE:NIFTY",
        {
            "bollinger_upper": 100.3,
            "bollinger_lower": 99.7,
            "bollinger_middle": 100.0,
            "close": 101.0,
            "open": 100.0,
            "atr": 1.0,
            "direction_bias": "CE",
            "volume": 120.0,
            "avg_volume": 100.0,
        },
        101.0,
    )
    assert signal is None


def test_bb_squeeze_emits_context_only_evidence() -> None:
    strategy = BBSqueezeStrategy(BBSqueezeStrategyConfig(squeeze_threshold_pct=0.5), object())
    signal = strategy._evaluate_signal(
        "NSE:NIFTY",
        {
            "bollinger_upper": 100.2,
            "bollinger_lower": 99.8,
            "bollinger_middle": 100.0,
            "close": 101.0,
            "open": 100.0,
            "atr": 1.0,
            "direction_bias": "CE",
            "volume": 120.0,
            "avg_volume": 100.0,
            "bb_width_percentile": 10.0,
        },
        101.0,
    )
    assert signal is not None
    assert signal.metadata["role"] == "context"
    assert signal.metadata["can_trigger"] is False
    assert signal.metadata["squeeze_threshold_pct"] == 0.5


def test_cpr_requires_configured_narrow_width() -> None:
    strategy = CPRBreakoutStrategy(CPRBreakoutStrategyConfig(narrow_cpr_threshold=0.25), object())
    signal = strategy._evaluate_signal(
        "NSE:NIFTY",
        {
            "bc": 9980.0,
            "tc": 10020.0,
            "pivot": 10000.0,
            "r1": 10100.0,
            "s1": 9900.0,
            "atr": 20.0,
            "direction_bias": "CE",
            "retest_confirmed": True,
        },
        10040.0,
    )
    assert signal is None


def test_cpr_does_not_invent_retest_from_breakout_penetration() -> None:
    strategy = CPRBreakoutStrategy(CPRBreakoutStrategyConfig(narrow_cpr_threshold=0.25), object())
    signal = strategy._evaluate_signal(
        "NSE:NIFTY",
        {
            "bc": 9990.0,
            "tc": 10010.0,
            "pivot": 10000.0,
            "r1": 10100.0,
            "s1": 9900.0,
            "atr": 20.0,
            "direction_bias": "CE",
            "retest_confirmed": False,
        },
        10030.0,
    )
    assert signal is not None
    assert signal.metadata["role"] == "context"
    assert signal.metadata["can_trigger"] is False
    assert "retest_confirmed" not in signal.metadata["score_reasons"]


def test_rsi_divergence_fails_closed_without_confirmed_swings() -> None:
    strategy = RSIDivergenceStrategy(RSIDivergenceStrategyConfig(), object())
    signal = strategy._evaluate_signal(
        "NSE:NIFTY",
        {
            "rsi": 35.0,
            "close": 95.0,
            "atr": 10.0,
            "regime": "range",
            "direction_bias": "CE",
            "confirmation_candle": True,
        },
        95.0,
    )
    assert signal is None


def test_rsi_divergence_uses_confirmed_completed_swing_evidence() -> None:
    strategy = RSIDivergenceStrategy(RSIDivergenceStrategyConfig(), object())
    signal = strategy._evaluate_signal(
        "NSE:NIFTY",
        {
            "rsi": 35.0,
            "close": 96.0,
            "atr": 10.0,
            "regime": "range",
            "direction_bias": "CE",
            "confirmation_candle": True,
            "rsi_swing_points": [
                {"kind": "low", "price": 100.0, "rsi": 30.0, "confirmed": True},
                {"kind": "low", "price": 95.0, "rsi": 36.0, "confirmed": True},
            ],
        },
        96.0,
    )
    assert signal is not None
    assert signal.metadata["divergence_type"] == "bullish"
    assert signal.metadata["role"] == "context"
    assert signal.metadata["can_trigger"] is False


def test_context_strategies_fail_closed_in_trigger_quality_scoring() -> None:
    for strategy_name in (
        "OrderFlow",
        "OIMaxPain",
        "BBSqueeze",
        "CPRBreakout",
        "RSIDivergence",
    ):
        score = score_signal_quality(
            direction_score=10.0,
            strategy_score=10.0,
            option_score=10.0,
            data_score=10.0,
            rr_score=10.0,
            strategy_name=strategy_name,
        )
        assert score.allowed is False
        assert "context_only_strategy" in score.reasons


def test_experimental_context_flags_are_reachable(monkeypatch) -> None:
    monkeypatch.setenv("STRATEGY_MODE", "directional_scalp")
    monkeypatch.delenv("ENABLE_BB_SQUEEZE_CONTEXT", raising=False)
    monkeypatch.delenv("ENABLE_CPR_EXPERIMENTAL", raising=False)
    monkeypatch.delenv("ENABLE_RSI_DIVERGENCE_EXPERIMENTAL", raising=False)

    assert (
        _strategy_runtime_role(
            "bb_squeeze",
            strategy_mode="directional_scalp",
            allow_expiry_gamma=False,
        )
        is None
    )
    assert (
        _strategy_runtime_role(
            "cpr",
            strategy_mode="directional_scalp",
            allow_expiry_gamma=False,
        )
        is None
    )

    monkeypatch.setenv("ENABLE_BB_SQUEEZE_CONTEXT", "true")
    monkeypatch.setenv("ENABLE_CPR_EXPERIMENTAL", "true")
    monkeypatch.setenv("ENABLE_RSI_DIVERGENCE_EXPERIMENTAL", "true")
    assert _strategy_runtime_role(
        "bb_squeeze",
        strategy_mode="directional_scalp",
        allow_expiry_gamma=False,
    ) == "context"
    assert _strategy_runtime_role(
        "cpr",
        strategy_mode="directional_scalp",
        allow_expiry_gamma=False,
    ) == "context"
    assert _strategy_runtime_role(
        "rsi_div",
        strategy_mode="directional_scalp",
        allow_expiry_gamma=False,
    ) == "context"
