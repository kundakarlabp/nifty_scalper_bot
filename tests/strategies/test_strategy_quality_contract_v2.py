from __future__ import annotations

from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    OrderFlowStrategyConfig,
    SMCStrategyConfig,
    VWAPProStrategyConfig,
)
from nifty_scalper_bot.strategies.signal_generator import Signal
from nifty_scalper_bot.strategies.strategy_quality_patch import normalise_strategy_signal


def _signal(strategy: str, metadata: dict[str, object], *, confidence: float = 0.9) -> Signal:
    payload = {"strategy": strategy, "strategy_name": strategy, **metadata}
    return Signal(
        action="BUY",
        symbol="NFO:NIFTY26SEP25000CE",
        quantity=1,
        confidence=confidence,
        reason=strategy,
        stop_loss=None,
        take_profit=None,
        metadata=payload,
    )


def test_orderflow_quality_removes_unconditional_point_and_uses_config() -> None:
    signal = _signal(
        "OrderFlow",
        {
            "role": "context",
            "strategy_score": 10.0,
            "context_score": 10.0,
            "depth_imbalance": 0.20,
            "spread_pct": 0.50,
            "trigger_max_spread_pct": 0.75,
            "tick_supports_direction": True,
            "effective_context_alignment": True,
            "effective_context_conflict": False,
            "context_quality_eligible": True,
            "stale_data_used": False,
        },
    )
    updated, reason = normalise_strategy_signal(
        signal,
        strategy_name="OrderFlow",
        config=OrderFlowStrategyConfig(),
        indicators={},
        current_price=100.0,
        live_mode=True,
    )

    assert reason is None
    assert updated is not None
    assert updated.metadata["quality_score"] == 8.0
    assert updated.metadata["context_evidence_score"] == 4.0
    assert updated.metadata["context_bonus_score"] == 2.0
    assert updated.metadata["depth_support_threshold"] == 0.15
    assert 0.47 < updated.metadata["strong_depth_support_threshold"] < 0.48
    assert updated.metadata["orderflow_unconditional_score_removed"] is True


def test_orderflow_large_order_threshold_is_not_a_dead_knob() -> None:
    signal = _signal(
        "OrderFlow",
        {
            "role": "context",
            "strategy_score": 9.0,
            "depth_imbalance": 0.20,
            "spread_pct": 0.50,
            "trigger_max_spread_pct": 0.75,
            "tick_supports_direction": True,
            "effective_context_alignment": True,
            "effective_context_conflict": False,
            "context_quality_eligible": True,
            "stale_data_used": False,
        },
    )
    updated, _ = normalise_strategy_signal(
        signal,
        strategy_name="OrderFlow",
        config=OrderFlowStrategyConfig(large_order_threshold_pct=30.0),
        indicators={},
        current_price=100.0,
        live_mode=True,
    )

    assert updated is not None
    assert updated.metadata["depth_support_threshold"] == 0.30
    assert updated.metadata["depth_supports_side"] is False
    assert updated.metadata["quality_score"] == 6.0


def test_smc_live_requires_quality_beyond_core_plus_direction() -> None:
    weak = _signal(
        "SMC",
        {
            "role": "trigger",
            "strategy_score": 6.5,
            "score_reasons": [
                "underlying_liquidity_sweep",
                "reclaim",
                "displacement_confirmation",
                "direction_alignment",
            ],
        },
    )
    rejected, reason = normalise_strategy_signal(
        weak,
        strategy_name="SMC",
        config=SMCStrategyConfig(),
        indicators={},
        current_price=100.0,
        live_mode=True,
    )
    assert rejected is None
    assert reason == "smc_quality_below_minimum"

    confirmed = _signal(
        "SMC",
        {
            "role": "trigger",
            "strategy_score": 7.5,
            "score_reasons": [
                "underlying_liquidity_sweep",
                "reclaim",
                "displacement_confirmation",
                "direction_alignment",
                "volume_confirmation",
            ],
        },
    )
    accepted, reason = normalise_strategy_signal(
        confirmed,
        strategy_name="SMC",
        config=SMCStrategyConfig(),
        indicators={},
        current_price=100.0,
        live_mode=True,
    )
    assert reason is None
    assert accepted is not None
    assert accepted.metadata["quality_score"] == 6.5
    assert accepted.metadata["setup_pass"] is True


def test_vwap_live_rejects_atr_overextension_and_consumes_proximity_config() -> None:
    overextended = _signal(
        "VWAPPro",
        {
            "role": "trigger",
            "strategy_score": 8.0,
            "setup_min": 5.5,
            "vwap": 100.0,
            "atr": 5.0,
            "premium_above_vwap": True,
            "trend_alignment": True,
            "futures_alignment": True,
            "vwap_event_confirmed": True,
            "score_reasons": ["volume_confirmation"],
        },
    )
    rejected, reason = normalise_strategy_signal(
        overextended,
        strategy_name="VWAPPro",
        config=VWAPProStrategyConfig(proximity_pct=0.15),
        indicators={},
        current_price=120.0,
        live_mode=True,
    )
    assert rejected is None
    assert reason == "vwap_quality_overextended_atr"

    near = _signal(
        "VWAPPro",
        {
            "role": "trigger",
            "strategy_score": 7.5,
            "setup_min": 5.5,
            "vwap": 100.0,
            "atr": 2.0,
            "premium_above_vwap": True,
            "trend_alignment": True,
            "futures_alignment": False,
            "vwap_event_confirmed": True,
            "score_reasons": ["volume_confirmation"],
        },
    )
    accepted, reason = normalise_strategy_signal(
        near,
        strategy_name="VWAPPro",
        config=VWAPProStrategyConfig(proximity_pct=0.15),
        indicators={},
        current_price=100.10,
        live_mode=True,
    )
    assert reason is None
    assert accepted is not None
    assert accepted.metadata["vwap_configured_proximity_pass"] is True
    assert accepted.metadata["vwap_configured_proximity_pct"] == 0.15


def test_orb_live_requires_one_independent_quality_domain() -> None:
    weak = _signal(
        "ORBPro",
        {
            "role": "trigger",
            "strategy_score": 6.0,
            "opening_range_high": 101.0,
            "opening_range_low": 100.9,
            "underlying_atr": 10.0,
            "score_reasons": [
                "underlying_opening_range_complete",
                "fresh_underlying_breakout",
                "momentum_acceptance",
            ],
        },
    )
    rejected, reason = normalise_strategy_signal(
        weak,
        strategy_name="ORBPro",
        config=None,
        indicators={},
        current_price=100.0,
        live_mode=True,
    )
    assert rejected is None
    assert reason == "orb_quality_below_minimum"

    balanced = _signal(
        "ORBPro",
        {
            "role": "trigger",
            "strategy_score": 6.0,
            "opening_range_high": 105.0,
            "opening_range_low": 100.0,
            "underlying_atr": 10.0,
            "score_reasons": [
                "underlying_opening_range_complete",
                "fresh_underlying_breakout",
                "momentum_acceptance",
            ],
        },
    )
    accepted, reason = normalise_strategy_signal(
        balanced,
        strategy_name="ORBPro",
        config=None,
        indicators={},
        current_price=100.0,
        live_mode=True,
    )
    assert reason is None
    assert accepted is not None
    assert accepted.metadata["opening_range_width_atr"] == 0.5
    assert accepted.metadata["opening_range_balanced"] is True
    assert accepted.metadata["quality_score"] == 6.0
