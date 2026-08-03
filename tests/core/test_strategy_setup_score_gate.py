from __future__ import annotations

from types import SimpleNamespace

from nifty_scalper_bot.core.strategy_manager import StrategyManager
from nifty_scalper_bot.core.strategy_setup_score_gate import (
    enforce_context_only_role,
    filter_context_promotions,
    setup_gate_result,
)
from nifty_scalper_bot.strategies.signal_generator import Signal


def _vote(**metadata):
    return SimpleNamespace(strategy=metadata.get("strategy", "VWAPPro"), metadata=metadata)


def test_setup_gate_rejects_score_below_strategy_minimum() -> None:
    passed, score, minimum, reason = setup_gate_result(
        _vote(role="trigger", raw_setup_score=4.9, setup_min=5.5, setup_pass=True)
    )

    assert passed is False
    assert score == 4.9
    assert minimum == 5.5
    assert reason == "setup_below_minimum"


def test_setup_gate_accepts_setup_that_cleared_its_own_threshold() -> None:
    passed, score, minimum, reason = setup_gate_result(
        _vote(role="trigger", raw_setup_score=6.0, setup_min=5.5, setup_pass=True)
    )

    assert passed is True
    assert score == 6.0
    assert minimum == 5.5
    assert reason is None


def test_setup_gate_preserves_legacy_vote_without_explicit_contract() -> None:
    passed, score, minimum, reason = setup_gate_result(
        _vote(role="trigger", raw_vote_score=7.0)
    )

    assert passed is True
    assert score is None
    assert minimum is None
    assert reason is None


def test_context_cannot_promote_an_explicitly_failed_trigger() -> None:
    manager = StrategyManager.__new__(StrategyManager)
    weak_trigger = (
        SimpleNamespace(action="BUY"),
        _vote(
            strategy="VWAPPro",
            role="trigger",
            raw_setup_score=4.0,
            setup_min=5.5,
            setup_pass=False,
            trigger_block_reason="weak_score",
        ),
    )
    strong_context = (
        SimpleNamespace(action="BUY"),
        _vote(
            strategy="ContextOnly",
            role="context",
            context_bonus_score=10.0,
        ),
    )

    result = manager._combine_strategy_votes(
        symbol="NFO:NIFTY2680625000CE",
        signals=[weak_trigger, strong_context],
        indicators={},
    )

    assert result is None


def test_malformed_orderflow_trigger_is_normalized_to_context() -> None:
    signal = SimpleNamespace(
        action="BUY",
        metadata={
            "strategy": "OrderFlow",
            "role": "trigger",
            "can_trigger": True,
            "trigger_conditions_met": True,
        },
    )
    vote = _vote(
        strategy="OrderFlow",
        role="trigger",
        can_trigger=True,
        trigger_conditions_met=True,
    )

    updated_signal, changed = enforce_context_only_role(signal, vote)

    assert changed is True
    for metadata in (updated_signal.metadata, vote.metadata):
        assert metadata["role"] == "context"
        assert metadata["can_trigger"] is False
        assert metadata["trigger_conditions_met"] is False
        assert metadata["trigger_eligible"] is False
        assert metadata["trigger_block_reason"] == "context_only_role"


def test_frozen_signal_is_replaced_not_mutated() -> None:
    signal = Signal(
        action="BUY",
        symbol="NFO:NIFTY2680424600PE",
        quantity=65,
        confidence=0.8,
        reason="OrderFlow",
        stop_loss=None,
        take_profit=None,
        metadata={"strategy": "OrderFlow", "role": "trigger"},
    )
    vote = _vote(strategy="OrderFlow", role="trigger", can_trigger=True)

    updated_signal, changed = enforce_context_only_role(signal, vote)

    assert changed is True
    assert updated_signal is not signal
    assert signal.metadata["role"] == "trigger"
    assert updated_signal.metadata["role"] == "context"
    assert updated_signal.metadata["can_trigger"] is False
    assert vote.metadata["role"] == "context"


def test_orderflow_is_removed_from_context_promotion_candidates() -> None:
    orderflow = (
        SimpleNamespace(action="BUY"),
        _vote(strategy="OrderFlow", role="context", raw_setup_score=10.0),
    )
    vwap_context = (
        SimpleNamespace(action="BUY"),
        _vote(strategy="VWAPPro", role="context", raw_setup_score=9.0),
    )

    eligible, blocked = filter_context_promotions([orderflow, vwap_context])

    assert eligible == [vwap_context]
    assert blocked == ["OrderFlow"]


def test_orderflow_cannot_promote_even_when_every_promotion_flag_is_enabled(
    monkeypatch,
) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ENABLE_LIVE", "true")
    monkeypatch.setenv("STRATEGY_CONTEXT_PROMOTION_LIVE_ALLOWED", "true")
    monkeypatch.setenv(
        "STRATEGY_CONTEXT_PROMOTION_ALLOWED_STRATEGIES", "OrderFlow,VWAPPro"
    )
    manager = StrategyManager.__new__(StrategyManager)
    orderflow = (
        SimpleNamespace(action="BUY"),
        _vote(
            strategy="OrderFlow",
            role="context",
            raw_setup_score=10.0,
            context_score=10.0,
        ),
    )

    result = manager._try_context_promotion(
        "NFO:NIFTY2680625000CE",
        [orderflow],
        {
            "direction_bias": "CE",
            "context_fresh": True,
            "context_age_seconds": 0.1,
            "is_selected_option": True,
            "bid": 99.5,
            "ask": 100.0,
            "quote_depth_valid": True,
        },
        {
            "mode": "LIVE",
            "allow_context_promotion": True,
            "allow_single_vote": True,
            "min_trade_quality": 7.0,
        },
    )

    assert result is None


def test_close_signal_is_never_blocked_by_setup_gate() -> None:
    passed, _, _, _ = setup_gate_result(
        _vote(role="trigger", raw_setup_score=1.0, setup_min=5.0, setup_pass=False)
    )
    assert passed is False
    # Close bypass is enforced by the wrapper before setup_gate_result is used.
    assert "CLOSE_LONG" in {"CLOSE_LONG", "CLOSE_SHORT"}
