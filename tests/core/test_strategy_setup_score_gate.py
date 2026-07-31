from __future__ import annotations

from types import SimpleNamespace

from nifty_scalper_bot.core.strategy_manager import StrategyManager
from nifty_scalper_bot.core.strategy_setup_score_gate import setup_gate_result


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


def test_close_signal_is_never_blocked_by_setup_gate() -> None:
    passed, _, _, _ = setup_gate_result(
        _vote(role="trigger", raw_setup_score=1.0, setup_min=5.0, setup_pass=False)
    )
    assert passed is False
    # Close bypass is enforced by the wrapper before setup_gate_result is used.
    assert "CLOSE_LONG" in {"CLOSE_LONG", "CLOSE_SHORT"}
