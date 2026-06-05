from __future__ import annotations

import logging

from nifty_scalper_bot.core.strategy_manager import StrategyManager, StrategyVote
from nifty_scalper_bot.strategies.signal_generator import Signal


def test_combiner_blocker_log_exposes_required_fields(monkeypatch, caplog) -> None:
    monkeypatch.setenv("STRATEGY_ALLOW_SINGLE_VOTE_SCALP", "false")
    manager = object.__new__(StrategyManager)
    signal = Signal(
        action="BUY",
        symbol="NFO:NIFTY25000CE",
        quantity=1,
        confidence=0.8,
        reason="OrderFlow",
        stop_loss=1.0,
        take_profit=2.0,
        metadata={"selected_ok_reason": "selected_option", "final_trade_score": 8.0},
    )
    vote = StrategyVote(
        strategy="OrderFlow",
        side="CE",
        score=8.0,
        confidence=0.8,
        reasons=[],
        metadata={"role": "trigger", "raw_setup_score": 8.0},
    )

    caplog.set_level(logging.INFO)
    manager._log_strategy_combiner_blocker(
        symbol=signal.symbol,
        signal_votes=[(signal, vote)],
        indicators={"selected_ce": signal.symbol},
        combined=None,
        blocked_reason="single_vote_scalp_disabled",
        no_vote_reason_counts={"smc_low_score": 1},
    )

    rec = next(r for r in caplog.records if getattr(r, "event", None) == "STRATEGY_COMBINER_BLOCKER")
    assert rec.strategy_vote_count == 1
    assert rec.trigger_vote_count == 1
    assert rec.context_vote_count == 0
    assert rec.best_strategy == "OrderFlow"
    assert rec.best_score == 8.0
    assert rec.best_confidence == 0.8
    assert rec.single_vote_allowed is False
    assert rec.selected_ok is True
    assert rec.selected_ok_reason == "selected_option"
    assert rec.final_trade_score == 8.0
    assert rec.final_trade_threshold == 4.5
    assert rec.blocked_reason == "single_vote_scalp_disabled"
