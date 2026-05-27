from __future__ import annotations

import logging

from nifty_scalper_bot.core.strategy_manager import StrategyManager, StrategyVote
from nifty_scalper_bot.strategies.signal_generator import Signal


def test_single_vote_scalp_disabled_reason(monkeypatch, caplog):
    monkeypatch.setenv('STRATEGY_ALLOW_SINGLE_VOTE_SCALP', 'false')
    monkeypatch.setenv('STRATEGY_SINGLE_VOTE_VWAP_MIN_SCORE', '5.5')
    monkeypatch.setenv('STRATEGY_SINGLE_VOTE_VWAP_MIN_CONFIDENCE', '0.45')

    manager = StrategyManager.__new__(StrategyManager)
    signal = Signal(
        action='BUY',
        symbol='NFO:NIFTY25000CE',
        quantity=1,
        confidence=0.60,
        reason='VWAPPro',
        stop_loss=1.0,
        take_profit=2.0,
        metadata={'is_selected_option': True},
    )
    vote = StrategyVote(
        strategy='VWAPPro',
        side='CE',
        score=5.5,
        confidence=0.60,
        reasons=[],
        metadata={'raw_setup_score': 5.5, 'regime_weight': 1.0},
    )
    with caplog.at_level(logging.INFO):
        combined = manager._combine_strategy_votes(
            symbol=signal.symbol,
            signals=[(signal, vote)],
            indicators={},
        )

    assert combined is None
    assert any('single_vote_scalp_disabled' in r.message for r in caplog.records)
    assert not any('blocked_reason=context_veto' in r.message for r in caplog.records)
    svd = next(r for r in caplog.records if "SINGLE_VOTE_DECISION" in r.message)
    assert getattr(svd, "final_allowed", False) is False
