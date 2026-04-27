from __future__ import annotations

from nifty_scalper_bot.core.strategy_manager import StrategyManager, StrategyVote
from nifty_scalper_bot.strategies.signal_generator import Signal


def _make_signal() -> Signal:
    return Signal(
        action='BUY',
        symbol='NFO:NIFTY25000CE',
        quantity=1,
        confidence=0.6,
        reason='test',
        stop_loss=10.0,
        take_profit=20.0,
        metadata={'strategy_score': 0.0},
    )


def _manager_stub() -> StrategyManager:
    manager = object.__new__(StrategyManager)
    manager._combine_signals = lambda signals: signals[0] if signals else None
    return manager


def test_single_low_score_vote_returns_none() -> None:
    manager = _manager_stub()
    signal = _make_signal()
    vote = StrategyVote(
        strategy='SMC', side='CE', score=8.0, confidence=0.8, reasons=[], metadata={}
    )
    combined = manager._combine_strategy_votes(
        symbol='NIFTY',
        signals=[(signal, vote)],
        indicators={'direction_score': 9.0, 'data_score': 9.0, 'option_score': 9.0},
    )
    assert combined is None


def test_single_high_score_vote_returns_preliminary_consensus() -> None:
    manager = _manager_stub()
    signal = _make_signal()
    vote = StrategyVote(
        strategy='SMC', side='CE', score=9.1, confidence=0.9, reasons=[], metadata={}
    )
    combined = manager._combine_strategy_votes(
        symbol='NIFTY',
        signals=[(signal, vote)],
        indicators={'direction_score': 6.0, 'data_score': 6.0, 'option_score': 6.0},
    )
    assert combined is not None
    assert combined.metadata is not None
    assert combined.metadata.get('consensus_stage') == 'preliminary_single_high_conviction'
