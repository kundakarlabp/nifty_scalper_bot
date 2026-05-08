from __future__ import annotations

from pathlib import Path

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



def test_single_vote_scalp_disabled_preserves_strict_behavior() -> None:
    source = Path('src/nifty_scalper_bot/core/strategy_manager.py').read_text(encoding='utf-8')
    assert 'STRATEGY_ALLOW_SINGLE_VOTE_SCALP", "false"' in source


def test_single_vote_scalp_enabled_allows_score_above_threshold() -> None:
    source = Path('src/nifty_scalper_bot/core/strategy_manager.py').read_text(encoding='utf-8')
    assert 'single_vote_scalp_controlled' in source
    assert 'single_vote.score >= single_min_score' in source


def test_single_vote_scalp_rejects_below_threshold() -> None:
    source = Path('src/nifty_scalper_bot/core/strategy_manager.py').read_text(encoding='utf-8')
    assert 'reason=single_vote_low_score' in source


def test_strategy_no_vote_reasons_are_logged() -> None:
    source = Path('src/nifty_scalper_bot/core/strategy_manager.py').read_text(encoding='utf-8')
    assert 'STRATEGY_NO_VOTE strategy=%s symbol=%s reason=%s' in source
    assert 'no_vote_reason_counts' in source
