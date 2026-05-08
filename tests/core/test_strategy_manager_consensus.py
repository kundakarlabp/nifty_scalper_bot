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


def test_strategy_manager_single_vote_uses_indicator_selected_context(monkeypatch) -> None:
    monkeypatch.setenv('STRATEGY_ALLOW_SINGLE_VOTE_SCALP', 'true')
    manager = _manager_stub()
    signal = _make_signal()
    vote = StrategyVote(strategy='SMC', side='CE', score=6.8, confidence=0.8, reasons=[], metadata={})
    combined = manager._combine_strategy_votes(
        symbol='NIFTY',
        signals=[(signal, vote)],
        indicators={'selected_ce': 'NFO:NIFTY25000CE', 'selected_pe': 'NFO:NIFTY25000PE', 'atm_strike': 25000, 'strike_distance_from_atm': 0},
    )
    assert combined is not None
    assert combined.metadata is not None
    assert combined.metadata.get('consensus_stage') == 'single_vote_scalp_controlled'


def test_single_vote_scalp_enabled_allows_valid_near_atm_vote(monkeypatch) -> None:
    monkeypatch.setenv('STRATEGY_ALLOW_SINGLE_VOTE_SCALP', 'true')
    manager = _manager_stub()
    signal = _make_signal()
    vote = StrategyVote(strategy='VWAPPro', side='CE', score=6.6, confidence=0.7, reasons=[], metadata={})
    combined = manager._combine_strategy_votes(
        symbol='NIFTY',
        signals=[(signal, vote)],
        indicators={'atm_strike': 25000, 'strike_distance_from_atm': 50, 'selected_ce': None, 'selected_pe': None},
    )
    assert combined is not None
    assert combined.metadata is not None
    assert combined.metadata.get('consensus_stage') == 'single_vote_scalp_controlled'


def test_single_vote_scalp_disabled_rejects_single_vote(monkeypatch) -> None:
    monkeypatch.setenv('STRATEGY_ALLOW_SINGLE_VOTE_SCALP', 'false')
    manager = _manager_stub()
    signal = _make_signal()
    vote = StrategyVote(strategy='VWAPPro', side='CE', score=7.5, confidence=0.8, reasons=[], metadata={})
    combined = manager._combine_strategy_votes(
        symbol='NIFTY',
        signals=[(signal, vote)],
        indicators={'selected_ce': 'NFO:NIFTY25000CE', 'selected_pe': 'NFO:NIFTY25000PE', 'atm_strike': 25000},
    )
    assert combined is None
