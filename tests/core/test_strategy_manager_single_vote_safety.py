from __future__ import annotations

from nifty_scalper_bot.core.strategy_manager import StrategyManager, StrategyVote
from nifty_scalper_bot.strategies.signal_generator import Signal


def _make_signal(selected: bool) -> Signal:
    return Signal(
        action='BUY',
        symbol='NFO:NIFTY26MAY23500CE',
        quantity=1,
        confidence=0.95,
        metadata={
            'strategy': 'VWAPPro',
            'role': 'trigger',
            'side': 'CE',
            'strategy_score': 9.2,
            'is_selected_option': selected,
            'strike_distance_from_atm': 10.0 if selected else 200.0,
        },
    )


def _make_vote() -> StrategyVote:
    return StrategyVote(
        strategy='VWAPPro',
        side='CE',
        score=9.2,
        confidence=0.95,
        reasons=['test'],
        metadata={'role': 'trigger'},
    )


def test_single_high_vote_requires_selected_or_near_atm(monkeypatch) -> None:
    monkeypatch.setenv('STRATEGY_SINGLE_VOTE_HIGH_CONVICTION', '8.8')
    manager = StrategyManager()
    combined = manager._combine_votes([(_make_signal(False), _make_vote())], [], {})
    assert combined is None


def test_single_high_vote_passes_when_selected(monkeypatch) -> None:
    monkeypatch.setenv('STRATEGY_SINGLE_VOTE_HIGH_CONVICTION', '8.8')
    manager = StrategyManager()
    combined = manager._combine_votes([(_make_signal(True), _make_vote())], [], {})
    assert combined is not None
    assert combined.metadata.get('consensus_stage') == 'preliminary_single_high_conviction'
