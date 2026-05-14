from nifty_scalper_bot.core.strategy_manager import StrategyManager, StrategyVote
from nifty_scalper_bot.strategies.signal_generator import Signal


def _signal() -> Signal:
    return Signal(action='BUY', symbol='NFO:NIFTY25100CE', quantity=1, confidence=0.65, reason='VWAPPro', stop_loss=1.0, take_profit=2.0, metadata={'is_selected_option': False, 'strike_distance_from_atm': 80, 'quote_depth_valid': True})


def test_nonselected_high_score_blocked_default(monkeypatch):
    monkeypatch.delenv('STRATEGY_ALLOW_CANDIDATE_SWITCH_ON_HIGH_SCORE', raising=False)
    monkeypatch.setenv('STRATEGY_ALLOW_SINGLE_VOTE_SCALP', 'true')
    manager = StrategyManager.__new__(StrategyManager)
    vote = StrategyVote(strategy='VWAPPro', side='CE', score=7.0, confidence=0.65, reasons=[], metadata={'raw_setup_score': 7.0})
    assert manager._combine_strategy_votes(symbol='NFO:NIFTY25100CE', signals=[(_signal(), vote)], indicators={}) is None


def test_nonselected_high_score_candidate_switch(monkeypatch):
    monkeypatch.setenv('STRATEGY_ALLOW_CANDIDATE_SWITCH_ON_HIGH_SCORE', 'true')
    monkeypatch.setenv('STRATEGY_ALLOW_SINGLE_VOTE_SCALP', 'true')
    manager = StrategyManager.__new__(StrategyManager)
    vote = StrategyVote(strategy='VWAPPro', side='CE', score=7.0, confidence=0.65, reasons=[], metadata={'raw_setup_score': 7.0})
    out = manager._combine_strategy_votes(symbol='NFO:NIFTY25100CE', signals=[(_signal(), vote)], indicators={})
    assert out is not None
    assert out.metadata.get('candidate_switch_requested') is True
