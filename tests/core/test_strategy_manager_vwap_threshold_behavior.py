from nifty_scalper_bot.core.strategy_manager import StrategyManager, StrategyVote
from nifty_scalper_bot.strategies.signal_generator import Signal


def _base_signal() -> Signal:
    return Signal(action='BUY', symbol='NFO:NIFTY25000CE', quantity=1, confidence=0.55, reason='VWAPPro', stop_loss=1.0, take_profit=2.0, metadata={'is_selected_option': True})


def test_vwap_threshold_blocks_default(monkeypatch):
    monkeypatch.setenv('STRATEGY_ALLOW_SINGLE_VOTE_SCALP', 'true')
    monkeypatch.setenv('STRATEGY_SINGLE_VOTE_VWAP_MIN_SCORE', '5.8')
    manager = StrategyManager.__new__(StrategyManager)
    vote = StrategyVote(strategy='VWAPPro', side='CE', score=4.4, confidence=0.55, reasons=[], metadata={'raw_setup_score': 5.5})
    assert manager._combine_strategy_votes(symbol='NFO:NIFTY25000CE', signals=[(_base_signal(), vote)], indicators={}) is None


def test_vwap_threshold_allows_when_lowered(monkeypatch):
    monkeypatch.setenv('STRATEGY_ALLOW_SINGLE_VOTE_SCALP', 'true')
    monkeypatch.setenv('STRATEGY_SINGLE_VOTE_VWAP_MIN_SCORE', '5.5')
    manager = StrategyManager.__new__(StrategyManager)
    vote = StrategyVote(strategy='VWAPPro', side='CE', score=4.4, confidence=0.55, reasons=[], metadata={'raw_setup_score': 5.5})
    assert manager._combine_strategy_votes(symbol='NFO:NIFTY25000CE', signals=[(_base_signal(), vote)], indicators={}) is not None
