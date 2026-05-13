from nifty_scalper_bot.core.strategy_manager import StrategyManager, StrategyVote
from nifty_scalper_bot.strategies.signal_generator import Signal


def test_single_vote_uses_raw_score_for_threshold(monkeypatch):
    monkeypatch.setenv('STRATEGY_ALLOW_SINGLE_VOTE_SCALP', 'true')
    monkeypatch.setenv('STRATEGY_SINGLE_VOTE_VWAP_MIN_SCORE', '5.5')
    manager = StrategyManager.__new__(StrategyManager)
    signal = Signal(action='BUY', symbol='NFO:NIFTY25000CE', quantity=1, confidence=0.55, reason='VWAPPro', stop_loss=1.0, take_profit=2.0, metadata={'is_selected_option': True})
    vote = StrategyVote(strategy='VWAPPro', side='CE', score=4.4, confidence=0.55, reasons=[], metadata={'raw_setup_score': 5.5, 'regime_weight': 0.8})
    combined = manager._combine_strategy_votes(symbol=signal.symbol, signals=[(signal, vote)], indicators={})
    assert combined is not None
    assert combined.metadata['raw_setup_score'] == 5.5
    assert combined.metadata['regime_weighted_score'] == 4.4
    assert combined.metadata['regime_weight'] == 0.8
