from nifty_scalper_bot.core.strategy_manager import StrategyManager, StrategyVote
from nifty_scalper_bot.strategies.signal_generator import Signal


def test_selected_context_fallback_near_same_side(monkeypatch):
    monkeypatch.setenv('STRATEGY_ALLOW_SINGLE_VOTE_SCALP', 'true')
    manager = StrategyManager.__new__(StrategyManager)
    signal = Signal(action='BUY', symbol='NFO:NIFTY26MAY23500CE', quantity=1, confidence=0.7, reason='SMC', stop_loss=1.0, take_profit=2.0, metadata={})
    vote = StrategyVote(strategy='SMC', side='CE', score=6.5, confidence=0.7, reasons=[], metadata={'raw_setup_score': 6.5})
    combined = manager._combine_strategy_votes(symbol=signal.symbol, signals=[(signal, vote)], indicators={'selected_ce': 'NFO:NIFTY26MAY23450CE'})
    assert combined is not None
    assert combined.metadata['selected_ce'] == 'NFO:NIFTY26MAY23450CE'
    assert combined.metadata['strike_distance_from_atm'] == 50.0
    assert combined.metadata['selected_ok_reason'] == 'near_atm'
