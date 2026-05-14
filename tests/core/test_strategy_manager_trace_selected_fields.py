from unittest.mock import patch

from nifty_scalper_bot.core.strategy_manager import StrategyManager, StrategyVote
from nifty_scalper_bot.strategies.signal_generator import Signal


def test_trace_uses_indicator_selected_fields(monkeypatch):
    monkeypatch.setenv('STRATEGY_ALLOW_SINGLE_VOTE_SCALP', 'false')
    manager = StrategyManager.__new__(StrategyManager)
    signal = Signal(action='BUY', symbol='NFO:NIFTY26MAY24000CE', quantity=1, confidence=0.4, reason='VWAPPro', stop_loss=1.0, take_profit=2.0, metadata={})
    vote = StrategyVote(strategy='VWAPPro', side='CE', score=3.0, confidence=0.4, reasons=[], metadata={'raw_setup_score': 3.0})
    with patch('nifty_scalper_bot.core.strategy_manager.log.info') as log_info:
        combined = manager._combine_strategy_votes(symbol=signal.symbol, signals=[(signal, vote)], indicators={'selected_ce': 'NFO:NIFTY26MAY23450CE', 'selected_pe': 'NFO:NIFTY26MAY23450PE'})
    assert combined is None
    calls = ' '.join(str(c.args[0]) for c in log_info.call_args_list if c.args)
    assert 'TRADE_DECISION_TRACE' in calls
    found = any('selected_ce=%s selected_pe=%s' in str(c.args[0]) for c in log_info.call_args_list if c.args)
    assert found
