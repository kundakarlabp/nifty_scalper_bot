from nifty_scalper_bot.core.strategy_manager import StrategyManager, StrategyVote
from nifty_scalper_bot.strategies.signal_generator import Signal


def test_context_only_vote_returns_none_with_trace(caplog):
    manager = StrategyManager.__new__(StrategyManager)
    signal = Signal(action='BUY', symbol='NFO:NIFTY25000CE', quantity=1, confidence=0.55, reason='OrderFlow', stop_loss=1.0, take_profit=2.0, metadata={})
    vote = StrategyVote(strategy='OrderFlow', side='CE', score=6.0, confidence=0.55, reasons=[], metadata={'role': 'context'})
    caplog.set_level('INFO')
    out = manager._combine_strategy_votes(symbol=signal.symbol, signals=[(signal, vote)], indicators={})
    assert out is None
    assert any('TRADE_DECISION_TRACE' in rec.message and 'no_trigger_vote' in rec.message for rec in caplog.records)
