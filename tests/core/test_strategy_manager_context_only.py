from nifty_scalper_bot.core.strategy_manager import StrategyManager, StrategyVote
from nifty_scalper_bot.strategies.signal_generator import Signal


def _context_signal() -> tuple[Signal, StrategyVote]:
    signal = Signal(action='BUY', symbol='NFO:NIFTY25000CE', quantity=1, confidence=0.52, reason='OrderFlow', stop_loss=1.0, take_profit=2.0, metadata={'is_selected_option': True})
    vote = StrategyVote(strategy='OrderFlow', side='CE', score=5.2, confidence=0.52, reasons=[], metadata={'role': 'context', 'raw_setup_score': 5.2})
    return signal, vote


def test_context_only_vote_returns_none_with_trace(caplog):
    manager = StrategyManager.__new__(StrategyManager)
    signal, vote = _context_signal()
    caplog.set_level('INFO')
    out = manager._combine_strategy_votes(symbol=signal.symbol, signals=[(signal, vote)], indicators={})
    assert out is None
    assert any('TRADE_DECISION_TRACE' in rec.message and 'no_trigger_vote' in rec.message for rec in caplog.records)


def test_context_promotion_enabled(monkeypatch):
    manager = StrategyManager.__new__(StrategyManager)
    signal, vote = _context_signal()
    monkeypatch.setenv('STRATEGY_ALLOW_CONTEXT_PROMOTION', 'true')
    out = manager._combine_strategy_votes(symbol=signal.symbol, signals=[(signal, vote)], indicators={'is_selected_option': True})
    assert out is not None
    assert out.metadata['promoted_from_context'] is True
    assert out.metadata['approval_path'] == 'context_promotion'
    assert out.metadata['quality_pass'] is True
    assert out.metadata['consensus_stage'] == 'context_promoted_controlled'


def test_context_promotion_quality_reject(monkeypatch, caplog):
    manager = StrategyManager.__new__(StrategyManager)
    signal, vote = _context_signal()
    monkeypatch.setenv('STRATEGY_ALLOW_CONTEXT_PROMOTION', 'true')
    monkeypatch.setenv('STRATEGY_MIN_TRADE_QUALITY_SHADOW', '9.5')
    caplog.set_level('INFO')
    out = manager._combine_strategy_votes(symbol=signal.symbol, signals=[(signal, vote)], indicators={'is_selected_option': True})
    assert out is None
    assert any('STRATEGY_QUALITY_REJECT' in rec.message for rec in caplog.records)
    assert not any('reason=ok' in rec.message for rec in caplog.records if 'STRATEGY_QUALITY_REJECT' in rec.message)


def test_context_promotion_blocked_in_live(monkeypatch):
    manager = StrategyManager.__new__(StrategyManager)
    signal, vote = _context_signal()
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    monkeypatch.setenv('STRATEGY_ALLOW_CONTEXT_PROMOTION', 'true')
    monkeypatch.setenv('STRATEGY_CONTEXT_PROMOTION_LIVE_ALLOWED', 'false')
    out = manager._combine_strategy_votes(symbol=signal.symbol, signals=[(signal, vote)], indicators={'is_selected_option': True})
    assert out is None
