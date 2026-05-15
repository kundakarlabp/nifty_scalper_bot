from nifty_scalper_bot.core.strategy_manager import StrategyManager, StrategyVote
from nifty_scalper_bot.strategies.runner import Signal


def _mgr():
    return StrategyManager.__new__(StrategyManager)


def test_vwappro_single_vote_rejected_low_score(monkeypatch):
    monkeypatch.setenv('SCALPER_MODE', 'true')
    mgr = _mgr()
    sig = Signal(action='BUY', symbol='NFO:NIFTY26MAY24100CE', quantity=1, confidence=0.5, reason='x', stop_loss=90, take_profit=110, metadata={'candidate_selected': True, 'spread_pct': 1.0})
    vote = StrategyVote(strategy='VWAPPro', side='CE', score=4.0, confidence=0.5)
    out = mgr._combine_strategy_votes(symbol=sig.symbol, signals=[(sig, vote)], indicators={})
    assert out is None


def test_single_trigger_path_not_context_promoted(monkeypatch):
    monkeypatch.setenv('STRATEGY_ALLOW_SINGLE_VOTE_SCALP', 'true')
    mgr = _mgr()
    sig = Signal(action='BUY', symbol='NFO:NIFTY26MAY24100CE', quantity=1, confidence=0.6, reason='x', stop_loss=90, take_profit=110, metadata={'is_selected_option': True, 'spread_pct': 1.0})
    trigger = StrategyVote(strategy='VWAPPro', side='CE', score=6.5, confidence=0.7, reasons=[], metadata={'role': 'trigger', 'raw_setup_score': 6.5, 'strategy_score': 6.5})
    context = StrategyVote(strategy='OrderFlow', side='CE', score=5.2, confidence=0.52, reasons=[], metadata={'role': 'context', 'raw_setup_score': 5.2})
    out = mgr._combine_strategy_votes(symbol=sig.symbol, signals=[(sig, trigger), (sig, context)], indicators={'is_selected_option': True})
    assert out is not None
    assert out.metadata['approval_path'] in {'single_vote_fallback', 'single_trigger'}
    assert out.metadata.get('promoted_from_context') is not True
