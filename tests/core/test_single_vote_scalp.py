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
