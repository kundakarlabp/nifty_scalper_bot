import pytest
from nifty_scalper_bot.core.strategy_manager import signal_to_vote, StrategyManager
from nifty_scalper_bot.strategies.signal_generator import Signal

class D:
    def get_indicators(self,s,n): return {}
class P:
    def get_position(self,s): return None

def test_signal_to_vote_uses_max_metadata_strategy_score_and_confidence():
    sig=Signal(action='BUY',symbol='NFO:XCE',quantity=1,confidence=0.4,reason='r',metadata={'strategy_score':7.0})
    vote=signal_to_vote(sig,'VWAPPro')
    assert vote.score==pytest.approx(7.0)

def test_apply_weighted_confidence_does_not_overwrite_setup_strategy_score():
    m=StrategyManager([],D(),P())
    entry=next(iter(m.get_strategy_scores()), None)
    sig=Signal(action='BUY',symbol='NFO:XCE',quantity=1,confidence=0.5,reason='r',metadata={'strategy_score':8.0})
    out=m._apply_weighted_confidence(sig,'VWAPPro',entry)
    assert out.metadata['strategy_score']==8.0
