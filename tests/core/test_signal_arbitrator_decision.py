from nifty_scalper_bot.core.signal_arbitrator import SignalArbitrator, StrategyVote


def vote(name, direction, score=7.0, conf=0.8):
    return StrategyVote(name, direction, score, conf, ['x'], 1.0, {})


def test_weak_single_vote_hold():
    d = SignalArbitrator().decide(underlying='NIFTY', votes=[vote('vwap', 'CE', 5.5, 0.6)], option_candidates=[], market_context={}, trace_id='t')
    assert d.action == 'HOLD'


def test_conflicting_vote_hold():
    votes = [vote('vwap', 'CE', 8, 0.9), vote('orb', 'PE', 8, 0.9), vote('rsi', 'CE', 8, 0.9), vote('bb', 'PE', 8, 0.9)]
    d = SignalArbitrator().decide(underlying='NIFTY', votes=votes, option_candidates=[], market_context={}, trace_id='t')
    assert d.action == 'HOLD' and 'conflicting_direction' in d.reasons


def test_strong_buy():
    votes = [vote('vwap', 'CE', 8, 0.9), vote('orb', 'CE', 8, 0.8)]
    c = [{'symbol': 'NFO:NIFTYCE', 'side': 'CE', 'rr': 1.8, 'entry_price': 100, 'stop_loss': 92, 'target': 113, 'final_score': 9}]
    d = SignalArbitrator().decide(underlying='NIFTY', votes=votes, option_candidates=c, market_context={}, trace_id='t')
    assert d.action == 'BUY' and d.direction == 'CE'


def test_low_rr_hold():
    votes = [vote('vwap', 'CE', 8, 0.9), vote('orb', 'CE', 8, 0.8)]
    c = [{'symbol': 'NFO:NIFTYCE', 'side': 'CE', 'rr': 1.2, 'entry_price': 100, 'stop_loss': 95, 'target': 106, 'final_score': 9}]
    d = SignalArbitrator().decide(underlying='NIFTY', votes=votes, option_candidates=c, market_context={}, trace_id='t')
    assert d.action == 'HOLD' and 'rr_low' in d.reasons
