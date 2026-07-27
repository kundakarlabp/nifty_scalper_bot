from nifty_scalper_bot.core import signal_arbitrator as arbitrator_module
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


def test_release_starts_reentry_cooldown_from_exit_time(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(arbitrator_module.time, 'time', lambda: now[0])
    arb = SignalArbitrator(
        cooldown_seconds=3.0,
        stale_active_seconds=120.0,
        reentry_cooldown_seconds=300.0,
    )
    ce = 'NFO:NIFTY26JUL23950CE'

    assert arb.allow(ce, 'BUY') is True
    arb.register(ce, 'BUY')
    now[0] += 180.0
    arb.release(ce)

    now[0] += 119.0
    assert arb.allow(ce, 'BUY') is False
    now[0] += 181.0
    assert arb.allow(ce, 'BUY') is True


def test_nifty_ce_and_pe_share_one_entry_reservation(monkeypatch):
    now = [2000.0]
    monkeypatch.setattr(arbitrator_module.time, 'time', lambda: now[0])
    arb = SignalArbitrator(reentry_cooldown_seconds=300.0)
    ce = 'NFO:NIFTY26JUL23950CE'
    pe = 'NFO:NIFTY26JUL23950PE'

    assert arb.allow(ce, 'BUY') is True
    arb.register(ce, 'BUY')
    assert arb.allow(pe, 'BUY') is False

    arb.release(ce)
    now[0] += 120.0
    assert arb.allow(pe, 'BUY') is False
    now[0] += 180.0
    assert arb.allow(pe, 'BUY') is True


def test_stale_active_reservation_fails_closed_for_reentry_window(monkeypatch):
    now = [3000.0]
    monkeypatch.setattr(arbitrator_module.time, 'time', lambda: now[0])
    arb = SignalArbitrator(
        stale_active_seconds=120.0,
        reentry_cooldown_seconds=300.0,
    )
    symbol = 'NFO:NIFTY26JUL23950CE'

    arb.register(symbol, 'BUY')
    now[0] += 121.0
    assert arb.allow(symbol, 'BUY') is False
    now[0] += 300.0
    assert arb.allow(symbol, 'BUY') is True
