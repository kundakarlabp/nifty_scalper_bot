from nifty_scalper_bot.core import signal_arbitrator as arbitrator_module
from nifty_scalper_bot.core.signal_arbitrator import SignalArbitrator


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


def test_stale_active_reservation_does_not_restart_reentry_window(monkeypatch):
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
    now[0] += 178.0
    assert arb.allow(symbol, 'BUY') is False
    now[0] += 1.0
    assert arb.allow(symbol, 'BUY') is True


def test_clear_removes_failed_entry_without_reentry_cooldown(monkeypatch):
    monkeypatch.setattr(arbitrator_module.time, 'time', lambda: 4000.0)
    arb = SignalArbitrator(reentry_cooldown_seconds=300.0)
    symbol = 'NFO:NIFTY26JUL23950CE'

    arb.register(symbol, 'BUY')
    arb.clear(symbol)

    assert arb.allow(symbol, 'BUY') is True
