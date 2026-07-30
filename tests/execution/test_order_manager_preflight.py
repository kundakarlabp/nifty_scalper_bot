from nifty_scalper_bot.execution.order_manager import OrderManager
from nifty_scalper_bot.core import signal_arbitrator as arbitrator_module


class _Broker: pass
class _Pos: pass
class _Limiter: pass


def test_extract_quote_diagnostics_handles_malformed_fields():
    om = OrderManager(_Broker(), _Pos(), _Limiter())
    q = {'bid': 'bad', 'ask': None, 'ltp': '12.5', 'bid_qty': 'bad', 'timestamp': 'bad-ts'}
    out = om._extract_quote_diagnostics(q)
    assert out['bid'] == 0.0
    assert out['ask'] == 0.0
    assert out['ltp'] == 12.5
    assert out['bid_qty'] == 0


def test_order_manager_arbitrator_does_not_add_five_minute_reentry_cooldown(
    monkeypatch, tmp_path
):
    now = [1000.0]
    monkeypatch.setattr(arbitrator_module.time, "time", lambda: now[0])
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    om = OrderManager(_Broker(), _Pos(), _Limiter())
    ce = "NFO:NIFTY26AUG24250CE"
    pe = "NFO:NIFTY26AUG24250PE"

    assert om._signal_arbitrator.allow(ce, "BUY") is True
    om._signal_arbitrator.register(ce, "BUY")
    assert om._signal_arbitrator.allow(pe, "BUY") is False

    om._signal_arbitrator.release(ce)
    now[0] += 3.0

    assert om._signal_arbitrator.allow(pe, "BUY") is True
