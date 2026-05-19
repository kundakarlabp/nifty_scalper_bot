from nifty_scalper_bot.execution.order_manager import OrderManager


class _Broker: pass
class _Pos: pass
class _Limiter: pass


def test_place_managed_order_passes_trace_id(monkeypatch):
    om = OrderManager(_Broker(), _Pos(), _Limiter())
    monkeypatch.setattr(om, '_lot_size_for_symbol', lambda s: 1)
    captured = {}
    def _fake_place_order(**kwargs):
        captured.update(kwargs)
        return 'OID'
    monkeypatch.setattr(om, 'place_order', _fake_place_order)
    om.place_managed_order(symbol='NFO:NIFTY26MAY23750CE', side='BUY', quantity=1, entry_price=100.0, stop_loss=90.0, take_profit=110.0, trace_id='abc123')
    assert captured['trace_id'] == 'abc123'
