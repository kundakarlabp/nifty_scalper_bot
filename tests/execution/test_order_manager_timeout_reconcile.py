from nifty_scalper_bot.execution.order_manager import OrderManager, OrderType


class _BrokerHang:
    def __init__(self): self.calls = 0
    def place_order(self, **kwargs):
        self.calls += 1
        return None


class _Pos:
    def has_open_position(self, symbol): return False


class _Limiter: pass


def test_timeout_reconcile_returns_existing_order(monkeypatch):
    broker = _BrokerHang()
    om = OrderManager(broker, _Pos(), _Limiter())
    monkeypatch.setattr(om, '_lot_size_for_symbol', lambda s: 1)
    monkeypatch.setattr(om, '_validate_live_execution_safety', lambda: True)
    monkeypatch.setattr(om, '_confirm_fill_fast', lambda order_id, timeout_ms=300: False)
    monkeypatch.setattr(om, '_find_open_order', lambda cid: {'order_id': 'OID_TIMEOUT'})
    oid = om.place_order('NFO:NIFTY26MAY23750CE', 'BUY', 1, order_type=OrderType.MARKET, stop_loss=10, signal_id='st')
    assert oid == 'OID_TIMEOUT'
    assert broker.calls == 1
