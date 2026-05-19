from nifty_scalper_bot.execution.order_manager import OrderManager, OrderType


class _BrokerHang:
    def __init__(self): self.calls = 0
    def place_order(self, **kwargs):
        self.calls += 1
        return None


class _BrokerError:
    def __init__(self):
        self.calls = 0

    def place_order(self, **kwargs):
        self.calls += 1
        raise TimeoutError("broker timeout")


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


def test_find_open_order_filters_statuses():
    class _BrokerOrders:
        def get_orders(self):
            return [
                {"client_order_id": "bot_x", "status": "CANCELLED", "order_id": "C1"},
                {"client_order_id": "bot_x", "status": "REJECTED", "order_id": "R1"},
                {"client_order_id": "bot_x", "status": "COMPLETE", "order_id": "D1"},
                {"client_order_id": "bot_x", "status": "OPEN", "order_id": "O1"},
            ]

    om = OrderManager(_BrokerOrders(), _Pos(), _Limiter())
    found = om._find_open_order("bot_x")
    assert found is not None
    assert found["order_id"] == "O1"


def test_find_open_order_ignores_terminal_when_no_open():
    class _BrokerOrders:
        def get_orders(self):
            return [
                {"client_order_id": "bot_x", "status": "CANCELLED", "order_id": "C1"},
                {"client_order_id": "bot_x", "status": "REJECTED", "order_id": "R1"},
                {"client_order_id": "bot_x", "status": "COMPLETE", "order_id": "D1"},
            ]

    om = OrderManager(_BrokerOrders(), _Pos(), _Limiter())
    assert om._find_open_order("bot_x") is None


def test_exception_reconcile_returns_existing_order(monkeypatch):
    broker = _BrokerError()
    om = OrderManager(broker, _Pos(), _Limiter())
    monkeypatch.setattr(om, '_lot_size_for_symbol', lambda s: 1)
    monkeypatch.setattr(om, '_validate_live_execution_safety', lambda: True)
    monkeypatch.setattr(om, '_confirm_fill_fast', lambda order_id, timeout_ms=300: False)
    monkeypatch.setattr(om, '_find_open_order', lambda cid: {'order_id': 'OID_TIMEOUT'})
    oid = om.place_order('NFO:NIFTY26MAY23750CE', 'BUY', 1, order_type=OrderType.MARKET, stop_loss=10, signal_id='se')
    assert oid == 'OID_TIMEOUT'
    assert broker.calls == 1
