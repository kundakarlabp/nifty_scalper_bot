from nifty_scalper_bot.execution.order_manager import OrderManager, OrderType


class _Pos:
    def has_open_position(self, symbol):
        return False


class _Limiter:
    pass


class _KwargsBroker:
    def __init__(self):
        self.received = None

    def place_order(self, **kwargs):
        self.received = kwargs
        return {"order_id": "OID_KWARGS"}


class _PayloadBroker:
    def __init__(self):
        self.payloads = []

    def place_order(self, payload):
        self.payloads.append(payload)
        return {"order_id": "OID_PAYLOAD"}


class _InternalTypeErrorBroker:
    def __init__(self):
        self.calls = 0

    def place_order(self, **kwargs):
        self.calls += 1
        raise TypeError("unsupported operand type after broker response")


class _RejectBroker:
    def __init__(self):
        self.calls = 0

    def place_order(self, **kwargs):
        self.calls += 1
        raise ValueError("insufficient margin")


def test_submit_broker_order_supports_kwargs_broker():
    broker = _KwargsBroker()
    om = OrderManager(broker, _Pos(), _Limiter())

    result = om._submit_broker_order({"symbol": "NFO:NIFTY", "quantity": 75})

    assert result["order_id"] == "OID_KWARGS"
    assert broker.received == {"symbol": "NFO:NIFTY", "quantity": 75}


def test_submit_broker_order_supports_payload_dict_broker():
    broker = _PayloadBroker()
    om = OrderManager(broker, _Pos(), _Limiter())

    result = om._submit_broker_order({"symbol": "NFO:NIFTY", "quantity": 75})

    assert result["order_id"] == "OID_PAYLOAD"
    assert broker.payloads == [{"symbol": "NFO:NIFTY", "quantity": 75}]


def test_signature_type_error_allows_payload_fallback():
    broker = _PayloadBroker()
    om = OrderManager(broker, _Pos(), _Limiter())

    result = om._submit_broker_order({"symbol": "NFO:NIFTY", "quantity": 75})

    assert result["order_id"] == "OID_PAYLOAD"
    assert broker.payloads == [{"symbol": "NFO:NIFTY", "quantity": 75}]


def test_submit_broker_order_does_not_swallow_real_rejection():
    broker = _RejectBroker()
    om = OrderManager(broker, _Pos(), _Limiter())

    try:
        om._submit_broker_order({"symbol": "NFO:NIFTY", "quantity": 75})
    except ValueError as exc:
        assert str(exc) == "insufficient margin"
    else:
        raise AssertionError("ValueError should propagate")

    assert broker.calls == 1


def test_submit_broker_order_does_not_fallback_on_internal_type_error():
    broker = _InternalTypeErrorBroker()
    om = OrderManager(broker, _Pos(), _Limiter())

    try:
        om._submit_broker_order({"symbol": "NFO:NIFTY", "quantity": 75})
    except TypeError as exc:
        assert "unsupported operand type" in str(exc)
    else:
        raise AssertionError("internal TypeError should propagate")

    assert broker.calls == 1


def test_place_order_supports_payload_dict_broker(monkeypatch):
    broker = _PayloadBroker()
    om = OrderManager(broker, _Pos(), _Limiter())
    monkeypatch.setattr(om, "_lot_size_for_symbol", lambda s: 1)
    monkeypatch.setattr(om, "_validate_live_execution_safety", lambda: True)
    monkeypatch.setattr(om, "_confirm_fill_fast", lambda order_id, timeout_ms=300: False)

    oid = om.place_order(
        "NFO:NIFTY26MAY23750CE",
        "BUY",
        1,
        order_type=OrderType.MARKET,
        stop_loss=10,
        signal_id="compat_case",
    )

    assert oid == "OID_PAYLOAD"
    assert broker.payloads, "payload fallback should be used"
