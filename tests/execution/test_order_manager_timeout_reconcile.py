from nifty_scalper_bot.execution.broker_rejects import (
    BrokerReject,
    RecoveryAction,
    parse_broker_error,
    recovery_decision,
)
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


def test_unresolved_timeout_marks_uncertain_and_does_not_retry(monkeypatch):
    class _SlowBroker:
        def __init__(self): self.calls = 0
        def place_order(self, **kwargs):
            import time as _t
            self.calls += 1
            _t.sleep(20)
            return None
    broker = _SlowBroker()
    om = OrderManager(broker, _Pos(), _Limiter())
    monkeypatch.setattr(om, '_lot_size_for_symbol', lambda s: 1)
    monkeypatch.setattr(om, '_validate_live_execution_safety', lambda: True)
    monkeypatch.setattr(om, '_confirm_fill_fast', lambda order_id, timeout_ms=300: False)
    monkeypatch.setattr(om, '_find_open_order', lambda cid: None)
    result = om.place_order('NFO:NIFTY26MAY23750CE', 'BUY', 1, order_type=OrderType.MARKET, stop_loss=10, signal_id='sig_timeout_case')
    assert result is None
    assert broker.calls == 1
    assert 'bot__timeout_case' in om._uncertain_client_order_ids


def test_uncertain_order_blocks_immediate_retry_without_broker_call(monkeypatch):
    class _Broker:
        def __init__(self): self.calls = 0
        def place_order(self, **kwargs):
            self.calls += 1
            return {"order_id": "SHOULD_NOT_HAPPEN"}
    broker = _Broker()
    om = OrderManager(broker, _Pos(), _Limiter())
    monkeypatch.setattr(om, '_lot_size_for_symbol', lambda s: 1)
    monkeypatch.setattr(om, '_validate_live_execution_safety', lambda: True)
    monkeypatch.setattr(om, '_find_open_order', lambda cid: None)
    om._mark_order_uncertain('bot_certainretry')
    result = om.place_order('NFO:NIFTY26MAY23750CE', 'BUY', 1, order_type=OrderType.MARKET, stop_loss=10, signal_id='manual_certainretry')
    assert result is None
    assert broker.calls == 0


def test_uncertain_order_retry_returns_existing_order(monkeypatch):
    broker = _BrokerHang()
    om = OrderManager(broker, _Pos(), _Limiter())
    monkeypatch.setattr(om, '_lot_size_for_symbol', lambda s: 1)
    monkeypatch.setattr(om, '_validate_live_execution_safety', lambda: True)
    monkeypatch.setattr(om, '_find_open_order', lambda cid: {'order_id': 'OID_RECOVERED'})
    om._mark_order_uncertain('bot_case1234')
    oid = om.place_order('NFO:NIFTY26MAY23750CE', 'BUY', 1, order_type=OrderType.MARKET, stop_loss=10, signal_id='manual_case1234')
    assert oid == 'OID_RECOVERED'
    assert 'bot_case1234' not in om._uncertain_client_order_ids


def test_find_open_order_matches_tag_suffix_case_insensitive():
    class _BrokerOrders:
        def get_orders(self):
            return [{"tag": "BOT_ABCD1234", "status": "OPEN", "order_id": "OID_TAG"}]
    om = OrderManager(_BrokerOrders(), _Pos(), _Limiter())
    found = om._find_open_order("bot_abcd1234")
    assert found is not None
    assert found["order_id"] == "OID_TAG"


def test_broker_tag_keeps_unique_suffix_with_long_tag(monkeypatch):
    captured = {}
    class _BrokerCapture:
        def place_order(self, **kwargs):
            captured.update(kwargs)
            return {"order_id": "OID1"}
    om = OrderManager(_BrokerCapture(), _Pos(), _Limiter())
    monkeypatch.setattr(om, '_lot_size_for_symbol', lambda s: 1)
    monkeypatch.setattr(om, '_validate_live_execution_safety', lambda: True)
    monkeypatch.setattr(om, '_confirm_fill_fast', lambda order_id, timeout_ms=300: False)
    oid = om.place_order('NFO:NIFTY26MAY23750CE', 'BUY', 1, order_type=OrderType.MARKET, stop_loss=10, signal_id='manual_1234abcd', tag='superlongtagname_with_extra_chars')
    assert oid == "OID1"
    assert str(captured["tag"]).endswith("1234abcd")


async def test_broker_reject_classifier_covers_actionable_entry_failures():
    cases = {
        "Insufficient funds. Required margin is 12500": BrokerReject.INSUFFICIENT_FUNDS,
        "Price is outside the allowed range": BrokerReject.PRICE_OUT_OF_RANGE,
        "Trigger price should be lower than limit price": BrokerReject.TRIGGER_PRICE_INVALID,
        "Order quantity exceeds freeze limit": BrokerReject.FREEZE_LIMIT,
        "Quantity is not a multiple of lot size": BrokerReject.QUANTITY_INVALID,
        "Invalid tradingsymbol": BrokerReject.INSTRUMENT_INVALID,
        "Duplicate order request": BrokerReject.DUPLICATE_ORDER,
        "Order cannot be modified because order is complete": BrokerReject.ORDER_STATE_INVALID,
        "429 too many requests": BrokerReject.THROTTLED,
        "504 gateway timeout": BrokerReject.TRANSIENT_BROKER,
        "Invalid access token": BrokerReject.AUTHENTICATION,
    }
    for message, expected in cases.items():
        assert parse_broker_error(message) is expected


async def test_price_reject_requires_fresh_market_and_full_revalidation():
    decision = recovery_decision("Price is outside the allowed range")
    assert decision.retryable is True
    assert decision.action is RecoveryAction.REFRESH_MARKET_AND_REVALIDATE
    assert decision.reconcile_before_retry is False
    assert decision.count_toward_kill_switch is False


async def test_timeout_and_duplicate_ack_require_reconciliation_before_retry():
    timeout = recovery_decision("504 gateway timeout")
    duplicate = recovery_decision("Duplicate order request")
    assert timeout.action is RecoveryAction.RECONCILE_ORDERBOOK
    assert timeout.reconcile_before_retry is True
    assert timeout.count_toward_kill_switch is True
    assert duplicate.action is RecoveryAction.RECONCILE_ORDERBOOK
    assert duplicate.reconcile_before_retry is True
    assert duplicate.count_toward_kill_switch is False


async def test_invalid_instrument_and_unknown_errors_are_terminal():
    invalid = recovery_decision("Invalid tradingsymbol")
    unknown = recovery_decision("unexpected broker response")
    assert invalid.retryable is False
    assert invalid.action is RecoveryAction.STOP_AND_ALERT
    assert unknown.reason is BrokerReject.UNKNOWN
    assert unknown.retryable is False
    assert unknown.count_toward_kill_switch is True
