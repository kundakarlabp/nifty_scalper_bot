from types import SimpleNamespace

from nifty_scalper_bot.execution.order_manager import OrderManager, OrderPreflightResult, TradePlan


def _manager_stub():
    m = SimpleNamespace()
    m._logger = SimpleNamespace(warning=lambda *a, **k: None, error=lambda *a, **k: None)
    return m


def test_trade_plan_dataclass_defaults():
    plan = TradePlan(symbol='NFO:NIFTY', side='BUY', quantity=75, entry_price=100.0, stop_loss=90.0, take_profit=110.0)
    assert plan.allow_market_entry is False
    assert isinstance(OrderPreflightResult(True), OrderPreflightResult)


def test_submit_trade_plan_protected_price_invalidates_buy_bracket() -> None:
    m = _manager_stub()
    m._validate_trade_plan = lambda p: OrderPreflightResult(True)
    m._protected_limit_price = lambda p: 111.0
    out = OrderManager.submit_trade_plan_result(m, TradePlan(symbol='NFO:NIFTY', side='BUY', quantity=75, entry_price=100.0, stop_loss=95.0, take_profit=105.0))
    assert out.accepted is False
    assert out.reason == 'protected_price_invalidates_bracket'
    assert out.broker_attempted is False
    assert out.details['violation'] == 'take_profit_below_or_equal_entry'


def test_submit_trade_plan_protected_price_invalidates_sell_bracket() -> None:
    m = _manager_stub()
    m._validate_trade_plan = lambda p: OrderPreflightResult(True)
    m._protected_limit_price = lambda p: 99.0
    out = OrderManager.submit_trade_plan_result(m, TradePlan(symbol='NFO:NIFTY', side='SELL', quantity=75, entry_price=100.0, stop_loss=101.0, take_profit=98.0))
    assert out.accepted is False
    assert out.reason == 'protected_price_invalidates_bracket'
    assert out.broker_attempted is False


def test_preflight_reject_does_not_attempt_broker() -> None:
    m = _manager_stub()
    m._validate_trade_plan = lambda p: OrderPreflightResult(False, 'quote_unavailable', {})
    out = OrderManager.submit_trade_plan_result(m, TradePlan(symbol='NFO:NIFTY', side='BUY', quantity=75, entry_price=100.0, stop_loss=90.0, take_profit=110.0))
    assert out.reason == 'quote_unavailable'
    assert out.broker_attempted is False


def test_managed_order_local_reject_does_not_attempt_broker() -> None:
    m = _manager_stub()
    m._lot_size_for_symbol = lambda s: 50
    out = OrderManager.place_managed_order_result(m, symbol='NFO:NIFTY', side='BUY', quantity=75, entry_price=100.0, stop_loss=90.0, take_profit=110.0)
    assert out.reason == 'invalid_lot_quantity'
    assert out.broker_attempted is False


def test_managed_order_uses_last_decision_for_broker_attempted_none() -> None:
    m = _manager_stub()
    m._lot_size_for_symbol = lambda s: 75

    def _place_order(**kwargs):
        m._last_order_decision = {'block_reason': 'risk_manager_blocked', 'details': {'x': 1}, 'broker_attempted': True}
        return None

    m.place_order = _place_order
    out = OrderManager.place_managed_order_result(m, symbol='NFO:NIFTY', side='BUY', quantity=75, entry_price=100.0, stop_loss=90.0, take_profit=110.0)
    assert out.reason == 'risk_manager_blocked'
    assert out.broker_attempted is True


def test_stale_last_order_decision_does_not_leak() -> None:
    m = _manager_stub()
    m._lot_size_for_symbol = lambda s: 75
    m._last_order_decision = {'block_reason': 'stale_old_reason', 'broker_attempted': True}
    m.place_order = lambda **kwargs: None
    out = OrderManager.place_managed_order_result(m, symbol='NFO:NIFTY', side='BUY', quantity=75, entry_price=100.0, stop_loss=90.0, take_profit=110.0, trace_id='t1')
    assert out.reason == 'place_order_rejected_without_decision'
    assert out.broker_attempted is False

from datetime import datetime, timezone
from collections import deque
import time


def test_kill_switch_history_helpers_capture_last_failure() -> None:
    om = OrderManager.__new__(OrderManager)
    om._lock = __import__('threading').RLock()
    om._kill_switch_failure_history = deque(maxlen=20)
    om._kill_switch_engaged_at = datetime.now(timezone.utc)
    om._kill_switch_allow_auto_reset = False
    om._kill_switch_auto_reset_seconds = 900
    om._kill_switch_reason = 'unexpected_exception'
    om._consecutive_failures = 3
    om._kill_switch_last_reset = None
    om._record_kill_switch_failure({
        'trace_id': 't-1',
        'symbol': 'NFO:NIFTY26MAY23700PE',
        'failure_class': 'unexpected_exception',
        'exception_type': 'RuntimeError',
        'exception_message': 'boom',
    })
    status = om.get_kill_switch_status()
    assert status['active'] is True
    assert status['last_failure']['exception_type'] == 'RuntimeError'
    assert 'boom' in status['last_failure']['exception_message']
    assert status['last_failure']['trace_id'] == 't-1'


def test_margin_api_502_uses_fresh_cache() -> None:
    om = OrderManager.__new__(OrderManager)
    om._data_hub = SimpleNamespace(
        refresh_margin_snapshot=lambda: (_ for _ in ()).throw(RuntimeError("HTTP 502")),
        get_available_balance=lambda: 5000.0,
    )
    om._market_data = None
    om._risk_manager = None
    om._logger = SimpleNamespace(error=lambda *a, **k: None, info=lambda *a, **k: None)
    om._margin_cache_max_age_seconds = 120
    om._allow_entry_with_stale_margin = False
    om._last_margin_success_ts = time.time()
    om._margin_circuit_open = False
    om._margin_circuit_until_ts = None
    om._emit_broker_health_status = lambda **kwargs: None
    available, source = OrderManager._resolve_available_margin(om)
    assert available == 5000.0
    assert source == "mdm"


def test_margin_api_502_blocks_entry_when_cache_stale() -> None:
    om = OrderManager.__new__(OrderManager)
    om._data_hub = SimpleNamespace(
        refresh_margin_snapshot=lambda: (_ for _ in ()).throw(RuntimeError("HTTP 502")),
        get_available_balance=lambda: None,
    )
    om._market_data = None
    om._risk_manager = None
    om._logger = SimpleNamespace(error=lambda *a, **k: None, info=lambda *a, **k: None)
    om._margin_cache_max_age_seconds = 120
    om._allow_entry_with_stale_margin = False
    om._last_margin_success_ts = time.time() - 1000
    om._margin_circuit_open = False
    om._margin_circuit_until_ts = None
    om._emit_broker_health_status = lambda **kwargs: None
    available, source = OrderManager._resolve_available_margin(om)
    assert available is None
    assert source == "margin_unavailable_stale"


def test_broker_health_status_emitted_on_margin_circuit_open() -> None:
    records: list[dict] = []
    om = OrderManager.__new__(OrderManager)
    om._logger = SimpleNamespace(info=lambda *_a, **k: records.append(k.get("extra", {})))
    om._broker = SimpleNamespace(is_connected=True)
    om._last_margin_success_ts = time.time() - 10
    om._last_margin_refresh_ts = time.time()
    om._last_margin_error_type = "RuntimeError"
    om._last_margin_error = "HTTP 502"
    om._margin_circuit_open = True
    om._margin_circuit_until_ts = time.time() + 10
    om._margin_cache_max_age_seconds = 120
    om._allow_entry_with_stale_margin = False
    om._last_order_api_error_type = None
    om._last_order_api_error = None
    om._last_broker_health_emit_ts = 0.0
    om._last_broker_health_effect = "none"
    om._last_broker_health_circuit_state = False
    om._resolve_available_margin_raw = lambda: (5000.0, "mdm")
    OrderManager._emit_broker_health_status(om, force=True)
    assert any(r.get("event") == "BROKER_HEALTH_STATUS" for r in records)
