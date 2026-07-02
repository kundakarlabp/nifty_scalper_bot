from types import SimpleNamespace

from nifty_scalper_bot.execution.order_manager_core import (
    OrderManager,
    OrderPreflightResult,
    TradePlan,
)


def _manager_stub():
    m = SimpleNamespace()
    m._logger = SimpleNamespace(warning=lambda *a, **k: None, error=lambda *a, **k: None)
    m.is_kill_switch_active = lambda: False
    m._reanchor_bracket_to_price = lambda plan, _price: plan
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
    m._protected_limit_price = lambda p: 97.0
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


def test_submit_trade_plan_kill_switch_returns_kill_switch_reason_without_broker_attempt() -> None:
    m = _manager_stub()
    m.is_kill_switch_active = lambda: True
    m.get_kill_switch_status = lambda: {"active": True, "consecutive_failures": 3, "last_reason": "unexpected_exception"}
    out = OrderManager.submit_trade_plan_result(m, TradePlan(symbol='NFO:NIFTY', side='BUY', quantity=75, entry_price=100.0, stop_loss=90.0, take_profit=110.0, trace_id="t1"))
    assert out.reason == "order_manager_kill_switch_active"
    assert out.broker_attempted is False
    assert out.details.get("kill_reason") == "unexpected_exception"


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

from datetime import datetime, timedelta, timezone
from collections import deque
import threading
import time
import nifty_scalper_bot.execution.order_manager as order_manager_module


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
    old_success_ts = time.time() - 5
    om._last_margin_success_ts = old_success_ts
    om._last_margin_available_balance = 4900.0
    om._margin_circuit_open = False
    om._margin_circuit_until_ts = None
    om._last_margin_balance_source = "mdm"
    om._emit_broker_health_status = lambda **kwargs: None
    available, source = OrderManager._resolve_available_margin(om)
    assert available == 4900.0
    assert source == "margin_cache_used"
    assert om._margin_circuit_open is True
    assert om._last_margin_error_type == "RuntimeError"
    assert om._last_margin_success_ts == old_success_ts
    assert om._last_margin_available_balance == 4900.0


def test_margin_api_502_does_not_clear_circuit_when_mdm_cache_read_returns_value() -> None:
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
    om._last_margin_success_ts = time.time() - 10
    om._last_margin_available_balance = 4900.0
    om._margin_circuit_open = False
    om._margin_circuit_until_ts = None
    om._emit_broker_health_status = lambda **kwargs: None
    _, source = OrderManager._resolve_available_margin(om)
    assert source == "margin_cache_used"
    assert om._margin_circuit_open is True
    assert om._last_margin_error_type == "RuntimeError"


def test_margin_api_502_blocks_entry_when_cache_stale() -> None:
    om = OrderManager.__new__(OrderManager)
    om._data_hub = SimpleNamespace(
        refresh_margin_snapshot=lambda: (_ for _ in ()).throw(RuntimeError("HTTP 502")),
        get_available_balance=lambda: None,
    )
    om._market_data = None
    om._risk_manager = SimpleNamespace(current_balance=99999.0)
    om._logger = SimpleNamespace(error=lambda *a, **k: None, info=lambda *a, **k: None)
    om._margin_cache_max_age_seconds = 120
    om._allow_entry_with_stale_margin = False
    om._last_margin_success_ts = time.time() - 1000
    om._last_margin_available_balance = 5000.0
    om._margin_circuit_open = False
    om._margin_circuit_until_ts = None
    om._emit_broker_health_status = lambda **kwargs: None
    available, source = OrderManager._resolve_available_margin(om)
    assert available is None
    assert source == "margin_unavailable_stale"


def test_margin_api_502_allows_stale_only_when_configured() -> None:
    om = OrderManager.__new__(OrderManager)
    om._data_hub = SimpleNamespace(
        refresh_margin_snapshot=lambda: (_ for _ in ()).throw(RuntimeError("HTTP 502")),
        get_available_balance=lambda: None,
    )
    om._market_data = None
    om._risk_manager = SimpleNamespace(current_balance=99999.0)
    om._logger = SimpleNamespace(error=lambda *a, **k: None, info=lambda *a, **k: None)
    om._margin_cache_max_age_seconds = 120
    om._allow_entry_with_stale_margin = True
    om._last_margin_success_ts = time.time() - 1000
    om._last_margin_available_balance = 5000.0
    om._margin_circuit_open = False
    om._margin_circuit_until_ts = None
    om._emit_broker_health_status = lambda **kwargs: None
    available, source = OrderManager._resolve_available_margin(om)
    assert available == 5000.0
    assert source == "margin_cache_stale_allowed"


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
    om._last_margin_available_balance = 5000.0
    om._last_margin_balance_source = "margin_cache_used"
    OrderManager._emit_broker_health_status(om, force=False)
    assert any(r.get("event") == "BROKER_HEALTH_STATUS" for r in records)


def test_broker_health_snapshot_is_read_only() -> None:
    om = OrderManager.__new__(OrderManager)
    om._broker = SimpleNamespace(is_connected=True)
    om._last_margin_success_ts = time.time()
    om._margin_cache_max_age_seconds = 120
    om._margin_circuit_open = False
    om._last_margin_error_type = None
    om._allow_entry_with_stale_margin = False
    om._last_margin_refresh_ts = time.time()
    om._last_margin_error = None
    om._margin_circuit_until_ts = None
    om._last_margin_available_balance = 4200.0
    om._last_margin_balance_source = "mdm"
    om._last_order_api_error_type = None
    om._last_order_api_error = None
    snap = OrderManager.get_broker_health_snapshot(om)
    assert snap["available_balance"] == 4200.0
    assert snap["balance_source"] == "mdm"


def test_exit_path_not_blocked_by_margin_cache_if_existing_policy_allows_exits() -> None:
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
    om._last_margin_available_balance = 5000.0
    om._margin_circuit_open = False
    om._margin_circuit_until_ts = None
    om._emit_broker_health_status = lambda **kwargs: None
    old_success_ts = om._last_margin_success_ts
    available, source = OrderManager._resolve_available_margin(om, for_entry=False)
    assert source == "margin_unavailable_stale_exit_allowed"
    assert om._last_margin_success_ts == old_success_ts
    assert om._margin_circuit_open is True


def test_order_api_failure_returns_structured_rejection() -> None:
    m = _manager_stub()
    m._validate_trade_plan = lambda p: OrderPreflightResult(True)
    m._protected_limit_price = lambda p: 100.0
    m.place_order = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("broker down token=secret"))
    m._sanitize_broker_error = lambda exc: OrderManager._sanitize_broker_error(exc)
    m._emit_broker_health_status = lambda **kwargs: None
    m._last_order_api_error_type = None
    m._last_order_api_error = None
    m._last_margin_success_ts = time.time()
    m._margin_cache_max_age_seconds = 120
    m._margin_circuit_open = False
    m._last_margin_error_type = None
    m._allow_entry_with_stale_margin = False
    m._last_margin_refresh_ts = time.time()
    m._last_margin_error = None
    m._margin_circuit_until_ts = None
    m._last_margin_available_balance = 5000.0
    m._last_margin_balance_source = "mdm"
    m._broker = SimpleNamespace(is_connected=True)
    result = OrderManager.submit_trade_plan_result(m, TradePlan(symbol='NFO:NIFTY', side='BUY', quantity=75, entry_price=100.0, stop_loss=90.0, take_profit=110.0))
    assert result.accepted is False
    assert result.broker_attempted is True
    assert result.reason == "broker_placement_exception"
    assert "secret" not in str(result.details)
    assert "token=secret" not in str(result.details)
    snap = OrderManager.get_broker_health_snapshot(m)
    assert snap["order_api_available"] is False
    assert snap["last_order_api_error_type"] == "RuntimeError"
    assert "secret" not in str(snap["last_order_api_error"])
    assert "token=secret" not in str(snap["last_order_api_error"])


def test_broker_error_sanitizer_redacts_common_secret_shapes() -> None:
    raw = (
        "broker failed token=secret access_token=abc123 "
        "request_token=req456 enctoken=enc789 password=pw "
        "Authorization: Bearer eyJabc.secret"
    )
    sanitized = OrderManager._sanitize_broker_error(raw)
    leaked_values = [
        "token=secret",
        "abc123",
        "req456",
        "enc789",
        "password=pw",
        "Bearer eyJabc.secret",
        "eyJabc.secret",
    ]
    for leaked in leaked_values:
        assert leaked not in sanitized
    assert "REDACTED" in sanitized


def test_broker_error_sanitizer_redacts_bare_bearer_token() -> None:
    sanitized = OrderManager._sanitize_broker_error("broker error Bearer abc.def-ghi")
    assert "abc.def-ghi" not in sanitized
    assert "Bearer [REDACTED]" in sanitized or "Bearer[REDACTED]" in sanitized


def test_broker_health_status_uses_two_tuple_time_status(monkeypatch) -> None:
    records: list[dict] = []
    om = OrderManager.__new__(OrderManager)
    om._logger = SimpleNamespace(info=lambda *_a, **k: records.append(k.get("extra", {})), debug=lambda *a, **k: None)
    om._last_broker_health_emit_ts = 0.0
    om._last_broker_health_effect = "none"
    om._last_broker_health_circuit_state = False
    om.get_broker_health_snapshot = lambda: {"trading_allowed_effect": "none", "margin_circuit_open": False}
    monkeypatch.setattr(order_manager_module, "get_time_status", lambda: (True, "Within safe entry window"))
    OrderManager._emit_broker_health_status(om, force=True)
    assert any(r.get("event") == "BROKER_HEALTH_STATUS" for r in records)


def test_broker_health_emit_never_raises(monkeypatch) -> None:
    om = OrderManager.__new__(OrderManager)
    om._logger = SimpleNamespace(info=lambda *a, **k: None, debug=lambda *a, **k: None)
    om._last_broker_health_emit_ts = 0.0
    om._last_broker_health_effect = "none"
    om._last_broker_health_circuit_state = False
    om._sanitize_broker_error = lambda e: str(e)
    om.get_broker_health_snapshot = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    monkeypatch.setattr(order_manager_module, "get_time_status", lambda: (True, "x"))
    OrderManager._emit_broker_health_status(om, force=True)


def test_live_kill_switch_auto_resets_after_cooldown(monkeypatch) -> None:
    # An unattended live bot must self-heal: a tripped kill switch (e.g. from a
    # transient broker/IP failure) auto-resets after the cooldown rather than
    # halting trading for the rest of the day. Within the cooldown it stays active;
    # after it, it resets — even in live mode.
    import time as _t
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    om = OrderManager.__new__(OrderManager)
    om._kill_switch_engaged_at = datetime.now(timezone.utc)
    om._kill_switch_allow_auto_reset = True
    om._kill_switch_auto_reset_seconds = 1
    om._logger = SimpleNamespace(info=lambda *a, **k: None)
    om._lock = threading.RLock()
    om._kill_switch_failure_history = deque(maxlen=20)
    # within cooldown -> still active
    assert OrderManager.is_kill_switch_active(om) is True
    # after cooldown -> auto-resets
    om._kill_switch_engaged_at = datetime.now(timezone.utc) - timedelta(seconds=5)
    reset_called = {"n": 0}
    real_reset = OrderManager.reset_kill_switch
    def _spy_reset(self, reason="manual"):
        reset_called["n"] += 1
        return real_reset(self, reason=reason)
    om.reset_kill_switch = lambda reason="manual": _spy_reset(om, reason=reason)
    om._last_kill_switch_log_ts = 0.0
    assert OrderManager.is_kill_switch_active(om) is False
    assert reset_called["n"] == 1
    assert om._kill_switch_engaged_at is None


def test_kill_switch_blocks_within_cooldown_when_auto_reset_disabled(monkeypatch) -> None:
    # With auto-reset disabled, it stays active (opt-out preserved).
    om = OrderManager.__new__(OrderManager)
    om._kill_switch_engaged_at = datetime.now(timezone.utc)
    om._kill_switch_allow_auto_reset = False
    om._kill_switch_auto_reset_seconds = 1
    assert OrderManager.is_kill_switch_active(om) is True


def test_risk_fallback_records_margin_success_clears_stale() -> None:
    # Root cause of BROKER_HEALTH_LIVE_ORDERS_BLOCKED with balance_stale=True /
    # margin_age_s=None: when the MDM margin snapshot isn't primed yet (returns
    # None, no error), the risk_manager fallback returned a balance WITHOUT
    # recording success, so _last_margin_success_ts stayed None forever and live
    # orders were permanently blocked though the broker was healthy. A positive
    # risk balance must record success.
    om = OrderManager.__new__(OrderManager)
    om._data_hub = SimpleNamespace(
        refresh_margin_snapshot=lambda: None,
        get_available_balance=lambda: None,  # MDM not primed yet
    )
    om._market_data = None
    om._risk_manager = SimpleNamespace(current_balance=22792.35)
    om._logger = SimpleNamespace(error=lambda *a, **k: None, info=lambda *a, **k: None, debug=lambda *a, **k: None)
    om._last_margin_success_ts = None
    om._last_margin_refresh_ts = None
    om._last_margin_available_balance = None
    om._last_margin_balance_source = None
    om._last_margin_error_type = None
    om._last_margin_error = None
    om._margin_circuit_open = False
    om._margin_circuit_until_ts = None

    available, source = OrderManager._resolve_available_margin_raw(om)
    assert available == 22792.35 and source == "risk"
    assert om._last_margin_success_ts is not None, "risk balance must record margin success"

    # broker-health snapshot must now report fresh, not stale -> live orders allowed
    om._broker = SimpleNamespace(is_connected=True)
    om._margin_cache_max_age_seconds = 120
    om._allow_entry_with_stale_margin = False
    om._last_order_api_error = None
    om._last_order_api_error_type = None
    om._last_broker_health_emit_ts = 0.0
    om._last_broker_health_effect = "none"
    om._last_broker_health_circuit_state = False
    om._emit_broker_health_status = lambda **k: None
    snap = OrderManager.get_broker_health_snapshot(om)
    assert snap["balance_stale"] is False
    assert snap["last_margin_success_age_s"] is not None
    assert snap["trading_allowed_effect"] != "live_orders_blocked"
