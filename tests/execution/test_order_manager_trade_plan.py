import pytest
import types
from types import SimpleNamespace
from typing import Any

from nifty_scalper_bot.execution.order_manager_core import (
    OrderManager,
    OrderType,
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
    m._protected_limit_price = lambda p: 107.0
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


# ==================== FINAL LIVE-ENTRY MARGIN GATE ====================
# The gate runs in submit_trade_plan_result AFTER protected price + bracket
# re-anchoring and BEFORE any managed order, lifecycle, recovery or bracket
# state exists. It consumes the full MarginDecision including decision.quantity.


class _GateBroker:
    """Records every broker placement so call count/quantity are assertable."""

    def __init__(self) -> None:
        self.orders: list[dict[str, Any]] = []

    def place_order(self, **kwargs: Any) -> dict[str, Any]:
        self.orders.append(dict(kwargs))
        return {"order_id": f"ORD-{len(self.orders)}", "status": "success"}


def _gate_manager(monkeypatch, *, decision, balance=(1_000_000.0, "mdm"), lot_size=65):
    """Real OrderManager with only external boundaries faked."""
    from nifty_scalper_bot.execution.order_manager import OrderPreflightResult
    from nifty_scalper_bot.execution.position_manager import PositionManager
    from nifty_scalper_bot.utils.rate_limiter import RateLimiter

    broker = _GateBroker()
    mgr = OrderManager(broker, PositionManager(), RateLimiter())
    captured: dict[str, Any] = {}
    margin_inputs: list[Any] = []

    def _fake_managed(**kwargs: Any) -> Any:
        captured.update(kwargs)
        qty = int(kwargs.get("quantity") or 0)
        broker.place_order(symbol=kwargs.get("symbol"), quantity=qty)
        return types.SimpleNamespace(
            accepted=True,
            order_id="ORD-1",
            reason="accepted",
            details={},
            broker_attempted=True,
        )

    def _plan_capture(inputs):
        margin_inputs.append(inputs)
        return decision

    monkeypatch.setattr(mgr, "is_kill_switch_active", lambda: False)
    monkeypatch.setattr(
        mgr, "_validate_trade_plan", lambda plan: OrderPreflightResult(True, "ok", {})
    )
    monkeypatch.setattr(mgr, "_protected_limit_price", lambda plan: 107.0)
    monkeypatch.setattr(mgr, "place_managed_order_result", _fake_managed)
    monkeypatch.setattr(mgr, "_lot_size_for_symbol", lambda symbol: lot_size)
    monkeypatch.setattr(mgr, "_resolve_available_margin", lambda **kw: balance)
    monkeypatch.setattr(
        mgr, "_margin_engine", types.SimpleNamespace(plan=_plan_capture)
    )
    return mgr, broker, captured, margin_inputs


def _gate_plan(quantity=130, lot_size=65, intent="ENTRY"):
    return TradePlan(
        symbol="NFO:NIFTY26AUG25000CE",
        side="BUY",
        quantity=quantity,
        entry_price=100.0,
        stop_loss=80.0,
        take_profit=140.0,
        intent=intent,  # type: ignore[arg-type]
        requested_lots=quantity // lot_size,
        resolved_lot_size=lot_size,
    )


def test_entry_gate_two_lots_requested_one_allowed(monkeypatch) -> None:
    """Test 1: 130 requested, 65 allowed -> broker receives exactly 65."""
    decision = types.SimpleNamespace(
        ok=True, quantity=65, reason=None, est_required=6955.0
    )
    mgr, broker, captured, _ = _gate_manager(monkeypatch, decision=decision)

    result = mgr.submit_trade_plan_result(_gate_plan(quantity=130))

    assert result.accepted
    assert len(broker.orders) == 1
    assert broker.orders[0]["quantity"] == 65
    assert captured["quantity"] == 65
    assert all(o["quantity"] != 130 for o in broker.orders)


def test_entry_gate_blocks_when_less_than_one_lot_affordable(monkeypatch) -> None:
    """Test 2: zero allowed -> rejected, no broker call at all."""
    decision = types.SimpleNamespace(
        ok=False, quantity=0, reason="MARGIN no_qty_after_risk", est_required=0.0
    )
    mgr, broker, _, _ = _gate_manager(monkeypatch, decision=decision)

    result = mgr.submit_trade_plan_result(_gate_plan(quantity=65))

    assert result.accepted is False
    assert result.broker_attempted is False
    assert result.reason == "MARGIN no_qty_after_risk"
    assert broker.orders == []


def test_entry_gate_blocks_when_balance_unavailable(monkeypatch) -> None:
    """Test 3: no trusted balance -> reject; no synthetic fallback accepted."""
    decision = types.SimpleNamespace(
        ok=True, quantity=65, reason=None, est_required=6955.0
    )
    mgr, broker, _, margin_inputs = _gate_manager(
        monkeypatch, decision=decision, balance=(None, "margin_unavailable_stale")
    )

    result = mgr.submit_trade_plan_result(_gate_plan(quantity=65))

    assert result.accepted is False
    assert result.broker_attempted is False
    assert result.reason == "available_balance_unavailable"
    assert broker.orders == []
    # Sizing must not even be attempted without a trusted balance.
    assert margin_inputs == []


def test_entry_gate_uses_protected_price_and_reanchored_stop(monkeypatch) -> None:
    """Test 6: sizing uses protected price + re-anchored stop, not signal price."""
    decision = types.SimpleNamespace(
        ok=True, quantity=65, reason=None, est_required=6955.0
    )
    mgr, _broker, _captured, margin_inputs = _gate_manager(
        monkeypatch, decision=decision
    )

    mgr.submit_trade_plan_result(_gate_plan(quantity=65))

    assert len(margin_inputs) == 1
    inputs = margin_inputs[0]
    assert inputs.price == 107.0
    assert inputs.price != 100.0
    assert inputs.symbol == "NFO:NIFTY26AUG25000CE"
    assert inputs.lot_size == 65
    # Sizing sees the final distance-based stop after it moves from the
    # signal entry (100) to the protected entry (107).
    assert inputs.stop_loss == 87.0


def test_entry_gate_passes_no_atr_when_option_stop_available(monkeypatch) -> None:
    """Test 12: never mix underlying ATR points with option-premium prices."""
    decision = types.SimpleNamespace(
        ok=True, quantity=65, reason=None, est_required=6955.0
    )
    mgr, _broker, _captured, margin_inputs = _gate_manager(
        monkeypatch, decision=decision
    )

    mgr.submit_trade_plan_result(_gate_plan(quantity=65))

    assert margin_inputs[0].atr is None


def test_entry_gate_rejects_non_lot_multiple_decision(monkeypatch) -> None:
    """A malformed engine result must fail closed, never round upward."""
    decision = types.SimpleNamespace(
        ok=True, quantity=70, reason=None, est_required=7000.0
    )
    mgr, broker, _, _ = _gate_manager(monkeypatch, decision=decision)

    result = mgr.submit_trade_plan_result(_gate_plan(quantity=130))

    assert result.accepted is False
    assert result.broker_attempted is False
    assert result.reason == "invalid_lot_quantity"
    assert broker.orders == []


def test_entry_gate_unchanged_quantity_is_unaffected(monkeypatch) -> None:
    """Test 10: main non-regression -- allowed == requested passes through."""
    decision = types.SimpleNamespace(
        ok=True, quantity=65, reason=None, est_required=6955.0
    )
    mgr, broker, captured, _ = _gate_manager(monkeypatch, decision=decision)

    result = mgr.submit_trade_plan_result(_gate_plan(quantity=65))

    assert result.accepted
    assert len(broker.orders) == 1
    assert broker.orders[0]["quantity"] == 65
    assert captured["quantity"] == 65


def test_entry_gate_skips_exposure_reducing_intents(monkeypatch) -> None:
    """Test 5: protective exits bypass the affordability gate entirely."""
    decision = types.SimpleNamespace(
        ok=False, quantity=0, reason="MARGIN no_qty_after_risk", est_required=0.0
    )
    for intent in ("EXIT", "REDUCE", "FLATTEN"):
        mgr, broker, captured, margin_inputs = _gate_manager(
            monkeypatch,
            decision=decision,
            balance=(None, "margin_unavailable_stale"),
        )
        result = mgr.submit_trade_plan_result(
            _gate_plan(quantity=65, intent=intent)
        )
        # Balance gate must not reject an exposure-reducing action.
        assert result.reason != "available_balance_unavailable", intent
        assert margin_inputs == [], intent
        assert captured.get("quantity") == 65, intent


def test_entry_gate_session_reason_rejects_with_zero_broker_calls(monkeypatch) -> None:
    """MIS_WINDOW_CLOSED must reject before placement, not be deferred.

    An out-of-window entry is never allowed on the assumption that some other
    guard will catch it later.
    """
    decision = types.SimpleNamespace(
        ok=False,
        quantity=65,
        reason="MIS_WINDOW_CLOSED",
        est_required=6955.0,
    )
    mgr, broker, _captured, _ = _gate_manager(monkeypatch, decision=decision)

    result = mgr.submit_trade_plan_result(_gate_plan(quantity=130))

    assert result.accepted is False
    assert result.reason == "MIS_WINDOW_CLOSED"
    assert result.broker_attempted is False
    assert len(broker.orders) == 0


def test_entry_gate_unknown_failure_reason_is_fail_closed(monkeypatch) -> None:
    """Only allowlisted session reasons defer; unknown ok=False blocks."""
    decision = types.SimpleNamespace(
        ok=False,
        quantity=65,
        reason="SOME_UNKNOWN_BROKER_STATE",
        est_required=6955.0,
    )
    mgr, broker, _, _ = _gate_manager(monkeypatch, decision=decision)

    result = mgr.submit_trade_plan_result(_gate_plan(quantity=65))

    assert result.accepted is False
    assert result.broker_attempted is False
    assert result.reason == "SOME_UNKNOWN_BROKER_STATE"
    assert broker.orders == []


def test_entry_gate_session_reason_with_zero_quantity_still_blocks(
    monkeypatch,
) -> None:
    """Deferral requires a safely sized positive quantity."""
    decision = types.SimpleNamespace(
        ok=False, quantity=0, reason="MIS_WINDOW_CLOSED", est_required=0.0
    )
    mgr, broker, _, _ = _gate_manager(monkeypatch, decision=decision)

    result = mgr.submit_trade_plan_result(_gate_plan(quantity=65))

    assert result.accepted is False
    assert result.broker_attempted is False
    assert broker.orders == []


@pytest.mark.parametrize(
    "reason",
    [
        "MARGIN no_qty_after_risk",
        "margin_no_qty",
        "insufficient_risk_capacity",
        "invalid_requested_quantity",
    ],
)
def test_entry_gate_sizing_failures_block_with_zero_broker_calls(
    monkeypatch, reason: str
) -> None:
    """Every sizing/affordability failure blocks with no broker call."""
    decision = types.SimpleNamespace(
        ok=False, quantity=0, reason=reason, est_required=0.0
    )
    mgr, broker, _, _ = _gate_manager(monkeypatch, decision=decision)

    result = mgr.submit_trade_plan_result(_gate_plan(quantity=65))

    assert result.accepted is False
    assert result.broker_attempted is False
    assert result.reason == reason
    assert broker.orders == []


def test_entry_gate_uses_final_candidate_contract(monkeypatch) -> None:
    """Test 9: sizing uses the FINAL submitted contract, not an earlier one."""
    decision = types.SimpleNamespace(
        ok=True, quantity=50, reason=None, est_required=5350.0
    )
    mgr, broker, captured, margin_inputs = _gate_manager(
        monkeypatch, decision=decision, lot_size=50
    )
    final_plan = TradePlan(
        symbol="NFO:NIFTY26AUG25200PE",  # candidate B, not the original
        side="BUY",
        quantity=100,
        entry_price=104.0,
        stop_loss=70.0,
        take_profit=140.0,
        intent="ENTRY",
        requested_lots=2,
        resolved_lot_size=50,
    )

    mgr.submit_trade_plan_result(final_plan)

    inputs = margin_inputs[0]
    assert inputs.symbol == "NFO:NIFTY26AUG25200PE"
    assert inputs.lot_size == 50
    assert inputs.price == 107.0  # protected price, not the 104.0 signal price
    assert captured["symbol"] == "NFO:NIFTY26AUG25200PE"
    assert broker.orders[0]["quantity"] == 50


def test_entry_gate_uses_canonical_lowercase_risk_settings(monkeypatch) -> None:
    """The gate must consume the SAME risk policy as _pre_trade_decision().

    Regression: it previously read uppercase names (RISK_PER_TRADE_PCT etc.)
    which never matched the lowercase settings fields, so every lookup missed
    and the gate silently applied its own wider defaults.
    """
    decision = types.SimpleNamespace(
        ok=True, quantity=65, reason=None, est_required=6955.0
    )
    mgr, _broker, _captured, margin_inputs = _gate_manager(
        monkeypatch, decision=decision
    )
    mgr._risk_manager = types.SimpleNamespace(
        settings=types.SimpleNamespace(
            per_trade_risk_pct=0.5,
            per_trade_cap_pct=25.0,
            min_lots_per_trade=1,
            max_lots_per_trade=2,
            atr_stop_multiple=2.25,
        )
    )
    mgr._margin_factor = 1.15
    mgr._margin_buffer = 0.90

    mgr.submit_trade_plan_result(_gate_plan(quantity=65))

    inputs = margin_inputs[0]
    assert inputs.per_trade_risk_pct == 0.5
    assert inputs.per_trade_cap_pct == 25.0
    assert inputs.min_lots_per_trade == 1
    assert inputs.max_lots_per_trade == 2
    assert inputs.atr_multiple == 2.25
    assert inputs.margin_factor == 1.15
    assert inputs.margin_buffer == 0.90
    assert inputs.atr is None
    assert inputs.symbol == "NFO:NIFTY26AUG25000CE"
    assert inputs.price == 107.0
    assert inputs.stop_loss == 87.0


def test_entry_gate_result_carries_frozen_sizing_details(monkeypatch) -> None:
    """3A: the first gate-approved sizing is stamped on THIS result."""
    decision = types.SimpleNamespace(
        ok=True, quantity=65, reason=None, est_required=6955.0
    )
    mgr, _broker, _captured, _ = _gate_manager(monkeypatch, decision=decision)
    plan = TradePlan(
        symbol="NFO:NIFTY26AUG25000CE",
        side="BUY",
        quantity=130,
        entry_price=100.0,
        stop_loss=80.0,
        take_profit=140.0,
        intent="ENTRY",  # type: ignore[arg-type]
        requested_lots=2,
        resolved_lot_size=65,
        trace_id="trace-A",
        signal_id="signal-A",
        trade_lifecycle_id="tl-A",
    )

    result = mgr.submit_trade_plan_result(plan)

    assert result.details["entry_sizing_requested_quantity"] == 130
    assert result.details["entry_sizing_effective_quantity"] == 65
    assert result.details["entry_sizing_lot_size"] == 65
    assert result.details["entry_sizing_symbol"] == "NFO:NIFTY26AUG25000CE"
    assert result.details["entry_sizing_trace_id"] == "trace-A"
    assert result.details["entry_sizing_signal_id"] == "signal-A"
    assert result.details["entry_sizing_trade_lifecycle_id"] == "tl-A"


def test_entry_gate_pre_broker_telemetry_does_not_claim_attempt(monkeypatch) -> None:
    """6: the pre-broker event must not report broker_attempted=True."""
    decision = types.SimpleNamespace(
        ok=True, quantity=65, reason=None, est_required=6955.0
    )
    mgr, _broker, _captured, _ = _gate_manager(monkeypatch, decision=decision)
    events: list[dict] = []

    class _Rec:
        def info(self, *a, **k):
            extra = k.get("extra") or {}
            if extra.get("event") == "ENTRY_MARGIN_DECISION":
                events.append(extra)

        def warning(self, *a, **k):
            extra = k.get("extra") or {}
            if extra.get("event") == "ENTRY_MARGIN_DECISION":
                events.append(extra)

        def error(self, *a, **k):
            pass

        def debug(self, *a, **k):
            pass

    monkeypatch.setattr(mgr, "_logger", _Rec())
    mgr.submit_trade_plan_result(_gate_plan(quantity=65))

    assert events, "expected an ENTRY_MARGIN_DECISION event"
    assert events[-1]["broker_attempted"] is False
    assert events[-1]["broker_attempt_pending"] is True


def test_gate_to_created_order_to_partial_fill_uses_effective_quantity(
    monkeypatch,
) -> None:
    """END-TO-END PR1 PROOF: 130 requested -> gate approves 65 -> the REAL
    managed-order path creates the operative OrderDetails with 65 -> the REAL
    partial-fill reconciliation persists 65.

    Only external boundaries are faked (broker transport, balance, protected
    price, margin engine decision). The entry gate, place_managed_order_result,
    place_order, the OrderDetails constructor and the lifecycle handoff all
    run for real.
    """
    from nifty_scalper_bot.execution.order_manager import OrderPreflightResult
    from nifty_scalper_bot.execution.position_manager import PositionManager
    from nifty_scalper_bot.utils.rate_limiter import RateLimiter

    broker_calls: list[dict[str, Any]] = []

    class _Broker:
        """Transport double honouring the real adapter contract: place_order
        must return a response dict (see _submit_broker_order)."""

        def place_order(self, **kwargs: Any) -> dict[str, Any]:
            broker_calls.append(dict(kwargs))
            return {"order_id": "BROKER-ORD-1", "status": "OPEN"}

    # Canonical runtime wiring: entry recovery / partial-fill reconciliation
    # is installed onto the order-manager class (same pattern as
    # tests/execution/test_canonical_entry_recovery.py). The production
    # functions themselves are NOT mocked.
    from nifty_scalper_bot.execution.entry_recovery import install_entry_recovery

    class _IntegrationOrderManager(OrderManager):
        pass

    install_entry_recovery(_IntegrationOrderManager)

    mgr = _IntegrationOrderManager(_Broker(), PositionManager(), RateLimiter())
    monkeypatch.setattr(mgr, "is_kill_switch_active", lambda: False)
    monkeypatch.setattr(
        mgr, "_validate_trade_plan", lambda plan: OrderPreflightResult(True, "ok", {})
    )
    monkeypatch.setattr(mgr, "_protected_limit_price", lambda plan: 107.0)
    monkeypatch.setattr(mgr, "_lot_size_for_symbol", lambda symbol: 65)
    monkeypatch.setattr(
        mgr, "_resolve_available_margin", lambda **kw: (1_000_000.0, "mdm")
    )
    # External boundary: the sizing engine's verdict.
    monkeypatch.setattr(
        mgr,
        "_margin_engine",
        types.SimpleNamespace(
            plan=lambda inputs: types.SimpleNamespace(
                ok=True, quantity=65, reason=None, est_required=6955.0
            )
        ),
    )

    plan = TradePlan(
        symbol="NFO:NIFTY26AUG25000CE",
        side="BUY",
        quantity=130,
        entry_price=105.0,
        stop_loss=80.0,
        take_profit=140.0,
        intent="ENTRY",  # type: ignore[arg-type]
        requested_lots=2,
        resolved_lot_size=65,
        trace_id="trace-INT",
        signal_id="signal-INT",
        trade_lifecycle_id="tl-INT",
    )

    result = mgr.submit_trade_plan_result(plan)
    assert result.accepted, result.reason

    # --- the genuinely created operative order ---
    created = mgr._orders["BROKER-ORD-1"]
    assert created.quantity == 65
    assert created.requested_lots == 1
    assert created.resolved_lot_size == 65
    assert broker_calls and int(broker_calls[0]["quantity"]) == 65

    # --- audit provenance keeps 130, operative state does not ---
    assert result.details["entry_sizing_requested_quantity"] == 130
    assert result.details["entry_sizing_effective_quantity"] == 65
    assert result.details["entry_sizing_lot_size"] == 65

    # --- real partial-fill reconciliation ---
    mgr._update_from_response(
        created,
        {
            "status": "PARTIALLY FILLED",
            "filled_quantity": 20,
            "pending_quantity": 45,
        },
    )
    state = created.entry_lifecycle_state
    assert state["requested_quantity"] == 65
    assert state["requested_lots"] == 1
    assert state["resolved_lot_size"] == 65
    assert state["broker_filled_quantity"] == 20
    assert state["broker_pending_quantity"] == 45
    assert state["broker_filled_quantity"] + state["broker_pending_quantity"] == 65
    assert int(state.get("protected_quantity") or 0) <= 65

    # No operative numeric quantity anywhere equals the pre-gate 130.
    operative = [v for k, v in state.items() if isinstance(v, int) and "original" not in k]
    assert 130 not in operative
    assert created.quantity != 130


def test_legacy_place_order_without_lot_metadata_keeps_safe_defaults(
    monkeypatch,
) -> None:
    """Signature compatibility: a direct legacy/protective place_order() call
    that omits the new keyword arguments must still work and default the lot
    metadata to 0. Exits must never be required to resolve entry lot data."""
    from nifty_scalper_bot.execution.position_manager import PositionManager
    from nifty_scalper_bot.utils.rate_limiter import RateLimiter

    class _Broker:
        def place_order(self, **kwargs: Any) -> dict[str, Any]:
            return {"order_id": "LEGACY-1", "status": "OPEN"}

    mgr = OrderManager(_Broker(), PositionManager(), RateLimiter())
    monkeypatch.setattr(mgr, "is_kill_switch_active", lambda: False)

    # Must not raise TypeError: the new lot-metadata parameters are optional
    # keyword-only additions with safe defaults, so exposure-reducing callers
    # that never resolve entry lot data keep working unchanged.
    mgr.place_order(
        symbol="NFO:NIFTY26AUG25000CE",
        side="SELL",
        quantity=65,
        order_type=OrderType.MARKET,
        intent="EXIT",
        check_risk=False,
    )

    # Whatever the surrounding guards decide, any order this legacy path does
    # register carries the safe defaults rather than inheriting entry sizing.
    for created in mgr._orders.values():
        assert created.requested_lots == 0
        assert created.resolved_lot_size == 0

    import inspect

    sig = inspect.signature(OrderManager.place_order)
    assert sig.parameters["requested_lots"].default == 0
    assert sig.parameters["resolved_lot_size"].default == 0
