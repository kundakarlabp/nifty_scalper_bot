from __future__ import annotations

from dataclasses import replace
import importlib
import json
import os
import subprocess
import sys
import threading
from types import SimpleNamespace
from typing import Any

import nifty_scalper_bot.execution as execution
from nifty_scalper_bot.execution import order_manager, order_manager_core
from nifty_scalper_bot.execution.native_entry_gate import NO_BLOCK
from nifty_scalper_bot.execution.runtime_order_manager import RuntimeOrderManager


class _Logger:
    def debug(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def info(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def warning(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def critical(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def error(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class _Provider:
    def __init__(self, unresolved: bool = True) -> None:
        self.unresolved = unresolved

    def has_unresolved_exit(self) -> bool:
        return self.unresolved

    def get_first_unresolved_exit_bracket_id(self) -> str:
        return "entry-1"


def _manager(provider: Any | None = None) -> RuntimeOrderManager:
    manager = object.__new__(RuntimeOrderManager)
    manager._logger = _Logger()
    manager._last_order_decision = {}
    manager._unresolved_exit_provider = provider
    manager._unresolved_exit_guard_installed = provider is not None
    return manager


def test_operator_controls_are_native_runtime_methods() -> None:
    for name in (
        "emergency_stop",
        "engage_kill_switch",
        "kill_switch",
        "cancel_pending_orders",
        "cancel_all_open_orders",
        "cancel_non_protective_orders",
        "flatten_all",
        "flatten_positions",
        "close_all_positions",
    ):
        method = getattr(RuntimeOrderManager, name)
        assert method.__module__ == "nifty_scalper_bot.execution.runtime_order_manager"
    assert not hasattr(RuntimeOrderManager, "_operator_control_patch")


def test_public_order_import_has_one_stable_runtime_identity() -> None:
    assert order_manager.OrderManager is RuntimeOrderManager
    assert execution.OrderManager is RuntimeOrderManager
    assert issubclass(RuntimeOrderManager, order_manager_core.OrderManager)
    assert not hasattr(order_manager, "LegacyOrderManager")
    assert RuntimeOrderManager.submit_trade_plan_result.__module__ == (
        "nifty_scalper_bot.execution.runtime_order_manager"
    )
    assert RuntimeOrderManager._update_from_response.__module__ == (
        "nifty_scalper_bot.execution.runtime_order_manager"
    )
    assert not hasattr(RuntimeOrderManager, "_canonical_entry_recovery_installed")


def test_importing_package_does_not_replace_order_methods() -> None:
    before = order_manager.OrderManager.submit_trade_plan_result
    imported = importlib.import_module("nifty_scalper_bot.execution")
    assert imported.OrderManager is order_manager.OrderManager
    assert order_manager.OrderManager.submit_trade_plan_result is before


def test_native_gate_blocks_new_entry_without_calling_base_engine() -> None:
    manager = _manager(_Provider(True))
    result = RuntimeOrderManager.submit_trade_plan_result(
        manager,
        SimpleNamespace(symbol="NFO:NIFTY26JUN24000CE"),
    )
    assert result.accepted is False
    assert result.reason == "unresolved_exit_position"
    assert result.broker_attempted is False
    assert manager._last_order_decision["bracket_id"] == "entry-1"


def test_native_gate_allows_protective_exit_but_blocks_normal_order() -> None:
    manager = _manager(_Provider(True))
    protective = manager._blocked(
        "place_order",
        (),
        {
            "symbol": "NFO:NIFTY26JUN24000CE",
            "side": "SELL",
            "quantity": 50,
            "tag": "EXIT_HARD_SL",
            "reduce_only": True,
        },
    )
    normal = manager._blocked(
        "place_order",
        (),
        {
            "symbol": "NFO:NIFTY26JUN24000CE",
            "side": "BUY",
            "quantity": 50,
            "tag": "runner_entry",
        },
    )
    assert protective is NO_BLOCK
    assert normal is None


def test_unresolved_exit_provider_is_canonical_reconciliation_owner() -> None:
    manager = _manager(None)
    provider = _Provider(True)

    manager.set_unresolved_exit_provider(provider)

    assert manager._unresolved_exit_provider is provider
    assert manager._bracket_manager is provider

    manager.set_unresolved_exit_provider(None)

    assert manager._unresolved_exit_provider is None
    assert manager._bracket_manager is None


def test_successful_native_order_marks_endpoint_verified(monkeypatch) -> None:
    manager = _manager(None)

    monkeypatch.setattr(
        order_manager_core.OrderManager,
        "place_order",
        lambda self, *args, **kwargs: "OID-PROOF",
    )

    result = manager.place_order(
        symbol="NFO:NIFTY26SEP22700CE",
        side="BUY",
        quantity=65,
        intent="ENTRY",
        check_risk=False,
    )

    assert result == "OID-PROOF"
    assert manager.order_endpoint_verified is True
    assert manager.broker_order_endpoint_verified is True


def test_rejected_native_order_does_not_mark_endpoint_verified(monkeypatch) -> None:
    manager = _manager(None)

    monkeypatch.setattr(
        order_manager_core.OrderManager,
        "place_order",
        lambda self, *args, **kwargs: None,
    )

    result = manager.place_order(
        symbol="NFO:NIFTY26SEP22700CE",
        side="BUY",
        quantity=65,
        intent="ENTRY",
        check_risk=False,
    )

    assert result is None
    assert getattr(manager, "order_endpoint_verified", False) is False
    assert getattr(manager, "broker_order_endpoint_verified", False) is False


def test_managed_order_preserves_approved_strategy_name(monkeypatch) -> None:
    manager = _manager(None)
    captured: dict[str, Any] = {}

    def core_place_order(self: Any, *args: Any, **kwargs: Any) -> str:
        captured.update(kwargs)
        return "OID-1"

    def core_managed(self: Any, *args: Any, **kwargs: Any) -> Any:
        order_id = self.place_order(
            symbol=kwargs["symbol"],
            side=kwargs["side"],
            quantity=kwargs["quantity"],
            signal_id=kwargs.get("signal_id"),
            intent="ENTRY",
        )
        return SimpleNamespace(
            accepted=bool(order_id),
            order_id=order_id,
            reason="accepted",
            details={},
            broker_attempted=True,
        )

    monkeypatch.setattr(
        order_manager_core.OrderManager, "place_order", core_place_order
    )
    monkeypatch.setattr(
        order_manager_core.OrderManager,
        "place_managed_order_result",
        core_managed,
    )

    result = manager.place_managed_order_result(
        symbol="NFO:NIFTY26JUL23950PE",
        side="BUY",
        quantity=65,
        strategy_name="OrderFlow",
        signal_id="sig-1",
    )

    assert result.accepted is True
    assert captured["strategy_name"] == "OrderFlow"


def test_live_env_normalizes_per_trade_risk_to_five_percent(monkeypatch) -> None:
    from nifty_scalper_bot.config.env_utils import normalise_live_env_defaults

    monkeypatch.setenv("ENABLE_LIVE", "true")
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("RISK__PER_TRADE_RISK_PCT", "4.0")
    monkeypatch.setenv("RISK_PER_TRADE_PCT", "4.0")

    normalise_live_env_defaults()

    assert os.environ["RISK__PER_TRADE_RISK_PCT"] == "5.0"
    assert os.environ["RISK_PER_TRADE_PCT"] == "5.0"


def test_order_module_is_safe_when_imported_before_package() -> None:
    code = r"""
import importlib
import json
om = importlib.import_module("nifty_scalper_bot.execution.order_manager")
before_class = id(om.OrderManager)
before_method = id(om.OrderManager.submit_trade_plan_result)
execution = importlib.import_module("nifty_scalper_bot.execution")
print(json.dumps({
    "before_class": before_class,
    "after_class": id(om.OrderManager),
    "package_class": id(execution.OrderManager),
    "before_method": before_method,
    "after_method": id(om.OrderManager.submit_trade_plan_result),
    "module": om.OrderManager.__module__,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                filter(None, ["src", os.environ.get("PYTHONPATH", "")])
            ),
        },
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["before_class"] == payload["after_class"] == payload["package_class"]
    assert payload["before_method"] == payload["after_method"]
    assert payload["module"] == "nifty_scalper_bot.execution.runtime_order_manager"


def test_exit_identity_reaches_core_place_order(monkeypatch) -> None:
    manager = _manager(None)
    captured: dict[str, Any] = {}

    def core_place_order(self: Any, *args: Any, **kwargs: Any) -> str:
        captured.update(kwargs)
        return "EXIT-1"

    monkeypatch.setattr(
        order_manager_core.OrderManager, "place_order", core_place_order
    )

    result = manager.place_order(
        symbol="NFO:NIFTY2681124500CE",
        side="SELL",
        quantity=65,
        intent="EXIT",
        bracket_id="ENTRY-1",
        linked_entry_order_id="ENTRY-1",
        trade_lifecycle_id="ENTRY-1",
        tag="exit_test",
        check_risk=False,
    )

    assert result == "EXIT-1"
    assert captured["bracket_id"] == "ENTRY-1"
    assert captured["linked_entry_order_id"] == "ENTRY-1"
    assert captured["trade_lifecycle_id"] == "ENTRY-1"


def test_filled_exit_update_notifies_runtime_bracket_owner(monkeypatch) -> None:
    class _ExitProvider(_Provider):
        def __init__(self) -> None:
            super().__init__(False)
            self.calls: list[tuple[Any, dict[str, Any]]] = []

        def reconcile_filled_exit_order(
            self, order: Any, payload: dict[str, Any]
        ) -> bool:
            self.calls.append((order, dict(payload)))
            return True

    provider = _ExitProvider()
    manager = _manager(provider)
    manager._bracket_manager = provider
    filled = SimpleNamespace(
        order_id="EXIT-1",
        symbol="NFO:NIFTY2681124500CE",
        side="SELL",
        quantity=65,
        filled_quantity=65,
        fill_price=95.0,
        status=order_manager_core.OrderStatus.FILLED,
        intent="EXIT",
        bracket_id="ENTRY-1",
        linked_entry_order_id="ENTRY-1",
        trade_lifecycle_id="ENTRY-1",
    )

    def core_apply(self: Any, payload: dict[str, Any]) -> Any:
        return filled

    monkeypatch.setattr(
        order_manager_core.OrderManager,
        "_apply_broker_order_update",
        core_apply,
    )

    payload = {
        "order_id": "EXIT-1",
        "status": "COMPLETE",
        "average_price": 95.0,
        "filled_quantity": 65,
    }
    result = RuntimeOrderManager._apply_broker_order_update(manager, payload)

    assert result is filled
    assert provider.calls == [(filled, payload)]


def test_real_broker_update_returns_order_and_reconciles_filled_exit() -> None:
    class _ExitProvider(_Provider):
        def __init__(self) -> None:
            super().__init__(False)
            self.calls: list[tuple[Any, dict[str, Any]]] = []

        def reconcile_filled_exit_order(
            self, order: Any, payload: dict[str, Any]
        ) -> bool:
            self.calls.append((order, dict(payload)))
            return True

    provider = _ExitProvider()
    manager = _manager(provider)
    manager._bracket_manager = provider
    manager._lock = threading.RLock()
    exit_order = order_manager_core.OrderDetails(
        order_id="EXIT-1",
        symbol="NFO:NIFTY2681124500CE",
        side="SELL",
        quantity=65,
        order_type=order_manager_core.OrderType.MARKET,
        status=order_manager_core.OrderStatus.SUBMITTED,
        intent="EXIT",
        bracket_id="ENTRY-1",
        linked_entry_order_id="ENTRY-1",
        trade_lifecycle_id="ENTRY-1",
    )
    manager._orders = {exit_order.order_id: exit_order}
    manager._positions = SimpleNamespace(
        apply_broker_order_update=lambda *_args, **_kwargs: None
    )
    manager._register_virtual_bracket_for_fill = lambda *_args, **_kwargs: None
    manager._confirm_position_protection_for_fill = lambda *_args, **_kwargs: None
    manager._notify_failed_entry_terminal = lambda *_args, **_kwargs: None
    manager.save_orders = lambda: None

    payload = {
        "order_id": "EXIT-1",
        "status": "COMPLETE",
        "average_price": 95.0,
        "filled_quantity": 65,
    }
    result = RuntimeOrderManager._apply_broker_order_update(manager, payload)

    assert result is exit_order
    assert exit_order.status is order_manager_core.OrderStatus.FILLED
    assert provider.calls == [(exit_order, payload)]


def test_real_broker_update_supplies_order_to_partial_fill_reconciler(
    monkeypatch,
) -> None:
    reconciled: list[tuple[Any, dict[str, Any]]] = []
    monkeypatch.setattr(
        "nifty_scalper_bot.execution.runtime_order_manager._finalize_partial_entry",
        lambda manager, order, payload: reconciled.append((order, dict(payload))),
    )
    manager = _manager(None)
    manager._bracket_manager = None
    manager._lock = threading.RLock()
    entry_order = order_manager_core.OrderDetails(
        order_id="ENTRY-1",
        symbol="NFO:NIFTY2681124500CE",
        side="BUY",
        quantity=130,
        order_type=order_manager_core.OrderType.LIMIT,
        status=order_manager_core.OrderStatus.SUBMITTED,
        intent="ENTRY",
        requested_lots=2,
        resolved_lot_size=65,
    )
    manager._orders = {entry_order.order_id: entry_order}
    manager._positions = SimpleNamespace(
        apply_broker_order_update=lambda *_args, **_kwargs: None
    )
    manager._register_virtual_bracket_for_fill = lambda *_args, **_kwargs: None
    manager._confirm_position_protection_for_fill = lambda *_args, **_kwargs: None
    manager._notify_failed_entry_terminal = lambda *_args, **_kwargs: None
    manager.save_orders = lambda: None

    payload = {
        "order_id": "ENTRY-1",
        "status": "PARTIALLY FILLED",
        "average_price": 100.0,
        "filled_quantity": 65,
        "pending_quantity": 65,
    }
    result = RuntimeOrderManager._apply_broker_order_update(manager, payload)

    assert result is entry_order
    assert entry_order.status is order_manager_core.OrderStatus.PARTIALLY_FILLED
    assert reconciled == [(entry_order, payload)]


def test_fast_fill_rejects_status_only_complete_payload() -> None:
    manager = _manager(None)
    updates: list[dict[str, Any]] = []

    class _Broker:
        def get_order_status(self, _order_id: str) -> dict[str, Any]:
            return {"status": "COMPLETE", "average_price": 101.5}

    manager._broker = _Broker()
    manager.on_order_update = lambda payload: updates.append(dict(payload))

    assert manager._confirm_fill_fast("ENTRY-STATUS-ONLY", timeout_ms=1) is False
    assert updates == []


def test_fast_fill_accepts_quantitative_broker_execution() -> None:
    manager = _manager(None)
    updates: list[dict[str, Any]] = []

    class _Broker:
        def get_order_status(self, _order_id: str) -> dict[str, Any]:
            return {
                "status": "COMPLETE",
                "average_price": 101.5,
                "filled_quantity": 65,
            }

    manager._broker = _Broker()
    manager.on_order_update = lambda payload: updates.append(dict(payload))

    assert manager._confirm_fill_fast("ENTRY-1", timeout_ms=50) is True
    assert updates == [
        {
            "status": "COMPLETE",
            "average_price": 101.5,
            "filled_quantity": 65,
        }
    ]


def test_entry_fill_journal_uses_actual_broker_fill_values() -> None:
    manager = _manager(None)
    events: list[dict[str, Any]] = []
    manager._trade_journal = SimpleNamespace(
        log_event=lambda payload: events.append(dict(payload))
    )
    manager._orders = {
        "ENTRY-1": SimpleNamespace(
            filled_quantity=65,
            fill_price=101.5,
            average_price=101.5,
        )
    }

    manager._log_trade_event(
        "ORDER_FILL_CONFIRMED",
        symbol="NFO:NIFTY26JUL23950CE",
        side="BUY",
        qty=130,
        price=99.0,
        order_id="ENTRY-1",
        meta={
            "trade_id": "TRD_sig-1",
            "signal_id": "sig-1",
            "trace_id": "trace-1",
            "strategy": "VWAP",
        },
    )

    assert len(events) == 1
    assert events[0]["qty"] == 65
    assert events[0]["price"] == 101.5
    assert events[0]["meta"]["broker_confirmed_fill"] is True


def test_open_live_entry_reprices_same_order_within_small_budget(monkeypatch) -> None:
    monkeypatch.setenv("ENTRY_OPEN_REPRICE_ENABLED", "true")
    monkeypatch.setenv("ENTRY_OPEN_REPRICE_MAX_MODIFICATIONS", "2")
    monkeypatch.setenv("ENTRY_OPEN_REPRICE_MIN_INTERVAL_SECONDS", "0")
    monkeypatch.setenv("ENTRY_OPEN_REPRICE_MAX_DEVIATION_PCT", "0.75")
    manager = _manager(None)
    manager.is_live_mode = lambda: True
    quote = {"bid": 100.8, "ask": 101.0, "age_ms": 10.0}
    manager._get_latest_quote_safe = lambda _symbol: dict(quote)
    manager._extract_quote_diagnostics = lambda payload: dict(payload)
    manager._lock = threading.RLock()
    entry = order_manager_core.OrderDetails(
        order_id="ENTRY-REPRICE",
        symbol="NFO:NIFTY2681124500CE",
        side="BUY",
        quantity=65,
        order_type=order_manager_core.OrderType.LIMIT,
        status=order_manager_core.OrderStatus.SUBMITTED,
        price=100.5,
        intent="ENTRY",
        resolved_lot_size=65,
    )
    manager._orders = {entry.order_id: entry}
    manager._positions = SimpleNamespace(
        apply_broker_order_update=lambda *_args, **_kwargs: None
    )
    manager._register_virtual_bracket_for_fill = lambda *_args, **_kwargs: None
    manager._confirm_position_protection_for_fill = lambda *_args, **_kwargs: None
    manager._notify_failed_entry_terminal = lambda *_args, **_kwargs: None
    manager.save_orders = lambda: None
    modified: list[tuple[str, dict[str, Any]]] = []

    monkeypatch.setattr(
        order_manager_core.OrderManager,
        "modify_order",
        lambda _self, order_id, **changes: modified.append(
            (str(order_id), dict(changes))
        )
        or True,
    )

    payload = {
        "order_id": entry.order_id,
        "status": "OPEN",
        "filled_quantity": 0,
        "pending_quantity": 65,
    }
    RuntimeOrderManager._apply_broker_order_update(manager, payload)
    assert modified == [("ENTRY-REPRICE", {"price": 101.0})]
    assert entry.price == 101.0

    quote.update({"bid": 101.15, "ask": 101.25})
    RuntimeOrderManager._apply_broker_order_update(manager, payload)
    assert modified[-1] == ("ENTRY-REPRICE", {"price": 101.25})

    quote.update({"bid": 101.35, "ask": 101.45})
    RuntimeOrderManager._apply_broker_order_update(manager, payload)
    assert len(modified) == 2
    assert entry.trade_provenance["entry_open_reprice_count"] == 2
    assert entry.trade_provenance["entry_open_reprice_anchor_price"] == 100.5


def test_absolute_level_entry_is_never_repriced(monkeypatch) -> None:
    monkeypatch.setenv("ENTRY_OPEN_REPRICE_ENABLED", "true")
    monkeypatch.setenv("ENTRY_OPEN_REPRICE_MIN_INTERVAL_SECONDS", "0")
    manager = _manager(None)
    manager.is_live_mode = lambda: True
    manager._get_latest_quote_safe = lambda _symbol: {
        "bid": 101.0,
        "ask": 101.2,
        "age_ms": 10.0,
    }
    manager._extract_quote_diagnostics = lambda payload: dict(payload)
    manager._lock = threading.RLock()
    entry = order_manager_core.OrderDetails(
        order_id="ENTRY-ABSOLUTE",
        symbol="NFO:NIFTY2681124500CE",
        side="BUY",
        quantity=65,
        order_type=order_manager_core.OrderType.LIMIT,
        status=order_manager_core.OrderStatus.SUBMITTED,
        price=100.5,
        intent="ENTRY",
        resolved_lot_size=65,
        trade_provenance={"bracket_anchor_mode": "absolute_level"},
    )
    manager._orders = {entry.order_id: entry}
    manager._positions = SimpleNamespace(
        apply_broker_order_update=lambda *_args, **_kwargs: None
    )
    manager._register_virtual_bracket_for_fill = lambda *_args, **_kwargs: None
    manager._confirm_position_protection_for_fill = lambda *_args, **_kwargs: None
    manager._notify_failed_entry_terminal = lambda *_args, **_kwargs: None
    manager.save_orders = lambda: None
    modified: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        order_manager_core.OrderManager,
        "modify_order",
        lambda _self, order_id, **changes: modified.append(
            (str(order_id), dict(changes))
        )
        or True,
    )

    RuntimeOrderManager._apply_broker_order_update(
        manager,
        {
            "order_id": entry.order_id,
            "status": "OPEN",
            "filled_quantity": 0,
            "pending_quantity": 65,
        },
    )

    assert modified == []


def test_partially_filled_entry_is_never_repriced(monkeypatch) -> None:
    monkeypatch.setenv("ENTRY_OPEN_REPRICE_ENABLED", "true")
    monkeypatch.setenv("ENTRY_OPEN_REPRICE_MIN_INTERVAL_SECONDS", "0")
    manager = _manager(None)
    manager.is_live_mode = lambda: True
    manager._get_latest_quote_safe = lambda _symbol: {
        "bid": 101.0,
        "ask": 101.2,
        "age_ms": 10.0,
    }
    manager._extract_quote_diagnostics = lambda payload: dict(payload)
    manager._lock = threading.RLock()
    entry = order_manager_core.OrderDetails(
        order_id="ENTRY-PARTIAL-NO-CHASE",
        symbol="NFO:NIFTY2681124500CE",
        side="BUY",
        quantity=130,
        order_type=order_manager_core.OrderType.LIMIT,
        status=order_manager_core.OrderStatus.SUBMITTED,
        price=100.5,
        intent="ENTRY",
        resolved_lot_size=65,
    )
    manager._orders = {entry.order_id: entry}
    manager._positions = SimpleNamespace(
        apply_broker_order_update=lambda *_args, **_kwargs: None
    )
    manager._register_virtual_bracket_for_fill = lambda *_args, **_kwargs: None
    manager._confirm_position_protection_for_fill = lambda *_args, **_kwargs: None
    manager._notify_failed_entry_terminal = lambda *_args, **_kwargs: None
    manager.save_orders = lambda: None
    modified: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        "nifty_scalper_bot.execution.runtime_order_manager._finalize_partial_entry",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        order_manager_core.OrderManager,
        "modify_order",
        lambda _self, order_id, **changes: modified.append(
            (str(order_id), dict(changes))
        )
        or True,
    )

    RuntimeOrderManager._apply_broker_order_update(
        manager,
        {
            "order_id": entry.order_id,
            "status": "PARTIALLY FILLED",
            "filled_quantity": 65,
            "pending_quantity": 65,
        },
    )

    assert modified == []

def _reprice_ready_entry() -> order_manager_core.OrderDetails:
    return order_manager_core.OrderDetails(
        order_id="ENTRY-HARDENED-REPRICE",
        symbol="NFO:NIFTY2681124500CE",
        side="BUY",
        quantity=65,
        order_type=order_manager_core.OrderType.LIMIT,
        status=order_manager_core.OrderStatus.SUBMITTED,
        price=100.5,
        stop_loss=90.5,
        take_profit=120.5,
        intent="ENTRY",
        signal_id="sig-reprice",
        client_order_id="client-reprice",
        trade_lifecycle_id="trade-reprice",
        instrument_token=12345,
        requested_lots=1,
        resolved_lot_size=65,
        trade_provenance={
            "bracket_anchor_mode": "distance",
            "entry_max_quote_age_ms": 1000,
            "entry_max_spread_pct": 1.0,
            "entry_min_depth_qty": 65,
        },
    )


def _configure_reprice_manager(manager, quote, modified, persisted=None) -> None:
    manager.is_live_mode = lambda: True
    manager._get_latest_quote_safe = lambda _symbol: dict(quote)
    manager._extract_quote_diagnostics = lambda payload: dict(payload)
    manager._lock = threading.RLock()
    manager._positions = SimpleNamespace(
        apply_broker_order_update=lambda *_args, **_kwargs: None
    )
    manager._register_virtual_bracket_for_fill = lambda *_args, **_kwargs: None
    manager._confirm_position_protection_for_fill = lambda *_args, **_kwargs: None
    manager._notify_failed_entry_terminal = lambda *_args, **_kwargs: None
    manager.save_orders = (
        (lambda: persisted.append("save_orders")) if persisted is not None else lambda: None
    )
    manager._persist_order_snapshot = (
        (lambda _order: persisted.append("snapshot"))
        if persisted is not None
        else lambda _order: None
    )
    manager._apply_entry_margin_gate = lambda plan, _price: (plan, None)
    manager._reanchor_bracket_to_price = (
        lambda plan, price: replace(
            plan,
            entry_price=price,
            stop_loss=round(price - 10.0, 2),
            take_profit=round(price + 20.0, 2),
        )
    )
    manager.modify_order = lambda order_id, **changes: modified.append(
        (str(order_id), dict(changes))
    ) or True


def test_open_entry_reprice_requires_ws_tradable_depth(monkeypatch) -> None:
    monkeypatch.setenv("ENTRY_OPEN_REPRICE_ENABLED", "true")
    monkeypatch.setenv("ENTRY_OPEN_REPRICE_MIN_INTERVAL_SECONDS", "0")
    manager = _manager(None)
    entry = _reprice_ready_entry()
    manager._orders = {entry.order_id: entry}
    modified: list[tuple[str, dict[str, Any]]] = []
    quote = {
        "bid": 100.9,
        "ask": 101.0,
        "age_ms": 10.0,
        "ask_qty": 130,
        "bid_qty": 130,
        "spread_pct": 0.1,
        "source": "poll",
        "tradable_quote": True,
        "depth_available": True,
    }
    _configure_reprice_manager(manager, quote, modified)

    RuntimeOrderManager._apply_broker_order_update(
        manager,
        {"order_id": entry.order_id, "status": "OPEN", "filled_quantity": 0},
    )
    assert modified == []

    quote["source"] = "ws"
    quote["tradable_quote"] = False
    RuntimeOrderManager._apply_broker_order_update(
        manager,
        {"order_id": entry.order_id, "status": "OPEN", "filled_quantity": 0},
    )
    assert modified == []

    quote["tradable_quote"] = True
    RuntimeOrderManager._apply_broker_order_update(
        manager,
        {"order_id": entry.order_id, "status": "OPEN", "filled_quantity": 0},
    )
    assert modified == [(entry.order_id, {"price": 101.0})]


def test_open_entry_reprice_reapplies_spread_depth_and_risk(monkeypatch) -> None:
    monkeypatch.setenv("ENTRY_OPEN_REPRICE_ENABLED", "true")
    monkeypatch.setenv("ENTRY_OPEN_REPRICE_MIN_INTERVAL_SECONDS", "0")
    manager = _manager(None)
    entry = _reprice_ready_entry()
    manager._orders = {entry.order_id: entry}
    modified: list[tuple[str, dict[str, Any]]] = []
    quote = {
        "bid": 99.0,
        "ask": 101.0,
        "age_ms": 10.0,
        "ask_qty": 130,
        "bid_qty": 130,
        "spread_pct": 2.0,
        "source": "ws",
        "tradable_quote": True,
        "depth_available": True,
    }
    _configure_reprice_manager(manager, quote, modified)

    RuntimeOrderManager._apply_broker_order_update(
        manager,
        {"order_id": entry.order_id, "status": "OPEN", "filled_quantity": 0},
    )
    assert modified == []

    quote.update({"bid": 100.9, "spread_pct": 0.1, "ask_qty": 10})
    RuntimeOrderManager._apply_broker_order_update(
        manager,
        {"order_id": entry.order_id, "status": "OPEN", "filled_quantity": 0},
    )
    assert modified == []

    quote["ask_qty"] = 130
    manager._apply_entry_margin_gate = lambda plan, _price: (
        replace(plan, quantity=0),
        None,
    )
    RuntimeOrderManager._apply_broker_order_update(
        manager,
        {"order_id": entry.order_id, "status": "OPEN", "filled_quantity": 0},
    )
    assert modified == []


def test_open_entry_reprice_persists_budget_and_reanchored_geometry(monkeypatch) -> None:
    monkeypatch.setenv("ENTRY_OPEN_REPRICE_ENABLED", "true")
    monkeypatch.setenv("ENTRY_OPEN_REPRICE_MIN_INTERVAL_SECONDS", "0")
    manager = _manager(None)
    entry = _reprice_ready_entry()
    manager._orders = {entry.order_id: entry}
    modified: list[tuple[str, dict[str, Any]]] = []
    persisted: list[str] = []
    quote = {
        "bid": 100.9,
        "ask": 101.0,
        "age_ms": 10.0,
        "ask_qty": 130,
        "bid_qty": 130,
        "spread_pct": 0.1,
        "source": "ws",
        "tradable_quote": True,
        "depth_available": True,
    }
    _configure_reprice_manager(manager, quote, modified, persisted)

    RuntimeOrderManager._apply_broker_order_update(
        manager,
        {"order_id": entry.order_id, "status": "OPEN", "filled_quantity": 0},
    )

    assert entry.price == 101.0
    assert entry.stop_loss == 91.0
    assert entry.take_profit == 121.0
    assert entry.trade_provenance["entry_open_reprice_count"] == 1
    assert persisted == ["snapshot", "save_orders"]

    payload = order_manager_core.OrderManager._serialize(manager, entry)
    restored = order_manager_core.OrderManager._order_from_dict(manager, payload)
    assert restored.stop_loss == 91.0
    assert restored.take_profit == 121.0
    assert restored.trade_provenance["entry_open_reprice_count"] == 1
    assert restored.trade_provenance["entry_open_reprice_anchor_price"] == 100.5
