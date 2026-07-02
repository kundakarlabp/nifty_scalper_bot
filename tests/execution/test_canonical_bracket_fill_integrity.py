from __future__ import annotations

import time
from typing import Any

import pytest

from nifty_scalper_bot.execution.bracket_manager import (
    BracketExitLifecycle,
    BracketManager,
)
from nifty_scalper_bot.execution.canonical_bracket_manager import (
    CanonicalBracketManager,
)


SYMBOL = "NFO:NIFTY2662324050PE"


@pytest.fixture(autouse=True)
def isolated_bracket_store(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))


class _Broker:
    def __init__(self) -> None:
        self.statuses: dict[str, dict[str, Any]] = {}
        self.positions: list[dict[str, Any]] = [
            {"symbol": SYMBOL, "quantity": 65}
        ]
        self.cancel_terminal_status = "CANCELLED"

    def get_order_status(self, order_id: str) -> dict[str, Any]:
        return dict(self.statuses.get(order_id, {"status": ""}))

    def get_positions(self) -> list[dict[str, Any]]:
        return list(self.positions)

    def cancel_order(self, order_id: str, *args: Any, **kwargs: Any) -> bool:
        payload = dict(self.statuses.get(order_id, {}))
        payload["status"] = self.cancel_terminal_status
        self.statuses[order_id] = payload
        return True


class _OrderManager:
    def __init__(self, broker: _Broker) -> None:
        self._broker = broker
        self._last_order_decision: dict[str, Any] = {}
        self.place_calls: list[dict[str, Any]] = []

    def place_order(self, **kwargs: Any) -> str:
        self.place_calls.append(dict(kwargs))
        order_id = f"exit-{len(self.place_calls)}"
        self._broker.statuses[order_id] = {"status": "OPEN"}
        return order_id

    def cancel_order(self, order_id: str, *args: Any, **kwargs: Any) -> bool:
        return self._broker.cancel_order(order_id, *args, **kwargs)

    def set_last_skip_reason(self, _reason: str) -> None:
        return None


def _manager(
    *,
    fill_price: float = 100.0,
    sync_grace_seconds: float = 0.0,
) -> tuple[CanonicalBracketManager, _OrderManager, _Broker]:
    broker = _Broker()
    order_manager = _OrderManager(broker)
    manager = BracketManager(order_manager=order_manager)
    assert isinstance(manager, CanonicalBracketManager)
    manager._running = False
    manager._watchdog_thread.join(timeout=1.0)
    manager._exit_cancel_confirm_timeout_seconds = 0.01
    manager._exit_cancel_poll_interval_seconds = 0.001
    manager._filled_position_sync_grace_seconds = sync_grace_seconds
    manager.register_virtual_bracket(
        order_id="entry-1",
        symbol=SYMBOL,
        side="BUY",
        qty=65,
        price=100.0,
        sl=90.0,
        tp=120.0,
        tp1_price=110.0,
        tp1_qty=25,
        activate_immediately=False,
    )
    manager.confirm_entry_fill("entry-1", fill_price)
    return manager, order_manager, broker


def _mark_exit_submitted(
    manager: CanonicalBracketManager,
    broker: _Broker,
    *,
    order_id: str,
    reason: str,
    residual_quantity: int,
    average_price: float,
) -> Any:
    bracket = manager.get_bracket("entry-1")
    assert bracket is not None
    bracket.exit_pending = True
    bracket.exit_reason = reason
    bracket.exit_state = BracketExitLifecycle.EXIT_ORDER_SUBMITTED.value
    bracket.entry_status = bracket.exit_state
    bracket.exit_order_id = order_id
    bracket.pending_exit_order_id = order_id
    bracket.exit_triggered_at = time.time()
    broker.statuses[order_id] = {
        "status": "COMPLETE",
        "average_price": average_price,
    }
    broker.positions = [{"symbol": SYMBOL, "quantity": residual_quantity}]
    return bracket


def test_confirmed_fill_reanchors_sl_final_tp_and_tp1() -> None:
    manager, _order_manager, _broker = _manager(fill_price=102.0)
    bracket = manager.get_bracket("entry-1")

    assert bracket is not None
    assert bracket.entry_price == 102.0
    assert bracket.sl_trigger_price == 91.8
    assert bracket.tp_trigger_price == 122.4
    assert len(bracket.tp_levels) == 1
    assert bracket.tp_levels[0].price == 112.2


def test_confirmed_tp1_fill_keeps_residual_position_open_and_protected() -> None:
    manager, order_manager, broker = _manager()
    events: list[tuple[str, dict[str, Any]]] = []
    close_calls: list[str] = []
    manager.set_notifier(lambda event, payload: events.append((event, dict(payload))))
    manager._on_exit_complete_hook = close_calls.append
    bracket = _mark_exit_submitted(
        manager,
        broker,
        order_id="tp1-exit",
        reason="TP1 Hit (110.00)",
        residual_quantity=40,
        average_price=110.10,
    )

    closed = manager._reconcile_exit_state(bracket, requested_by="post_submit")

    assert closed is False
    assert bracket.remaining_quantity == 40
    assert bracket.tp_levels[0].executed is True
    assert bracket.sl_trigger_price == bracket.entry_price
    assert bracket.exit_state == BracketExitLifecycle.OPEN_ACTIVE.value
    assert bracket.entry_status == "ACTIVE"
    assert bracket.exit_pending is False
    assert bracket.exit_order_id is None
    assert bracket.pending_exit_order_id is None
    assert bracket.position_flat_confirmed is False
    assert manager.has_unresolved_exit() is False
    assert close_calls == []
    assert order_manager.place_calls == []
    assert any(event == "PARTIAL_EXIT_CONFIRMED" for event, _ in events)


def test_tp1_tick_submit_then_fill_reconciles_through_live_path() -> None:
    manager, order_manager, broker = _manager()

    manager.on_tick(SYMBOL, 110.0)
    bracket = manager.get_bracket("entry-1")
    assert bracket is not None
    assert len(order_manager.place_calls) == 1
    assert bracket.exit_order_id == "exit-1"
    assert bracket.exit_pending is True

    broker.statuses["exit-1"] = {
        "status": "COMPLETE",
        "average_price": 110.05,
    }
    broker.positions = [{"symbol": SYMBOL, "quantity": 40}]
    manager.on_tick(SYMBOL, 110.5)

    assert bracket.remaining_quantity == 40
    assert bracket.tp_levels[0].executed is True
    assert bracket.exit_state == BracketExitLifecycle.OPEN_ACTIVE.value
    assert bracket.exit_pending is False
    assert len(order_manager.place_calls) == 1


def test_completed_order_waits_for_positions_endpoint_to_catch_up() -> None:
    manager, order_manager, broker = _manager(sync_grace_seconds=5.0)
    bracket = _mark_exit_submitted(
        manager,
        broker,
        order_id="tp1-sync",
        reason="TP1 Hit (110.00)",
        residual_quantity=65,
        average_price=110.05,
    )

    assert manager._reconcile_exit_state(bracket, requested_by="sync_first") is False
    assert bracket.exit_state == BracketExitLifecycle.EXIT_PARTIALLY_FILLED.value
    assert bracket.exit_order_id == "tp1-sync"
    assert bracket.tp_levels[0].executed is False
    assert order_manager.place_calls == []

    broker.positions = [{"symbol": SYMBOL, "quantity": 40}]
    assert manager._reconcile_exit_state(bracket, requested_by="sync_second") is False
    assert bracket.exit_state == BracketExitLifecycle.OPEN_ACTIVE.value
    assert bracket.remaining_quantity == 40
    assert bracket.tp_levels[0].executed is True
    assert bracket.exit_order_id is None


def test_filled_full_exit_with_residual_never_closes_or_rearms() -> None:
    manager, order_manager, broker = _manager()
    close_calls: list[str] = []
    manager._on_exit_complete_hook = close_calls.append
    bracket = _mark_exit_submitted(
        manager,
        broker,
        order_id="sl-exit",
        reason="HARD_SL_BREACH",
        residual_quantity=20,
        average_price=89.75,
    )

    closed = manager._reconcile_exit_state(bracket, requested_by="post_submit")

    assert closed is False
    assert bracket.remaining_quantity == 20
    assert bracket.exit_state == BracketExitLifecycle.EXIT_FAILED_ESCALATED.value
    assert bracket.exit_pending is True
    assert bracket.position_flat_confirmed is False
    assert bracket.exit_order_id == "sl-exit"
    assert manager.has_unresolved_exit() is True
    assert close_calls == []
    assert order_manager.place_calls == []

    broker.positions = []
    assert manager._reconcile_exit_state(bracket, requested_by="post_submit") is True
    assert bracket.exit_state == BracketExitLifecycle.CLOSED.value
    assert bracket.position_flat_confirmed is True
    assert close_calls == [SYMBOL]


def test_stale_rescue_fill_with_residual_uses_same_fail_closed_rule() -> None:
    manager, order_manager, broker = _manager()
    bracket = _mark_exit_submitted(
        manager,
        broker,
        order_id="stale-exit",
        reason="HARD_SL_BREACH",
        residual_quantity=20,
        average_price=89.50,
    )
    broker.statuses["stale-exit"]["status"] = "OPEN PENDING"
    broker.cancel_terminal_status = "COMPLETE"

    manager._rescue_stale_exit_order(
        bracket,
        order_id="stale-exit",
        qty=65,
        status="OPEN PENDING",
    )

    assert bracket.exit_state == BracketExitLifecycle.EXIT_FAILED_ESCALATED.value
    assert bracket.exit_pending is True
    assert bracket.remaining_quantity == 20
    assert bracket.position_flat_confirmed is False
    assert order_manager.place_calls == []
