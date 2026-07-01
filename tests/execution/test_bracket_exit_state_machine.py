"""Exit-state safety coverage for virtual brackets."""

from __future__ import annotations

import logging
import time
from typing import Any

import pytest

from nifty_scalper_bot.execution.bracket_manager import BracketExitLifecycle, BracketManager


@pytest.fixture(autouse=True)
def isolated_bracket_store(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))


class _Broker:
    def __init__(self, *, status: str = "OPEN", positions: list[dict[str, Any]] | None = None) -> None:
        self.status = status
        self.positions = positions if positions is not None else [
            {"symbol": "NFO:NIFTY2660923100CE", "quantity": 65}
        ]

    def get_order_status(self, _order_id: str) -> dict[str, Any]:
        return {"status": self.status, "average_price": 157.10}

    def get_positions(self) -> list[dict[str, Any]]:
        return list(self.positions)


class _OrderManager:
    def __init__(self, *, broker: _Broker | None = None, order_id: str | None = "exit-1", exc: Exception | None = None) -> None:
        self._broker = broker if broker is not None else _Broker()
        self.order_id = order_id
        self.exc = exc
        self.calls: list[dict[str, Any]] = []

    def place_order(self, **kwargs: Any) -> str | None:
        self.calls.append(kwargs)
        if self.exc is not None:
            raise self.exc
        return self.order_id


def _active_manager(order_manager: _OrderManager) -> BracketManager:
    manager = BracketManager(order_manager=order_manager)
    manager.register_virtual_bracket(
        order_id="entry-1",
        symbol="NFO:NIFTY2660923100CE",
        side="BUY",
        qty=65,
        price=150.65,
        sl=157.10,
        tp=175.00,
    )
    manager.confirm_entry_fill("entry-1", 150.65)
    return manager


def test_first_sl_breach_latches_and_submits_once(caplog) -> None:
    broker = _Broker(status="OPEN")
    om = _OrderManager(broker=broker, order_id="exit-1")
    manager = _active_manager(om)

    with caplog.at_level(logging.INFO):
        manager.on_tick("NFO:NIFTY2660923100CE", 157.10)
        manager.on_tick("NFO:NIFTY2660923100CE", 156.90)
        manager.on_tick("NFO:NIFTY2660923100CE", 156.80)

    bracket = manager.get_bracket("entry-1")
    assert bracket is not None
    assert bracket.exit_pending is True
    assert bracket.exit_state == BracketExitLifecycle.EXIT_ORDER_SUBMITTED.value
    assert bracket.exit_order_id == "exit-1"
    assert len(om.calls) == 1
    assert om.calls[0]["side"] == "SELL"
    assert caplog.text.count("EXIT_TRIGGERED") == 1


def test_exit_fill_closes_bracket_and_unfreezes_entries() -> None:
    broker = _Broker(status="COMPLETE", positions=[])
    om = _OrderManager(broker=broker, order_id="exit-filled")
    manager = _active_manager(om)

    manager.on_tick("NFO:NIFTY2660923100CE", 157.10)

    bracket = manager.get_bracket("entry-1")
    assert bracket is not None
    assert bracket.exit_state == BracketExitLifecycle.CLOSED.value
    assert bracket.exit_pending is False
    assert bracket.position_flat_confirmed is True
    assert bracket.close_source == "broker_fill"
    assert manager.has_unresolved_exit() is False


def test_retryable_submit_failure_uses_backoff_without_infinite_retry() -> None:
    om = _OrderManager(order_id=None)
    manager = _active_manager(om)
    manager._exit_retry_backoffs = [10.0, 20.0, 30.0]

    manager.on_tick("NFO:NIFTY2660923100CE", 157.10)
    manager.on_tick("NFO:NIFTY2660923100CE", 156.90)

    bracket = manager.get_bracket("entry-1")
    assert bracket is not None
    assert bracket.exit_pending is True
    assert bracket.exit_state == BracketExitLifecycle.EXIT_REJECTED_RETRYABLE.value
    assert bracket.exit_attempt_count == 1
    assert len(om.calls) == 1
    assert bracket.next_exit_attempt_at is not None

    bracket.next_exit_attempt_at = time.time() - 0.1
    manager.on_tick("NFO:NIFTY2660923100CE", 156.80)
    assert bracket.exit_attempt_count == 2
    assert len(om.calls) == 2


def test_fatal_submit_failure_escalates_and_freezes_entries() -> None:
    events: list[tuple[str, dict[str, object]]] = []
    om = _OrderManager(exc=RuntimeError("invalid symbol rejected"))
    manager = _active_manager(om)
    manager.set_notifier(lambda event, payload: events.append((event, dict(payload or {}))))

    manager.on_tick("NFO:NIFTY2660923100CE", 157.10)

    bracket = manager.get_bracket("entry-1")
    assert bracket is not None
    assert bracket.exit_state == BracketExitLifecycle.EXIT_FAILED_ESCALATED.value
    assert manager.has_unresolved_exit() is True
    assert any(event == "EXIT_ESCALATED" for event, _ in events)


def test_position_already_flat_closes_without_exit_submit() -> None:
    broker = _Broker(status="", positions=[])
    om = _OrderManager(broker=broker, order_id="should-not-submit")
    manager = _active_manager(om)

    manager.on_tick("NFO:NIFTY2660923100CE", 157.10)

    bracket = manager.get_bracket("entry-1")
    assert bracket is not None
    assert bracket.exit_state == BracketExitLifecycle.CLOSED.value
    assert bracket.close_source == "reconciled_flat"
    assert len(om.calls) == 0


def test_duplicate_exit_order_is_reconciled_not_resubmitted() -> None:
    broker = _Broker(status="OPEN")
    om = _OrderManager(broker=broker, order_id="new-exit")
    manager = _active_manager(om)
    bracket = manager.get_bracket("entry-1")
    assert bracket is not None
    bracket.exit_pending = True
    bracket.exit_order_id = "existing-exit"
    bracket.pending_exit_order_id = "existing-exit"
    bracket.exit_state = BracketExitLifecycle.EXIT_ORDER_SUBMITTED.value
    bracket.exit_reason = "HARD_SL_BREACH"
    bracket.exit_triggered_at = time.time()

    manager.on_tick("NFO:NIFTY2660923100CE", 156.50)

    assert len(om.calls) == 0
    assert bracket.exit_order_id == "existing-exit"
    assert bracket.exit_pending is True


def test_position_reconcile_failure_keeps_pending_then_escalates() -> None:
    class _FailingBroker(_Broker):
        def get_positions(self) -> list[dict[str, Any]]:
            raise RuntimeError("positions unavailable")

    om = _OrderManager(broker=_FailingBroker(status="OPEN"), order_id="exit-open")
    manager = _active_manager(om)
    manager._exit_unresolved_escalation_seconds = 0.01

    manager.on_tick("NFO:NIFTY2660923100CE", 157.10)
    bracket = manager.get_bracket("entry-1")
    assert bracket is not None
    assert bracket.exit_pending is True
    assert bracket.position_flat_confirmed is False

    bracket.exit_triggered_at = time.time() - 1.0
    manager._reconcile_exit_state(bracket, requested_by="test_timeout")

    assert bracket.exit_state == BracketExitLifecycle.EXIT_FAILED_ESCALATED.value
    assert bracket.exit_pending is True


def test_missing_order_id_failure_carries_order_manager_reason_and_payload(caplog) -> None:
    om = _OrderManager(order_id=None)
    om._last_order_decision = {
        "block_reason": "missing_order_id",
        "details": {"broker_payload": {"status": "error", "message": "IP not allowed"}},
        "broker_attempted": True,
        "retryable": False,
    }
    manager = _active_manager(om)

    with caplog.at_level(logging.ERROR):
        manager.on_tick("NFO:NIFTY2660923100CE", 157.10)

    bracket = manager.get_bracket("entry-1")
    assert bracket is not None
    assert bracket.last_exit_error == "IP not allowed"
    assert manager.has_unresolved_exit() is True
    assert "BRACKET_EXIT_ORDER_FAILED" in caplog.text


def test_repeated_failed_exits_set_unresolved_until_broker_confirms_flat() -> None:
    broker = _Broker(status="OPEN", positions=[{"symbol": "NFO:NIFTY2660923100CE", "quantity": 65}])
    om = _OrderManager(broker=broker, order_id=None)
    manager = _active_manager(om)
    manager._exit_retry_backoffs = [0.0, 0.0, 0.0]

    manager.on_tick("NFO:NIFTY2660923100CE", 157.10)
    bracket = manager.get_bracket("entry-1")
    assert bracket is not None
    for _ in range(4):
        bracket.next_exit_attempt_at = time.time() - 0.1
        manager.on_tick("NFO:NIFTY2660923100CE", 156.80)

    assert bracket.exit_state == BracketExitLifecycle.EXIT_FAILED_ESCALATED.value
    assert manager.has_unresolved_exit() is True

    broker.positions = []
    manager._reconcile_exit_state(bracket, requested_by="test_flat")
    assert bracket.exit_state == BracketExitLifecycle.CLOSED.value
    assert manager.has_unresolved_exit() is False
