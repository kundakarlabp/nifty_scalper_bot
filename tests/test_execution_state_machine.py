"""Tests for execution order state machine transitions."""

from nifty_scalper_bot.execution.order_state_machine import (
    ExecutionState,
    OrderStateMachine,
)


def test_invalid_transition_rejected() -> None:
    machine = OrderStateMachine()

    assert machine.transition(ExecutionState.ORDER_PENDING) is False
    assert machine.state == ExecutionState.IDLE


def test_happy_path_transitions() -> None:
    machine = OrderStateMachine()

    assert machine.transition(ExecutionState.SIGNAL_RECEIVED) is True
    assert machine.transition(ExecutionState.ORDER_PENDING) is True
    assert machine.transition(ExecutionState.POSITION_OPEN) is True
    assert machine.transition(ExecutionState.EXIT_PENDING) is True
    assert machine.transition(ExecutionState.IDLE) is True


def test_ready_state_accepts_signal() -> None:
    machine = OrderStateMachine()

    assert machine.transition(ExecutionState.READY) is True
    assert machine.can_accept_signal() is True
    assert machine.transition(ExecutionState.SIGNAL_RECEIVED) is True


# ---- Stale-state metadata + recovery (async so they execute under conftest hook) ----

async def test_force_idle_records_reason_and_clears_order_id() -> None:
    machine = OrderStateMachine()
    machine.transition(ExecutionState.SIGNAL_RECEIVED)
    machine.transition(ExecutionState.ORDER_PENDING, order_id="X1")
    assert machine.order_id == "X1"
    machine.force_idle(reason="stale_order_pending_no_active_order")
    assert machine.state == ExecutionState.IDLE
    assert machine.order_id is None
    details = machine.current_state_details()
    assert details["reason"] == "stale_order_pending_no_active_order"


async def test_state_age_seconds_nonnegative_and_resets_on_transition() -> None:
    machine = OrderStateMachine()
    assert machine.state_age_seconds() >= 0.0
    machine.transition(ExecutionState.SIGNAL_RECEIVED)
    # freshly entered -> small age
    assert machine.state_age_seconds() < 5.0


async def test_transition_stamps_order_id_and_trace() -> None:
    machine = OrderStateMachine()
    machine.transition(ExecutionState.SIGNAL_RECEIVED, trace_id="t-1")
    machine.transition(ExecutionState.ORDER_PENDING, order_id="O-9", reason="order_submit", trace_id="t-1")
    d = machine.current_state_details()
    assert d["order_id"] == "O-9"
    assert d["trace_id"] == "t-1"
    assert d["reason"] == "order_submit"


async def test_set_order_id_attaches_to_current_state() -> None:
    machine = OrderStateMachine()
    machine.transition(ExecutionState.SIGNAL_RECEIVED)
    machine.transition(ExecutionState.ORDER_PENDING)
    machine.set_order_id("BROKER-123")
    assert machine.order_id == "BROKER-123"
