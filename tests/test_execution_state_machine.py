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
