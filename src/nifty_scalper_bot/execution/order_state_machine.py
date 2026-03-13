"""Execution state machine to prevent duplicate and invalid order flows."""

from __future__ import annotations

import threading
from enum import Enum


class ExecutionState(str, Enum):
    """Supported execution lifecycle states."""

    IDLE = "IDLE"
    SIGNAL_RECEIVED = "SIGNAL_RECEIVED"
    ORDER_PENDING = "ORDER_PENDING"
    POSITION_OPEN = "POSITION_OPEN"
    EXIT_PENDING = "EXIT_PENDING"


class OrderStateMachine:
    """Track and validate order lifecycle transitions per symbol."""

    _VALID_TRANSITIONS: dict[ExecutionState, set[ExecutionState]] = {
        ExecutionState.IDLE: {ExecutionState.SIGNAL_RECEIVED},
        ExecutionState.SIGNAL_RECEIVED: {
            ExecutionState.ORDER_PENDING,
            ExecutionState.IDLE,
        },
        ExecutionState.ORDER_PENDING: {
            ExecutionState.POSITION_OPEN,
            ExecutionState.IDLE,
        },
        ExecutionState.POSITION_OPEN: {ExecutionState.EXIT_PENDING},
        ExecutionState.EXIT_PENDING: {
            ExecutionState.IDLE,
            ExecutionState.POSITION_OPEN,
        },
    }

    def __init__(self) -> None:
        """Initialize with IDLE state. Args: none. Returns: none. Raises: none."""
        self._state = ExecutionState.IDLE
        self._lock = threading.RLock()

    @property
    def state(self) -> ExecutionState:
        """Return current state. Args: none. Returns: ExecutionState. Raises: none."""
        with self._lock:
            return self._state

    def can_accept_signal(self) -> bool:
        """Check whether a new signal can be accepted."""
        with self._lock:
            return self._state == ExecutionState.IDLE

    def transition(self, new_state: ExecutionState) -> bool:
        """Apply validated transition. Args: new_state. Returns: bool. Raises: none."""
        with self._lock:
            allowed = self._VALID_TRANSITIONS.get(self._state, set())
            if new_state not in allowed:
                return False
            self._state = new_state
            return True

    def force_idle(self) -> None:
        """Force reset to IDLE state. Args: none. Returns: none. Raises: none."""
        with self._lock:
            self._state = ExecutionState.IDLE
