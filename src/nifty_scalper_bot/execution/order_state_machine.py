"""Execution state machine to prevent duplicate and invalid order flows."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ExecutionState(str, Enum):
    """Supported execution lifecycle states."""

    IDLE = "IDLE"
    READY = "READY"
    SIGNAL_RECEIVED = "SIGNAL_RECEIVED"
    ORDER_PENDING = "ORDER_PENDING"
    POSITION_OPEN = "POSITION_OPEN"
    EXIT_PENDING = "EXIT_PENDING"


class OrderStateMachine:
    """Track and validate order lifecycle transitions per symbol."""

    _VALID_TRANSITIONS: dict[ExecutionState, set[ExecutionState]] = {
        ExecutionState.IDLE: {ExecutionState.READY, ExecutionState.SIGNAL_RECEIVED},
        ExecutionState.READY: {ExecutionState.SIGNAL_RECEIVED, ExecutionState.IDLE},
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
        self._last_transition_ts: str | None = None
        self._last_reject: dict[str, Any] | None = None

    @property
    def state(self) -> ExecutionState:
        """Return current state. Args: none. Returns: ExecutionState. Raises: none."""
        with self._lock:
            return self._state

    def can_accept_signal(self) -> bool:
        """Check whether a new signal can be accepted."""
        with self._lock:
            return self._state in (ExecutionState.IDLE, ExecutionState.READY)

    def transition(self, new_state: ExecutionState) -> bool:
        """Apply validated transition. Args: new_state. Returns: bool. Raises: none."""
        with self._lock:
            allowed = self._VALID_TRANSITIONS.get(self._state, set())
            if new_state not in allowed:
                self._last_reject = {
                    "from": self._state.value,
                    "to": new_state.value,
                    "allowed_next": sorted(state.value for state in allowed),
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
                return False
            self._state = new_state
            self._last_transition_ts = datetime.now(timezone.utc).isoformat()
            self._last_reject = None
            return True

    def force_idle(self) -> None:
        """Force reset to IDLE state. Args: none. Returns: none. Raises: none."""
        with self._lock:
            self._state = ExecutionState.IDLE
            self._last_transition_ts = datetime.now(timezone.utc).isoformat()

    def current_state_details(self) -> dict[str, Any]:
        """Return state diagnostic details for transition logging."""
        with self._lock:
            allowed = self._VALID_TRANSITIONS.get(self._state, set())
            return {
                "state": self._state.value,
                "allowed_next": sorted(state.value for state in allowed),
                "last_transition_ts": self._last_transition_ts,
                "last_reject": dict(self._last_reject) if self._last_reject else None,
            }
