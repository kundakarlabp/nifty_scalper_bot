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
        # Per-state metadata for stale-state reconciliation and observability.
        self._entered_at: datetime = datetime.now(timezone.utc)
        self._order_id: str | None = None
        self._reason: str | None = None
        self._trace_id: str | None = None

    @property
    def state(self) -> ExecutionState:
        """Return current state. Args: none. Returns: ExecutionState. Raises: none."""
        with self._lock:
            return self._state

    def can_accept_signal(self) -> bool:
        """Check whether a new signal can be accepted."""
        with self._lock:
            return self._state in (ExecutionState.IDLE, ExecutionState.READY)

    def transition(
        self,
        new_state: ExecutionState,
        *,
        order_id: str | None = None,
        reason: str | None = None,
        trace_id: str | None = None,
    ) -> bool:
        """Apply validated transition. Args: new_state and optional order_id/
        reason/trace_id metadata. Returns: bool. Raises: none."""
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
            self._entered_at = datetime.now(timezone.utc)
            if order_id is not None:
                self._order_id = order_id
            if reason is not None:
                self._reason = reason
            if trace_id is not None:
                self._trace_id = trace_id
            self._last_reject = None
            return True

    def force_idle(self, reason: str | None = None) -> None:
        """Force reset to IDLE state. Args: optional reason. Returns: none. Raises: none."""
        with self._lock:
            self._state = ExecutionState.IDLE
            self._last_transition_ts = datetime.now(timezone.utc).isoformat()
            self._entered_at = datetime.now(timezone.utc)
            self._order_id = None
            self._reason = reason
            self._trace_id = None

    def state_age_seconds(self) -> float:
        """Return seconds since the current state was entered. Args: none.
        Returns: float seconds. Raises: none."""
        with self._lock:
            return max(0.0, (datetime.now(timezone.utc) - self._entered_at).total_seconds())

    @property
    def order_id(self) -> str | None:
        """Return the order id associated with the current state, if any."""
        with self._lock:
            return self._order_id

    def set_order_id(self, order_id: str) -> None:
        """Attach a broker order id to the current state. Args: order_id.
        Returns: none. Raises: none."""
        with self._lock:
            self._order_id = str(order_id)

    def current_state_details(self) -> dict[str, Any]:
        """Return state diagnostic details for transition logging."""
        with self._lock:
            allowed = self._VALID_TRANSITIONS.get(self._state, set())
            return {
                "state": self._state.value,
                "allowed_next": sorted(state.value for state in allowed),
                "last_transition_ts": self._last_transition_ts,
                "last_reject": dict(self._last_reject) if self._last_reject else None,
                "entered_at": self._entered_at.isoformat(),
                "state_age_seconds": max(0.0, (datetime.now(timezone.utc) - self._entered_at).total_seconds()),
                "order_id": self._order_id,
                "reason": self._reason,
                "trace_id": self._trace_id,
            }
