"""Execution circuit breaker for temporary strategy pauses after repeated failures."""

from __future__ import annotations

import threading
import time


class ExecutionCircuitBreaker:
    """Track consecutive order failures and open for a cooldown window."""

    def __init__(
        self,
        *,
        max_failures: int = 3,
        cooldown_seconds: float = 60.0,
    ) -> None:
        self._max_failures = max(1, int(max_failures))
        self._cooldown_seconds = max(1.0, float(cooldown_seconds))
        self._lock = threading.RLock()
        self._consecutive_failures = 0
        self._opened_until = 0.0

    def is_open(self) -> bool:
        """Return True when breaker is active and strategy submissions must pause."""
        with self._lock:
            return time.time() < self._opened_until

    def remaining_seconds(self) -> float:
        """Return remaining open duration in seconds."""
        with self._lock:
            return max(0.0, self._opened_until - time.time())

    def record_success(self) -> None:
        """Reset failure counters on a successful order submission."""
        with self._lock:
            self._consecutive_failures = 0

    def record_failure(self) -> bool:
        """Record one failure and return True when breaker transitions to open."""
        with self._lock:
            self._consecutive_failures += 1
            if self._consecutive_failures < self._max_failures:
                return False
            self._opened_until = time.time() + self._cooldown_seconds
            self._consecutive_failures = 0
            return True
