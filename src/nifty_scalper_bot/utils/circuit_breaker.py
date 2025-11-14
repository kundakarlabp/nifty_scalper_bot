"""Utility circuit breaker for broker integrations."""

from __future__ import annotations

import time
from typing import Callable


class CircuitBreaker:
    """Track broker failures and gate outbound requests."""

    def __init__(
        self,
        *,
        fail_threshold: int = 5,
        window_s: float = 30.0,
        cooldown_s: float = 15.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if fail_threshold <= 0:
            raise ValueError("fail_threshold must be positive")
        if window_s <= 0:
            raise ValueError("window_s must be positive")
        if cooldown_s <= 0:
            raise ValueError("cooldown_s must be positive")
        self._fail_threshold = int(fail_threshold)
        self._window_s = float(window_s)
        self._cooldown_s = float(cooldown_s)
        self._clock = clock or time.time
        self._failure_count = 0
        self._window_anchor = self._clock()
        self._open_until = 0.0

    def allow(self) -> bool:
        """Return ``True`` if the circuit permits a call."""

        now = self._clock()
        if self._open_until and now < self._open_until:
            return False
        if now - self._window_anchor >= self._window_s:
            self._failure_count = 0
            self._window_anchor = now
        return True

    def on_success(self) -> None:
        """Reset failure state after a successful call."""

        self._failure_count = 0
        self._open_until = 0.0
        self._window_anchor = self._clock()

    def on_failure(self) -> None:
        """Record a broker failure and open the circuit if needed."""

        now = self._clock()
        if now - self._window_anchor >= self._window_s:
            self._failure_count = 0
            self._window_anchor = now
        self._failure_count += 1
        if self._failure_count >= self._fail_threshold:
            self._open_until = now + self._cooldown_s
            self._failure_count = 0
            self._window_anchor = now

    @property
    def is_open(self) -> bool:
        """Return ``True`` if the circuit is currently open."""

        return bool(self._open_until and self._clock() < self._open_until)


__all__ = ["CircuitBreaker"]
