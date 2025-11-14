"""Condition based leaky bucket rate limiter."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Condition, Lock
from typing import Dict

from nifty_scalper_bot.utils.errors import RateLimitError


@dataclass
class LeakyBucket:
    """Thread-safe leaky bucket implementation."""

    capacity: int
    refill_rate_per_sec: float
    tokens: float = field(init=False)
    last_refill_ts: float = field(default_factory=time.monotonic)
    lock: Lock = field(default_factory=Lock)
    condition: Condition = field(init=False)

    def __post_init__(self) -> None:
        self.tokens = float(self.capacity)
        self.condition = Condition(self.lock)

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_refill_ts
        if elapsed <= 0:
            return
        self.tokens = min(
            self.capacity, self.tokens + elapsed * self.refill_rate_per_sec
        )
        self.last_refill_ts = now
        if self.tokens >= 1.0:
            self.condition.notify_all()

    def acquire(self, timeout: float = 2.0) -> None:
        """Acquire a token, blocking until available or timeout."""

        end_time = time.monotonic() + timeout
        with self.condition:
            while True:
                self._refill()
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                remaining = end_time - time.monotonic()
                if remaining <= 0:
                    raise RateLimitError("Rate limit exceeded")
                wait_time = min(1.0 / self.refill_rate_per_sec, remaining)
                self.condition.wait(wait_time)

    def snapshot(self) -> dict[str, float]:
        """Return a thread-safe snapshot of bucket state."""

        with self.lock:
            self._refill()
            return {
                "capacity": float(self.capacity),
                "tokens": float(self.tokens),
                "refill_rate_per_sec": float(self.refill_rate_per_sec),
            }


class RateLimiter:
    """Manage multiple named leaky buckets."""

    def __init__(self) -> None:
        self._buckets: Dict[str, LeakyBucket] = {}
        self._lock = Lock()

    def configure_bucket(
        self, name: str, capacity: int, refill_rate_per_sec: float
    ) -> None:
        if capacity <= 0 or refill_rate_per_sec <= 0:
            raise RateLimitError("Capacity and refill rate must be positive")
        bucket = LeakyBucket(capacity=capacity, refill_rate_per_sec=refill_rate_per_sec)
        with self._lock:
            self._buckets[name] = bucket

    def acquire(self, name: str, timeout: float = 2.0) -> None:
        bucket = self._buckets.get(name)
        if bucket is None:
            raise RateLimitError(f"Rate limit bucket not configured: {name}")
        bucket.acquire(timeout=timeout)

    def snapshot(self) -> dict[str, dict[str, float]]:
        """Return shallow copy of all bucket states for telemetry."""

        with self._lock:
            items = list(self._buckets.items())
        return {name: bucket.snapshot() for name, bucket in items}


__all__ = ["LeakyBucket", "RateLimiter"]
