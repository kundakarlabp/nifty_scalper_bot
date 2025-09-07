from __future__ import annotations

import random
import threading
import time
from typing import Callable, Iterable, Optional, Type, TypeVar

T = TypeVar("T")


class RateLimiter:
    """Simple token bucket rate limiter."""

    def __init__(self, max_per_min: int = 30) -> None:
        self.max = int(max_per_min)
        self.bucket = float(self.max)
        self.ts = time.time()
        self.lock = threading.Lock()

    def allow(self) -> bool:
        """Return True if an action is permitted, else False."""
        with self.lock:
            now = time.time()
            refill = (now - self.ts) * (self.max / 60.0)
            if refill > 0:
                self.bucket = min(self.max, self.bucket + refill)
                self.ts = now
            if self.bucket >= 1.0:
                self.bucket -= 1.0
                return True
            return False


class CircuitBreaker:
    """Fail-fast circuit breaker."""

    def __init__(self, fail_threshold: int = 5, cooldown_s: float = 30.0) -> None:
        self.fail_th = max(1, int(fail_threshold))
        self.cool = float(cooldown_s)
        self.fail = 0
        self.open_until = 0.0
        self.lock = threading.Lock()

    def call(self, fn: Callable[..., T], *a: object, **k: object) -> T:
        """Run ``fn`` unless the circuit is open."""
        now = time.time()
        with self.lock:
            if now < self.open_until:
                raise RuntimeError("circuit_open")
        try:
            result = fn(*a, **k)
            with self.lock:
                self.fail = 0
            return result
        except Exception:
            with self.lock:
                self.fail += 1
                if self.fail >= self.fail_th:
                    self.open_until = time.time() + self.cool
            raise


def retry(
    attempts: int = 3,
    base: float = 0.25,
    max_sleep: float = 2.0,
    only: Optional[Iterable[Type[BaseException]]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Retry decorator with exponential backoff."""
    only_tuple = tuple(only) if only else (Exception,)

    def deco(fn: Callable[..., T]) -> Callable[..., T]:
        def wrap(*a: object, **k: object) -> T:
            for i in range(attempts):
                try:
                    return fn(*a, **k)
                except only_tuple:
                    if i == attempts - 1:
                        raise
                    sleep_s = min(max_sleep, base * (2**i) + random.random() * 0.05)
                    time.sleep(sleep_s)
            raise RuntimeError("unreachable")

        return wrap

    return deco


__all__ = ["RateLimiter", "CircuitBreaker", "retry"]
