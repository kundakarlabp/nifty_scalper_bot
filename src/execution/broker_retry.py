"""Lightweight retry helper for broker API calls."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import ParamSpec, TypeVar

from src.utils.circuit_breaker import CircuitBreaker

P = ParamSpec("P")
T = TypeVar("T")


CB = CircuitBreaker("broker_retry")


def call(fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    """Invoke ``fn`` with classified retries and circuit breaker.

    Raises the last encountered exception if all retries are exhausted or the
    circuit breaker is open.
    """
    last_exc: Exception | None = None
    for i in range(3):
        if not CB.allow():
            raise RuntimeError("breaker_open")
        t0 = time.monotonic()
        try:
            result = fn(*args, **kwargs)
            CB.record_success(int((time.monotonic() - t0) * 1000))
            return result
        except Exception as e:  # pragma: no cover - broad to classify
            last_exc = e
            CB.record_failure(int((time.monotonic() - t0) * 1000), reason=str(e))
            m = str(e).lower()
            if any(
                key in m
                for key in (
                    "429",
                    "rate",
                    "throttle",
                    "timeout",
                    "temporar",
                    "connection",
                )
            ):
                sleep = 0.5 * (2**i) + random.uniform(0, 0.05)
                time.sleep(sleep)
                continue
            if "session" in m and "expired" in m:
                raise
            if i == 2:
                break
            time.sleep(0.5 * (2**i) + random.uniform(0, 0.05))
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("broker_retry.call exhausted retries")


__all__ = ["call"]
