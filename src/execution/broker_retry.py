"""Lightweight retry helper for broker API calls."""

from __future__ import annotations

import time
from typing import Callable, TypeVar, ParamSpec

P = ParamSpec("P")
T = TypeVar("T")


def call(fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    """Invoke ``fn`` with simple classified retries."""
    for i in range(3):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # pragma: no cover - broad to classify
            m = str(e).lower()
            if any(key in m for key in ("429", "rate", "throttle")):
                time.sleep(0.5 * (2**i))
                continue
            if any(key in m for key in ("timeout", "temporar", "connection")):
                time.sleep(0.5 * (2**i))
                continue
            if "session" in m and "expired" in m:
                raise
            if i == 2:
                raise
            time.sleep(0.5 * (2**i))
    # Should never reach here but raise for safety
    raise RuntimeError("broker_retry.call exhausted retries")


__all__ = ["call"]
