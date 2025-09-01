"""Lightweight retry helper for broker API calls."""

from __future__ import annotations

import time
from typing import Callable, TypeVar, ParamSpec, Optional

P = ParamSpec("P")
T = TypeVar("T")


def call(fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    """Invoke ``fn`` with simple classified retries.

    Raises the last encountered exception if all retries are exhausted.
    """
    last_exc: Optional[Exception] = None
    for i in range(3):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # pragma: no cover - broad to classify
            last_exc = e
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
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("broker_retry.call exhausted retries")


__all__ = ["call"]
