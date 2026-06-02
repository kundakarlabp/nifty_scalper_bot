"""Exponential backoff generator utilities.

Provides ``exp_backoff`` — a simple generator that yields increasing wait
intervals with jitter.  This is a thin compatibility shim so that
``data.polling.manager`` can import ``exp_backoff`` from a dedicated module
while the heavier ``retry_with_backoff`` machinery lives in ``utils.retry``.
"""
from __future__ import annotations

import random
from typing import Generator


def exp_backoff(
    base_ms: float = 500,
    max_ms: float = 30_000,
    jitter_pct: float = 0.25,
) -> Generator[float, None, None]:
    """Yield exponentially increasing wait times (in seconds) with jitter.

    Args:
        base_ms:    Initial backoff in milliseconds.
        max_ms:     Cap on backoff in milliseconds.
        jitter_pct: Fraction of current delay to randomise (0–1).

    Yields:
        float: Seconds to sleep before the next retry attempt.
    """
    delay_ms = float(base_ms)
    while True:
        jitter = delay_ms * jitter_pct * (2 * random.random() - 1)
        yield max(0.0, (delay_ms + jitter) / 1000.0)
        delay_ms = min(delay_ms * 2, max_ms)


__all__ = ["exp_backoff"]
