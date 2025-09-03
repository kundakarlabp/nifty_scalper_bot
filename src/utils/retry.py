# src/utils/retry.py
from __future__ import annotations

import time
import logging
import random
import asyncio
from functools import wraps
from typing import Callable, Optional, Tuple, TypeVar, ParamSpec

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


def _compute_sleep(
    base_delay: float,
    backoff: float,
    attempt_idx: int,
    max_delay: Optional[float],
    jitter: Optional[float | Tuple[float, float] | Callable[[], float]],
) -> float:
    """Calculate delay for a retry attempt with optional jitter.

    The delay grows exponentially with ``backoff`` while respecting
    ``max_delay`` and optionally adding ``jitter`` noise.
    """
    # exponential backoff: delay * (backoff ** (attempt_idx-1))
    delay = base_delay * (backoff ** max(0, attempt_idx - 1))
    if max_delay is not None:
        delay = min(delay, max_delay)

    # jitter options:
    if jitter is None:
        return delay
    if callable(jitter):
        return max(0.0, delay + float(jitter()))
    if isinstance(jitter, tuple):
        lo, hi = jitter
        lo_f, hi_f = float(lo), float(hi)
        if hi_f < lo_f:
            lo_f, hi_f = hi_f, lo_f
        return max(0.0, delay + random.uniform(lo_f, hi_f))
    # numeric => uniform [0, jitter]
    return max(0.0, delay + random.uniform(0.0, float(jitter)))


def retry(
    *,
    tries: int = 3,
    delay: float = 2.0,
    backoff: float = 2.0,
    max_delay: Optional[float] = None,
    exceptions: Tuple[type[BaseException], ...] = (Exception,),
    exclude_exceptions: Tuple[type[BaseException], ...] = (),
    jitter: Optional[float | Tuple[float, float] | Callable[[], float]] = None,
    on_retry: Optional[Callable[[int, BaseException, float], None]] = None,
    reraise: bool = True,
    log: Optional[logging.Logger] = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Retry decorator with exponential backoff (sync & async).

    Args:
      tries: total attempts (>=1)
      delay: base delay (seconds) before first retry
      backoff: exponential factor (>0)
      max_delay: cap for sleep (seconds)
      exceptions: exception types that trigger retry
      exclude_exceptions: exception types that should NOT be retried
      jitter: None | float(max extra) | (low, high) | callable()->float
      on_retry: callback(attempt_index, exception, sleep_seconds)
      reraise: if True, re-raise last exception after final attempt
      log: custom logger (defaults to module logger)

    Usage (backward-compatible):
      @retry(tries=3, delay=1, backoff=2)
      def fn(...): ...
    """
    if tries < 1:
        raise ValueError("tries must be >= 1")
    if delay < 0:
        raise ValueError("delay must be >= 0")
    if backoff <= 0:
        raise ValueError("backoff must be > 0")
    if max_delay is not None and max_delay < 0:
        raise ValueError("max_delay must be >= 0")

    lg = log or logger
    tries = int(tries)
    delay = float(delay)
    backoff = float(backoff)

    def deco_retry(func: Callable[P, T]) -> Callable[P, T]:
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:  # type: ignore[misc]
                last_exc: Optional[BaseException] = None
                for attempt in range(1, tries + 1):
                    try:
                        return await func(*args, **kwargs)
                    except exclude_exceptions:
                        # explicitly do not retry
                        raise
                    except exceptions as e:
                        last_exc = e
                        if attempt >= tries:
                            if reraise:
                                raise
                            lg.error("Retry: giving up after %d attempts on %s: %s", attempt, func.__name__, e)
                            return None  # type: ignore[return-value]
                        sleep_s = _compute_sleep(delay, backoff, attempt, max_delay, jitter)
                        lg.warning(
                            "Retry %s attempt %d/%d failed: %s — sleeping %.2fs",
                            func.__name__, attempt, tries, e, sleep_s,
                        )
                        if on_retry:
                            try:
                                on_retry(attempt, e, sleep_s)
                            except Exception:
                                pass
                        await asyncio.sleep(sleep_s)
                # should not reach
                if reraise and last_exc:
                    raise last_exc
                return None  # type: ignore[return-value]

            return async_wrapper  # type: ignore[return-value]

        @wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:  # type: ignore[misc]
            last_exc: Optional[BaseException] = None
            for attempt in range(1, tries + 1):
                try:
                    return func(*args, **kwargs)
                except exclude_exceptions:
                    # explicitly do not retry
                    raise
                except exceptions as e:
                    last_exc = e
                    if attempt >= tries:
                        if reraise:
                            raise
                        lg.error("Retry: giving up after %d attempts on %s: %s", attempt, func.__name__, e)
                        return None  # type: ignore[return-value]
                    sleep_s = _compute_sleep(delay, backoff, attempt, max_delay, jitter)
                    lg.warning(
                        "Retry %s attempt %d/%d failed: %s — sleeping %.2fs",
                        func.__name__, attempt, tries, e, sleep_s,
                    )
                    if on_retry:
                        try:
                            on_retry(attempt, e, sleep_s)
                        except Exception:
                            pass
                    time.sleep(sleep_s)
            # should not reach
            if reraise and last_exc:
                raise last_exc
            return None  # type: ignore[return-value]

        return sync_wrapper  # type: ignore[return-value]

    return deco_retry


__all__ = ["retry"]
