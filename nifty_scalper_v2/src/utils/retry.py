"""
Async retry decorator with exponential back-off for NiftyScalperBot v2.

Usage::

    @retry(max_retries=3, base_delay=1.0, exceptions=(aiohttp.ClientError,))
    async def fetch_quote(symbol: str) -> dict:
        ...

    # one-off:
    result = await retry_call(fetch_quote, "NIFTY", max_retries=5)
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
from collections.abc import Callable, Coroutine
from typing import Any, ParamSpec, TypeVar

log = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")

_DEFAULT_EXCEPTIONS: tuple[type[BaseException], ...] = (Exception,)


def retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    exceptions: tuple[type[BaseException], ...] = _DEFAULT_EXCEPTIONS,
    on_retry: Callable[[int, BaseException], None] | None = None,
) -> Callable[[Callable[P, Coroutine[Any, Any, T]]], Callable[P, Coroutine[Any, Any, T]]]:
    """
    Decorator that retries an ``async`` function on failure.

    Args:
        max_retries:    Maximum number of *additional* attempts after the first.
                        Total calls = max_retries + 1.
        base_delay:     Initial back-off delay in seconds.
        max_delay:      Cap on back-off delay in seconds.
        backoff_factor: Multiplier applied after each failure (exponential growth).
        jitter:         Add up to ±25 % random jitter to avoid thundering-herd.
        exceptions:     Tuple of exception types that trigger a retry.
                        Non-matching exceptions propagate immediately.
        on_retry:       Optional callback ``(attempt_number, exc) -> None``
                        called before the sleep.  Useful for metrics/logging.

    Returns:
        Decorated coroutine function with identical signature.

    Example::

        @retry(max_retries=5, base_delay=0.5, exceptions=(TimeoutError, IOError))
        async def call_broker_api() -> dict:
            return await session.get("/orders")
    """

    def decorator(
        func: Callable[P, Coroutine[Any, Any, T]]
    ) -> Callable[P, Coroutine[Any, Any, T]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exc: BaseException | None = None
            delay = base_delay

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:  # type: ignore[misc]
                    last_exc = exc
                    if attempt >= max_retries:
                        break

                    sleep_time = min(delay, max_delay)
                    if jitter:
                        sleep_time *= 1.0 + random.uniform(-0.25, 0.25)
                        sleep_time = max(0.0, sleep_time)

                    if on_retry is not None:
                        try:
                            on_retry(attempt + 1, exc)
                        except Exception:  # noqa: BLE001
                            pass

                    log.warning(
                        "retry attempt %d/%d for %s after %.2fs — %s: %s",
                        attempt + 1,
                        max_retries,
                        func.__qualname__,
                        sleep_time,
                        type(exc).__name__,
                        exc,
                    )
                    await asyncio.sleep(sleep_time)
                    delay = min(delay * backoff_factor, max_delay)

            assert last_exc is not None
            raise last_exc

        return wrapper

    return decorator


async def retry_call(
    func: Callable[..., Coroutine[Any, Any, T]],
    *args: Any,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    exceptions: tuple[type[BaseException], ...] = _DEFAULT_EXCEPTIONS,
    on_retry: Callable[[int, BaseException], None] | None = None,
    **kwargs: Any,
) -> T:
    """
    Functional variant of :func:`retry` for one-off calls.

    Example::

        result = await retry_call(
            client.get_quote,
            "NIFTY",
            max_retries=5,
            exceptions=(TimeoutError,),
        )
    """
    wrapped = retry(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        backoff_factor=backoff_factor,
        jitter=jitter,
        exceptions=exceptions,
        on_retry=on_retry,
    )(func)
    return await wrapped(*args, **kwargs)
