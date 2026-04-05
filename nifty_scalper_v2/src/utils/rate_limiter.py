"""
Async token-bucket rate limiter for NiftyScalperBot v2.

Handles two separate concerns:
- Order placement rate   (e.g. 5 req/s — Zerodha hard limit)
- REST API calls         (e.g. 10 req/s)

Usage::

    limiter = TokenBucketRateLimiter(rate=5.0, burst=5)

    async with limiter:          # acquire one token
        await broker.place_order(...)

    # or explicitly:
    await limiter.acquire()
    await broker.place_order(...)

    # named multi-bucket convenience:
    limiters = build_broker_limiters()
    async with limiters.orders:
        await broker.place_order(...)
    async with limiters.rest:
        data = await broker.get_quote(...)
"""

from __future__ import annotations

import asyncio
import time
from types import TracebackType


class TokenBucketRateLimiter:
    """
    Async token-bucket rate limiter.

    Tokens are replenished continuously at ``rate`` tokens per second up to
    a maximum of ``burst`` tokens.  When the bucket is empty, :meth:`acquire`
    sleeps until enough tokens have accumulated.

    Thread-safety: designed for single-threaded ``asyncio`` event loops.
    Do **not** share instances across threads.

    Args:
        rate:  Sustained replenishment rate in tokens per second (must be > 0).
        burst: Maximum token accumulation (bucket capacity).
               Defaults to ``rate`` (no burst headroom).
    """

    def __init__(self, rate: float, burst: float | None = None) -> None:
        if rate <= 0:
            raise ValueError(f"rate must be positive, got {rate}")
        self._rate = rate
        self._burst = burst if burst is not None else rate
        if self._burst < 1.0:
            raise ValueError(f"burst must be >= 1.0, got {self._burst}")

        # Start with a full bucket
        self._tokens: float = self._burst
        self._last_refill: float = time.monotonic()
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def rate(self) -> float:
        """Configured replenishment rate (tokens / second)."""
        return self._rate

    @property
    def burst(self) -> float:
        """Bucket capacity (maximum token accumulation)."""
        return self._burst

    @property
    def available_tokens(self) -> float:
        """Current token count (approximate, without holding the lock)."""
        elapsed = time.monotonic() - self._last_refill
        return min(self._burst, self._tokens + elapsed * self._rate)

    # ------------------------------------------------------------------
    # Acquire
    # ------------------------------------------------------------------

    async def acquire(self, tokens: float = 1.0) -> None:
        """
        Acquire ``tokens`` from the bucket, sleeping if necessary.

        Args:
            tokens: Number of tokens to consume (default 1).

        Raises:
            ValueError: If ``tokens`` exceeds burst capacity or is non-positive.
        """
        if tokens <= 0:
            raise ValueError(f"tokens must be positive, got {tokens}")
        if tokens > self._burst:
            raise ValueError(
                f"Requested {tokens} tokens exceeds bucket capacity {self._burst}"
            )

        async with self._lock:
            while True:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                # Sleep until enough tokens are available
                deficit = tokens - self._tokens
                sleep_secs = deficit / self._rate
                await asyncio.sleep(sleep_secs)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "TokenBucketRateLimiter":
        await self.acquire()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        pass  # tokens already consumed on entry

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _refill(self) -> None:
        """Add tokens proportional to elapsed time since last refill."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
        self._last_refill = now

    def __repr__(self) -> str:
        return (
            f"TokenBucketRateLimiter("
            f"rate={self._rate}/s, "
            f"burst={self._burst}, "
            f"tokens={self._tokens:.2f}"
            f")"
        )


# ---------------------------------------------------------------------------
# Multi-bucket convenience
# ---------------------------------------------------------------------------


class MultiRateLimiter:
    """
    Named collection of :class:`TokenBucketRateLimiter` instances.

    Attributes are created dynamically from kwargs so callers get IDE
    completion when using the standard ``build_broker_limiters()`` factory.

    Usage::

        lims = MultiRateLimiter(
            orders=TokenBucketRateLimiter(rate=5.0),
            rest=TokenBucketRateLimiter(rate=10.0),
        )
        async with lims.orders:
            await broker.place_order(...)
        async with lims.rest:
            data = await broker.get_quote(...)
    """

    def __init__(self, **limiters: TokenBucketRateLimiter) -> None:
        self._limiters: dict[str, TokenBucketRateLimiter] = limiters
        for name, lim in limiters.items():
            setattr(self, name, lim)

    def __getitem__(self, name: str) -> TokenBucketRateLimiter:
        return self._limiters[name]

    def __repr__(self) -> str:
        inner = ", ".join(f"{k}={v!r}" for k, v in self._limiters.items())
        return f"MultiRateLimiter({inner})"


def build_broker_limiters(
    orders_per_sec: float = 5.0,
    rest_per_sec: float = 10.0,
) -> MultiRateLimiter:
    """
    Factory that creates the standard two-bucket limiter used by broker clients.

    Args:
        orders_per_sec: Order-placement rate limit (Zerodha allows ~5/s).
        rest_per_sec:   General REST API rate limit.

    Returns:
        :class:`MultiRateLimiter` with ``.orders`` and ``.rest`` attributes.
    """
    return MultiRateLimiter(
        orders=TokenBucketRateLimiter(rate=orders_per_sec),
        rest=TokenBucketRateLimiter(rate=rest_per_sec),
    )
