"""Broker REST client wrappers with rate limiting and retries."""

from __future__ import annotations

import time
from typing import Any, Dict, Mapping, Protocol, Sequence, cast, runtime_checkable

from nifty_scalper_bot.utils.errors import BrokerError, RateLimitError
from nifty_scalper_bot.utils.logging import get_logger
from nifty_scalper_bot.utils.rate_limiter import RateLimiter

LOGGER = get_logger(__name__)


class ExponentialBackoff:
    """Deterministic exponential backoff helper."""

    def __init__(self, initial: float = 0.5, maximum: float = 10.0) -> None:
        self._initial = max(0.0, float(initial))
        self._maximum = max(self._initial, float(maximum))
        self._current = self._initial

    def next(self) -> float:
        """Return next delay in seconds."""
        delay = self._current
        self._current = min(self._maximum, max(self._initial, self._current * 2))
        return delay


@runtime_checkable
class BaseBrokerClient(Protocol):
    """Protocol describing the required broker client interface."""

    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Return a quote dictionary containing symbol, ltp and ts_ms."""

    def place_order(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Place an order and return the broker response."""


class DummyBrokerClient:
    """Simple broker client for local testing with monotonic prices."""

    def __init__(self, start_price: float = 20_000.0, step: float = 1.0) -> None:
        self._price = start_price
        self._step = step

    def get_quote(self, symbol: str) -> Dict[str, Any]:
        self._price += self._step
        return {
            "symbol": symbol,
            "ltp": float(self._price),
            "ts_ms": int(time.time() * 1000),
        }

    def place_order(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "order_id": f"SIM-{int(time.time() * 1000)}",
            "status": "accepted",
            "payload": payload,
        }


class ThrottledBrokerClient:
    """Wrap a broker client with retry and rate limit handling."""

    def __init__(self, inner: BaseBrokerClient, limiter: RateLimiter) -> None:
        self._inner = inner
        self._limiter = limiter
        self._logger = LOGGER

    def _retry(
        self,
        fn: Any,
        bucket: str,
        *args: Any,
        tries: int = 3,
        timeout: float = 2.0,
        **kwargs: Any,
    ) -> Any:
        last_error: Exception | None = None
        backoff = ExponentialBackoff(initial=0.5, maximum=10.0)
        for attempt in range(1, tries + 1):
            self._limiter.acquire(bucket, timeout=timeout)
            try:
                return fn(*args, **kwargs)
            except RateLimitError as exc:
                last_error = exc
                if attempt >= tries:
                    break
                delay = backoff.next()
                self._logger.warning(
                    'Rate limit backoff',
                    extra={
                        'symbol': str(args[0]) if args else '',
                        'service': bucket,
                        'backoff_ms': int(delay * 1000),
                    },
                )
                time.sleep(delay)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                self._logger.warning(
                    "Broker call failed (attempt %s/%s): %s", attempt, tries, exc
                )
                if attempt >= tries:
                    break
                backoff = 0.5 * (2 ** (attempt - 1))
                time.sleep(backoff)
        self._logger.error("Broker call exhausted retries: %s", last_error)
        raise BrokerError("Broker call failed") from last_error

    def get_quote(self, symbol: str) -> Dict[str, Any]:
        result = self._retry(self._inner.get_quote, "rest", symbol)
        return cast(Dict[str, Any], result)

    def quote_any(self, items: Sequence[object]) -> Mapping[str, Any] | None:
        """Return mixed token/symbol quotes when supported by the inner client.

        Args:
            items: Sequence of Zerodha identifiers such as instrument tokens or
                ``EXCHANGE:SYMBOL`` strings.

        Returns:
            Mapping[str, Any] | None: Quote payloads keyed by identifier when the
            inner client exposes ``quote_any``; ``None`` otherwise.

        Raises:
            BrokerError: If the wrapped client fails after exhausting retries.
        """

        self._logger.debug(
            "Entered ThrottledBrokerClient.quote_any",
            extra={"event": "throttled_quote_any_enter", "count": len(items)},
        )
        inner_quote_any = getattr(self._inner, "quote_any", None)
        if inner_quote_any is None:
            self._logger.info(
                "Condition met: throttled_quote_any_unsupported",
                extra={"event": "throttled_quote_any_unsupported"},
            )
            return None
        try:
            result = self._retry(inner_quote_any, "rest", items)
        except BrokerError as exc:
            self._logger.error(
                "Failure in ThrottledBrokerClient.quote_any: %s",
                exc,
                extra={"event": "throttled_quote_any_failed"},
                exc_info=exc,
            )
            raise
        return cast(Mapping[str, Any] | None, result)

    def place_order(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        result = self._retry(self._inner.place_order, "orders", payload)
        return cast(Dict[str, Any], result)


__all__ = ["BaseBrokerClient", "DummyBrokerClient", "ThrottledBrokerClient"]
