"""Broker REST client wrappers with rate limiting and retries."""

from __future__ import annotations

import time
from typing import Any, Dict, Protocol, cast, runtime_checkable

from nifty_scalper_bot.utils.errors import BrokerError
from nifty_scalper_bot.utils.logging import get_logger
from nifty_scalper_bot.utils.rate_limiter import RateLimiter

LOGGER = get_logger(__name__)


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
        for attempt in range(1, tries + 1):
            self._limiter.acquire(bucket, timeout=timeout)
            try:
                return fn(*args, **kwargs)
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

    def place_order(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        result = self._retry(self._inner.place_order, "orders", payload)
        return cast(Dict[str, Any], result)


__all__ = ["BaseBrokerClient", "DummyBrokerClient", "ThrottledBrokerClient"]
