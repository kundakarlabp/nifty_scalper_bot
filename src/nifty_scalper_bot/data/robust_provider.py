"""World-class data provider with validation, retry, and circuit breaking."""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, TypeVar
from dataclasses import dataclass

try:
    from tenacity import (
        retry,
        stop_after_attempt,
        wait_exponential,
        retry_if_exception_type,
        before_sleep_log,
    )
except Exception:  # dependency-free fallback for clean or partially-initialised envs
    import sys
    import time

    for _name in tuple(sys.modules):
        if _name == "tenacity" or _name.startswith("tenacity."):
            sys.modules.pop(_name, None)

    def stop_after_attempt(attempts: int) -> int:
        return int(attempts)

    def wait_exponential(*, multiplier: float = 1.0, min: float = 1.0, max: float = 10.0):
        return (float(multiplier), float(min), float(max))

    def retry_if_exception_type(exc_types):
        return exc_types

    def before_sleep_log(logger, level):
        def _before_sleep(retry_state) -> None:
            try:
                logger.log(
                    level,
                    "ROBUST_PROVIDER_RETRY_FALLBACK attempt=%s",
                    getattr(retry_state, "attempt_number", None),
                )
            except Exception:
                pass

        return _before_sleep

    def retry(*, stop=3, wait=(1.0, 1.0, 10.0), retry=Exception, before_sleep=None):
        attempts = int(stop if isinstance(stop, int) else 3)
        exc_types = retry if isinstance(retry, tuple) else (retry,)

        def decorator(fn):
            if asyncio.iscoroutinefunction(fn):
                async def _wrapped(*args, **kwargs):
                    mult, min_wait, max_wait = wait
                    for attempt in range(1, attempts + 1):
                        try:
                            return await fn(*args, **kwargs)
                        except exc_types as exc:
                            if attempt >= attempts:
                                raise
                            if before_sleep:
                                try:
                                    state = type(
                                        "RetryState",
                                        (),
                                        {"attempt_number": attempt, "fn": fn, "exception": exc},
                                    )()
                                    before_sleep(state)
                                except Exception:
                                    pass
                            delay = min(max_wait, max(min_wait, mult * (2 ** (attempt - 1))))
                            await asyncio.sleep(delay)

                return _wrapped

            def _wrapped(*args, **kwargs):
                mult, min_wait, max_wait = wait
                for attempt in range(1, attempts + 1):
                    try:
                        return fn(*args, **kwargs)
                    except exc_types as exc:
                        if attempt >= attempts:
                            raise
                        if before_sleep:
                            try:
                                state = type(
                                    "RetryState",
                                    (),
                                    {"attempt_number": attempt, "fn": fn, "exception": exc},
                                )()
                                before_sleep(state)
                            except Exception:
                                pass
                        delay = min(max_wait, max(min_wait, mult * (2 ** (attempt - 1))))
                        time.sleep(delay)

            return _wrapped

        return decorator

LOGGER = logging.getLogger(__name__)
T = TypeVar('T')


class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5          # Consecutive failures to open
    success_threshold: int = 2          # Successes to close from half-open
    timeout_seconds: float = 60.0       # Time before attempting recovery
    

class DataFetchError(Exception):
    """Raised when broker returns unexpected structure."""
    pass


class CircuitBreaker:
    """Circuit breaker pattern for API protection."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: datetime | None = None
        
    def record_success(self) -> None:
        """Record successful API call."""
        self.failure_count = 0
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.success_count = 0
                LOGGER.info("Circuit breaker CLOSED - service recovered")
                
    def record_failure(self) -> None:
        """Record failed API call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        self.success_count = 0
        
        if self.failure_count >= self.config.failure_threshold:
            if self.state != CircuitState.OPEN:
                self.state = CircuitState.OPEN
                LOGGER.error(
                    f"Circuit breaker OPEN after {self.failure_count} failures"
                )
                
    def allow_request(self) -> bool:
        """Check if request should be allowed."""
        if self.state == CircuitState.CLOSED:
            return True
            
        if self.state == CircuitState.OPEN:
            if self.last_failure_time:
                elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                if elapsed >= self.config.timeout_seconds:
                    self.state = CircuitState.HALF_OPEN
                    LOGGER.info("Circuit breaker HALF-OPEN - testing recovery")
                    return True
            return False
            
        # HALF_OPEN state
        return True


class RobustDataProvider:
    """
    Universal Proxy Wrapper.
    Intercepts specific calls for robustness (get_positions), 
    but auto-delegates EVERYTHING else to the underlying broker.
    """
    
    def __init__(
        self,
        broker_client: Any,
        circuit_config: Any | None = None,
        notifier: Callable[[str, dict], None] | None = None
    ):
        # We store the real client as 'client' AND '_broker' to be safe
        self.client = broker_client
        self._broker = broker_client 
        if isinstance(circuit_config, CircuitBreaker):
            self.circuit = circuit_config
        else:
            self.circuit = CircuitBreaker(circuit_config or CircuitBreakerConfig())
        self.notifier = notifier
        self._logger = logging.getLogger(__name__)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.client, name)

    def _notify_failure(self, event: str, payload: dict[str, Any]) -> None:
        """Emit provider failure notification safely. Args: event/payload. Returns: None. Raises: none."""
        if self.notifier is None:
            return
        try:
            result = self.notifier(event, payload)
            if asyncio.iscoroutine(result):
                asyncio.create_task(result)
        except Exception:
            LOGGER.exception("DATA_PROVIDER_NOTIFY_FAILED event=%s", event)

    
    def get_profile(self) -> dict:
        """
        World-class FIX: Proxies non-critical status calls (like get_profile) 
        synchronously to the underlying broker client for startup checks.
        """
        profile_fn = getattr(self._broker, 'get_profile', None) 
        if callable(profile_fn):
            try:
                # Assuming the underlying call is synchronous, use to_thread to be safe 
                # if this code is ever called from an async context outside startup.
                # However, for the simple startup probe, we rely on the client being synchronous.
                return profile_fn()
            except Exception as e:
                self._logger.warning("BROKER_PROFILE_FETCH_FAILED: %s", e)
                raise
        raise AttributeError("Underlying broker client lacks get_profile")

    
    def get_margins(self) -> dict:
        """Proxy broker margins with startup diagnostics. Args: none. Returns: dict. Raises: passthrough."""
        margins_fn = getattr(self._broker, 'get_margins', None)
        if callable(margins_fn):
            try:
                return margins_fn()
            except Exception as e:
                self._logger.warning("BROKER_MARGINS_FETCH_FAILED: %s", e)
                raise
        raise AttributeError("Underlying broker client lacks get_margins")

    def get_positions(self) -> list[dict]:
        """Fetch broker positions with circuit breaker + retry.

        Returns:
            list[dict]: Normalised broker positions.

        Raises:
            DataFetchError: When broker response shape is unexpected.
            Exception: Broker exceptions after retry exhaustion.
        """
        if not self.circuit.allow_request():
            raise DataFetchError("Circuit breaker OPEN - broker service unavailable")
        try:
            return self._fetch_positions_with_retry()
        except Exception as exc:
            self.circuit.record_failure()
            self._notify_failure(
                "BROKER_POSITION_FETCH_FAILED",
                {"error": str(exc), "error_type": type(exc).__name__},
            )
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((Exception,)),
        before_sleep=before_sleep_log(LOGGER, logging.WARNING)
    )
    def _fetch_positions_with_retry(self) -> list[dict]:
        raw = self.client.get_positions()
        positions = raw.get('net', raw) if isinstance(raw, dict) else raw
        if positions is None:
            return []
        if not isinstance(positions, list):
            raise DataFetchError(f"Invalid positions response type: {type(positions)}")
        self.circuit.record_success()
        return positions

    async def get_positions_async(self) -> list[dict]:
        return await asyncio.to_thread(self.get_positions)

    async def safe_quote_any(self, symbols: list[str]) -> dict[str, Any]:
        """Fetch quotes using broker quote_any when present.

        Args:
            symbols: Broker symbols to fetch.

        Returns:
            Mapping of symbol to quote payload. Empty mapping on failure.
        """
        quote_any = getattr(self.client, 'quote_any', None)
        if not callable(quote_any):
            return {}
        try:
            result = quote_any(symbols)
            if asyncio.iscoroutine(result):
                result = await result
            return result if isinstance(result, dict) else {}
        except Exception as exc:  # noqa: BLE001
            self._notify_failure(
                "BROKER_QUOTE_FETCH_FAILED",
                {"error": str(exc), "error_type": type(exc).__name__, "symbols": list(symbols)},
            )
            return {}
