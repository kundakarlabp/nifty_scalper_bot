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
except ImportError:  # dependency-free fallback for clean envs
    import asyncio
    import time

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
                # Since the current environment is synchronous during init:
                return profile_fn()
            except Exception as exc:
                LOGGER.warning("Profile fetch failed via proxy: %s", exc)
        
        # Fallback if method or client is missing/failing
        return {"user_id": "unavailable", "user_name": "unavailable", "broker": "unknown"}    
    def _validate_response(
        self,
        response: Any,
        expected_key: str = "result"
    ) -> dict[str, Any]:
        """
        Validate broker response structure.
        
        Args:
            response: Raw broker response
            expected_key: Key that must exist (default: "result")
            
        Returns:
            Validated response dict
            
        Raises:
            DataFetchError: If response structure invalid
        """
        # Handle None response
        if response is None:
            raise DataFetchError("Broker returned None response")
            
        # Handle error responses
        if isinstance(response, dict):
            # Check for explicit error
            if "error" in response or "status" in response and response["status"] == "error":
                error_msg = response.get("error", "Unknown error")
                error_code = response.get("code", "NO_CODE")
                LOGGER.error(
                    f"Broker API error: {error_msg} (code: {error_code})"
                )
                raise DataFetchError(f"Broker error: {error_msg}")
                
            # Validate expected key exists
            if expected_key not in response:
                LOGGER.error(
                    f"Response missing '{expected_key}' key. "
                    f"Full response: {response}"
                )
                raise DataFetchError(
                    f"Response missing expected key: {expected_key}"
                )
                
        return response
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((DataFetchError, ConnectionError)),
        before_sleep=before_sleep_log(LOGGER, logging.WARNING)
    )
    async def fetch_with_validation(
        self,
        fetch_fn: Callable[[], Any],
        operation_name: str = "fetch",
        expected_key: str = "result"
    ) -> Any:
        """
        Fetch data with full protection.
        
        Args:
            fetch_fn: Async function to call broker API
            operation_name: Human-readable operation name for logging
            expected_key: Expected response key to validate
            
        Returns:
            Validated data from response[expected_key]
            
        Raises:
            DataFetchError: If all retries exhausted or circuit open
        """
        # Check circuit breaker
        if not self.circuit.allow_request():
            raise DataFetchError(
                f"Circuit breaker OPEN - {operation_name} blocked"
            )
            
        try:
            # Call broker API
            response = await asyncio.to_thread(fetch_fn)
            
            # Validate response structure
            validated = self._validate_response(response, expected_key)
            
            # Record success
            self.circuit.record_success()
            
            # Return extracted data
            return validated[expected_key]
            
        except Exception as exc:
            # Record failure
            self.circuit.record_failure()
            
            # Notify if critical
            if self.circuit.state == CircuitState.OPEN and self.notifier:
                self._notify_failure(
                    "DATA_PROVIDER_FAILURE",
                    {
                        "operation": operation_name,
                        "error": str(exc),
                        "circuit_state": self.circuit.state.value,
                    },
                )
                
            # Re-raise for retry logic
            raise
            
    async def get_positions(self) -> list[dict[str, Any]]:
        """
        Fetch positions with intelligent fallback (sync/async, list/dict, method names).
        """
        try:
            # 1. Access the real client
            real_client = getattr(self, "client", getattr(self, "_broker", None))
            if not real_client:
                return []

            # 2. Intelligent Fetcher
            async def _execute_fetch():
                # Try standard method name 'positions' (Zerodha) first
                method = getattr(real_client, "positions", getattr(real_client, "get_positions", None))
                
                if not method:
                    self._logger.warning("Broker client missing positions/get_positions method")
                    return []

                if asyncio.iscoroutinefunction(method):
                    return await method()
                else:
                    return await asyncio.to_thread(method)

            raw_response = await _execute_fetch()

            # 3. Robust Normalization (The "World-Class" Part)
            # Zerodha returns {'net': [...], 'day': [...]}
            # Some brokers return [...]
            data = []
            if isinstance(raw_response, dict):
                data = raw_response.get("net", raw_response.get("data", []))
            elif isinstance(raw_response, list):
                data = raw_response

            if not isinstance(data, list):
                return []
                
            return data

        except Exception as exc:
            self._logger.error(f"Robust position fetch failed: {exc}")
            return [] # Return empty list to prevent crash

    async def get_quotes(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        """Fetch quotes with full validation."""
        
        def _fetch() -> Any:
            return self._broker.quote(symbols)
            
        try:
            quotes = await self.fetch_with_validation(
                _fetch,
                operation_name="get_quotes",
                expected_key="data"
            )
            
            if not isinstance(quotes, dict):
                raise DataFetchError(
                    f"Expected dict of quotes, got {type(quotes)}"
                )
                
            return quotes
            
        except DataFetchError:
            LOGGER.error("Quote fetch failed after retries")
            return {}
