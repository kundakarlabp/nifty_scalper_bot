"""World-class data provider with validation, retry, and circuit breaking."""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, TypeVar
from dataclasses import dataclass

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

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
        self.circuit = circuit_config 
        self.notifier = notifier
        self._logger = logging.getLogger(__name__)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.client, name)
    
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
                asyncio.create_task(
                    self.notifier(
                        "DATA_PROVIDER_FAILURE",
                        {
                            "operation": operation_name,
                            "error": str(exc),
                            "circuit_state": self.circuit.state.value
                        }
                    )
                )
                
            # Re-raise for retry logic
            raise
            
    async def get_positions(self) -> list[dict[str, Any]]:
        """Fetch positions safely, handling sync/async and data normalization."""
        try:
            # Helper to run the fetch
            async def _fetch():
                if asyncio.iscoroutinefunction(self.client.get_positions):
                    return await self.client.get_positions()
                else:
                    # Run synchronous Zerodha call in a thread to stop blocking
                    return await asyncio.to_thread(self.client.get_positions)

            raw_response = await _fetch()

            # ✅ 3. Data Normalization: Fixes 'list indices' errors
            # Handle Zerodha's {'net': [...]} vs direct list
            data = raw_response
            if isinstance(raw_response, dict):
                data = raw_response.get("net", raw_response.get("data", []))
            
            if not isinstance(data, list):
                self._logger.warning(f"Unexpected position data type: {type(data)}")
                return []
                
            return data

        except Exception as exc:
            self._logger.error(f"Robust position fetch failed: {exc}")
            return []

    async def get_quotes(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        """Fetch quotes with full validation."""
        
        def _fetch() -> Any:
            return self.broker.quote(symbols)
            
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
