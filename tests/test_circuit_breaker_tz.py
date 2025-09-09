from datetime import datetime, timezone, timedelta

from src.utils.circuit_breaker import CircuitBreaker


def test_tick_handles_naive_and_aware_datetime() -> None:
    cb = CircuitBreaker("t", min_samples=1, open_cooldown_sec=1)
    cb.record_failure(100, reason="boom")
    assert cb.state == CircuitBreaker.OPEN

    cb.tick(datetime.now())  # naive
    assert cb.state == CircuitBreaker.OPEN

    cb.tick(datetime.now(timezone.utc) + timedelta(seconds=2))
    assert cb.state == CircuitBreaker.HALF_OPEN

