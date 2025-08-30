from datetime import datetime, timedelta

from src.utils.circuit_breaker import CircuitBreaker


def test_closed_to_open_on_error_rate() -> None:
    cb = CircuitBreaker("t", min_samples=30, error_rate_threshold=0.1)
    for _ in range(30):
        cb.record_failure(100, reason="boom")
    assert cb.state == CircuitBreaker.OPEN


def test_open_to_half_open_after_cooldown() -> None:
    cb = CircuitBreaker("t", min_samples=1, open_cooldown_sec=1)
    cb.record_failure(100, reason="boom")
    assert cb.state == CircuitBreaker.OPEN
    cb.tick(datetime.utcnow() + timedelta(seconds=2))
    assert cb.state == CircuitBreaker.HALF_OPEN


def test_half_open_to_closed_after_successes() -> None:
    cb = CircuitBreaker("t", min_samples=1, open_cooldown_sec=1, half_open_probe=2)
    cb.record_failure(100, reason="boom")
    cb.tick(datetime.utcnow() + timedelta(seconds=2))
    assert cb.state == CircuitBreaker.HALF_OPEN
    cb.record_success(50)
    assert cb.state == CircuitBreaker.HALF_OPEN
    cb.record_success(40)
    assert cb.state == CircuitBreaker.CLOSED
