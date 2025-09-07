import time

import pytest

from src.utils.reliability import CircuitBreaker, RateLimiter, retry


def test_rate_limiter_refills() -> None:
    rl = RateLimiter(max_per_min=1)
    assert rl.allow()
    assert not rl.allow()
    rl.ts -= 60
    assert rl.allow()


def test_circuit_breaker_opens_and_recovers() -> None:
    cb = CircuitBreaker(fail_threshold=2, cooldown_s=0.1)

    def boom() -> None:
        raise ValueError("boom")

    with pytest.raises(ValueError):
        cb.call(boom)
    with pytest.raises(ValueError):
        cb.call(boom)
    with pytest.raises(RuntimeError):
        cb.call(lambda: None)
    cb.open_until = time.time() - 1
    assert cb.call(lambda: "ok") == "ok"


def test_retry_eventually_succeeds() -> None:
    calls: list[int] = []

    @retry(attempts=3, base=0, max_sleep=0)
    def flaky() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise ValueError("err")
        return "ok"

    assert flaky() == "ok"
    assert len(calls) == 3
