from __future__ import annotations

from typing import Callable

import pytest

from nifty_scalper_bot.utils.circuit_breaker import CircuitBreaker


def _clock_factory(
    start: float = 0.0,
) -> tuple[Callable[[], float], Callable[[float], None]]:
    current = start

    def now() -> float:
        return current

    def advance(delta: float) -> None:
        nonlocal current
        current += delta

    return now, advance


def test_circuit_breaker_opens_after_failures() -> None:
    clock, advance = _clock_factory()
    breaker = CircuitBreaker(
        fail_threshold=2, window_s=10.0, cooldown_s=5.0, clock=clock
    )

    assert breaker.allow()
    breaker.on_failure()
    assert breaker.allow()
    breaker.on_failure()
    assert not breaker.allow()
    assert breaker.is_open
    advance(5.1)
    assert breaker.allow()
    breaker.on_success()
    assert not breaker.is_open


def test_circuit_breaker_resets_window() -> None:
    clock, advance = _clock_factory()
    breaker = CircuitBreaker(
        fail_threshold=3, window_s=2.0, cooldown_s=5.0, clock=clock
    )

    for _ in range(2):
        breaker.on_failure()
        advance(0.5)
    assert breaker.allow()
    advance(2.1)
    assert breaker.allow()
    breaker.on_failure()
    breaker.on_failure()
    breaker.on_failure()
    assert breaker.is_open


def test_invalid_configuration() -> None:
    with pytest.raises(ValueError):
        CircuitBreaker(fail_threshold=0)
    with pytest.raises(ValueError):
        CircuitBreaker(window_s=0)
    with pytest.raises(ValueError):
        CircuitBreaker(cooldown_s=0)
