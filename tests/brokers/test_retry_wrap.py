"""Tests covering the retry_with_backoff helper."""

from __future__ import annotations

import logging

import pytest

from nifty_scalper_bot.infra.metrics import METRICS
from nifty_scalper_bot.utils.retry import (
    RetryableError,
    RetryErrorContext,
    retry_with_backoff,
)

LOGGER = logging.getLogger("tests.retry")


def test_retry_with_backoff_eventually_succeeds(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure transient failures recover with jittered backoff."""

    caplog.set_level(logging.ERROR)
    attempts: list[int] = []
    delays: list[float] = []
    metrics_calls: list[tuple[str, str, str]] = []

    def sleep(delay: float) -> None:
        delays.append(delay)

    def operation() -> str:
        attempts.append(1)
        if len(attempts) < 3:
            raise ValueError("transient issue")
        return "success"

    def should_retry(exc: Exception) -> bool:
        return isinstance(exc, ValueError)

    def fake_increment(*, label: str, stage: str, outcome: str) -> None:
        metrics_calls.append((label, stage, outcome))

    monkeypatch.setattr(METRICS, "increment_retry_event", fake_increment, raising=False)

    result = retry_with_backoff(
        operation=operation,
        retries=5,
        base_delay=0.1,
        max_delay=1.0,
        jitter=0.0,
        logger=LOGGER,
        label="test.eventual",
        sleep=sleep,
        should_retry=should_retry,
    )

    assert result == "success"
    assert len(attempts) == 3
    assert delays == [0.1, 0.2]
    failure_events = [
        record.getMessage()
        for record in caplog.records
        if "broker_call_failed" in record.getMessage()
    ]
    assert any("attempt=1/5" in message for message in failure_events)
    assert any("attempt=2/5" in message for message in failure_events)
    assert metrics_calls == [
        ("test.eventual", "attempt", "retry"),
        ("test.eventual", "attempt", "retry"),
        ("test.eventual", "result", "success"),
    ]


def test_retry_with_backoff_exhausts_retries(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure retry helper raises after exhausting attempts."""

    caplog.set_level(logging.ERROR)
    delays: list[float] = []
    attempt_counter = 0
    metrics_calls: list[tuple[str, str, str]] = []

    def sleep(delay: float) -> None:
        delays.append(delay)

    def operation() -> str:
        nonlocal attempt_counter
        attempt_counter += 1
        raise RetryableError(
            "persistent failure",
            context=RetryErrorContext(status=500, endpoint="/orders/test"),
        )

    def fake_increment(*, label: str, stage: str, outcome: str) -> None:
        metrics_calls.append((label, stage, outcome))

    monkeypatch.setattr(METRICS, "increment_retry_event", fake_increment, raising=False)

    with pytest.raises(RetryableError):
        retry_with_backoff(
            operation=operation,
            retries=3,
            base_delay=0.2,
            max_delay=0.2,
            jitter=0.0,
            logger=LOGGER,
            label="test.exhaust",
            sleep=sleep,
        )

    assert attempt_counter == 3
    assert delays == [0.2, 0.2]
    failure_events = [
        record.getMessage()
        for record in caplog.records
        if "broker_call_failed" in record.getMessage()
    ]
    assert any("attempt=3/3" in message for message in failure_events)
    assert len(failure_events) == 3
    assert metrics_calls == [
        ("test.exhaust", "attempt", "retry"),
        ("test.exhaust", "attempt", "retry"),
        ("test.exhaust", "result", "failure"),
    ]


def test_retry_with_backoff_stops_when_not_retryable(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure non-retryable errors propagate without delay."""

    caplog.set_level(logging.ERROR)
    delays: list[float] = []
    metrics_calls: list[tuple[str, str, str]] = []

    def sleep(delay: float) -> None:
        delays.append(delay)

    def operation() -> str:
        raise ValueError("permanent failure")

    def should_retry(exc: Exception) -> bool:
        return isinstance(exc, RetryableError)

    def fake_increment(*, label: str, stage: str, outcome: str) -> None:
        metrics_calls.append((label, stage, outcome))

    monkeypatch.setattr(METRICS, "increment_retry_event", fake_increment, raising=False)

    with pytest.raises(ValueError):
        retry_with_backoff(
            operation=operation,
            retries=4,
            base_delay=0.05,
            max_delay=0.2,
            jitter=0.0,
            logger=LOGGER,
            label="test.no_retry",
            sleep=sleep,
            should_retry=should_retry,
        )

    assert delays == []
    failure_events = [
        record.getMessage()
        for record in caplog.records
        if "broker_call_failed" in record.getMessage()
    ]
    assert len(failure_events) == 1
    assert "terminal=true" in failure_events[0]
    assert metrics_calls == [("test.no_retry", "result", "failure")]
