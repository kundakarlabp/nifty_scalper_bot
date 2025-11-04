"""Retry helpers with bounded exponential backoff and jitter."""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

from nifty_scalper_bot.infra.metrics import METRICS

T = TypeVar("T")


@dataclass(slots=True)
class RetryErrorContext:
    """Encapsulate metadata for retry decisions."""

    status: int | None = None
    endpoint: str | None = None
    delay_hint: float | None = None
    error: Exception | None = None


class RetryableError(Exception):
    """Signal that an operation can be retried with optional metadata."""

    def __init__(
        self, message: str, *, context: RetryErrorContext | None = None
    ) -> None:
        super().__init__(message)
        self.context = context or RetryErrorContext()


def retry_with_backoff(
    *,
    operation: Callable[[], T],
    retries: int,
    base_delay: float,
    max_delay: float,
    jitter: float,
    logger: logging.Logger,
    label: str,
    sleep: Callable[[float], None] | None = None,
    should_retry: Callable[[Exception], bool] | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
) -> T:
    """Execute *operation* with jittered exponential backoff.

    Args:
        operation: Callable invoked for every attempt.
        retries: Total attempts allowed (minimum of one).
        base_delay: Base delay in seconds for the first retry.
        max_delay: Maximum delay cap in seconds.
        jitter: Symmetric jitter factor applied to the delay.
        logger: Logger used for structured diagnostics.
        label: Operation label for diagnostics and logging.
        sleep: Optional callable to sleep for the computed delay.
        should_retry: Optional predicate deciding if an exception is retryable.
        on_retry: Optional callback receiving attempt index, exception, and delay.

    Returns:
        T: Result returned by ``operation`` on success.

    Raises:
        Exception: Propagates the last exception if retries are exhausted.
    """

    sleeper = sleep or time.sleep
    total_attempts = max(1, int(retries))
    normalized_base = max(0.001, float(base_delay))
    normalized_cap = max(normalized_base, float(max_delay))
    normalized_jitter = max(0.0, float(jitter))
    attempt = 0
    last_error: Exception | None = None

    while attempt < total_attempts:
        attempt += 1
        try:
            result = operation()
        except Exception as exc:  # noqa: BLE001 - explicit logging required
            retry_decision = should_retry(exc) if should_retry is not None else True
            if not retry_decision:
                _log_failure(
                    logger=logger,
                    label=label,
                    attempt=attempt,
                    total_attempts=total_attempts,
                    error=exc,
                    delay=0.0,
                    terminal=True,
                )
                _increment_metrics(
                    label=label, stage="result", outcome="failure", logger=logger
                )
                raise

            last_error = exc
            delay = _compute_delay(
                attempt=attempt,
                base_delay=normalized_base,
                cap=normalized_cap,
                jitter=normalized_jitter,
                error=exc,
            )
            terminal_attempt = attempt >= total_attempts
            _log_failure(
                logger=logger,
                label=label,
                attempt=attempt,
                total_attempts=total_attempts,
                error=exc,
                delay=delay,
                terminal=terminal_attempt,
            )
            callback_delay = 0.0 if terminal_attempt else delay
            if terminal_attempt:
                _increment_metrics(
                    label=label,
                    stage="result",
                    outcome="failure",
                    logger=logger,
                )
            else:
                _increment_metrics(
                    label=label,
                    stage="attempt",
                    outcome="retry",
                    logger=logger,
                )

            if on_retry is not None:
                try:
                    on_retry(attempt, exc, callback_delay)
                except Exception as callback_exc:  # pragma: no cover - defensive
                    logger.error(
                        "retry_callback_failed",
                        extra={
                            "event": "retry_callback_failed",
                            "label": label,
                            "attempt": attempt,
                            "error": repr(callback_exc),
                        },
                    )

            if terminal_attempt:
                raise

            if callback_delay > 0.0:
                sleeper(callback_delay)
        else:
            _increment_metrics(
                label=label,
                stage="result",
                outcome="success",
                logger=logger,
            )
            return result

    if last_error is None:
        raise RuntimeError("retry_with_backoff executed without attempts")
    raise last_error


def _compute_delay(
    *,
    attempt: int,
    base_delay: float,
    cap: float,
    jitter: float,
    error: Exception,
) -> float:
    """Compute jittered delay for the given attempt.

    Args:
        attempt: Current attempt number (1-indexed).
        base_delay: Base delay in seconds before jitter.
        cap: Maximum allowed delay in seconds.
        jitter: Symmetric jitter factor applied to the exponential delay.
        error: Exception raised by the operation.

    Returns:
        float: Delay value in seconds respecting jitter and cap constraints.

    Raises:
        None.
    """

    if isinstance(error, RetryableError) and error.context.delay_hint is not None:
        delay = max(0.0, float(error.context.delay_hint))
    else:
        exponent = max(0, attempt - 1)
        delay = base_delay * (2**exponent)
    bounded = min(delay, cap)
    if jitter <= 0.0:
        return bounded
    low = max(0.0, 1.0 - min(jitter, 0.99))
    high = 1.0 + min(jitter, 0.99)
    factor = random.uniform(low, high)
    return max(0.0, min(cap, bounded * factor))


def _increment_metrics(
    *, label: str, stage: str, outcome: str, logger: logging.Logger
) -> None:
    """Increment retry metrics counter while shielding metrics errors.

    Args:
        label: Operation label for Prometheus dimensions.
        stage: Lifecycle stage (``attempt`` or ``result``).
        outcome: Result tag such as ``retry``, ``success``, or ``failure``.
        logger: Logger for diagnostics when metrics fail.

    Returns:
        None.

    Raises:
        None.
    """

    try:
        METRICS.increment_retry_event(label=label, stage=stage, outcome=outcome)
    except Exception as exc:  # noqa: BLE001 - defensive metrics logging
        logger.debug(
            "retry_metrics_increment_failed",
            extra={
                "event": "retry_metrics_increment_failed",
                "label": label,
                "stage": stage,
                "outcome": outcome,
                "error": repr(exc),
            },
        )


def _log_failure(
    *,
    logger: logging.Logger,
    label: str,
    attempt: int,
    total_attempts: int,
    error: Exception,
    delay: float,
    terminal: bool,
) -> None:
    """Emit structured failure log for broker retry attempts.

    Args:
        logger: Logger used for structured diagnostics.
        label: Operation label for diagnostics and logging.
        attempt: Current attempt index.
        total_attempts: Total allowed attempts.
        error: Exception triggering the retry.
        delay: Planned delay before the next attempt.
        terminal: Flag indicating whether no further retries will occur.

    Returns:
        None.

    Raises:
        None.
    """

    logger.error(
        "broker_call_failed label=%s attempt=%d/%d delay=%0.3fs terminal=%s error=%s",
        label,
        attempt,
        total_attempts,
        delay,
        str(terminal).lower(),
        error,
        extra={
            "event": "broker_call_failed",
            "label": label,
            "attempt": attempt,
            "total_attempts": total_attempts,
            "delay": delay,
            "terminal": terminal,
        },
    )


__all__ = ["RetryableError", "RetryErrorContext", "retry_with_backoff"]
