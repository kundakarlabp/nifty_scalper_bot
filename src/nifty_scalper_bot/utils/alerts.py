"""Utilities for deduplicating and routing alert notifications."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable, Mapping, MutableMapping

from nifty_scalper_bot.utils.errors import RateLimitError
from nifty_scalper_bot.utils.logging import get_logger
from nifty_scalper_bot.utils.rate_limiter import LeakyBucket

log = get_logger(__name__)


@dataclass(slots=True)
class AggregatedAlert:
    """Accumulate alert statistics for aggregated delivery."""

    key: str
    severity: str
    first_seen: datetime
    last_seen: datetime
    count: int = 0
    messages: list[str] = field(default_factory=list)
    category: str = "misc"


class AlertDeduplicator:
    """Decide whether alert events should bypass aggregation."""

    def __init__(
        self,
        quiet_window: timedelta,
        *,
        bucket_capacity: int = 3,
        bucket_refill_seconds: float = 300.0,
    ) -> None:
        """Initialise deduplicator with quiet window and rate limiter.

        Args:
            quiet_window: Duration suppressing duplicate immediate sends.
            bucket_capacity: Burst capacity for immediate dispatch tokens.
            bucket_refill_seconds: Seconds per token refill for immediate dispatch.

        Returns:
            None.

        Raises:
            None.
        """

        self._quiet_window = quiet_window
        refill_seconds = max(bucket_refill_seconds, 1.0)
        refill_rate = max(1.0 / refill_seconds, 0.01)
        self._bucket = LeakyBucket(
            capacity=bucket_capacity, refill_rate_per_sec=refill_rate
        )
        self._last_seen: MutableMapping[str, datetime] = {}

    def should_immediate(
        self,
        key: str,
        severity: str,
        *,
        hint_immediate: bool,
        now: datetime | None = None,
    ) -> bool:
        """Return ``True`` when an alert should dispatch immediately.

        Args:
            key: Deduplication key for the alert source.
            severity: Severity string (info|warning|critical).
            hint_immediate: Caller preference for immediate delivery.
            now: Optional override for the current timestamp.

        Returns:
            ``True`` when the alert should bypass aggregation.

        Raises:
            None.
        """

        try:
            moment = now or datetime.now(timezone.utc)
            normalized_key = str(key)
            normalized_severity = (severity or "info").strip().lower()
            last_seen = self._last_seen.get(normalized_key)
            quiet_elapsed = False
            if last_seen is not None:
                quiet_elapsed = moment - last_seen >= self._quiet_window

            immediate_requested = hint_immediate or normalized_severity == "critical"

            if immediate_requested:
                if last_seen is None or quiet_elapsed:
                    if self._acquire_token():
                        self._last_seen[normalized_key] = moment
                        return True
                    self._last_seen[normalized_key] = moment
                    log.debug(
                        "Alert token bucket exhausted; aggregating",
                        extra={
                            "event": "alert_token_exhausted",
                            "key": normalized_key,
                        },
                    )
                    return False
                self._last_seen[normalized_key] = moment
                log.debug(
                    "Alert immediate suppressed within quiet window",
                    extra={
                        "event": "alert_immediate_suppressed",
                        "key": normalized_key,
                    },
                )
                return False

            if last_seen is None:
                self._last_seen[normalized_key] = moment
                return False

            if quiet_elapsed and self._acquire_token():
                self._last_seen[normalized_key] = moment
                return True

            self._last_seen[normalized_key] = moment
            if quiet_elapsed:
                log.debug(
                    "Alert quiet window reached; aggregating",
                    extra={
                        "event": "alert_quiet_window_aggregate",
                        "key": normalized_key,
                    },
                )
            return False
        except Exception as exc:  # noqa: BLE001 - defensive catch
            log.error(
                "Failure in AlertDeduplicator.should_immediate: %s",
                exc,
                extra={"event": "alert_dedupe_error", "key": key},
            )
            return hint_immediate

    def prune(self, *, now: datetime | None = None) -> None:
        """Drop dedupe entries older than the quiet window.

        Args:
            now: Optional override for the current timestamp.

        Returns:
            None.

        Raises:
            None.
        """

        try:
            moment = now or datetime.now(timezone.utc)
            threshold = moment - self._quiet_window
            stale = [key for key, seen in self._last_seen.items() if seen < threshold]
            for key in stale:
                self._last_seen.pop(key, None)
        except Exception as exc:  # noqa: BLE001 - defensive catch
            log.error(
                "Failure in AlertDeduplicator.prune: %s",
                exc,
                extra={"event": "alert_dedupe_prune_error"},
            )

    def snapshot(self) -> dict[str, str]:
        """Return a shallow snapshot of dedupe state for diagnostics.

        Args:
            None.

        Returns:
            Mapping of dedupe keys to ISO-8601 timestamps.

        Raises:
            None.
        """

        try:
            return {key: value.isoformat() for key, value in self._last_seen.items()}
        except Exception as exc:  # noqa: BLE001 - defensive catch
            log.error(
                "Failure in AlertDeduplicator.snapshot: %s",
                exc,
                extra={"event": "alert_dedupe_snapshot_error"},
            )
            return {}

    def _acquire_token(self) -> bool:
        """Attempt to reserve a rate-limit token for immediate dispatch.

        Args:
            None.

        Returns:
            ``True`` when a token is acquired, else ``False``.

        Raises:
            None.
        """

        try:
            self._bucket.acquire(timeout=0.01)
            return True
        except RateLimitError:
            return False
        except Exception as exc:  # noqa: BLE001 - defensive catch
            log.error(
                "Failure in AlertDeduplicator._acquire_token: %s",
                exc,
                extra={"event": "alert_token_error"},
            )
            return False


class AlertLogHandler(logging.Handler):
    """Bridge warning/error log records into alert queue events."""

    def __init__(
        self,
        emit_callback: Callable[[Mapping[str, str]], None],
        *,
        exclude: Iterable[str] | None = None,
    ) -> None:
        """Configure handler with callback and optional exclusions.

        Args:
            emit_callback: Callable invoked with alert payload mapping.
            exclude: Optional iterable of logger names to suppress.

        Returns:
            None.

        Raises:
            None.
        """

        super().__init__(level=logging.WARNING)
        self._emit = emit_callback
        self._exclude = set(exclude or [])
        self._exclude.add(__name__)

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        """Format record and forward to the alert callback.

        Args:
            record: Log record forwarded from the logging framework.

        Returns:
            None.

        Raises:
            None.
        """

        if record.name in self._exclude:
            return
        if record.levelno < logging.WARNING:
            return
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - defensive formatting
            message = str(record.msg)
        severity = "critical" if record.levelno >= logging.ERROR else "warning"
        func = getattr(record, "funcName", "") or "unknown"
        key = f"log:{record.name}:{record.levelno}:{func}"
        payload = {
            "key": key,
            "message": message,
            "severity": severity,
        }
        try:
            self._emit(payload)
        except Exception as exc:  # noqa: BLE001 - defensive catch
            log.error(
                "Failure in AlertLogHandler.emit: %s",
                exc,
                extra={"event": "alert_log_handler_error", "key": key},
            )


__all__ = ["AggregatedAlert", "AlertDeduplicator", "AlertLogHandler"]
