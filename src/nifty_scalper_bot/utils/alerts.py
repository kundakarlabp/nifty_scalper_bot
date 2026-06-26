"""Utilities for deduplicating and routing alert notifications."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Callable, Iterable, Mapping, MutableMapping

from nifty_scalper_bot.utils.errors import RateLimitError
from nifty_scalper_bot.utils.logging import get_logger
from nifty_scalper_bot.utils.rate_limiter import LeakyBucket

log = get_logger(__name__)

_CONDITION_EVENT_RE = re.compile(
    r"^\s*Condition met:\s*(?P<event>[A-Za-z0-9_.:-]+)"
)


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
        bucket_capacity: int = 100,
        bucket_refill_seconds: float = 100.0,
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
        self._last_severity: MutableMapping[str, str] = {}
        self._last_outage_class: MutableMapping[str, str] = {}
        self._flood_hold_until: MutableMapping[str, datetime] = {}

    def should_immediate(
        self,
        key: str,
        severity: str,
        *,
        hint_immediate: bool,
        outage_class: str | None = None,
        recovery: bool = False,
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
            normalized_outage = (outage_class or "").strip().lower()
            family_key = self._family_key(normalized_key)
            last_seen = self._last_seen.get(normalized_key)
            quiet_elapsed = False
            if last_seen is not None:
                quiet_elapsed = moment - last_seen >= self._quiet_window
            family_seen = self._last_seen.get(family_key)
            family_quiet_elapsed = False
            if family_seen is not None:
                family_quiet_elapsed = moment - family_seen >= self._quiet_window

            flood_hold_until = self._flood_hold_until.get(family_key)
            if (
                flood_hold_until is not None
                and moment < flood_hold_until
                and not recovery
            ):
                self._last_seen[normalized_key] = moment
                self._last_seen[family_key] = moment
                return False

            immediate_requested = hint_immediate or normalized_severity == "critical"
            severity_changed = self._last_severity.get(family_key) != normalized_severity
            outage_changed = (
                bool(normalized_outage)
                and self._last_outage_class.get(family_key) != normalized_outage
            )
            if recovery:
                immediate_requested = True
                severity_changed = True
            if severity_changed or outage_changed:
                immediate_requested = True

            if immediate_requested:
                if (
                    last_seen is None
                    or family_seen is None
                    or quiet_elapsed
                    or family_quiet_elapsed
                    or severity_changed
                    or outage_changed
                ):
                    if self._acquire_token():
                        self._last_seen[normalized_key] = moment
                        self._last_seen[family_key] = moment
                        self._last_severity[family_key] = normalized_severity
                        if normalized_outage:
                            self._last_outage_class[family_key] = normalized_outage
                        if recovery:
                            self._flood_hold_until.pop(family_key, None)
                        return True
                    self._last_seen[normalized_key] = moment
                    self._last_seen[family_key] = moment
                    log.debug(
                        "Alert token bucket exhausted; aggregating",
                        extra={
                            "event": "alert_token_exhausted",
                            "key": normalized_key,
                        },
                    )
                    return False
                self._last_seen[normalized_key] = moment
                self._last_seen[family_key] = moment
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
                self._last_seen[family_key] = moment
                return False

            if quiet_elapsed and self._acquire_token():
                self._last_seen[normalized_key] = moment
                self._last_seen[family_key] = moment
                self._last_severity[family_key] = normalized_severity
                if normalized_outage:
                    self._last_outage_class[family_key] = normalized_outage
                return True

            self._last_seen[normalized_key] = moment
            self._last_seen[family_key] = moment
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

    def mark_flood_limited(
        self, key: str, *, now: datetime | None = None, hold_for: timedelta | None = None
    ) -> None:
        """Record send flood-control and suppress immediate retries for family."""

        moment = now or datetime.now(timezone.utc)
        family_key = self._family_key(str(key))
        effective_hold = (
            hold_for
            if hold_for is not None and hold_for > self._quiet_window
            else self._quiet_window
        )
        hold_until = moment + effective_hold
        self._flood_hold_until[family_key] = hold_until
        # Apply hold to all severities under the family to stop critical bypass spam.
        for severity in ("critical", "warning", "info"):
            self._flood_hold_until[f"{family_key}:{severity}"] = hold_until

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
                self._last_severity.pop(key, None)
                self._last_outage_class.pop(key, None)
            expired = [
                key
                for key, hold_until in self._flood_hold_until.items()
                if hold_until < moment
            ]
            for key in expired:
                self._flood_hold_until.pop(key, None)
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

    @staticmethod
    def _family_key(key: str) -> str:
        """Collapse alert keys into outage-family buckets."""

        normalized = str(key or "").strip().lower()
        if normalized.startswith("market_data."):
            parts = normalized.split(".")
            if len(parts) >= 3:
                return ".".join(parts[:2]) + f".{parts[2]}"
        return normalized.split(":", 1)[0] if ":" in normalized else normalized


class AlertLogHandler(logging.Handler):
    """Bridge warning/error log records into alert queue events.

    Repeated ``Condition met: <event>`` records are collapsed before they enter
    Telegram's queue. Dynamic fields such as ``age=`` therefore do not create a
    new notification for every watchdog cycle. The original application log is
    untouched, and the same condition may notify again after the repeat window.
    """

    def __init__(
        self,
        emit_callback: Callable[[Mapping[str, str]], None],
        *,
        exclude: Iterable[str] | None = None,
        repeat_window_seconds: float = 300.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """Configure handler with callback and optional exclusions.

        Args:
            emit_callback: Callable invoked with alert payload mapping.
            exclude: Optional iterable of logger names to suppress.
            repeat_window_seconds: Cooldown for the same semantic condition.
            clock: Monotonic clock override for deterministic tests.

        Returns:
            None.

        Raises:
            None.
        """

        super().__init__(level=logging.WARNING)
        self._emit = emit_callback
        self._exclude = set(exclude or [])
        self._exclude.add(__name__)
        self._repeat_window_seconds = max(0.0, float(repeat_window_seconds))
        self._clock = clock or time.monotonic
        self._last_condition_emit: MutableMapping[str, float] = {}
        self._condition_lock = Lock()

    @staticmethod
    def _condition_event(message: str) -> str | None:
        """Return the semantic condition name embedded in a log message."""

        match = _CONDITION_EVENT_RE.match(str(message or ""))
        if match is None:
            return None
        event = match.group("event").strip().lower()
        return event or None

    def _allow_condition_emit(self, signature: str) -> bool:
        """Return whether a semantic condition is outside its repeat window."""

        if self._repeat_window_seconds <= 0.0:
            return True
        try:
            now = float(self._clock())
        except Exception:  # noqa: BLE001 - alert delivery must remain fail-open
            return True
        with self._condition_lock:
            last = self._last_condition_emit.get(signature)
            if last is not None and now - last < self._repeat_window_seconds:
                return False
            self._last_condition_emit[signature] = now
            cutoff = now - max(self._repeat_window_seconds * 4.0, 60.0)
            stale = [
                key
                for key, emitted_at in self._last_condition_emit.items()
                if emitted_at < cutoff
            ]
            for key in stale:
                self._last_condition_emit.pop(key, None)
            return True

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
        condition_event = self._condition_event(message)
        key = f"log:{record.name}:{record.levelno}:{func}"
        if condition_event is not None:
            key = f"{key}:{condition_event}"
            if not self._allow_condition_emit(key):
                return
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
