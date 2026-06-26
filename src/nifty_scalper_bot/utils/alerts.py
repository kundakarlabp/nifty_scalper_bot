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
    r"^\s*Condition met:\s*(?P<event>[A-Za-z0-9_.:-]+)", re.IGNORECASE
)
_PERSISTENT_WARNING_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"strategy evaluation stalled", re.IGNORECASE),
        "strategy_evaluation_stalled",
    ),
    (
        re.compile(
            r"strategy eval genuinely stalled while ticks flowing", re.IGNORECASE
        ),
        "strategy_eval_ticks_flowing_stalled",
    ),
    (
        re.compile(
            r"websocket_degraded\s+code=1006|closing[_ ]handshake[_ ]timeout",
            re.IGNORECASE,
        ),
        "websocket_degraded_1006",
    ),
    (
        re.compile(r"websocket stale\.\s*reconnecting", re.IGNORECASE),
        "websocket_tick_stale",
    ),
)
_GENERIC_RECORD_EVENTS = {"", "app.log", "log", "warning", "error"}


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
        self._quiet_window = quiet_window
        refill_seconds = max(bucket_refill_seconds, 1.0)
        refill_rate = max(1.0 / refill_seconds, 0.01)
        self._bucket = LeakyBucket(
            capacity=bucket_capacity,
            refill_rate_per_sec=refill_rate,
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
        """Return ``True`` when an alert should dispatch immediately."""
        try:
            moment = now or datetime.now(timezone.utc)
            normalized_key = str(key)
            normalized_severity = (severity or "info").strip().lower()
            normalized_outage = (outage_class or "").strip().lower()
            family_key = self._family_key(normalized_key)
            last_seen = self._last_seen.get(normalized_key)
            quiet_elapsed = bool(
                last_seen is not None
                and moment - last_seen >= self._quiet_window
            )
            family_seen = self._last_seen.get(family_key)
            family_quiet_elapsed = bool(
                family_seen is not None
                and moment - family_seen >= self._quiet_window
            )

            flood_hold_until = self._flood_hold_until.get(family_key)
            if flood_hold_until is not None and moment < flood_hold_until and not recovery:
                self._last_seen[normalized_key] = moment
                self._last_seen[family_key] = moment
                return False

            immediate_requested = hint_immediate or normalized_severity == "critical"
            severity_changed = (
                self._last_severity.get(family_key) != normalized_severity
            )
            outage_changed = bool(normalized_outage) and (
                self._last_outage_class.get(family_key) != normalized_outage
            )
            if recovery:
                immediate_requested = True
                severity_changed = True
            if severity_changed or outage_changed:
                immediate_requested = True

            if immediate_requested:
                eligible = (
                    last_seen is None
                    or family_seen is None
                    or quiet_elapsed
                    or family_quiet_elapsed
                    or severity_changed
                    or outage_changed
                )
                if eligible and self._acquire_token():
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
            return False
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in AlertDeduplicator.should_immediate: %s",
                exc,
                extra={"event": "alert_dedupe_error", "key": key},
            )
            return hint_immediate

    def mark_flood_limited(
        self,
        key: str,
        *,
        now: datetime | None = None,
        hold_for: timedelta | None = None,
    ) -> None:
        """Suppress immediate retries for an alert family after flood control."""
        moment = now or datetime.now(timezone.utc)
        family_key = self._family_key(str(key))
        effective_hold = (
            hold_for
            if hold_for is not None and hold_for > self._quiet_window
            else self._quiet_window
        )
        hold_until = moment + effective_hold
        self._flood_hold_until[family_key] = hold_until
        for severity in ("critical", "warning", "info"):
            self._flood_hold_until[f"{family_key}:{severity}"] = hold_until

    def prune(self, *, now: datetime | None = None) -> None:
        """Drop expired dedupe and flood-control entries."""
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
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in AlertDeduplicator.prune: %s",
                exc,
                extra={"event": "alert_dedupe_prune_error"},
            )

    def snapshot(self) -> dict[str, str]:
        """Return a shallow snapshot of dedupe state."""
        try:
            return {key: value.isoformat() for key, value in self._last_seen.items()}
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in AlertDeduplicator.snapshot: %s",
                exc,
                extra={"event": "alert_dedupe_snapshot_error"},
            )
            return {}

    def _acquire_token(self) -> bool:
        try:
            self._bucket.acquire(timeout=0.01)
            return True
        except RateLimitError:
            return False
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in AlertDeduplicator._acquire_token: %s",
                exc,
                extra={"event": "alert_token_error"},
            )
            return False

    @staticmethod
    def _family_key(key: str) -> str:
        normalized = str(key or "").strip().lower()
        if normalized.startswith("market_data."):
            parts = normalized.split(".")
            if len(parts) >= 3:
                return ".".join(parts[:3])
        return normalized.split(":", 1)[0] if ":" in normalized else normalized


class AlertLogHandler(logging.Handler):
    """Bridge warning/error records into Telegram with semantic throttling.

    Persistent watchdog messages often include changing ages or counters. They
    are assigned stable event identities and throttled before queue insertion,
    while unmatched warnings remain fully visible.
    """

    def __init__(
        self,
        emit_callback: Callable[[Mapping[str, str]], None],
        *,
        exclude: Iterable[str] | None = None,
        repeat_window_seconds: float = 300.0,
        persistent_repeat_window_seconds: float = 900.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        super().__init__(level=logging.WARNING)
        self._emit = emit_callback
        self._exclude = set(exclude or [])
        self._exclude.add(__name__)
        self._repeat_window_seconds = max(0.0, float(repeat_window_seconds))
        self._persistent_repeat_window_seconds = max(
            self._repeat_window_seconds,
            float(persistent_repeat_window_seconds),
        )
        self._clock = clock or time.monotonic
        self._last_semantic_emit: MutableMapping[str, float] = {}
        self._semantic_lock = Lock()

    @staticmethod
    def _condition_event(message: str) -> str | None:
        match = _CONDITION_EVENT_RE.match(str(message or ""))
        if match is None:
            return None
        event = match.group("event").strip().lower()
        return event or None

    @staticmethod
    def _persistent_warning_event(message: str) -> str | None:
        text = str(message or "")
        for pattern, event in _PERSISTENT_WARNING_PATTERNS:
            if pattern.search(text):
                return event
        return None

    @staticmethod
    def _structured_event(record: logging.LogRecord) -> str | None:
        event = str(getattr(record, "event", "") or "").strip().lower()
        if event in _GENERIC_RECORD_EVENTS:
            return None
        return event or None

    def _semantic_event(
        self,
        record: logging.LogRecord,
        message: str,
    ) -> tuple[str | None, bool]:
        persistent = self._persistent_warning_event(message)
        if persistent is not None:
            return persistent, True
        condition = self._condition_event(message)
        if condition is not None:
            return condition, False
        structured = self._structured_event(record)
        if structured is not None:
            return structured, False
        return None, False

    def _allow_semantic_emit(self, signature: str, window_seconds: float) -> bool:
        if window_seconds <= 0.0:
            return True
        try:
            now = float(self._clock())
        except Exception:  # noqa: BLE001
            return True
        with self._semantic_lock:
            last = self._last_semantic_emit.get(signature)
            if last is not None and now - last < window_seconds:
                return False
            self._last_semantic_emit[signature] = now
            cutoff = now - max(window_seconds * 4.0, 60.0)
            stale = [
                key
                for key, emitted_at in self._last_semantic_emit.items()
                if emitted_at < cutoff
            ]
            for key in stale:
                self._last_semantic_emit.pop(key, None)
            return True

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        if record.name in self._exclude or record.levelno < logging.WARNING:
            return
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001
            message = str(record.msg)
        severity = "critical" if record.levelno >= logging.ERROR else "warning"
        func = getattr(record, "funcName", "") or "unknown"
        semantic_event, persistent = self._semantic_event(record, message)
        key = f"log:{record.name}:{record.levelno}:{func}"
        if semantic_event is not None:
            key = f"{key}:{semantic_event}"
            window = (
                self._persistent_repeat_window_seconds
                if persistent
                else self._repeat_window_seconds
            )
            if not self._allow_semantic_emit(key, window):
                return
        payload = {
            "key": key,
            "message": message,
            "severity": severity,
        }
        try:
            self._emit(payload)
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in AlertLogHandler.emit: %s",
                exc,
                extra={"event": "alert_log_handler_error", "key": key},
            )


__all__ = ["AggregatedAlert", "AlertDeduplicator", "AlertLogHandler"]
