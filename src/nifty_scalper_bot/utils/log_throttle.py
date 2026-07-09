"""Canonical thread-safe monotonic log throttling utilities."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

NEVER_THROTTLE_EVENTS = {
    "ORDER_SUBMITTED",
    "ORDER_REJECTED",
    "ORDER_REJECTED_FATAL",
    "ORDER_FILLED",
    "ORDER_CANCEL_FAILED",
    "ORDER_MODIFY_FAILED",
    "ORDER_KILL_SWITCH_ENGAGED",
    "POSITION_OPENED",
    "POSITION_CLOSED",
    "POSITION_CLOSE_FAILED",
    "POSITION_RECONCILE_FAILED",
    "UNPROTECTED_POSITION_BLOCKING_ENTRIES",
    "BRACKET_EXIT_ORDER_FAILED",
    "BRACKET_EXIT_FAILED",
    "EXIT_ORDER_REJECTED",
    "EXIT_FAILED_ESCALATED",
    "RISK_LIMIT_BREACHED",
    "MARGIN_REJECTED",
    "BROKER_REJECTED",
    "AUTHENTICATION_FAILED",
    "AUTH_FAILURE",
    "SESSION_EXPIRED",
    "SESSION_REFRESH_FAILED",
    "UNCAUGHT_EXCEPTION",
    "WEBSOCKET_DISCONNECTED",
    "WEBSOCKET_RECONNECTED",
    "WEBSOCKET_RECONNECT_FAILED",
}
# Backwards-compatible alias for imports/tests; do not add prefix matching.
CRITICAL_EVENTS = NEVER_THROTTLE_EVENTS


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ThrottleState:
    first_seen: str
    last_seen: str
    interval_seconds: float
    last_emit_mono: float = 0.0
    suppressed_count: int = 0


@dataclass
class ChangeState:
    state: Any
    first_seen: str
    last_seen: str
    last_emit_mono: float
    suppressed_count: int = 0


@dataclass
class StrategyRejectionStats:
    evaluation_count: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    by_reason: Counter[str] = field(default_factory=Counter)
    by_strategy: Counter[str] = field(default_factory=Counter)
    by_symbol: Counter[str] = field(default_factory=Counter)
    first_score: float | None = None
    latest_score: float | None = None
    min_score: float | None = None
    max_score: float | None = None

    def record_score(self, score: Any) -> None:
        try:
            value = float(score)
        except (TypeError, ValueError):
            return
        if self.first_score is None:
            self.first_score = value
        self.latest_score = value
        self.min_score = value if self.min_score is None else min(self.min_score, value)
        self.max_score = value if self.max_score is None else max(self.max_score, value)


class LogThrottle:
    """Thread-safe per-key log throttle with monotonic timing and summaries."""

    def __init__(self, cooldown_seconds: float | None = None) -> None:
        self.default_interval_seconds = float(cooldown_seconds or 0.0)
        self._lock = threading.RLock()
        self._states: dict[str, ThrottleState] = {}
        self._change_states: dict[str, ChangeState] = {}
        self._summary_last_emit_mono: float = 0.0
        self._strategy_stats = StrategyRejectionStats()
        self._strategy_summary_last_emit_mono: float = 0.0
        # Backwards-compatible aliases for older tests/introspection only.
        self._last_emit_mono: dict[str, float] = {}
        self._suppressed: defaultdict[str, int] = defaultdict(int)

    def should_log(self, key: str, interval_seconds: float | None = None) -> bool:
        """Return True when the key's monotonic interval has elapsed."""
        interval = self.default_interval_seconds if interval_seconds is None else float(interval_seconds)
        now = time.monotonic()
        wall = _utc_iso()
        with self._lock:
            state = self._states.get(key)
            if state is None:
                state = ThrottleState(first_seen=wall, last_seen=wall, interval_seconds=max(0.0, interval))
                self._states[key] = state
            else:
                state.last_seen = wall
                state.interval_seconds = max(0.0, interval)
            if state.last_emit_mono > 0.0 and (now - state.last_emit_mono) < state.interval_seconds:
                return False
            state.last_emit_mono = now
            self._last_emit_mono[key] = now
            return True

    def record_suppressed(self, key: str) -> None:
        wall = _utc_iso()
        with self._lock:
            state = self._states.get(key)
            if state is None:
                state = ThrottleState(first_seen=wall, last_seen=wall, interval_seconds=0.0)
                self._states[key] = state
            state.last_seen = wall
            state.suppressed_count += 1
            self._suppressed[key] += 1

    def pop_suppressed(self, key: str) -> int:
        with self._lock:
            state = self._states.get(key)
            value = int(state.suppressed_count if state else self._suppressed.get(key, 0) or 0)
            if state:
                state.suppressed_count = 0
            self._suppressed[key] = 0
            return value

    def state_metadata(self, key: str) -> dict[str, Any]:
        with self._lock:
            state = self._states.get(key)
            if state is None:
                return {}
            return {
                "first_seen": state.first_seen,
                "last_seen": state.last_seen,
                "interval_seconds": state.interval_seconds,
                "suppressed_count": state.suppressed_count,
            }

    def maybe_emit_summary(self, logger: logging.Logger, *, interval_seconds: float = 60.0, top_n: int = 10) -> None:
        """Emit a compact aggregate suppression summary periodically."""
        now = time.monotonic()
        with self._lock:
            if self._summary_last_emit_mono > 0 and (now - self._summary_last_emit_mono) < float(interval_seconds):
                return
            pending = {k: s.suppressed_count for k, s in self._states.items() if s.suppressed_count > 0}
            if not pending:
                return
            self._summary_last_emit_mono = now
            for key in pending:
                self._states[key].suppressed_count = 0
                self._suppressed[key] = 0
        top = sorted(pending.items(), key=lambda item: item[1], reverse=True)[: max(1, int(top_n))]
        total_suppressed = sum(int(v) for v in pending.values())
        keys = ",".join(f"{k}:{v}" for k, v in top)
        logger.info(
            "LOG_THROTTLE_SUMMARY total_suppressed=%s top_keys=%s",
            total_suppressed,
            keys,
            extra={
                "event": "LOG_THROTTLE_SUMMARY",
                "reason": "throttle_summary",
                "total_suppressed": total_suppressed,
                "top_keys": keys,
                "keys_count": len(top),
            },
        )

    def log_on_change(self, logger: logging.Logger, *, key: str, state: Any, message: str, reminder_seconds: float = 600, level: int = logging.INFO, extra: dict[str, Any] | None = None) -> bool:
        now = time.monotonic()
        wall = _utc_iso()
        payload = dict(extra or {})
        with self._lock:
            previous = self._change_states.get(key)
            changed = previous is None or previous.state != state
            reminder_due = previous is not None and (now - previous.last_emit_mono) >= max(0.0, float(reminder_seconds))
            if not changed and not reminder_due:
                previous.suppressed_count += 1
                previous.last_seen = wall
                return False
            suppressed = int(previous.suppressed_count) if previous else 0
            first_seen = previous.first_seen if previous else wall
            self._change_states[key] = ChangeState(state=state, first_seen=first_seen, last_seen=wall, last_emit_mono=now, suppressed_count=0)
        payload.update({"log_key": key, "state": state, "previous_state": None if previous is None else previous.state, "suppressed_count": suppressed, "first_seen": first_seen, "last_seen": wall, "reminder_seconds": float(reminder_seconds)})
        logger.log(level, message, extra=payload)
        return True

    def record_strategy_evaluation(self, *, strategy: str, symbol: str, accepted: bool, reason: str | None = None, score: Any = None) -> None:
        with self._lock:
            self._strategy_stats.evaluation_count += 1
            if accepted:
                self._strategy_stats.accepted_count += 1
            else:
                self._strategy_stats.rejected_count += 1
                self._strategy_stats.by_reason[str(reason or "unknown")] += 1
                self._strategy_stats.by_strategy[str(strategy or "unknown")] += 1
                self._strategy_stats.by_symbol[str(symbol or "unknown")] += 1
                self._strategy_stats.record_score(score)

    def maybe_emit_strategy_rejection_summary(self, logger: logging.Logger, *, interval_seconds: float = 300.0, top_n: int = 5) -> bool:
        now = time.monotonic()
        with self._lock:
            if self._strategy_summary_last_emit_mono > 0 and (now - self._strategy_summary_last_emit_mono) < float(interval_seconds):
                return False
            stats = self._strategy_stats
            if stats.rejected_count <= 0:
                return False
            self._strategy_summary_last_emit_mono = now
            payload = {
                "event": "STRATEGY_REJECTION_SUMMARY",
                "evaluation_count": stats.evaluation_count,
                "accepted_count": stats.accepted_count,
                "rejected_count": stats.rejected_count,
                "top_reasons": dict(stats.by_reason.most_common(top_n)),
                "top_strategies": dict(stats.by_strategy.most_common(top_n)),
                "top_symbols": dict(stats.by_symbol.most_common(top_n)),
                "first_score": stats.first_score,
                "latest_score": stats.latest_score,
                "min_score": stats.min_score,
                "max_score": stats.max_score,
            }
            self._strategy_stats = StrategyRejectionStats()
        logger.info(
            "STRATEGY_REJECTION_SUMMARY evaluations=%s accepted=%s rejected=%s top_reasons=%s",
            payload["evaluation_count"], payload["accepted_count"], payload["rejected_count"], payload["top_reasons"],
            extra=payload,
        )
        return True


DEFAULT_LOG_THROTTLE = LogThrottle()


def event_is_never_throttled(event: str | None) -> bool:
    """Return True only for explicitly allow-listed safety events."""
    return str(event or "").upper() in NEVER_THROTTLE_EVENTS


def log_throttled(logger: logging.Logger, level: int, event: str, key: str, interval_seconds: float, message: str, *args: Any, **kwargs: Any) -> bool:
    """Emit a throttled log with shared state and ``suppressed_count`` metadata.

    Logging must never break trading paths; any logger/format/extra failure is
    swallowed after best-effort accounting.
    """
    try:
        throttle: LogThrottle = kwargs.pop("throttle", DEFAULT_LOG_THROTTLE)
        extra = dict(kwargs.pop("extra", {}) or {})
        extra.setdefault("event", event)
        if event_is_never_throttled(event):
            extra.setdefault("suppressed_count", 0)
            extra["bypass_filters"] = True
            logger.log(level, message, *args, extra=extra, **kwargs)
            return True
        if throttle.should_log(key, interval_seconds):
            suppressed = throttle.pop_suppressed(key)
            extra["suppressed_count"] = suppressed
            extra.update({k: v for k, v in throttle.state_metadata(key).items() if k != "suppressed_count"})
            logger.log(level, message, *args, extra=extra, **kwargs)
            enabled = os.getenv("LOG_THROTTLE_SUMMARY_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
            if enabled:
                throttle.maybe_emit_summary(logger, interval_seconds=float(os.getenv("LOG_THROTTLE_SUMMARY_SECONDS", "120") or "120"), top_n=int(os.getenv("LOG_THROTTLE_SUMMARY_TOP_N", "10") or "10"))
            return True
        throttle.record_suppressed(key)
        return False
    except Exception:
        return False


def log_on_change(logger: logging.Logger, *, key: str, state: Any, message: str, reminder_seconds: float = 600, level: int = logging.INFO, extra: dict[str, Any] | None = None, throttle: LogThrottle = DEFAULT_LOG_THROTTLE) -> bool:
    """Emit a state-change log without allowing logging failures to propagate."""
    try:
        return throttle.log_on_change(logger, key=key, state=state, message=message, reminder_seconds=reminder_seconds, level=level, extra=extra)
    except Exception:
        return False


def record_strategy_evaluation(*, strategy: str, symbol: str, accepted: bool, reason: str | None = None, score: Any = None, throttle: LogThrottle = DEFAULT_LOG_THROTTLE) -> None:
    try:
        throttle.record_strategy_evaluation(strategy=strategy, symbol=symbol, accepted=accepted, reason=reason, score=score)
    except Exception:
        return


def maybe_emit_strategy_rejection_summary(logger: logging.Logger, *, interval_seconds: float = 300.0, top_n: int = 5, throttle: LogThrottle = DEFAULT_LOG_THROTTLE) -> bool:
    try:
        return throttle.maybe_emit_strategy_rejection_summary(logger, interval_seconds=interval_seconds, top_n=top_n)
    except Exception:
        return False
