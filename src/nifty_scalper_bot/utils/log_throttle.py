"""Reusable monotonic log throttling utilities."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import defaultdict
from typing import Any


class LogThrottle:
    """Thread-safe per-key log throttle with suppression counters."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._last_emit_mono: dict[str, float] = {}
        self._suppressed: dict[str, int] = defaultdict(int)
        self._summary_last_emit_mono: float = 0.0

    def should_log(self, key: str, interval_seconds: float) -> bool:
        """Return True when interval elapsed for key using monotonic time."""
        now = time.monotonic()
        with self._lock:
            last = float(self._last_emit_mono.get(key, 0.0) or 0.0)
            if last > 0.0 and (now - last) < max(0.0, float(interval_seconds)):
                return False
            self._last_emit_mono[key] = now
            return True

    def record_suppressed(self, key: str) -> None:
        with self._lock:
            self._suppressed[key] += 1

    def pop_suppressed(self, key: str) -> int:
        with self._lock:
            value = int(self._suppressed.get(key, 0) or 0)
            if key in self._suppressed:
                self._suppressed[key] = 0
            return value

    def maybe_emit_summary(self, logger: logging.Logger, *, interval_seconds: float = 60.0, top_n: int = 10) -> None:
        """Emit periodic aggregate suppression summary."""
        now = time.monotonic()
        with self._lock:
            if self._summary_last_emit_mono > 0 and (now - self._summary_last_emit_mono) < interval_seconds:
                return
            self._summary_last_emit_mono = now
            pending = {k: v for k, v in self._suppressed.items() if int(v) > 0}
            for key in pending:
                self._suppressed[key] = 0
        if not pending:
            return
        top = sorted(pending.items(), key=lambda item: item[1], reverse=True)[: max(1, int(top_n))]
        total_suppressed = sum(int(v) for v in pending.values())
        keys = ",".join(f"{k}:{v}" for k, v in top)
        logger.info(
            "LOG_THROTTLE_SUMMARY total_suppressed=%s top_keys=%s",
            total_suppressed,
            keys,
            extra={"event": "LOG_THROTTLE_SUMMARY", "total_suppressed": total_suppressed, "top_keys": keys, "keys_count": len(top)},
        )


DEFAULT_LOG_THROTTLE = LogThrottle()


def log_throttled(
    logger: logging.Logger,
    level: int,
    event: str,
    key: str,
    interval_seconds: float,
    message: str,
    *args: Any,
    **kwargs: Any,
) -> bool:
    """Emit throttled log with suppressed_count from previous interval."""
    throttle: LogThrottle = kwargs.pop("throttle", DEFAULT_LOG_THROTTLE)
    extra = dict(kwargs.pop("extra", {}) or {})
    if throttle.should_log(key, interval_seconds):
        suppressed = throttle.pop_suppressed(key)
        extra.setdefault("event", event)
        extra["suppressed_count"] = suppressed
        logger.log(level, message, *args, extra=extra, **kwargs)
        enabled = os.getenv("LOG_THROTTLE_SUMMARY_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
        if enabled:
            throttle.maybe_emit_summary(
                logger,
                interval_seconds=float(os.getenv("LOG_THROTTLE_SUMMARY_SECONDS", "120") or "120"),
                top_n=int(os.getenv("LOG_THROTTLE_SUMMARY_TOP_N", "10") or "10"),
            )
        return True
    throttle.record_suppressed(key)
    return False
