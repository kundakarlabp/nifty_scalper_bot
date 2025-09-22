from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any, Deque, Dict, Optional

try:  # pragma: no cover - during tests settings may be absent
    from src.config import settings
except Exception:  # pragma: no cover
    settings = None  # type: ignore

log = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Return current time as an aware UTC ``datetime``."""
    return datetime.now(timezone.utc)


def _as_aware_utc(dt: datetime) -> datetime:
    """Convert ``dt`` to an aware UTC ``datetime``."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class CircuitBreaker:
    """Simple latency/error rate based circuit breaker.

    The breaker tracks a rolling window of request outcomes and transitions
    between ``CLOSED`` → ``OPEN`` → ``HALF_OPEN`` depending on error rate and
    p95 latency thresholds.
    """

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"

    def __init__(self, name: str, **cfg: float) -> None:
        env = {
            "error_rate_threshold": float(
                getattr(settings, "cb_error_rate", 0.10) if settings else 0.10
            ),
            "latency_p95_ms": int(
                getattr(settings, "cb_p95_ms", 1200) if settings else 1200
            ),
            "min_samples": int(
                getattr(settings, "cb_min_samples", 30) if settings else 30
            ),
            "open_cooldown_sec": int(
                getattr(settings, "cb_open_cooldown_sec", 30) if settings else 30
            ),
            "half_open_probe": int(
                getattr(settings, "cb_half_open_probe", 3) if settings else 3
            ),
            "window_size": 200,
        }
        env.update(cfg)

        self.name = name
        self.error_rate_threshold = float(env["error_rate_threshold"])
        self.latency_p95_ms = int(env["latency_p95_ms"])
        self.min_samples = int(env["min_samples"])
        self.open_cooldown_sec = int(env["open_cooldown_sec"])
        self.half_open_probe = int(env["half_open_probe"])
        self.window_size = int(env.get("window_size", 200))

        self.state: str = self.CLOSED
        self._last_state: str = self.state
        self._last_logged_state: str = self.state
        self._latencies: Deque[int] = deque(maxlen=self.window_size)
        self._errors = 0
        self._total = 0
        self.open_until: Optional[datetime] = (
            None  # aware UTC time when breaker can half-open
        )
        self._half_open_successes = 0
        self.last_reason: str = ""

    # ----- helpers -----
    def _p95(self) -> int:
        if not self._latencies:
            return 0
        arr = sorted(self._latencies)
        idx = max(0, int(0.95 * len(arr)) - 1)
        return arr[idx]

    def _err_rate(self) -> float:
        return (self._errors / self._total) if self._total else 0.0

    def allow(self) -> bool:
        """Return ``True`` if calls are permitted."""
        return self.state != self.OPEN

    # ----- recording -----
    def record_success(self, latency_ms: int) -> None:
        """Record a successful call latency."""
        self._latencies.append(int(latency_ms))
        self._total += 1
        if self.state == self.HALF_OPEN:
            self._half_open_successes += 1
            if self._half_open_successes >= self.half_open_probe:
                self._transition(
                    self.HALF_OPEN,
                    self.CLOSED,
                    reason=f"probe={self._half_open_successes}/{self.half_open_probe}",
                )
                self._reset_metrics()
        else:
            self._evaluate(latency_ms)

    def record_failure(self, latency_ms: int, reason: str = "") -> None:
        """Record a failed call latency and reason."""
        self._latencies.append(int(latency_ms))
        self._total += 1
        self._errors += 1
        self.last_reason = reason
        if self.state == self.HALF_OPEN:
            self._transition(self.HALF_OPEN, self.OPEN, reason=reason)
            self.open_until = _utcnow() + timedelta(seconds=self.open_cooldown_sec)
            self._half_open_successes = 0
        else:
            self._evaluate(latency_ms, reason)

    def _evaluate(self, latency_ms: int, reason: str = "") -> None:
        if self.state != self.CLOSED:
            return
        if self._total < self.min_samples:
            return
        err_rate = self._err_rate()
        p95 = self._p95()
        if err_rate > self.error_rate_threshold or p95 > self.latency_p95_ms:
            self._transition(
                self.CLOSED, self.OPEN, err_rate=err_rate, p95=p95, reason=reason
            )
            self.open_until = _utcnow() + timedelta(seconds=self.open_cooldown_sec)

    def _transition(self, src: str, dst: str, **extra: Any) -> None:
        if dst == self.state:
            return

        if dst != getattr(self, "_last_logged_state", None):
            log.info(
                "CB[%s] %s→%s err=%.1f%% p95=%sms n=%s reason=%s",
                self.name,
                src,
                dst,
                self._err_rate() * 100.0,
                self._p95(),
                self._total,
                extra.get("reason", ""),
            )
            self._last_logged_state = dst

        self._last_state = dst
        self.state = dst

    def _reset_metrics(self) -> None:
        self._latencies.clear()
        self._errors = 0
        self._total = 0
        self.last_reason = ""

    # ----- house keeping -----
    def tick(self, now: datetime | None = None) -> None:
        """Advance internal timers (OPEN→HALF_OPEN)."""
        now = _utcnow() if now is None else _as_aware_utc(now)
        if self.state == self.OPEN and self.open_until and now >= self.open_until:
            self._transition(self.OPEN, self.HALF_OPEN, reason="cooldown_done")
            self._half_open_successes = 0

    def health(self) -> Dict[str, object]:
        """Return current breaker metrics."""
        return {
            "state": self.state,
            "err_rate": self._err_rate(),
            "p95_ms": self._p95(),
            "n": self._total,
            "open_until": self.open_until.isoformat() if self.open_until else None,
            "last_reason": self.last_reason,
        }

    def force_open(self, seconds: int) -> None:
        """Force breaker to OPEN for ``seconds`` seconds."""
        self.state = self.OPEN
        self._last_state = self.state
        self._last_logged_state = self.state
        self.open_until = _utcnow() + timedelta(seconds=max(0, int(seconds)))
        self.last_reason = "forced_open"

    def reset(self) -> None:
        """Reset breaker to CLOSED and clear metrics."""
        self.state = self.CLOSED
        self._last_state = self.state
        self._last_logged_state = self.state
        self.open_until = None
        self._half_open_successes = 0
        self._reset_metrics()
