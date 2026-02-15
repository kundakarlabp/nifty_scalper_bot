"""Coordinator for polling-based market data streaming."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from enum import Enum
import threading
import time
from typing import Any, Callable, Iterable, Sequence

from nifty_scalper_bot.utils.logging import get_logger
from nifty_scalper_bot.utils.symbols import unique_normalized_symbols

LOG = get_logger(__name__)


class ConnectionLifecycleState(str, Enum):
    """Connection lifecycle state for singleton connect behavior."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    BACKOFF = "backoff"


@dataclass(slots=True)
class StreamHealth:
    """Structured telemetry describing the polling supervisor state."""

    running: bool
    tokens: int
    uptime_s: float | None
    last_tick_ts: float | None
    last_tick_age_s: float | None
    consecutive_failures: int
    restarts: int
    last_error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a dictionary representation for serialization."""

        return {
            "running": self.running,
            "tokens": self.tokens,
            "uptime_s": self.uptime_s,
            "last_tick_ts": self.last_tick_ts,
            "last_tick_age_s": self.last_tick_age_s,
            "consecutive_failures": self.consecutive_failures,
            "restarts": self.restarts,
            "last_error": self.last_error,
        }


class StreamSupervisor:
    """Single source of truth for polling lifecycle, health, and tokens."""

    def __init__(
        self,
        *,
        streamer: Any,
        resolver: Any,
        default_symbols: Sequence[str] | None = None,
        autostart: bool = True,
        monitor_interval_s: float = 300.0,
        risk_halt_getter: Callable[[], bool] | None = None,
    ) -> None:
        self.streamer = streamer
        self.resolver = resolver
        self._default_symbols = unique_normalized_symbols(default_symbols or [])
        self._autostart = bool(autostart)
        self._monitor_interval = max(float(monitor_interval_s), 0.2)
        self._risk_halt_getter = risk_halt_getter

        self._tokens: set[int] = set()
        self._lock = threading.RLock()
        self._monitor_stop = threading.Event()
        self._monitor_thread: threading.Thread | None = None

        self._started_at_mono: float | None = None
        self._last_tick_wall: float | None = None
        self._last_tick_mono: float | None = None
        self._consecutive_failures = 0
        self._start_count = 0
        self._last_error: str | None = None
        self._risk_halt_restart_logged = False
        self._started = False
        self._connection_state = ConnectionLifecycleState.DISCONNECTED
        self._connection_lock = threading.Lock()
        self._backoff_seconds = 2.0
        self._backoff_until_mono = 0.0

    # ------------------------------------------------------------------
    # Lifecycle helpers
    def bootstrap(self) -> None:
        """Subscribe default symbols and start the poller if configured."""

        symbols = list(self._default_symbols) or ["NIFTY"]
        self.subscribe_symbols(symbols)
        if self._autostart:
            self.ensure_started()

    def ensure_started(self) -> bool:
        """Start the poller when tokens are present."""

        with self._lock:
            if not self._tokens:
                return False
        return self.start()

    def start(self) -> bool:
        """Start the underlying streamer if not already running."""

        with self._connection_lock:
            now = time.monotonic()
            if self._connection_state in {
                ConnectionLifecycleState.CONNECTING,
                ConnectionLifecycleState.BACKOFF,
            }:
                return False
            if (
                self._connection_state == ConnectionLifecycleState.CONNECTED
                and self.is_running()
            ):
                return True
            if now < self._backoff_until_mono:
                self._connection_state = ConnectionLifecycleState.BACKOFF
                return False
            self._connection_state = ConnectionLifecycleState.CONNECTING
            with self._lock:
                if self._started:
                    self._connection_state = ConnectionLifecycleState.CONNECTED
                    return True
            if self.is_running():
                self._connection_state = ConnectionLifecycleState.CONNECTED
                LOG.debug(
                    "stream_supervisor_start_skipped",
                    extra={
                        "event": "stream_supervisor_start_skipped",
                        "reason": "already_running",
                    },
                )
                return True
            try:
                self.streamer.start()
            except Exception as exc:  # noqa: BLE001 - defensive guard
                with self._lock:
                    self._consecutive_failures += 1
                    self._last_error = str(exc)
                self._backoff_until_mono = time.monotonic() + self._backoff_seconds
                self._connection_state = ConnectionLifecycleState.BACKOFF
                LOG.error(
                    "stream_supervisor_start_failed",
                    extra={
                        "event": "stream_supervisor_start_failed",
                        "error": str(exc),
                    },
                )
                return False
            with self._lock:
                self._started = True
                self._started_at_mono = time.monotonic()
                self._consecutive_failures = 0
                self._last_error = None
                self._start_count += 1
            self._backoff_until_mono = 0.0
            self._connection_state = ConnectionLifecycleState.CONNECTED
        self._ensure_monitor_thread()
        LOG.info(
            "stream_supervisor_started", extra={"event": "stream_supervisor_started"}
        )
        return True

    def stop(self) -> None:
        """Stop the polling thread and monitor."""

        self._monitor_stop.set()
        monitor = self._monitor_thread
        if monitor is not None:
            monitor.join(timeout=2.0)
        self._monitor_thread = None
        try:
            self.streamer.stop()
        except Exception as exc:  # noqa: BLE001 - defensive guard
            LOG.warning(
                "stream_supervisor_stop_failed",
                extra={"event": "stream_supervisor_stop_failed", "error": str(exc)},
            )
        with self._lock:
            self._started = False
            self._started_at_mono = None
        with self._connection_lock:
            self._connection_state = ConnectionLifecycleState.DISCONNECTED
            self._backoff_until_mono = 0.0

    def is_running(self) -> bool:
        """Return ``True`` when the underlying streamer reports running."""

        try:
            return bool(self.streamer.is_running())
        except Exception:  # noqa: BLE001 - defensive guard
            return False

    # ------------------------------------------------------------------
    # Subscriptions
    def subscribe_symbols(self, symbols: Iterable[str]) -> int:
        """Resolve *symbols* into tokens and subscribe."""

        tokens, _, _ = self.resolve_symbols(symbols)
        return self.subscribe_tokens(tokens)

    def resolve_symbols(
        self, symbols: Iterable[str]
    ) -> tuple[list[int], list[str], dict[str, int]]:
        """Resolve symbols into tokens.

        Returns a tuple of ``(tokens, unresolved, mapping)`` where ``mapping`` maps
        upper-cased symbols to resolved instrument tokens.
        """

        cleaned = unique_normalized_symbols(symbols or [])
        if not cleaned:
            return ([], [], {})

        resolved: list[int] = []
        unresolved: list[str] = []
        resolved_map: dict[str, int] = {}
        resolver_many = getattr(self.resolver, "resolve_many", None)
        if callable(resolver_many):
            try:
                many = resolver_many(cleaned)
            except Exception as exc:  # noqa: BLE001 - defensive resolution
                LOG.warning(
                    "stream_supervisor_resolve_many_failed",
                    extra={
                        "event": "stream_supervisor_resolve_many_failed",
                        "error": str(exc),
                    },
                )
            else:
                for symbol, token in zip(cleaned, many, strict=False):
                    if token is None:
                        unresolved.append(symbol)
                    else:
                        resolved_value = int(token)
                        resolved.append(resolved_value)
                        resolved_map[symbol] = resolved_value
                return (resolved, unresolved, resolved_map)

        resolve_one = getattr(self.resolver, "resolve", None)
        for symbol in cleaned:
            if not callable(resolve_one):
                unresolved.append(symbol)
                continue
            try:
                token = resolve_one(symbol)
            except Exception as exc:  # noqa: BLE001 - defensive
                LOG.debug(
                    "stream_supervisor_resolve_failed",
                    extra={
                        "event": "stream_supervisor_resolve_failed",
                        "symbol": symbol,
                        "error": str(exc),
                    },
                )
                unresolved.append(symbol)
                continue
            if token is None:
                unresolved.append(symbol)
            else:
                resolved_value = int(token)
                resolved.append(resolved_value)
                resolved_map[symbol] = resolved_value
        return (resolved, unresolved, resolved_map)

    def subscribe_tokens(self, tokens: Iterable[int]) -> int:
        """Subscribe the provided tokens and return the new total size."""

        additions = {int(token) for token in tokens or []}
        if not additions:
            return self.token_count()
        with self._lock:
            new_tokens = additions - self._tokens
            if new_tokens:
                self._tokens.update(new_tokens)
        if new_tokens:
            payload = sorted(new_tokens)
            try:
                subscribe = getattr(self.streamer, "subscribe_tokens", None)
                if callable(subscribe):
                    subscribe(payload)
                else:
                    self.streamer.subscribe(payload)
            except Exception as exc:  # noqa: BLE001 - defensive guard
                LOG.warning(
                    "stream_supervisor_subscribe_failed",
                    extra={
                        "event": "stream_supervisor_subscribe_failed",
                        "tokens": payload,
                        "error": str(exc),
                    },
                )
        if self._autostart:
            self.ensure_started()
        return len(self._tokens)

    def unsubscribe_tokens(self, tokens: Iterable[int]) -> int:
        """Unsubscribe the provided tokens and return the remaining size."""

        removals = {int(token) for token in tokens or []}
        if not removals:
            return self.token_count()
        with self._lock:
            has_overlap = bool(self._tokens & removals)
            if has_overlap:
                self._tokens -= removals
        payload = sorted(removals)
        try:
            unsubscribe = getattr(self.streamer, "unsubscribe_tokens", None)
            if callable(unsubscribe):
                unsubscribe(payload)
            else:
                self.streamer.unsubscribe(payload)
        except Exception as exc:  # noqa: BLE001 - defensive guard
            LOG.warning(
                "stream_supervisor_unsubscribe_failed",
                extra={
                    "event": "stream_supervisor_unsubscribe_failed",
                    "tokens": payload,
                    "error": str(exc),
                },
            )
        if not self._tokens:
            with suppress(Exception):
                self.streamer.stop()
            with self._lock:
                self._started_at_mono = None
        return len(self._tokens)

    # ------------------------------------------------------------------
    # Telemetry helpers
    def token_count(self) -> int:
        """Return the number of tracked tokens."""

        with self._lock:
            return len(self._tokens)

    def tracked_tokens(self) -> list[int]:
        """Return the tracked tokens in sorted order."""

        with self._lock:
            return sorted(self._tokens)

    def on_tick(self, _tick: dict[str, Any]) -> None:
        """Record the last tick timestamps for health accounting."""

        now_wall = time.time()
        now_mono = time.monotonic()
        with self._lock:
            self._last_tick_wall = now_wall
            self._last_tick_mono = now_mono
            self._consecutive_failures = 0

    def get_health(self) -> StreamHealth:
        """Return the current supervisor health snapshot."""

        running = self.is_running()
        with self._lock:
            tokens = len(self._tokens)
            started_at = self._started_at_mono
            last_tick_wall = self._last_tick_wall
            last_tick_mono = self._last_tick_mono
            failures = self._consecutive_failures
            start_count = self._start_count
            last_error = self._last_error
        uptime = None
        if running and started_at is not None:
            uptime = max(0.0, time.monotonic() - started_at)
        last_tick_age = None
        if last_tick_mono is not None:
            last_tick_age = max(0.0, time.monotonic() - last_tick_mono)
        restarts = max(0, start_count - 1)
        return StreamHealth(
            running=running,
            tokens=tokens,
            uptime_s=uptime,
            last_tick_ts=last_tick_wall,
            last_tick_age_s=last_tick_age,
            consecutive_failures=failures,
            restarts=restarts,
            last_error=last_error,
        )

    def status_line(self) -> str:
        """Return a concise human-friendly status string."""

        health = self.get_health()
        heart = "💓 running" if health.running else "⛔ stopped"
        interval = float(getattr(self.streamer, "_interval_s", 0.0) or 0.0)
        uptime = (
            f"up={health.uptime_s:.0f}s" if health.uptime_s is not None else "up=n/a"
        )
        last_tick = (
            f"last={health.last_tick_age_s:.1f}s"
            if health.last_tick_age_s is not None
            else "last=n/a"
        )
        extras = [uptime, last_tick]
        if health.consecutive_failures > 0:
            extras.append(f"fails={health.consecutive_failures}")
        if health.restarts > 0:
            extras.append(f"restarts={health.restarts}")
        return (
            f"tokens={health.tokens} @{interval:.2f}s {' '.join(extras)} {heart}"
        ).strip()

    def status(self) -> str:
        """Compatibility alias returning :meth:`status_line`."""

        return self.status_line()

    def snapshot(self) -> dict[str, Any]:
        """Return a dictionary snapshot of health and configuration."""

        health = self.get_health()
        data = health.as_dict()
        data["status"] = self.status_line()
        data["interval_s"] = float(getattr(self.streamer, "_interval_s", 0.0) or 0.0)
        data["batch_size"] = int(getattr(self.streamer, "_batch_size", 0) or 0)
        return data

    # ------------------------------------------------------------------
    # Internal helpers
    @staticmethod
    def _is_trading_window() -> bool:
        """Check if current time is within NSE trading window (Mon-Fri 09:00-15:45 IST)."""
        from datetime import datetime, time as dt_time
        from zoneinfo import ZoneInfo

        ist = ZoneInfo("Asia/Kolkata")
        now_ist = datetime.now(ist)
        if now_ist.weekday() >= 5:
            return False
        t = now_ist.time()
        return dt_time(9, 0) <= t <= dt_time(15, 45)

    def _ensure_monitor_thread(self) -> None:
        thread = self._monitor_thread
        if thread is not None and thread.is_alive():
            return
        self._monitor_stop.clear()
        monitor = threading.Thread(
            target=self._monitor_loop,
            name="stream-supervisor-monitor",
            daemon=True,
        )
        self._monitor_thread = monitor
        monitor.start()

    def _monitor_loop(self) -> None:
        while not self._monitor_stop.wait(self._monitor_interval):
            if not self._autostart:
                continue
            # ✅ FIX: Skip restart attempts outside trading hours
            if not self._is_trading_window():
                continue
            with self._lock:
                active_tokens = bool(self._tokens)
            if not active_tokens:
                continue
            if self.is_running():
                with self._lock:
                    self._consecutive_failures = 0
                with self._connection_lock:
                    self._connection_state = ConnectionLifecycleState.CONNECTED
                    self._backoff_until_mono = 0.0
                self._risk_halt_restart_logged = False
                continue
            risk_halt_active = False
            if self._risk_halt_getter is not None:
                try:
                    risk_halt_active = bool(self._risk_halt_getter())
                except Exception:  # noqa: BLE001 - defensive callback surface
                    risk_halt_active = False
            if risk_halt_active:
                # During a global risk halt we intentionally avoid transport churn,
                # because protective lifecycle handling remains active without restarts.
                if not self._risk_halt_restart_logged:
                    self._risk_halt_restart_logged = True
                    LOG.warning(
                        "Stream restart suppressed due to risk halt",
                        extra={"event": "stream_supervisor_restart_suppressed"},
                    )
                continue
            self._risk_halt_restart_logged = False
            LOG.warning(
                "stream_supervisor_restart",
                extra={"event": "stream_supervisor_restart"},
            )
            restarted = self.start()
            if not restarted:
                with self._lock:
                    self._consecutive_failures += 1
                with self._connection_lock:
                    if self._connection_state != ConnectionLifecycleState.CONNECTED:
                        self._connection_state = ConnectionLifecycleState.BACKOFF


__all__ = ["StreamSupervisor", "StreamHealth"]
