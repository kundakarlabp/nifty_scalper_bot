"""WebSocket manager with watchdog, reconnect logic, subscriptions, and
flexible SDK adapters."""

from __future__ import annotations

from collections import defaultdict
from enum import Enum
import random
import threading
import time
import types
from typing import Any, Callable, Iterable

from nifty_scalper_bot.utils.errors import WebSocketError
from nifty_scalper_bot.utils.logging import get_logger
from nifty_scalper_bot.utils.metrics import Counter, Gauge


class ConnectionState(Enum):
    """Enumeration describing the websocket connection lifecycle."""

    # Keep enum values stable for metrics/health endpoints
    DISCONNECTED = 0
    CONNECTING = 1
    CONNECTED = 2
    ERROR = 3
    CLOSING = 4
    IDLE = DISCONNECTED  # Backwards compatibility alias


class HandshakeTimeoutError(WebSocketError):
    """Raised when the websocket handshake stalls beyond tolerance."""

    def __init__(self, elapsed: float, hb_delta: float) -> None:
        message = (
            f"Handshake timeout after {elapsed:.1f}s (hbΔ={hb_delta:.1f}s). "
            "Verify broker session/token validity."
        )
        super().__init__(message)
        self.code = 1006  # type: ignore[attr-defined]
        self.reason = "handshake_timeout"  # type: ignore[attr-defined]


class WebSocketManager:
    """Manage broker WebSocket callbacks with stale detection and fan-out."""

    def __init__(
        self,
        broker_ws_client: Any,
        on_tick: Callable[[dict[str, Any]], None],
        on_error: Callable[[Exception], None] | None = None,
        *,
        heartbeat_sec: float = 10.0,
        stale_sec: float = 30.0,
        backoff_min_sec: float = 5.0,
        backoff_max_sec: float = 120.0,
        backoff_factor: float = 2.0,
        backoff_jitter: float = 0.25,
        breaker_threshold: int = 5,
        breaker_open_sec: float = 180.0,
    ) -> None:
        self._client = broker_ws_client
        self._on_tick = on_tick
        self._user_on_error = on_error
        self._heartbeat_sec = heartbeat_sec
        self._stale_sec = stale_sec
        self._logger = get_logger(__name__)
        self._last_heartbeat = 0.0
        self._connecting_since = 0.0
        self._last_connect_attempt = 0.0
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        self._state_lock = threading.Lock()
        self._started = False
        self._watchdog_thread: threading.Thread | None = None
        self._state = ConnectionState.DISCONNECTED
        self._downtime_started = time.monotonic()
        self._subscription_tokens: set[Any] = set()
        self._subscriptions: dict[Any, str] = {}
        self._reconnect_lock = threading.Lock()
        self._build_client_fn: Callable[[], Any] = lambda: broker_ws_client
        self._configure_callbacks()
        # Reconnect & breaker state
        self._reconnect_failures = 0
        self._handshake_failures = 0
        self._backoff_min = float(backoff_min_sec)
        self._backoff_max = float(backoff_max_sec)
        self._backoff_factor = float(backoff_factor)
        self._backoff_jitter = float(backoff_jitter)
        self._breaker_threshold = max(1, int(breaker_threshold))
        self._breaker_open_sec = float(breaker_open_sec)
        self._breaker_open_until = 0.0
        self._backoff_base = max(
            1.0, self._backoff_min if self._backoff_min > 0 else 1.0
        )
        self._backoff_s = self._backoff_base
        self._max_backoff_s = max(
            self._backoff_base, self._backoff_max if self._backoff_max > 0 else 60.0
        )
        self._refresh_hook = getattr(broker_ws_client, "refresh_session", None)
        self._last_error: Exception | None = None
        self._reconnect_timer: threading.Timer | None = None
        self._handshake_timeout = 15.0
        self._handshake_hb_timeout = 10.0

        # Metrics
        self._m_reconnects = Counter(
            "ws_reconnect_attempts_total", "Number of websocket reconnect attempts"
        )
        self._m_reconnect_skipped = Counter(
            "ws_reconnect_skipped_total",
            "Reconnect attempts skipped due to breaker or shutdown",
        )
        self._m_errors = Counter("ws_errors_total", "WebSocket errors", labels=["code"])
        self._m_state = Gauge(
            "ws_state",
            "WebSocket state (0=disconnected, 1=connecting, 2=connected)",
        )
        try:
            self._m_state.set(float(ConnectionState.DISCONNECTED.value))
        except Exception:  # pragma: no cover - optional metrics
            pass
        self._m_handshake_failures = Counter(
            "ws_handshake_failures_total",
            "Number of websocket handshake failures",
        )
        self._user_on_connect: Callable[[], None] | None = None
        self._user_on_disconnect: Callable[[], None] | None = None

    def set_client_factory(self, fn: Callable[[], Any]) -> None:
        """Inject a callable that returns a fresh websocket client."""

        with self._lock:
            self._build_client_fn = fn

    def set_callbacks(
        self,
        *,
        on_connect: Callable[[], None] | None = None,
        on_disconnect: Callable[[], None] | None = None,
    ) -> None:
        """Register lifecycle callbacks for connect/disconnect events."""

        self._user_on_connect = on_connect
        self._user_on_disconnect = on_disconnect

    @property
    def on_tick(self) -> Callable[[dict[str, Any]], None]:
        """Return the currently registered tick callback."""

        return self._on_tick

    @on_tick.setter
    def on_tick(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Update the tick callback used to dispatch ticks."""

        self._on_tick = callback

    # ------------------------------------------------------------------
    # Client helpers
    def _client_is_connected(self) -> bool:
        if self._state != ConnectionState.CONNECTED:
            return False
        checker = getattr(self._client, "is_connected", None)
        if callable(checker):
            try:
                return bool(checker())
            except Exception:  # pragma: no cover - defensive
                return True
        return True

    def subscribe(self, tokens: Iterable[Any]) -> None:
        """Subscribe raw tokens without touching mode information."""

        with self._lock:
            new_tokens = [
                token
                for token in tokens
                if token is not None and token not in self._subscription_tokens
            ]
            if not new_tokens:
                return
            self._subscription_tokens.update(new_tokens)
            for token in new_tokens:
                self._subscriptions.setdefault(token, "ltp")
            if (
                self._client_is_connected()
                and getattr(self._client, "subscribe", None) is not None
            ):
                self._client.subscribe(list(new_tokens))

    def _configure_callbacks(self) -> None:
        # Always route SDK callbacks through adapters that tolerate arity/shape
        # differences.
        try:
            self._client.on_ticks = self._handle_tick
        except Exception:  # pragma: no cover - some clients expose only on_tick
            self._logger.debug("WS client rejected on_ticks assignment", exc_info=True)

        if hasattr(self._client, "on_tick"):
            try:
                self._client.on_tick = self._handle_tick
            except Exception:  # pragma: no cover - fallback defensive
                self._logger.debug(
                    "WS client rejected on_tick assignment", exc_info=True
                )

        def _adapter_on_error(*args: Any, **kwargs: Any) -> None:
            code = kwargs.get("code")
            reason = kwargs.get("reason")
            exc: Exception | None = kwargs.get("exc")

            for value in reversed(args):
                if isinstance(value, Exception):
                    exc = exc or value
                    code = code if code is not None else getattr(value, "code", None)
                    reason = (
                        reason if reason is not None else getattr(value, "reason", None)
                    )
                    break

            if code is None and len(args) >= 2:
                code, reason = args[-2], args[-1]
            elif code is None and args:
                code = args[-1]

            self._logger.error("WebSocket error: code=%s reason=%s", code, reason)
            payload: Any = (
                exc
                if isinstance(exc, Exception)
                else types.SimpleNamespace(code=code, reason=reason)
            )
            self._handle_error(payload, code=code, reason=reason, exc=exc)

        def _adapter_on_close(*args: Any, **kwargs: Any) -> None:
            code = kwargs.get("code")
            reason = kwargs.get("reason")
            self._logger.info("WebSocket closed: code=%s reason=%s", code, reason)
            _adapter_on_error(*args, **kwargs)

        try:
            self._client.on_error = _adapter_on_error
        except Exception:  # pragma: no cover - some clients block assignment
            pass

        if getattr(self._client, "on_close", None) is not None:
            try:
                self._client.on_close = _adapter_on_close
            except Exception:  # pragma: no cover - some clients block assignment
                pass

        if getattr(self._client, "on_reconnect", None) is not None:

            def _adapter_on_reconnect(*_: Any, **__: Any) -> None:
                self._transition_state(ConnectionState.CONNECTING)

            try:
                self._client.on_reconnect = _adapter_on_reconnect
            except Exception:  # pragma: no cover - some clients block assignment
                pass

        if getattr(self._client, "on_open", None) is not None:

            def _adapter_on_open(*_: Any, **__: Any) -> None:
                self._transition_state(ConnectionState.CONNECTED)
                try:
                    self._resubscribe_all()
                except Exception as exc:  # noqa: BLE001
                    self._logger.warning("Resubscribe on open failed: %s", exc)

            try:
                self._client.on_open = _adapter_on_open
            except Exception:  # pragma: no cover - some clients block assignment
                pass

        if getattr(self._client, "on_connect", None) is not None:
            try:
                self._client.on_connect = self._on_connect
            except Exception:  # pragma: no cover - some clients block assignment
                pass

    def start(self) -> None:
        # Guard against duplicate starts from different call sites
        with self._state_lock:
            if self._started:
                self._logger.debug(
                    "WebSocket start called but already running; ignoring"
                )
                return
            self._started = True
        self._stop_event.clear()
        if self._watchdog_thread is None or not self._watchdog_thread.is_alive():
            self._watchdog_thread = threading.Thread(
                target=self._watchdog_loop,
                daemon=True,
                name="ws-watchdog",
            )
            self._watchdog_thread.start()
        self._reconnect_failures = 0
        self._handshake_failures = 0
        self._breaker_open_until = 0.0
        self._backoff_s = self._backoff_base
        self._transition_state(ConnectionState.CONNECTING)
        self._logger.info("Starting WebSocket connection")
        self._last_heartbeat = time.monotonic()
        self._connecting_since = self._last_heartbeat
        self._last_connect_attempt = self._last_heartbeat
        self.refresh_session()
        try:
            self._connect_client()
        except Exception as exc:  # noqa: BLE001
            self._last_error = exc
            self._logger.error("Initial WS connect failed: %s", exc)
            self._schedule_reconnect(initial=True)

    def _on_connect(self, *_: Any, **__: Any) -> None:
        self._last_heartbeat = time.monotonic()
        self._last_error = None
        self._handshake_failures = 0
        self._connecting_since = 0.0
        self._transition_state(ConnectionState.CONNECTED)
        if not self._subscriptions:
            return
        try:
            self._resubscribe_all()
            self._logger.info("Resubscribed %d tokens", len(self._subscription_tokens))
        except Exception:  # noqa: BLE001
            self._logger.exception("Resubscribe failed on connect")
            self._force_reconnect()

    def stop(self) -> None:
        self._logger.info("Stopping WebSocket connection")
        self._stop_event.set()
        with self._state_lock:
            self._started = False
        self._cancel_reconnect_timer()
        if self._watchdog_thread is not None:
            self._watchdog_thread.join(timeout=self._heartbeat_sec)
        self._transition_state(ConnectionState.CLOSING)
        with self._lock:
            try:
                closer = getattr(self._client, "close", None)
                if callable(closer):
                    closer()
                else:
                    disconnector = getattr(self._client, "disconnect", None)
                    if callable(disconnector):
                        disconnector()
            except Exception as exc:  # noqa: BLE001
                self._logger.error("Error while disconnecting WebSocket: %s", exc)
                raise WebSocketError("Failed to disconnect WebSocket") from exc
            finally:
                self._transition_state(ConnectionState.DISCONNECTED)
                self._connecting_since = 0.0
                self._last_heartbeat = 0.0

    def _handle_tick(self, payload: Any) -> None:
        data: dict[str, Any]
        if isinstance(payload, dict):
            data = payload
        elif isinstance(payload, list) and payload and isinstance(payload[0], dict):
            data = payload[0]
        else:
            self._logger.debug("Ignoring non-dict WebSocket payload: %s", payload)
            return
        self._last_heartbeat = time.monotonic()
        self._transition_state(ConnectionState.CONNECTED)
        try:
            self._on_tick(data)
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("Tick handler raised an error: %s", exc)

    def _handle_error(
        self,
        *args: Any,
        code: Any | None = None,
        reason: Any | None = None,
        exc: Exception | None = None,
    ) -> None:
        """Central error handler that normalizes SDK callbacks for backoff logic."""

        err_obj: Exception | None = exc
        if not err_obj and args:
            first = args[0]
            if isinstance(first, Exception):
                err_obj = first
                code = code if code is not None else getattr(first, "code", None)
                reason = (
                    reason if reason is not None else getattr(first, "reason", None)
                )
            elif len(args) >= 2:
                code = code if code is not None else args[-2]
                reason = reason if reason is not None else args[-1]
            elif len(args) == 1:
                code = code if code is not None else args[0]

        if err_obj is None:

            class _WSExc(Exception):
                """Synthetic websocket exception populated from raw callback data."""

                pass

            message = f"ws_error code={code!r} reason={reason!r}"
            err_obj = _WSExc(message)
            setattr(err_obj, "code", code)
            setattr(err_obj, "reason", reason)

        code_attr = getattr(err_obj, "code", code)
        try:
            label = str(code_attr) if code_attr is not None else type(err_obj).__name__
            self._m_errors.labels(label).inc()
        except Exception:  # pragma: no cover - optional metrics
            pass

        reason_attr = getattr(err_obj, "reason", reason)
        self._logger.error(
            "WebSocket error handled: code=%s reason=%s", code_attr, reason_attr
        )

        self._last_error = err_obj

        if code_attr == 1006:
            self._reconnect_failures = max(self._reconnect_failures, 2)

        self._connecting_since = 0.0
        self._transition_state(ConnectionState.ERROR)
        self._increase_backoff()

        if self._user_on_error is not None:
            try:
                self._user_on_error(err_obj)
            except Exception as callback_error:  # noqa: BLE001
                self._logger.exception("Error handler raised: %s", callback_error)

        label = "error"
        if code_attr is not None:
            label = f"error:{code_attr}"
        elif reason_attr:
            label = f"error:{reason_attr}"
        self._force_reconnect(reason=label)

    def _watchdog_loop(self) -> None:
        while not self._stop_event.is_set():
            if self._stop_event.wait(self._heartbeat_sec):
                break
            now = time.monotonic()
            self._detect_stuck_connecting(now)
            if (
                self._state == ConnectionState.CONNECTED
                and self._last_heartbeat > 0.0
                and (now - self._last_heartbeat) > self._stale_sec
            ):
                self._logger.warning(
                    "WS heartbeat stale for %.1fs -> forcing reconnect",
                    now - self._last_heartbeat,
                )
                self._force_reconnect()

    def _detect_stuck_connecting(self, now: float) -> None:
        if self._state != ConnectionState.CONNECTING or self._connecting_since <= 0.0:
            return
        elapsed = now - self._connecting_since
        hb_delta = now - self._last_heartbeat if self._last_heartbeat > 0 else elapsed
        if elapsed < self._handshake_timeout and hb_delta < self._handshake_hb_timeout:
            return

        self._handshake_failures += 1
        self._last_error = HandshakeTimeoutError(elapsed, hb_delta)
        self._logger.warning(
            "WS handshake stuck; forcing reconnect",
            extra={
                "event": "ws_stuck_connecting",
                "elapsed_s": round(elapsed, 3),
                "hb_delta_s": round(hb_delta, 3),
                "failures": self._handshake_failures,
            },
        )
        try:
            self._m_handshake_failures.inc()
        except Exception:  # pragma: no cover - optional metrics
            pass
        self._increase_backoff()
        self.refresh_session()
        self._transition_state(ConnectionState.ERROR)
        self._connecting_since = 0.0
        self._force_reconnect(reason="handshake_timeout")

    def reconnect(self) -> None:
        """Force manual reconnection (e.g. from Telegram /ws command)."""

        self._logger.info("Manual reconnect triggered via command")
        self._force_reconnect(reason="manual")

    def refresh_session(self) -> None:
        """Refresh the underlying broker session before reconnecting."""

        if callable(self._refresh_hook):
            try:
                self._logger.info(
                    "Refreshing WS/broker session",
                    extra={"event": "ws_handshake_retry"},
                )
                self._refresh_hook()
            except Exception as exc:  # noqa: BLE001 - defensive logging
                self._logger.warning("WebSocket session refresh failed: %s", exc)

    def is_connected(self) -> bool:
        """Return True if the WebSocket heartbeat is within the stale window."""

        try:
            if self._state != ConnectionState.CONNECTED:
                return False
            return time.monotonic() - self._last_heartbeat < self._stale_sec
        except Exception:
            return False

    def queue_depths(self) -> dict[str, int]:
        """Return counts for queued subscribe/unsubscribe/mode operations."""

        client = self._client
        getter = getattr(client, "pending_counts", None)
        if callable(getter):
            try:
                counts = getter()
                if isinstance(counts, dict):
                    return {
                        "subs": int(counts.get("subs", 0)),
                        "unsubs": int(counts.get("unsubs", 0)),
                        "modes": int(counts.get("modes", 0)),
                    }
            except Exception:  # pragma: no cover - debug helper only
                self._logger.debug(
                    "Failed to read websocket queue depths", exc_info=True
                )
        return {"subs": 0, "unsubs": 0, "modes": 0}

    def last_error(self) -> Exception | None:
        """Return the most recent WebSocket error recorded."""

        client_error: Exception | None = None
        getter = getattr(self._client, "last_error", None)
        if callable(getter):
            try:
                client_error = getter()
            except Exception:  # pragma: no cover - defensive logging
                self._logger.debug("Failed to fetch client.last_error", exc_info=True)
        return client_error or self._last_error

    def last_heartbeat_monotonic(self) -> float:
        """Return the monotonic timestamp of the last heartbeat."""

        return self._last_heartbeat

    # ------------------------------------------------------------------
    # Subscription helpers
    def subscribe_tokens(self, tokens: Iterable[Any], mode: str = "ltp") -> None:
        """Subscribe to *tokens* and ensure requested mode is set."""

        state = self.connection_state()

        with self._lock:
            new_tokens: list[Any] = []
            for token in tokens:
                if token is None:
                    continue
                if token not in self._subscriptions:
                    new_tokens.append(token)
                self._subscriptions[token] = mode
            if not new_tokens:
                return
            self._subscription_tokens.update(new_tokens)
            active_tokens = len(self._subscription_tokens)

            if state != ConnectionState.CONNECTED:
                self._logger.info(
                    "WebSocket subscription queued",
                    extra={
                        "event": "ws_subscription_queued",
                        "queued_tokens": len(new_tokens),
                        "active_tokens": active_tokens,
                        "mode": mode,
                        "state": state.value,
                    },
                )
                return

            self._logger.info(
                "WebSocket subscribed",
                extra={
                    "event": "ws_subscription_updated",
                    "added_tokens": len(new_tokens),
                    "active_tokens": active_tokens,
                    "mode": mode,
                },
            )
            if getattr(self._client, "subscribe", None) is not None:
                self._client.subscribe(list(new_tokens))
            if getattr(self._client, "set_mode", None) is not None:
                self._client.set_mode(mode, list(new_tokens))

    def unsubscribe_tokens(self, tokens: Iterable[Any]) -> None:
        """Unsubscribe provided *tokens* if they are registered."""

        to_remove = [token for token in tokens if token in self._subscriptions]
        if not to_remove:
            return
        for token in to_remove:
            self._subscriptions.pop(token, None)

        with self._lock:
            self._logger.debug(
                "Unsubscribing websocket tokens", extra={"tokens": to_remove}
            )
            for token in to_remove:
                self._subscription_tokens.discard(token)
            active_tokens = len(self._subscription_tokens)
            if (
                self._client_is_connected()
                and getattr(self._client, "unsubscribe", None) is not None
            ):
                self._client.unsubscribe(list(to_remove))
        self._logger.info(
            "WebSocket unsubscribed",
            extra={
                "event": "ws_unsubscription_updated",
                "removed_tokens": len(to_remove),
                "active_tokens": active_tokens,
            },
        )

    def connection_state(self) -> ConnectionState:
        """Return the current websocket connection state."""

        with self._state_lock:
            return self._state

    # ------------------------------------------------------------------
    # Internal helpers
    def _transition_state(self, new_state: ConnectionState) -> None:
        callback: Callable[[], None] | None = None
        with self._state_lock:
            old_state = self._state
            if new_state == old_state:
                return
            self._state = new_state
            try:
                self._m_state.set(float(new_state.value))
            except Exception:  # pragma: no cover - optional metrics
                pass
            now = time.monotonic()
            if new_state in {
                ConnectionState.DISCONNECTED,
                ConnectionState.IDLE,
                ConnectionState.ERROR,
            }:
                self._downtime_started = now
                if old_state not in {
                    ConnectionState.DISCONNECTED,
                    ConnectionState.IDLE,
                }:
                    callback = self._user_on_disconnect
            if new_state == ConnectionState.CONNECTED:
                downtime = now - self._downtime_started
                self._logger.info(
                    "WebSocket connected",
                    extra={
                        "event": "websocket_connected",
                        "state": new_state.value,
                        "downtime_s": round(downtime, 3),
                    },
                )
                callback = self._user_on_connect or callback
            else:
                self._logger.debug(
                    "WebSocket state change",
                    extra={"event": "websocket_state", "state": new_state.value},
                )

        if callable(callback):
            try:
                callback()
            except Exception:  # pragma: no cover - defensive logging
                self._logger.exception("WebSocket lifecycle callback failed")

    def _resubscribe_all(self) -> None:
        if not self._subscriptions:
            return
        grouped: dict[str, list[Any]] = defaultdict(list)
        for token, mode in self._subscriptions.items():
            grouped[mode].append(token)
        batch_size = 400
        for mode, tokens in grouped.items():
            total = len(tokens)
            for start in range(0, total, batch_size):
                chunk = list(tokens[start : start + batch_size])
                try:
                    if getattr(self._client, "subscribe", None) is not None:
                        self._client.subscribe(chunk)
                    if getattr(self._client, "set_mode", None) is not None:
                        self._client.set_mode(mode, chunk)
                except Exception as exc:  # noqa: BLE001
                    self._logger.warning(
                        "Failed to resubscribe tokens",
                        extra={"mode": mode, "err": str(exc)},
                    )
                    break
                if total > batch_size and start + batch_size < total:
                    time.sleep(0.05)
            else:
                self._logger.info(
                    "Resubscribed websocket tokens",
                    extra={
                        "event": "ws_subs_flushed",
                        "mode": mode,
                        "resubscribed_tokens": total,
                        "active_tokens": len(self._subscription_tokens),
                    },
                )

    def _cancel_reconnect_timer(self) -> None:
        timer = self._reconnect_timer
        if timer is not None:
            timer.cancel()
            self._reconnect_timer = None

    def _schedule_reconnect(self, *, initial: bool = False) -> None:
        if self._stop_event.is_set():
            return
        delay = 0.0
        if initial:
            delay = min(1.0, self._backoff_base)
        else:
            delay = min(self._backoff_s, self._max_backoff_s)
        if delay <= 0:
            delay = 1.0 if initial else 0.0

        self._cancel_reconnect_timer()

        if delay <= 0:
            self._force_reconnect(reason="timer")
            return

        timer = threading.Timer(delay, self._force_reconnect)
        timer.daemon = True
        self._reconnect_timer = timer
        timer.start()
        self._logger.info(
            "WS reconnect scheduled",
            extra={"event": "ws_backoff_sleep", "delay_s": round(delay, 3)},
        )

    def _connect_client(self) -> None:
        try:
            now = time.monotonic()
            self._last_connect_attempt = now
            self._connecting_since = now
            connector = getattr(self._client, "_nsb_connect", None)
            if callable(connector):
                connector(threaded=True)
            else:
                try:
                    self._client.connect(threaded=True)
                except TypeError:  # pragma: no cover - backward compatibility
                    self._client.connect()
        except Exception as exc:  # noqa: BLE001
            self._last_error = exc
            raise

    def force_reconnect(self) -> None:
        """Force a websocket reconnect attempt regardless of breaker state."""

        self._logger.info("Manual websocket reconnect requested")
        self._force_reconnect(reason="manual")

    def _force_reconnect(self, *, reason: str | None = None) -> None:
        if self._stop_event.is_set():
            return
        self._cancel_reconnect_timer()
        if reason:
            self._logger.info(
                "Forcing websocket reconnect",
                extra={"event": "ws_client_rebuild", "reason": reason},
            )
        else:
            self._logger.info(
                "Forcing websocket reconnect", extra={"event": "ws_client_rebuild"}
            )
        with self._lock:
            try:
                closer = getattr(self._client, "close", None)
                if callable(closer):
                    closer()
                else:
                    disconnector = getattr(self._client, "disconnect", None)
                    if callable(disconnector):
                        disconnector()
            except Exception:  # pragma: no cover - best effort cleanup
                pass
        self._transition_state(ConnectionState.ERROR)
        try:
            self._reconnect()
        except WebSocketError:
            self._logger.debug("Force reconnect attempt failed", exc_info=True)
            self._schedule_reconnect()

    def _reconnect(self) -> None:
        if self._stop_event.is_set():
            return

        now = time.monotonic()
        if now < self._breaker_open_until:
            remaining = self._breaker_open_until - now
            self._logger.warning(
                "Reconnect suppressed by circuit breaker",
                extra={"cooldown_s": round(remaining, 3)},
            )
            try:
                self._m_reconnect_skipped.inc()
            except Exception:  # pragma: no cover - optional metrics
                pass
            return

        with self._reconnect_lock:
            delay = self._compute_backoff_delay()
            if delay > 0:
                wait_s = round(delay, 3)
                self._logger.info(
                    "WS reconnect backoff %.2fs",
                    wait_s,
                    extra={"event": "ws_backoff_sleep", "delay_s": wait_s},
                )
                if self._stop_event.wait(delay):
                    return

            if callable(self._refresh_hook):
                self.refresh_session()

            try:
                new_client = self._build_client_fn()
            except Exception as exc:  # noqa: BLE001
                self._logger.exception("Failed to build websocket client")
                self._reconnect_failures += 1
                raise WebSocketError("Unable to rebuild WebSocket client") from exc

            with self._lock:
                self._client = new_client
                self._configure_callbacks()

            try:
                self._last_heartbeat = time.monotonic()
                self._transition_state(ConnectionState.CONNECTING)
                self._connect_client()
                self._backoff_reset()
                try:
                    self._m_reconnects.inc()
                except Exception:  # pragma: no cover - optional metrics
                    pass
            except Exception as exc:  # noqa: BLE001
                self._reconnect_failures += 1
                self._last_error = exc
                code_attr = self._extract_error_code(exc)
                self._logger.error(
                    "Failed to reconnect WebSocket (attempt #%d): %s",
                    self._reconnect_failures,
                    exc,
                    extra={"code": code_attr},
                )
                if self._reconnect_failures >= self._breaker_threshold or (
                    code_attr in {429, 1006} and self._breaker_threshold <= 1
                ):
                    self._trip_breaker()
                raise WebSocketError("Unable to reconnect WebSocket") from exc

    def _trip_breaker(self) -> None:
        deadline = time.monotonic() + self._breaker_open_sec
        if deadline <= self._breaker_open_until:
            return
        self._breaker_open_until = deadline
        self._logger.error(
            "Circuit breaker OPEN",
            extra={"open_seconds": round(self._breaker_open_sec, 3)},
        )

    def _backoff_reset(self) -> None:
        self._reconnect_failures = 0
        self._backoff_s = self._backoff_base

    def _increase_backoff(self) -> None:
        self._backoff_s = min(
            self._backoff_s * self._backoff_factor, self._max_backoff_s
        )

    def _compute_backoff_delay(self) -> float:
        base_delay = min(self._backoff_s, self._max_backoff_s)
        if base_delay <= 0:
            return 0.0
        jitter = 0.0
        if self._backoff_jitter > 0:
            jitter = random.uniform(0.0, base_delay * self._backoff_jitter)
        return float(base_delay + jitter)

    def _extract_error_code(self, value: Any) -> int | None:
        if isinstance(value, int):
            return value
        code = getattr(value, "code", None)
        if isinstance(code, int):
            return code
        if isinstance(value, BaseException):
            for arg in getattr(value, "args", ()):
                extracted = self._extract_error_code(arg)
                if extracted is not None:
                    return extracted
        return None


__all__ = ["ConnectionState", "HandshakeTimeoutError", "WebSocketManager"]
