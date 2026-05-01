"""Hardened WebSocket manager for Zerodha KiteTicker streaming."""

from __future__ import annotations

import asyncio
from contextlib import suppress
import random
import time
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from dataclasses import dataclass
from datetime import time as dtime
from enum import Enum
from typing import Any
from zoneinfo import ZoneInfo

from kiteconnect import KiteTicker

from nifty_scalper_bot.utils.async_helpers import safe_task
from nifty_scalper_bot.utils.logging import get_logger

TickCallback = Callable[[dict[str, Any]], Awaitable[None] | None]


class ConnectionState(Enum):
    """Args: none; Returns: enum state; Raises: none."""

    DISCONNECTED = 0
    CONNECTING = 1
    CONNECTED = 2
    RECONNECTING = 3


@dataclass(slots=True)
class _CircuitState:
    """Args: none; Returns: breaker state; Raises: none."""

    failures: int = 0
    open_until_mono: float = 0.0


class WebSocketManager:
    """Transport-only KiteTicker session manager; business subscription intent lives upstream."""

    def __init__(
        self,
        api_key: str,
        access_token: str,
        tokens: Sequence[int] | None = None,
        *,
        on_tick: TickCallback | None = None,
        on_tick_callback: TickCallback | None = None,
        on_error: Callable[[Exception], None] | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
        max_backoff_seconds: float = 60.0,
        base_backoff_seconds: float = 1.0,
        heartbeat_interval_seconds: float = 20.0,
        stale_threshold_seconds: float = 10.0,
        heartbeat_timeout_seconds: float = 10.0,
        handshake_timeout_seconds: float = 30.0,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_cooldown_seconds: float = 60.0,
        reconnect_min_seconds: float | None = None,
        reconnect_max_seconds: float | None = None,
        trading_window_enabled: bool = True,
        trading_window_tz: str = "Asia/Kolkata",
        trading_start: dtime = dtime(hour=9, minute=15),
        trading_end: dtime = dtime(hour=15, minute=30),
        **kwargs: Any,
    ) -> None:
        """Args: constructor fields; Returns: none; Raises: ValueError."""

        backoff_min_alias = kwargs.get("backoff_min_sec")
        backoff_max_alias = kwargs.get("backoff_max_sec")
        if isinstance(backoff_min_alias, (int, float)):
            base_backoff_seconds = float(backoff_min_alias)
        if isinstance(backoff_max_alias, (int, float)):
            max_backoff_seconds = float(backoff_max_alias)
        if reconnect_min_seconds is not None:
            base_backoff_seconds = reconnect_min_seconds
        if reconnect_max_seconds is not None:
            max_backoff_seconds = reconnect_max_seconds

        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("api_key is required")
        if not isinstance(access_token, str) or not access_token.strip():
            raise ValueError("access_token is required")
        if base_backoff_seconds <= 0:
            raise ValueError("base_backoff_seconds must be > 0")
        if max_backoff_seconds < base_backoff_seconds:
            raise ValueError("max_backoff_seconds must be >= base_backoff_seconds")
        if heartbeat_interval_seconds <= 0:
            raise ValueError("heartbeat_interval_seconds must be > 0")
        if stale_threshold_seconds <= 0:
            raise ValueError("stale_threshold_seconds must be > 0")
        if heartbeat_timeout_seconds <= 0:
            raise ValueError("heartbeat_timeout_seconds must be > 0")
        if handshake_timeout_seconds < 15.0 or handshake_timeout_seconds > 45.0:
            raise ValueError("handshake_timeout_seconds must be within 15-45 seconds")
        if circuit_breaker_threshold <= 0:
            raise ValueError("circuit_breaker_threshold must be > 0")
        if circuit_breaker_cooldown_seconds <= 0:
            raise ValueError("circuit_breaker_cooldown_seconds must be > 0")

        self._logger = get_logger(__name__)
        self._api_key = api_key.strip()
        self._access_token = access_token.strip()
        self._tokens: set[int] = {int(token) for token in (tokens or [])}
        self._on_tick_callback = on_tick_callback or on_tick
        self._on_error_callback = on_error
        self._loop = loop

        self._base_backoff = float(base_backoff_seconds)
        self._max_backoff = float(max_backoff_seconds)
        self._heartbeat_interval = float(heartbeat_interval_seconds)
        self._stale_threshold = float(stale_threshold_seconds)
        self._heartbeat_timeout = float(heartbeat_timeout_seconds)
        self._handshake_timeout = float(handshake_timeout_seconds)
        self._circuit_breaker_threshold = int(circuit_breaker_threshold)
        self._circuit_breaker_cooldown = float(circuit_breaker_cooldown_seconds)

        self._trading_window_enabled = bool(trading_window_enabled)
        self._trading_tz = ZoneInfo(trading_window_tz)
        self._trading_start = trading_start
        self._trading_end = trading_end

        self._state = ConnectionState.DISCONNECTED
        self._connected = asyncio.Event()
        self._connect_lock = asyncio.Lock()
        self._reconnect_lock = asyncio.Lock()
        self._shutdown = False
        self._manual_disconnect = False
        self._running = False
        self._connect_task: asyncio.Task[None] | None = None
        self._watchdog_task: asyncio.Task[None] | None = None
        self._reconnect_task: asyncio.Task[None] | None = None

        self._ticker: KiteTicker | None = None
        self._ticker_factory: Callable[[], KiteTicker] | None = None
        self._on_connect_callback: Callable[[], None] | None = None
        self._on_disconnect_callback: Callable[[], None] | None = None
        self._connect_started_mono = 0.0
        self._last_tick_mono = 0.0
        self._last_stale_log_time = 0.0
        self._last_pong_mono = 0.0
        self._last_backoff_delay = self._base_backoff
        self._circuit = _CircuitState()
        self._fallback_start_callback: Callable[[], None] | None = None
        self._fallback_stop_callback: Callable[[], None] | None = None
        self._fallback_active = False
        self._first_tick_logged = False
        self._subscription_log_tokens: set[int] = set()

    @property
    def on_tick(self) -> TickCallback | None:
        """Args: none; Returns: callback; Raises: none."""

        return self._on_tick_callback

    @on_tick.setter
    def on_tick(self, callback: TickCallback | None) -> None:
        """Args: callback; Returns: none; Raises: none."""

        self.set_tick_callback(callback)

    def set_tick_callback(
        self,
        callback: TickCallback | None,
        *,
        suppress_warning: bool = False,
    ) -> None:
        """Args: callback; Returns: none; Raises: none."""

        if self._on_tick_callback is not None and not suppress_warning:
            self._logger.warning("WS|Replacing existing tick callback")
        self._on_tick_callback = callback

    def set_fallback_callbacks(
        self,
        *,
        on_start: Callable[[], None] | None = None,
        on_stop: Callable[[], None] | None = None,
    ) -> None:
        """Args: callbacks; Returns: none; Raises: none."""

        self._fallback_start_callback = on_start
        self._fallback_stop_callback = on_stop

    @property
    def ticker(self) -> KiteTicker:
        """Args: none; Returns: active ticker; Raises: RuntimeError."""

        if self._ticker is None:
            raise RuntimeError("ticker is not initialized")
        return self._ticker

    async def connect(self) -> None:
        """Args: none; Returns: none; Raises: Exception."""

        try:
            self._logger.debug("Entered connect")
            if self._loop is None:
                self._loop = asyncio.get_running_loop()
            self._shutdown = False
            self._manual_disconnect = False
            self._running = True
            await self._ensure_watchdog()
            await self._connect_once(reason="initial")
        except Exception as e:
            self._logger.error("Failure in connect: %s", e)
            raise

    async def disconnect(self) -> None:
        """Args: none; Returns: none; Raises: Exception."""

        try:
            self._logger.debug("Entered disconnect")
            self._manual_disconnect = True
            self._shutdown = True
            self._running = False
            self._state = ConnectionState.DISCONNECTED
            self._connected.clear()
            await self._cancel_task(self._connect_task)
            await self._cancel_task(self._reconnect_task)
            await self._cancel_task(self._watchdog_task)
            self._connect_task = None
            self._reconnect_task = None
            self._watchdog_task = None
            ticker = self._ticker
            self._ticker = None
            if ticker is not None:
                await asyncio.to_thread(ticker.close)
        except Exception as e:
            self._logger.error("Failure in disconnect: %s", e)
            raise

    async def subscribe(self, tokens: Sequence[int]) -> None:
        """Args: tokens; Returns: none; Raises: Exception."""

        try:
            self._logger.debug("Entered subscribe")
            self.add_tokens(tokens)
        except Exception as e:
            self._logger.error("Failure in subscribe: %s", e)
            raise

    def start(self) -> None:
        """Args: none; Returns: none; Raises: RuntimeError."""

        try:
            if self._loop is None:
                self._loop = asyncio.get_running_loop()
            if self._connect_task is not None and not self._connect_task.done():
                return
            self._connect_task = safe_task(self.connect())
        except Exception as e:
            self._logger.error("Failure in start: %s", e)
            raise

    def stop(self) -> None:
        """Args: none; Returns: none; Raises: RuntimeError."""

        try:
            if self._loop is None:
                return
            safe_task(self.disconnect())
        except Exception as e:
            self._logger.error("Failure in stop: %s", e)
            raise

    def set_tokens(self, tokens: Sequence[int]) -> bool:
        """Reconcile websocket transport token state. Args: tokens. Returns: changed flag. Raises: none."""

        try:
            new_tokens = {int(token) for token in tokens if int(token) > 0}
        except Exception as e:
            self._logger.error("Failure in set_tokens normalization: %s", e)
            return False

        if new_tokens == self._tokens:
            return False

        old_tokens = set(self._tokens)
        to_add = sorted(new_tokens - old_tokens)
        to_remove = sorted(old_tokens - new_tokens)

        ticker = self._ticker
        connected = ticker is not None and self._connected.is_set()
        if connected and to_remove:
            with suppress(Exception):
                self._schedule_blocking(lambda: ticker.unsubscribe(to_remove))
        if connected and to_add:
            with suppress(Exception):
                self._schedule_blocking(lambda: ticker.subscribe(to_add))
                self._schedule_blocking(lambda: ticker.set_mode(ticker.MODE_FULL, to_add))
                self._log_ws_subscriptions(to_add)

        self._tokens = new_tokens
        self._logger.info(
            "ws_tokens_reconciled total=%d added=%d removed=%d",
            len(self._tokens),
            len(to_add),
            len(to_remove),
        )
        return True

    def add_tokens(self, tokens: Sequence[int]) -> bool:
        """Add transport tokens. Args: tokens. Returns: changed flag. Raises: none."""

        incoming = {int(token) for token in tokens if int(token) > 0}
        return self.set_tokens(sorted(self._tokens | incoming))

    def remove_tokens(self, tokens: Sequence[int]) -> bool:
        """Remove transport tokens. Args: tokens. Returns: changed flag. Raises: none."""

        outgoing = {int(token) for token in tokens if int(token) > 0}
        return self.set_tokens(sorted(self._tokens - outgoing))

    def subscribe_tokens(self, tokens: Sequence[int], mode: str = "full") -> None:
        """Compatibility wrapper for transport token additions. Args: tokens/mode. Returns: none. Raises: none."""

        del mode
        self.add_tokens(tokens)

    def resubscribe(self, tokens: list[int]) -> None:
        """Compatibility replay helper that delegates token-set reconciliation."""

        try:
            if not tokens:
                return
            self.set_tokens(tokens)
        except Exception as e:
            self._logger.error("WebSocket resubscribe failed: %s", e, exc_info=True)

    def unsubscribe(self, tokens: Sequence[int]) -> None:
        """Args: tokens; Returns: none; Raises: none."""

        self.unsubscribe_tokens(tokens)

    def unsubscribe_tokens(self, tokens: Sequence[int]) -> None:
        """Compatibility wrapper for transport token removals. Args: tokens; Returns: none; Raises: none."""

        try:
            self.remove_tokens(tokens)
        except Exception as e:
            self._logger.error("Failure in unsubscribe_tokens: %s", e)

    def set_callbacks(
        self,
        *,
        on_connect: Callable[[], None] | None = None,
        on_disconnect: Callable[[], None] | None = None,
    ) -> None:
        """Args: callbacks; Returns: none; Raises: none."""

        self._on_connect_callback = on_connect
        self._on_disconnect_callback = on_disconnect

    def set_client_factory(self, fn: Callable[[], Any]) -> None:
        """Args: factory callable; Returns: none; Raises: none."""

        try:

            def _factory() -> KiteTicker:
                built = fn()
                if not isinstance(built, KiteTicker):
                    raise TypeError("client factory must return KiteTicker")
                return built

            self._ticker_factory = _factory
        except Exception as e:
            self._logger.error("Failure in set_client_factory: %s", e)

    def force_reconnect(self) -> None:
        """Args: none; Returns: none; Raises: none."""

        self._schedule_reconnect("manual")

    def reconnect(self) -> None:
        """Reconnect websocket session. Args: none. Returns: none. Raises: none."""

        self.force_reconnect()

    def is_connected(self) -> bool:
        """Args: none; Returns: bool; Raises: none."""

        return self._connected.is_set()

    def is_running(self) -> bool:
        """Args: none; Returns: bool; Raises: none."""

        return self._running and not self._shutdown

    def connection_state(self) -> ConnectionState:
        """Args: none; Returns: state; Raises: none."""

        return self._state

    def health_snapshot(self) -> dict[str, float | int | bool]:
        """Args: none; Returns: health map; Raises: none."""

        now = time.monotonic()
        last_tick_age = now - self._last_tick_mono if self._last_tick_mono > 0 else 0.0
        hb_age = now - self._last_pong_mono if self._last_pong_mono > 0 else 0.0
        return {
            "connected": self._connected.is_set(),
            "state": int(self._state.value),
            "consecutive_failures": self._circuit.failures,
            "last_tick_age_seconds": max(0.0, last_tick_age),
            "last_heartbeat_age_seconds": max(0.0, hb_age),
            "circuit_open": self._circuit.open_until_mono > now,
        }

    async def _connect_once(self, reason: str) -> None:
        """Args: reason; Returns: none; Raises: Exception."""

        try:
            async with self._connect_lock:
                if self._shutdown or self._manual_disconnect:
                    return
        
                self._state = ConnectionState.CONNECTING
                self._connected.clear()
                self._connect_started_mono = time.monotonic()

                # --- PRODUCTION SAFETY BLOCK: Always rebuild ticker on reconnect. ---
                await self._replace_ticker()
                try:
                    await asyncio.to_thread(
                        self.ticker.connect,
                        True,
                        ping_interval=20,
                        ping_timeout=10,
                    )
                except TypeError:
                    await asyncio.to_thread(self.ticker.connect, True)
                await asyncio.wait_for(
                    self._connected.wait(), timeout=self._handshake_timeout
                )
                self._state = ConnectionState.CONNECTED
                await self._ensure_watchdog()
                self._circuit.failures = 0
                self._circuit.open_until_mono = 0.0
                self._last_backoff_delay = self._base_backoff
                self._logger.info(
                    "Condition met: websocket_connected reason=%s",
                    reason,
                )
                if not self._is_within_trading_window():
                    self._logger.warning(
                        "WebSocket connected outside trading hours; running in data-observation mode while execution remains runtime-gated"
                    )
        except Exception as e:
            self._record_failure()
            self._logger.error("Failure in _connect_once: %s", e)
            # Close the ticker that failed to connect — prevents orphaned
            # KiteTicker threads from producing late 1006 close events.
            ticker = self._ticker
            if ticker is not None:
                try:
                    await asyncio.to_thread(ticker.close)
                except Exception as e:
                    self._logger.exception("Unhandled exception", exc_info=True)
                    raise  # Best effort cleanup
            self._ticker = None
            self._schedule_reconnect("connect_failure")

    def _schedule_reconnect(self, reason: str) -> None:
        """Args: reason; Returns: none; Raises: none."""

        async def _ensure_task() -> None:
            async with self._reconnect_lock:
                if self._shutdown or self._manual_disconnect:
                    return
                if self._reconnect_task is not None and not self._reconnect_task.done():
                    return
                self._state = ConnectionState.RECONNECTING
                self._reconnect_task = safe_task(self._reconnect_loop(reason))

        self._schedule_async(_ensure_task())

    async def _reconnect_loop(self, reason: str) -> None:
        """Args: reason; Returns: none; Raises: none."""

        try:
            while not self._shutdown and not self._manual_disconnect:
                if self._connected.is_set():
                    return
                if self._circuit_is_open():
                    await asyncio.sleep(min(self._heartbeat_interval, 1.0))
                    continue
                if not self._is_within_trading_window():
                    await asyncio.sleep(30.0)
                    continue

                delay = self._next_backoff_delay()
                # Enforce minimum floor — sub-second reconnects overwhelm
                # Zerodha and cause cascading 1006 errors.
                delay = max(delay, 5.0)
                self._logger.info(
                    "Condition met: reconnect_scheduled reason=%s delay=%.2fs",
                    reason,
                    delay,
                )
                await asyncio.sleep(delay)
                if (
                    self._shutdown
                    or self._manual_disconnect
                    or self._connected.is_set()
                ):
                    return

                # Ensure any old ticker is fully closed before reconnect
                if self._ticker is not None:
                    try:
                        await asyncio.to_thread(self._ticker.close)
                    except Exception as e:
                        self._logger.exception("Unhandled exception", exc_info=True)
                        raise
                    self._ticker = None
                await self._connect_once(reason="reconnect")
                if self._connected.is_set():
                    return
        except Exception as e:
            self._logger.error("Failure in _reconnect_loop: %s", e)
        finally:
            async with self._reconnect_lock:
                if self._reconnect_task is asyncio.current_task():
                    self._reconnect_task = None

    async def _watchdog_loop(self) -> None:
        """Args: none; Returns: none; Raises: none."""

        try:
            while not self._shutdown:
                await asyncio.sleep(self._heartbeat_interval)
                if self._shutdown:
                    return
                now = time.monotonic()

                # --- PRODUCTION SAFETY BLOCK: Handshake timeout detection. ---
                if (
                    self._state == ConnectionState.CONNECTING
                    and self._connect_started_mono > 0.0
                    and (now - self._connect_started_mono) > self._handshake_timeout
                ):
                    self._logger.warning("Condition met: websocket_handshake_timeout")
                    self._connected.clear()
                    self._schedule_reconnect("watchdog_handshake_timeout")
                    continue

                if self._connected.is_set() and self._last_pong_mono > 0.0:
                    # Use max(pong, tick) as activity indicator.
                    # KiteTicker pong callbacks are unreliable through
                    # cloud proxies (Railway); flowing ticks prove the
                    # connection is alive even when pong frames are delayed.
                    last_activity = max(
                        self._last_pong_mono,
                        self._last_tick_mono if self._last_tick_mono > 0.0 else 0.0,
                    )
                    activity_age = now - last_activity
                    if activity_age > self._heartbeat_timeout:
                        if not self._is_within_trading_window():
                            self._logger.debug(
                                "WS heartbeat idle outside trading window - suppressing reconnect "
                                "age=%.2fs threshold=%.2fs",
                                activity_age,
                                self._heartbeat_timeout,
                            )
                            continue
                        self._logger.warning(
                            "Condition met: websocket_pong_timeout "
                            "age=%.2fs threshold=%.2fs",
                            activity_age,
                            self._heartbeat_timeout,
                        )
                        self._connected.clear()
                        self._schedule_reconnect("watchdog_pong_timeout")
                        continue

                if self._connected.is_set() and self._last_tick_mono > 0.0:
                    last_tick_age = now - self._last_tick_mono
                    if last_tick_age > self._stale_threshold:
                        if not self._is_within_trading_window():
                            # Outside market hours: no ticks are expected.
                            # Real dead-socket failures are handled by the pong-timeout path above.
                            self._logger.debug(
                                "WS stale outside trading window — skipping reconnect age=%.2fs",
                                last_tick_age,
                            )
                        else:
                            now_wall = time.time()
                            if now_wall - self._last_stale_log_time > 5.0:
                                self._last_stale_log_time = now_wall
                                self._logger.warning(
                                    "WebSocket stale. Reconnecting... age=%.2fs threshold=%.2fs",
                                    last_tick_age,
                                    self._stale_threshold,
                                )
                            if (
                                not self._fallback_active
                                and self._fallback_start_callback is not None
                            ):
                                self._fallback_active = True
                                try:
                                    self._fallback_start_callback()
                                except Exception as e:
                                    self._logger.error(
                                        "Failure in _watchdog_loop.fallback_start: %s", e
                                    )
                            self._schedule_reconnect("watchdog_tick_stale")
        except asyncio.CancelledError:
            return
        except Exception as e:
            self._logger.error("Failure in _watchdog_loop: %s", e)

    async def _ensure_watchdog(self) -> None:
        """Args: none; Returns: none; Raises: none."""

        if self._watchdog_task is None or self._watchdog_task.done():
            self._watchdog_task = safe_task(self._watchdog_loop())

    async def _replace_ticker(self) -> None:
        """Args: none; Returns: none; Raises: Exception."""

        old = self._ticker
        self._ticker = self._build_ticker()
        self._ticker.on_ticks = self._on_ticks
        self._ticker.on_connect = self._on_connect
        self._ticker.on_close = self._on_close
        self._ticker.on_error = self._on_error
        self._bind_handlers(self._ticker)
        self._logger.debug(
            "Condition met: websocket_ticker_replaced callback_bound=%s",
            bool(self._on_tick_callback),
        )
        if old is not None:
            try:
                await asyncio.to_thread(old.close)
            except Exception as e:
                self._logger.error("Failure in _replace_ticker.close_old: %s", e)

    def _build_ticker(self) -> KiteTicker:
        """Args: none; Returns: KiteTicker; Raises: Exception."""

        if self._ticker_factory is not None:
            return self._ticker_factory()
        return KiteTicker(self._api_key, self._access_token, reconnect=False)

    def _bind_handlers(self, ticker: KiteTicker) -> None:
        """Args: ticker; Returns: none; Raises: none."""

        ticker.on_connect = self._on_connect
        ticker.on_ticks = self._on_ticks
        ticker.on_error = self._on_error
        ticker.on_close = self._on_close
        if hasattr(ticker, "on_pong"):
            ticker.on_pong = self._on_pong
        if self._on_tick_callback is None:
            self._logger.debug(
                "Condition met: websocket_tick_callback_unbound "
                "(expected — process_ticks is the single WS ingress)"
            )
        else:
            self._logger.debug(
                "Condition met: websocket_handlers_bound callback=%s",
                getattr(
                    self._on_tick_callback,
                    "__name__",
                    type(self._on_tick_callback).__name__,
                ),
            )

    def _on_connect(self, ws: KiteTicker, response: dict[str, Any]) -> None:
        """Args: ws/response; Returns: none; Raises: none."""

        try:
            del response
            self._last_pong_mono = time.monotonic()
            self._state = ConnectionState.CONNECTING
            self._logger.info(
                "WebSocket CONNECTED successfully | tokens=%d",
                len(self._tokens),
            )
            if self._tokens:
                token_list = sorted(self._tokens)
                ws.subscribe(token_list)
                ws.set_mode(ws.MODE_FULL, token_list)
                self._log_ws_subscriptions(token_list)
                self._logger.info(
                    "ws_replay_subscriptions total=%d",
                    len(token_list),
                )
            self._connected.set()
            self._state = ConnectionState.CONNECTED
            if self._on_connect_callback is not None:
                self._on_connect_callback()
        except Exception as e:
            self._logger.error("Failure in _on_connect: %s", e)

    def _on_ticks(self, ws, ticks):
        """Args: ws, ticks; Returns: none; Raises: none."""

        del ws
        if not ticks:
            return

        market_data_manager = getattr(self, "_market_data_manager", None)
        process_ticks = getattr(market_data_manager, "process_ticks", None)
        if callable(process_ticks):
            try:
                process_ticks(ticks)
            except Exception as e:
                self._logger.error("Failure in _on_ticks.process_ticks: %s", e)

        if not isinstance(ticks, list):
            self._logger.error("Invalid ticks payload type: %s", type(ticks))
            return

        now = time.monotonic()
        self._last_tick_mono = now
        self._last_pong_mono = now
        update_authoritative = getattr(
            market_data_manager, "update_authoritative_ticks", None
        )
        if callable(update_authoritative):
            try:
                update_authoritative(ticks)
            except Exception as e:
                self._logger.error("Failure in _on_ticks.update_authoritative: %s", e)
        if self._fallback_active and self._fallback_stop_callback is not None:
            self._fallback_active = False
            try:
                self._fallback_stop_callback()
            except Exception as e:
                self._logger.error("Failure in _on_ticks.fallback_stop: %s", e)

        callback = self._on_tick_callback
        if not callable(callback):
            # process_ticks() already enqueued all ticks into MDM's queue —
            # the per-tick callback slot is intentionally unset to prevent
            # double-enqueue (see MarketDataManager.__init__ comment).
            return

        # Log first tick received — confirms pipeline is alive
        if not self._first_tick_logged and ticks:
            first_token = ticks[0].get("instrument_token", "?")
            self._first_tick_logged = True
            self._logger.info(
                "FIRST_TICK_RECEIVED instrument_token=%s — pipeline is live",
                first_token,
            )

        for tick in ticks:
            if "instrument_token" not in tick:
                self._logger.debug(
                    "Skipping tick without instrument_token: %s",
                    list(tick.keys())[:5],
                )
                continue
            try:
                result = callback(tick)
                if asyncio.iscoroutine(result):
                    loop = self._resolve_loop()
                    self._schedule_coroutine(loop, result)
            except Exception as exc:
                self._logger.error("Failure dispatching tick: %s", exc, exc_info=True)

    def _on_pong(self, _ws: KiteTicker, _payload: Any | None = None) -> None:
        """Args: ws/payload; Returns: none; Raises: none."""

        try:
            self._last_pong_mono = time.monotonic()
            if not self._connected.is_set():
                self._connected.set()
                self._state = ConnectionState.CONNECTED
        except Exception as e:
            self._logger.error("Failure in _on_pong: %s", e)

    def _on_error(self, _ws: KiteTicker, code: int, reason: str) -> None:
        """Args: ws/code/reason; Returns: none; Raises: none."""

        try:
            self._logger.error("Failure in websocket: code=%s reason=%s", code, reason)
            self._connected.clear()
            self._state = ConnectionState.DISCONNECTED
            self._record_failure()
            if self._on_error_callback is not None:
                self._on_error_callback(RuntimeError(f"code={code} reason={reason}"))
            # Don't schedule another reconnect if one is already running —
            # this prevents the storm where error + close + watchdog each
            # fire their own reconnect path simultaneously.
            if not self._manual_disconnect:
                if self._reconnect_task is not None and not self._reconnect_task.done():
                    self._logger.debug(
                        "Suppressed error reconnect — reconnect already in progress"
                    )
                else:
                    self._schedule_reconnect(f"error:{code}")
        except Exception as e:
            self._logger.error("Failure in _on_error: %s", e)

    def _on_close(self, _ws: KiteTicker, code: int, reason: str) -> None:
        """Args: ws/code/reason; Returns: none; Raises: none."""

        try:
            self._logger.info(
                "Condition met: websocket_closed code=%s reason=%s", code, reason
            )
            self._connected.clear()
            self._state = ConnectionState.DISCONNECTED
            # 1006 = old connection handshake timeout during normal teardown.
            # Don't count as failure and don't trigger redundant reconnect
            # if one is already in progress — this breaks the 1006 cascade.
            if code == 1006:
                if self._reconnect_task is not None and not self._reconnect_task.done():
                    self._logger.debug(
                        "Suppressed 1006 reconnect — reconnect already in progress"
                    )
                    return
            elif code != 1000:
                self._record_failure()
            if self._on_disconnect_callback is not None:
                self._on_disconnect_callback()
            if not self._manual_disconnect:
                self._schedule_reconnect(f"close:{code}")
        except Exception as e:
            self._logger.error("Failure in _on_close: %s", e)

    async def _resubscribe_if_connected(self) -> None:
        """Args: none; Returns: none; Raises: none."""

        if not self._tokens:
            self._logger.error(
                "WS connected but no tokens present — subscription skipped"
            )
            return

        ticker = self._ticker
        if ticker is None:
            return

        if not self._connected.is_set():
            return

        if not self._tokens:
            self._logger.error("WS connected but token set empty — no subscription")
            return

        payload = sorted(self._tokens)
        for idx in range(0, len(payload), 200):
            batch = payload[idx : idx + 200]
            await asyncio.to_thread(ticker.subscribe, batch)
            await asyncio.to_thread(ticker.set_mode, ticker.MODE_FULL, batch)
            self._log_ws_subscriptions(batch)

    def _log_ws_subscriptions(self, tokens: Sequence[int]) -> None:
        """Emit symbol/token subscription logs once per token. Args: tokens. Returns: none. Raises: none."""

        mdm = getattr(self, "_market_data_manager", None)
        symbol_map = getattr(mdm, "_symbol_by_token", {}) if mdm is not None else {}
        for raw_token in tokens:
            token = int(raw_token)
            if token in self._subscription_log_tokens:
                continue
            symbol = symbol_map.get(token)
            if not symbol:
                continue
            self._subscription_log_tokens.add(token)
            self._logger.info(
                "ws_transport_subscribed symbol=%s token=%s",
                symbol,
                token,
                extra={
                    "event": "ws_transport_subscribed",
                    "symbol": symbol,
                    "token": token,
                },
            )

    def _resolve_loop(self) -> asyncio.AbstractEventLoop:
        """Args: none; Returns: event loop; Raises: RuntimeError."""

        if self._loop is not None:
            return self._loop
        self._loop = asyncio.get_running_loop()
        return self._loop

    def _schedule_coroutine(
        self,
        loop: asyncio.AbstractEventLoop,
        coroutine: Coroutine[Any, Any, Any],
    ) -> None:
        """Args: loop/coroutine; Returns: none; Raises: none."""

        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            safe_task(coroutine)
            return
        loop.call_soon_threadsafe(lambda: safe_task(coroutine))

    def _schedule_async(self, coroutine: Coroutine[Any, Any, Any]) -> None:
        """Args: coroutine; Returns: none; Raises: none."""

        try:
            loop = self._resolve_loop()
            self._schedule_coroutine(loop, coroutine)
        except Exception as e:
            self._logger.error("Failure in _schedule_async: %s", e)

    def _schedule_blocking(self, fn: Callable[[], Any]) -> None:
        """Args: fn; Returns: none; Raises: none."""

        async def _run() -> None:
            await asyncio.to_thread(fn)

        self._schedule_async(_run())

    def _next_backoff_delay(self) -> float:
        """Args: none; Returns: delay; Raises: none."""

        sleep = self._next_backoff(self._circuit.failures)
        self._last_backoff_delay = sleep
        return sleep

    def _next_backoff(self, attempt: int) -> float:
        """Args: attempt; Returns: bounded reconnect backoff; Raises: none."""

        schedule = [2.0, 5.0, 10.0]
        base_delay = schedule[min(max(attempt, 0), len(schedule) - 1)]
        capped = min(30.0, max(self._base_backoff, base_delay), self._max_backoff)
        return random.uniform(base_delay * 0.8, capped)

    def _record_failure(self) -> None:
        """Args: none; Returns: none; Raises: none."""

        self._circuit.failures += 1
        if self._circuit.failures >= self._circuit_breaker_threshold:
            self._circuit.open_until_mono = (
                time.monotonic() + self._circuit_breaker_cooldown
            )
            self._logger.warning(
                "Condition met: websocket_circuit_open failures=%d cooldown=%.2fs",
                self._circuit.failures,
                self._circuit_breaker_cooldown,
            )

    def _circuit_is_open(self) -> bool:
        """Args: none; Returns: bool; Raises: none."""

        if self._circuit.open_until_mono <= 0.0:
            return False
        now = time.monotonic()
        if now >= self._circuit.open_until_mono:
            self._circuit.open_until_mono = 0.0
            return False
        return True

    def _is_within_trading_window(self) -> bool:
        """Determine whether WebSocket is allowed to connect."""

        if not self._trading_window_enabled:
            return True

        from datetime import datetime
        from zoneinfo import ZoneInfo

        ist = ZoneInfo("Asia/Kolkata")
        now = datetime.now(ist)

        # Weekends blocked
        if now.weekday() >= 5:
            return False

        now_time = now.time().replace(tzinfo=None)
        allowed = self._trading_start <= now_time <= self._trading_end

        if not allowed:
            self._logger.debug(
                "WS window blocked | now=%s | start=%s | end=%s",
                now_time,
                self._trading_start,
                self._trading_end,
            )

        return allowed

    async def _cancel_task(self, task: asyncio.Task[None] | None) -> None:
        """Args: task; Returns: none; Raises: none."""

        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            return
