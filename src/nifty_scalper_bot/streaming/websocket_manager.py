"""Hardened WebSocket manager for Zerodha KiteTicker streaming."""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from dataclasses import dataclass
from datetime import datetime
from datetime import time as dtime
from enum import Enum
from typing import Any
from zoneinfo import ZoneInfo

from kiteconnect import KiteTicker

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
    """Manage one resilient KiteTicker session with guarded reconnect logic."""

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
        heartbeat_interval_seconds: float = 10.0,
        stale_threshold_seconds: float = 5.0,
        heartbeat_timeout_seconds: float = 25.0,
        handshake_timeout_seconds: float = 20.0,
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
        if heartbeat_timeout_seconds <= heartbeat_interval_seconds:
            raise ValueError(
                "heartbeat_timeout_seconds must be > heartbeat_interval_seconds"
            )
        if handshake_timeout_seconds < 15.0 or handshake_timeout_seconds > 25.0:
            raise ValueError("handshake_timeout_seconds must be within 15-25 seconds")
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
        self._last_pong_mono = 0.0
        self._last_backoff_delay = self._base_backoff
        self._circuit = _CircuitState()

    @property
    def on_tick(self) -> TickCallback | None:
        """Args: none; Returns: callback; Raises: none."""

        return self._on_tick_callback

    @on_tick.setter
    def on_tick(self, callback: TickCallback | None) -> None:
        """Args: callback; Returns: none; Raises: none."""

        self._on_tick_callback = callback

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
            self._merge_tokens(tokens)
            await self._resubscribe_if_connected()
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
            self._connect_task = self._loop.create_task(self.connect())
        except Exception as e:
            self._logger.error("Failure in start: %s", e)
            raise

    def stop(self) -> None:
        """Args: none; Returns: none; Raises: RuntimeError."""

        try:
            if self._loop is None:
                return
            self._loop.create_task(self.disconnect())
        except Exception as e:
            self._logger.error("Failure in stop: %s", e)
            raise

    def subscribe_tokens(self, tokens: Sequence[int], mode: str = "ltp") -> None:
        """Args: tokens/mode; Returns: none; Raises: none."""

        del mode
        self._merge_tokens(tokens)
        self._schedule_async(self._resubscribe_if_connected())

    def unsubscribe_tokens(self, tokens: Sequence[int]) -> None:
        """Args: tokens; Returns: none; Raises: none."""

        try:
            remove_set = {int(token) for token in tokens}
            self._tokens -= remove_set
            ticker = self._ticker
            if ticker is not None and self._connected.is_set() and remove_set:
                payload = sorted(remove_set)
                self._schedule_blocking(lambda: ticker.unsubscribe(payload))
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
                if not self._is_within_trading_window():
                    self._logger.info(
                        "Condition met: reconnect_suppressed_outside_trading_window"
                    )
                    return
                self._state = ConnectionState.CONNECTING
                self._connected.clear()
                self._connect_started_mono = time.monotonic()

                # --- PRODUCTION SAFETY BLOCK: Always rebuild ticker on reconnect. ---
                await self._replace_ticker()
                await asyncio.to_thread(self.ticker.connect, True)
                await asyncio.wait_for(
                    self._connected.wait(), timeout=self._handshake_timeout
                )
                self._state = ConnectionState.CONNECTED
                self._circuit.failures = 0
                self._circuit.open_until_mono = 0.0
                self._last_backoff_delay = self._base_backoff
                self._logger.info(
                    "Condition met: websocket_connected reason=%s",
                    reason,
                )
                await self._resubscribe_if_connected()
        except Exception as e:
            self._record_failure()
            self._logger.error("Failure in _connect_once: %s", e)
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
                self._reconnect_task = asyncio.create_task(self._reconnect_loop(reason))

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
                    pong_age = now - self._last_pong_mono
                    if pong_age > self._heartbeat_timeout:
                        self._logger.warning(
                            "Condition met: websocket_pong_timeout "
                            "age=%.2fs threshold=%.2fs",
                            pong_age,
                            self._heartbeat_timeout,
                        )
                        self._connected.clear()
                        self._schedule_reconnect("watchdog_pong_timeout")
        except asyncio.CancelledError:
            return
        except Exception as e:
            self._logger.error("Failure in _watchdog_loop: %s", e)

    async def _ensure_watchdog(self) -> None:
        """Args: none; Returns: none; Raises: none."""

        if self._watchdog_task is None or self._watchdog_task.done():
            self._watchdog_task = asyncio.create_task(self._watchdog_loop())

    async def _replace_ticker(self) -> None:
        """Args: none; Returns: none; Raises: Exception."""

        old = self._ticker
        self._ticker = self._build_ticker()
        self._bind_handlers(self._ticker)
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

    def _on_connect(self, ws: KiteTicker, response: dict[str, Any]) -> None:
        """Args: ws/response; Returns: none; Raises: none."""

        try:
            del ws, response
            self._last_pong_mono = time.monotonic()
            self._last_tick_mono = time.monotonic()
            self._connected.set()
            self._state = ConnectionState.CONNECTED
            if self._on_connect_callback is not None:
                self._on_connect_callback()
            self._schedule_async(self._resubscribe_if_connected())
        except Exception as e:
            self._logger.error("Failure in _on_connect: %s", e)

    def _on_ticks(self, _ws: KiteTicker, ticks: list[dict[str, Any]]) -> None:
        """Args: ws/ticks; Returns: none; Raises: none."""

        try:
            now = time.monotonic()
            self._last_tick_mono = now
            callback = self._on_tick_callback
            if callback is None:
                return
            loop = self._resolve_loop()
            for tick in ticks:
                result = callback(tick)
                if asyncio.iscoroutine(result):
                    self._schedule_coroutine(loop, result)
        except Exception as e:
            self._logger.error("Failure in _on_ticks: %s", e)

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
            if not self._manual_disconnect:
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
            if code != 1000:
                self._record_failure()
            if self._on_disconnect_callback is not None:
                self._on_disconnect_callback()
            if not self._manual_disconnect:
                self._schedule_reconnect(f"close:{code}")
        except Exception as e:
            self._logger.error("Failure in _on_close: %s", e)

    async def _resubscribe_if_connected(self) -> None:
        """Args: none; Returns: none; Raises: none."""

        ticker = self._ticker
        if ticker is None or not self._connected.is_set() or not self._tokens:
            return
        payload = sorted(self._tokens)
        for idx in range(0, len(payload), 200):
            batch = payload[idx : idx + 200]
            await asyncio.to_thread(ticker.subscribe, batch)
            await asyncio.to_thread(ticker.set_mode, ticker.MODE_FULL, batch)

    def _merge_tokens(self, tokens: Sequence[int]) -> None:
        """Args: tokens; Returns: none; Raises: none."""

        for token in tokens:
            self._tokens.add(int(token))

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
            loop.create_task(coroutine)
            return
        loop.call_soon_threadsafe(lambda: loop.create_task(coroutine))

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

        low = self._base_backoff
        high = max(low, self._last_backoff_delay * 3.0)
        delay = random.uniform(low, high)
        delay = min(self._max_backoff, delay)
        self._last_backoff_delay = delay
        return delay

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
        """Args: none; Returns: bool; Raises: none."""

        if not self._trading_window_enabled:
            return True
        now = datetime.now(self._trading_tz)
        if now.weekday() >= 5:
            return False
        now_time = now.time().replace(tzinfo=None)
        return self._trading_start <= now_time <= self._trading_end

    async def _cancel_task(self, task: asyncio.Task[None] | None) -> None:
        """Args: task; Returns: none; Raises: none."""

        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            return
