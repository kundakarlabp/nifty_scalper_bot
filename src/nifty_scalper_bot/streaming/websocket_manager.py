"""Async WebSocket manager for Zerodha KiteTicker streaming."""

from __future__ import annotations

import asyncio
import random
import threading
import time
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, cast

from kiteconnect import KiteTicker

from nifty_scalper_bot.utils.logging import get_logger


class HandshakeTimeoutError(RuntimeError):
    """Args: elapsed/hb_delta; Returns: error; Raises: none."""

    def __init__(self, elapsed: float, hb_delta: float) -> None:
        super().__init__(
            f"Handshake timeout after {elapsed:.1f}s (hbΔ={hb_delta:.1f}s)."
        )
        self.code = 1006
        self.reason = "handshake_timeout"


TickCallback = Callable[[dict[str, Any]], Awaitable[None] | None]


class ConnectionState(Enum):
    """Args: none; Returns: enum; Raises: none."""

    DISCONNECTED = 0
    CONNECTING = 1
    CONNECTED = 2
    RECONNECTING = 3


@dataclass(slots=True)
class _ReconnectState:
    """Args: none; Returns: state; Raises: none."""

    attempts: int = 0
    task: asyncio.Task[None] | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass(slots=True)
class _HealthState:
    """Args: none; Returns: health state; Raises: none."""

    last_tick_monotonic: float = 0.0
    last_heartbeat_monotonic: float = 0.0
    last_connect_monotonic: float = 0.0
    consecutive_failures: int = 0
    circuit_open_until_monotonic: float = 0.0


class WebSocketManager:
    """Manage a single KiteTicker connection with async reconnect and tick dispatch."""

    def __init__(
        self,
        api_key: str | Any,
        access_token: str | None = None,
        tokens: Sequence[int] | None = None,
        *,
        on_tick: TickCallback | None = None,
        on_tick_callback: TickCallback | None = None,
        on_error: Callable[[Exception], None] | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
        max_backoff_seconds: float = 60.0,
        base_backoff_seconds: float = 1.0,
        heartbeat_interval_seconds: float = 2.0,
        stale_threshold_seconds: float = 5.0,
        handshake_timeout_seconds: float = 15.0,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_cooldown_seconds: float = 20.0,
        **_: Any,
    ) -> None:
        """Args: init fields; Returns: None; Raises: ValueError."""

        if base_backoff_seconds <= 0:
            raise ValueError("base_backoff_seconds must be > 0")
        if max_backoff_seconds < base_backoff_seconds:
            raise ValueError("max_backoff_seconds must be >= base_backoff_seconds")
        if heartbeat_interval_seconds <= 0:
            raise ValueError('heartbeat_interval_seconds must be > 0')
        if stale_threshold_seconds <= 0:
            raise ValueError('stale_threshold_seconds must be > 0')
        if handshake_timeout_seconds <= 0:
            raise ValueError('handshake_timeout_seconds must be > 0')
        if circuit_breaker_threshold <= 0:
            raise ValueError('circuit_breaker_threshold must be > 0')
        if circuit_breaker_cooldown_seconds <= 0:
            raise ValueError('circuit_breaker_cooldown_seconds must be > 0')

        self._logger = get_logger(__name__)
        if isinstance(api_key, str):
            resolved_api_key = api_key
            resolved_access_token = access_token or ""
        else:
            resolved_api_key = str(getattr(api_key, "api_key", "") or "")
            resolved_access_token = str(
                access_token or getattr(api_key, "access_token", "") or ""
            )
        if not resolved_api_key or not resolved_access_token:
            raise ValueError("api_key and access_token are required")

        self._loop = loop
        self._api_key = resolved_api_key
        self._access_token = resolved_access_token
        self._tokens: set[int] = {int(token) for token in (tokens or [])}
        self._on_tick_callback = on_tick_callback or on_tick
        self._on_error_callback = on_error
        self._max_backoff_seconds = max_backoff_seconds
        self._base_backoff_seconds = base_backoff_seconds
        self._heartbeat_interval_seconds = heartbeat_interval_seconds
        self._stale_threshold_seconds = stale_threshold_seconds
        self._handshake_timeout_seconds = handshake_timeout_seconds
        self._circuit_breaker_threshold = circuit_breaker_threshold
        self._circuit_breaker_cooldown_seconds = circuit_breaker_cooldown_seconds
        self._connected = asyncio.Event()
        self._shutdown = False
        self._manual_disconnect = False
        self._state = ConnectionState.DISCONNECTED
        self._reconnect = _ReconnectState()
        self._connect_lock = asyncio.Lock()
        self._health = _HealthState()
        self._health_lock = asyncio.Lock()
        self._watchdog_task: asyncio.Task[None] | None = None
        self._thread_loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._on_connect_callback: Callable[[], None] | None = None
        self._on_disconnect_callback: Callable[[], None] | None = None

        self._ticker = KiteTicker(self._api_key, self._access_token, reconnect=False)
        self._configure_handlers()

    @property
    def on_tick(self) -> TickCallback | None:
        """Args: none; Returns: callback; Raises: none."""

        return self._on_tick_callback

    @on_tick.setter
    def on_tick(self, callback: TickCallback | None) -> None:
        """Args: callback; Returns: none; Raises: none."""

        self._on_tick_callback = callback

    def _configure_handlers(self) -> None:
        """Args: none; Returns: None; Raises: none."""

        self._ticker.on_connect = self._on_connect
        self._ticker.on_ticks = self._on_ticks
        self._ticker.on_error = self._on_error
        self._ticker.on_close = self._on_close

    @property
    def ticker(self) -> KiteTicker:
        """Args: none; Returns: KiteTicker; Raises: none."""

        return self._ticker

    async def connect(self) -> None:
        """Args: none; Returns: None; Raises: Exception."""

        try:
            self._logger.debug("Entered connect")
            self._shutdown = False
            self._manual_disconnect = False
            self._state = ConnectionState.CONNECTING
            if self._loop is None:
                self._loop = asyncio.get_running_loop()
            self._configure_handlers()
            await self._start_watchdog()
            await self._run_ticker_connect()
        except Exception as e:
            await self._stop_watchdog()
            self._logger.error("Failure in connect: %s", e)
            raise

    async def disconnect(self) -> None:
        """Args: none; Returns: None; Raises: Exception."""

        try:
            self._logger.debug("Entered disconnect")
            self._manual_disconnect = True
            self._shutdown = True
            self._state = ConnectionState.DISCONNECTED
            async with self._reconnect.lock:
                if self._reconnect.task is not None:
                    self._reconnect.task.cancel()
                    self._reconnect.task = None
            await self._stop_watchdog()
            self._connected.clear()
            await asyncio.to_thread(self._ticker.close)
        except Exception as e:
            self._logger.error("Failure in disconnect: %s", e)
            raise

    async def subscribe(self, tokens: Sequence[int]) -> None:
        """Args: tokens; Returns: None; Raises: Exception."""

        try:
            self._logger.debug("Entered subscribe")
            new_tokens = [
                int(token) for token in tokens if int(token) not in self._tokens
            ]
            if not new_tokens:
                return
            self._tokens.update(new_tokens)
            if self._connected.is_set():
                await asyncio.to_thread(self._ticker.subscribe, new_tokens)
                await asyncio.to_thread(
                    self._ticker.set_mode,
                    self._ticker.MODE_FULL,
                    new_tokens,
                )
        except Exception as e:
            self._logger.error("Failure in subscribe: %s", e)
            raise

    async def _run_ticker_connect(self) -> None:
        """Args: none; Returns: None; Raises: Exception."""

        try:
            async with self._connect_lock:
                if self._shutdown:
                    return
                self._state = ConnectionState.CONNECTING
                await self._mark_connect_attempt()
                await asyncio.to_thread(self._ticker.connect, True)
                await self._await_handshake()
        except Exception as e:
            self._logger.error("Failure in _run_ticker_connect: %s", e)
            raise

    async def _await_handshake(self) -> None:
        """Args: none; Returns: None; Raises: HandshakeTimeoutError."""

        if self._connected.is_set():
            return
        started = time.monotonic()
        try:
            await asyncio.wait_for(
                self._connected.wait(), timeout=self._handshake_timeout_seconds
            )
        except TimeoutError as exc:
            hb_delta = await self._heartbeat_age_seconds()
            elapsed = time.monotonic() - started
            raise HandshakeTimeoutError(elapsed=elapsed, hb_delta=hb_delta) from exc

    async def _mark_connect_attempt(self) -> None:
        """Args: none; Returns: None; Raises: none."""

        async with self._health_lock:
            self._health.last_connect_monotonic = time.monotonic()

    async def _record_heartbeat(self) -> None:
        """Args: none; Returns: None; Raises: none."""

        now = time.monotonic()
        async with self._health_lock:
            self._health.last_heartbeat_monotonic = now

    async def _record_tick(self) -> None:
        """Args: none; Returns: None; Raises: none."""

        now = time.monotonic()
        async with self._health_lock:
            self._health.last_tick_monotonic = now
            self._health.last_heartbeat_monotonic = now

    async def _heartbeat_age_seconds(self) -> float:
        """Args: none; Returns: float; Raises: none."""

        async with self._health_lock:
            last = self._health.last_heartbeat_monotonic
        if last <= 0.0:
            return self._stale_threshold_seconds + 1.0
        return max(0.0, time.monotonic() - last)

    def _schedule_reconnect(self, reason: str) -> None:
        """Args: reason; Returns: None; Raises: none."""

        async def _inner() -> None:
            try:
                async with self._reconnect.lock:
                    if self._shutdown or self._manual_disconnect:
                        return
                    self._state = ConnectionState.RECONNECTING
                    if self._reconnect.task and not self._reconnect.task.done():
                        return
                    self._reconnect.task = asyncio.create_task(
                        self._reconnect_loop(reason)
                    )
            except Exception as e:
                self._logger.error("Failure in _schedule_reconnect._inner: %s", e)

        loop = self._resolve_loop()
        loop.call_soon_threadsafe(lambda: asyncio.create_task(_inner()))

    async def _reconnect_loop(self, reason: str) -> None:
        """Args: reason; Returns: None; Raises: Exception."""

        try:
            attempt = 0
            while not self._shutdown and not self._manual_disconnect:
                if self._connected.is_set():
                    return
                if await self._is_circuit_open():
                    continue
                delay = self._calculate_backoff(attempt)
                self._logger.info(
                    "Condition met: reconnect_scheduled "
                    "reason=%s delay=%.2fs attempt=%d",
                    reason,
                    delay,
                    attempt,
                )
                await asyncio.sleep(delay)
                if self._shutdown or self._manual_disconnect:
                    return
                try:
                    await self._run_ticker_connect()
                    await self._record_connect_success()
                except Exception as connect_error:
                    await self._record_connect_failure()
                    self._logger.error(
                        "Failure in _reconnect_loop.connect_attempt: %s",
                        connect_error,
                    )
                await asyncio.sleep(0)
                attempt += 1
        except Exception as e:
            self._logger.error("Failure in _reconnect_loop: %s", e)
            raise
        finally:
            async with self._reconnect.lock:
                if self._reconnect.task is asyncio.current_task():
                    self._reconnect.task = None

    def _calculate_backoff(self, attempts: int) -> float:
        """Args: attempts; Returns: delay; Raises: none."""

        exponential = 5.0 * (2 ** max(0, attempts))
        base_delay = min(exponential, self._max_backoff_seconds)
        jitter = random.uniform(0.0, 1.0)
        return base_delay + jitter

    async def _record_connect_success(self) -> None:
        """Args: none; Returns: None; Raises: none."""

        async with self._health_lock:
            self._health.consecutive_failures = 0
            self._health.circuit_open_until_monotonic = 0.0

    async def _record_connect_failure(self) -> None:
        """Args: none; Returns: None; Raises: none."""

        async with self._health_lock:
            self._health.consecutive_failures += 1
            failures = self._health.consecutive_failures
            if failures >= self._circuit_breaker_threshold:
                cooldown_until = (
                    time.monotonic() + self._circuit_breaker_cooldown_seconds
                )
                self._health.circuit_open_until_monotonic = cooldown_until
                self._logger.warning(
                    'Condition met: websocket_circuit_open failures=%d cooldown=%.2fs',
                    failures,
                    self._circuit_breaker_cooldown_seconds,
                )

    async def _is_circuit_open(self) -> bool:
        """Args: none; Returns: bool; Raises: none."""

        async with self._health_lock:
            open_until = self._health.circuit_open_until_monotonic
        if open_until <= 0.0:
            return False
        remaining = open_until - time.monotonic()
        if remaining <= 0.0:
            async with self._health_lock:
                self._health.circuit_open_until_monotonic = 0.0
            return False
        await asyncio.sleep(min(remaining, self._heartbeat_interval_seconds))
        return True

    async def _start_watchdog(self) -> None:
        """Args: none; Returns: None; Raises: none."""

        if self._watchdog_task is not None and not self._watchdog_task.done():
            return
        await self._record_heartbeat()
        self._watchdog_task = asyncio.create_task(self._watchdog_loop())

    async def _stop_watchdog(self) -> None:
        """Args: none; Returns: None; Raises: none."""

        if self._watchdog_task is None:
            return
        self._watchdog_task.cancel()
        try:
            await self._watchdog_task
        except asyncio.CancelledError:
            pass
        self._watchdog_task = None

    async def _watchdog_loop(self) -> None:
        """Args: none; Returns: None; Raises: none."""

        try:
            while not self._shutdown:
                await asyncio.sleep(self._heartbeat_interval_seconds)
                if self._shutdown:
                    return
                if not self._connected.is_set():
                    continue
                heartbeat_age = await self._heartbeat_age_seconds()
                if heartbeat_age > self._stale_threshold_seconds:
                    self._logger.warning(
                        'Condition met: websocket_stale_detected '
                        'age=%.2fs threshold=%.2fs',
                        heartbeat_age,
                        self._stale_threshold_seconds,
                    )
                    self._connected.clear()
                    self._schedule_reconnect('watchdog_stale')
        except asyncio.CancelledError:
            return
        except Exception as e:
            self._logger.error('Failure in _watchdog_loop: %s', e)

    def _resolve_loop(self) -> asyncio.AbstractEventLoop:
        """Args: none; Returns: loop; Raises: RuntimeError."""

        if self._loop is not None:
            return self._loop
        try:
            self._loop = asyncio.get_running_loop()
            return self._loop
        except RuntimeError:
            if self._loop is None:
                raise
            return self._loop

    def _on_connect(self, ws: KiteTicker, response: dict[str, Any]) -> None:
        """Args: ws,response; Returns: None; Raises: none."""

        try:
            self._logger.info("Condition met: websocket_connected")
            self._connected.set()
            self._state = ConnectionState.CONNECTED
            loop = self._resolve_loop()
            loop.call_soon_threadsafe(
                lambda: asyncio.create_task(self._record_connect_success())
            )
            loop.call_soon_threadsafe(
                lambda: asyncio.create_task(self._record_heartbeat())
            )
            if self._on_connect_callback is not None:
                self._on_connect_callback()
            self._reconnect.attempts = 0
            if self._tokens:
                ws.subscribe(list(self._tokens))
                ws.set_mode(ws.MODE_FULL, list(self._tokens))
        except Exception as e:
            self._logger.error("Failure in _on_connect: %s", e)
            raise

    def _on_ticks(self, _ws: KiteTicker, ticks: list[dict[str, Any]]) -> None:
        """Args: ws,ticks; Returns: None; Raises: none."""

        try:
            if not self._on_tick_callback:
                return
            loop = self._resolve_loop()
            for tick in ticks:
                loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(self._record_tick())
                )
                maybe = self._on_tick_callback(tick)
                if asyncio.iscoroutine(maybe):
                    try:
                        running_loop = asyncio.get_running_loop()
                    except RuntimeError:
                        running_loop = None
                    if running_loop is loop:
                        loop.create_task(maybe)
                    else:
                        self._schedule_coroutine(
                            loop, cast(Coroutine[Any, Any, Any], maybe)
                        )
        except Exception as e:
            self._logger.error("Failure in _on_ticks: %s", e)
            raise


    def _schedule_coroutine(
        self,
        loop: asyncio.AbstractEventLoop,
        coroutine: Coroutine[Any, Any, Any],
    ) -> None:
        """Args: loop/coroutine; Returns: none; Raises: none."""

        def _create_task() -> None:
            loop.create_task(coroutine)

        loop.call_soon_threadsafe(_create_task)

    def _on_error(self, _ws: KiteTicker, code: int, reason: str) -> None:
        """Args: ws,code,reason; Returns: None; Raises: none."""

        try:
            self._logger.error("Failure in websocket: code=%s reason=%s", code, reason)
            self._connected.clear()
            if self._on_error_callback is not None:
                self._on_error_callback(RuntimeError(f"code={code} reason={reason}"))
            if not self._manual_disconnect:
                self._schedule_reconnect(f"error:{code}")
        except Exception as e:
            self._logger.error("Failure in _on_error: %s", e)
            raise

    def _on_close(self, _ws: KiteTicker, code: int, reason: str) -> None:
        """Args: ws,code,reason; Returns: None; Raises: none."""

        try:
            self._logger.info(
                "Condition met: websocket_closed code=%s reason=%s", code, reason
            )
            self._connected.clear()
            if self._on_disconnect_callback is not None:
                self._on_disconnect_callback()
            if not self._manual_disconnect:
                self._schedule_reconnect(f"close:{code}")
        except Exception as e:
            self._logger.error("Failure in _on_close: %s", e)
            raise

    def start(self) -> None:
        """Args: none; Returns: none; Raises: none."""

        try:
            if self._thread_loop is not None and self._thread is not None:
                return
            loop = asyncio.new_event_loop()
            self._thread_loop = loop

            def _run() -> None:
                asyncio.set_event_loop(loop)
                loop.run_forever()

            self._thread = threading.Thread(target=_run, daemon=True, name='ws-loop')
            self._thread.start()
            asyncio.run_coroutine_threadsafe(self.connect(), loop)
        except Exception as e:
            self._logger.error('Failure in start: %s', e)
            raise

    def stop(self) -> None:
        """Args: none; Returns: none; Raises: none."""

        try:
            if self._thread_loop is None:
                return
            future = asyncio.run_coroutine_threadsafe(
                self.disconnect(),
                self._thread_loop,
            )
            future.result(timeout=5)
            self._thread_loop.call_soon_threadsafe(self._thread_loop.stop)
            if self._thread is not None:
                self._thread.join(timeout=2)
            self._thread = None
            self._thread_loop = None
        except Exception as e:
            self._logger.error('Failure in stop: %s', e)
            raise

    def subscribe_tokens(self, tokens: Sequence[int], mode: str = "ltp") -> None:
        """Args: tokens/mode; Returns: none; Raises: none."""

        try:
            del mode
            new_tokens = [
                int(token) for token in tokens if int(token) not in self._tokens
            ]
            if not new_tokens:
                return
            self._tokens.update(new_tokens)
            if self._connected.is_set():
                self._ticker.subscribe(new_tokens)
                self._ticker.set_mode(self._ticker.MODE_FULL, new_tokens)
        except Exception as e:
            self._logger.error("Failure in subscribe_tokens: %s", e)

    def unsubscribe_tokens(self, tokens: Sequence[int]) -> None:
        """Args: tokens; Returns: none; Raises: none."""

        try:
            token_set = {int(token) for token in tokens}
            self._tokens = {token for token in self._tokens if token not in token_set}
            if self._connected.is_set() and token_set:
                self._ticker.unsubscribe(list(token_set))
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

    def force_reconnect(self) -> None:
        """Args: none; Returns: none; Raises: none."""

        self._schedule_reconnect('manual')

    def is_connected(self) -> bool:
        """Args: none; Returns: bool; Raises: none."""

        return self._connected.is_set()

    def connection_state(self) -> ConnectionState:
        """Args: none; Returns: state; Raises: none."""

        return self._state

    def set_client_factory(self, _fn: Callable[[], Any]) -> None:
        """Args: factory; Returns: none; Raises: none."""

        return

    def health_snapshot(self) -> dict[str, float | int | bool]:
        """Args: none; Returns: health map; Raises: none."""

        last_tick_age = 0.0
        last_heartbeat_age = 0.0
        now = time.monotonic()
        if self._health.last_tick_monotonic > 0.0:
            last_tick_age = now - self._health.last_tick_monotonic
        if self._health.last_heartbeat_monotonic > 0.0:
            last_heartbeat_age = now - self._health.last_heartbeat_monotonic
        return {
            'connected': self._connected.is_set(),
            'state': int(self._state.value),
            'consecutive_failures': self._health.consecutive_failures,
            'last_tick_age_seconds': max(0.0, last_tick_age),
            'last_heartbeat_age_seconds': max(0.0, last_heartbeat_age),
            'circuit_open': self._health.circuit_open_until_monotonic > now,
        }
