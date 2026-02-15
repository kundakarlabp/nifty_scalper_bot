"""Async WebSocket manager for Zerodha KiteTicker streaming."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from kiteconnect import KiteTicker

from nifty_scalper_bot.utils.logging import get_logger

TickCallback = Callable[[dict[str, Any]], Awaitable[None] | None]


@dataclass(slots=True)
class _ReconnectState:
    """Args: none; Returns: state; Raises: none."""

    attempts: int = 0
    task: asyncio.Task[None] | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class WebSocketManager:
    """Manage a single KiteTicker connection with async reconnect and tick dispatch."""

    def __init__(
        self,
        api_key: str,
        access_token: str,
        tokens: Sequence[int] | None = None,
        *,
        on_tick_callback: TickCallback | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
        max_backoff_seconds: float = 60.0,
        base_backoff_seconds: float = 1.0,
    ) -> None:
        """Args: init fields; Returns: None; Raises: ValueError."""

        if base_backoff_seconds <= 0:
            raise ValueError("base_backoff_seconds must be > 0")
        if max_backoff_seconds < base_backoff_seconds:
            raise ValueError("max_backoff_seconds must be >= base_backoff_seconds")

        self._logger = get_logger(__name__)
        self._loop = loop
        self._api_key = api_key
        self._access_token = access_token
        self._tokens: set[int] = {int(token) for token in (tokens or [])}
        self._on_tick_callback = on_tick_callback
        self._max_backoff_seconds = max_backoff_seconds
        self._base_backoff_seconds = base_backoff_seconds
        self._connected = asyncio.Event()
        self._shutdown = False
        self._manual_disconnect = False
        self._reconnect = _ReconnectState()
        self._connect_lock = asyncio.Lock()

        self._ticker = KiteTicker(api_key=api_key, access_token=access_token)
        self._configure_handlers()

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
            self._configure_handlers()
            await self._run_ticker_connect()
        except Exception as e:
            self._logger.error("Failure in connect: %s", e)
            raise

    async def disconnect(self) -> None:
        """Args: none; Returns: None; Raises: Exception."""

        try:
            self._logger.debug("Entered disconnect")
            self._manual_disconnect = True
            self._shutdown = True
            async with self._reconnect.lock:
                if self._reconnect.task is not None:
                    self._reconnect.task.cancel()
                    self._reconnect.task = None
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
                await asyncio.to_thread(self._ticker.connect, True)
        except Exception as e:
            self._logger.error("Failure in _run_ticker_connect: %s", e)
            raise

    def _schedule_reconnect(self, reason: str) -> None:
        """Args: reason; Returns: None; Raises: none."""

        async def _inner() -> None:
            try:
                async with self._reconnect.lock:
                    if self._shutdown or self._manual_disconnect:
                        return
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
                except Exception as connect_error:
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
                maybe = self._on_tick_callback(tick)
                if asyncio.iscoroutine(maybe):
                    try:
                        running_loop = asyncio.get_running_loop()
                    except RuntimeError:
                        running_loop = None
                    if running_loop is loop:
                        loop.create_task(maybe)
                    else:
                        loop.call_soon_threadsafe(lambda c=maybe: loop.create_task(c))
        except Exception as e:
            self._logger.error("Failure in _on_ticks: %s", e)
            raise

    def _on_error(self, _ws: KiteTicker, code: int, reason: str) -> None:
        """Args: ws,code,reason; Returns: None; Raises: none."""

        try:
            self._logger.error("Failure in websocket: code=%s reason=%s", code, reason)
            self._connected.clear()
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
            if not self._manual_disconnect:
                self._schedule_reconnect(f"close:{code}")
        except Exception as e:
            self._logger.error("Failure in _on_close: %s", e)
            raise
