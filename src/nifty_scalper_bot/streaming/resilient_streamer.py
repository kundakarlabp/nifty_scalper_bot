"""High level resilient streamer built on top of :class:`WebSocketManager`.

The goal is to keep the existing websocket connection logic untouched
while layering additional safeguards: automatic re-subscription, gap
detection, bounded REST backfill and instrumentation hooks for health
checks.  The streamer exposes an ``asyncio`` friendly interface so that
tick dispatching never blocks the hot path.
"""

from __future__ import annotations

import asyncio
from asyncio import QueueEmpty, QueueFull
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable

from nifty_scalper_bot.config.settings import Settings
from nifty_scalper_bot.data.rest.zerodha_client import ZerodhaKiteClient
from nifty_scalper_bot.data.websocket.manager import WebSocketManager
from nifty_scalper_bot.infra.metrics import METRICS
from nifty_scalper_bot.utils.logging import get_logger
from nifty_scalper_bot.utils.metrics import Counter, Gauge, Histogram

TickHandler = Callable[[dict[str, object]], Awaitable[None] | None]


@dataclass(slots=True)
class StreamMetrics:
    """Prometheus metric handles shared with the health endpoint."""

    tick_latency_ms: Any
    queue_depth: Any
    gap_events: Any
    backfill_requests: Any
    deduplicated: Any


class ResilientStreamer:
    """Wrap :class:`WebSocketManager` with dedupe, gap detection and backfill."""

    def __init__(
        self,
        websocket: WebSocketManager,
        rest_client: ZerodhaKiteClient,
        settings: Settings,
        *,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._ws = websocket
        self._rest = rest_client
        self._settings = settings
        self._loop = loop
        self._logger = get_logger(__name__)
        self._queue: asyncio.Queue[dict[str, object]] = asyncio.Queue(
            maxsize=settings.streamer.queue_size
        )
        self._handlers: list[TickHandler] = []
        self._dispatcher_task: asyncio.Task[None] | None = None
        self._backfill_tasks: set[asyncio.Task[None]] = set()
        self._last_tick_signature: dict[str, tuple[int, float | None]] = {}
        self._last_gap_at: dict[str, float] = {}
        self._latency_ms = Histogram(
            "resilient_tick_latency_ms",
            "Latency between tick timestamp and processing (ms)",
            buckets=(1, 5, 10, 50, 100, 250, 500, 1000, 2500),
        )
        self._queue_depth = Gauge(
            "resilient_tick_queue_depth",
            "Queued ticks awaiting processing",
        )
        self._gaps = Counter(
            "resilient_gap_events_total",
            "Count of detected timestamp gaps triggering backfill",
        )
        self._backfills = Counter(
            "resilient_backfill_requests_total",
            "Historical data backfill requests",
        )
        self._deduped = Counter(
            "resilient_ticks_deduplicated_total",
            "Ticks dropped due to deduplication",
        )
        self.metrics = StreamMetrics(
            tick_latency_ms=self._latency_ms,
            queue_depth=self._queue_depth,
            gap_events=self._gaps,
            backfill_requests=self._backfills,
            deduplicated=self._deduped,
        )
        self._ws.on_tick = self._handle_tick

    # ------------------------------------------------------------------
    # Lifecycle management
    def register_handler(self, handler: TickHandler) -> None:
        """Register a coroutine or synchronous handler for processed ticks."""

        self._handlers.append(handler)

    async def start(self) -> None:
        """Start dispatch loop and underlying websocket."""

        if self._dispatcher_task is not None:
            return
        self._loop = asyncio.get_running_loop()
        self._dispatcher_task = self._loop.create_task(self._dispatch_loop())
        self._ws.start()

    async def stop(self) -> None:
        """Stop dispatch loop and cancel outstanding tasks."""

        self._ws.stop()
        if self._dispatcher_task is not None:
            self._dispatcher_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._dispatcher_task
        self._dispatcher_task = None
        while self._backfill_tasks:
            task = self._backfill_tasks.pop()
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        self._loop = None

    # ------------------------------------------------------------------
    # Public helpers exposed to tests/health endpoints
    def simulate_disconnect(self) -> None:  # pragma: no cover
        """Force a reconnect cycle for testing resiliency."""

        self._ws.force_reconnect()

    def is_connected(self) -> bool:  # pragma: no cover
        """Return ``True`` if websocket is within heartbeat tolerance."""

        return self._ws.is_connected()

    def backlog_size(self) -> int:  # pragma: no cover
        """Return queued tick backlog."""

        return self._queue.qsize()

    # ------------------------------------------------------------------
    def _handle_tick(self, tick: dict[str, object]) -> None:  # pragma: no cover
        payload = dict(tick)
        loop = self._loop
        if loop is None:
            self._enqueue_tick(payload)
            return
        if loop.is_closed():
            self._logger.debug("Dropping tick because resilient loop is closed")
            return
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is loop:
            self._enqueue_tick(payload)
            return
        try:
            loop.call_soon_threadsafe(self._enqueue_tick, payload)
        except RuntimeError:
            self._logger.debug("Dropping tick because resilient loop is unavailable")

    def _enqueue_tick(self, tick: dict[str, object]) -> None:
        symbol = str(tick.get("symbol") or tick.get("tradingsymbol") or "")
        timestamp = self._extract_timestamp_ms(tick)
        price = self._extract_price(tick)
        signature = (timestamp, price)
        if symbol:
            previous = self._last_tick_signature.get(symbol)
            if previous == signature:
                try:
                    self._deduped.inc()
                except Exception:  # pragma: no cover - optional metrics
                    pass
                return
            self._last_tick_signature[symbol] = signature
            self._detect_gap(symbol, timestamp)

        if self._queue.full():
            with suppress(QueueEmpty):
                _ = self._queue.get_nowait()

        try:
            self._queue.put_nowait(dict(tick))
        except QueueFull:  # pragma: no cover - defensive
            with suppress(QueueEmpty):
                _ = self._queue.get_nowait()
            self._queue.put_nowait(dict(tick))
        try:
            self._queue_depth.set(self._queue.qsize())
        except Exception:  # pragma: no cover - optional metrics
            pass

    async def _dispatch_loop(self) -> None:  # pragma: no cover
        while True:
            tick = await self._queue.get()
            try:
                self._queue_depth.set(self._queue.qsize())
            except Exception:  # pragma: no cover - optional metrics
                pass
            latency_ms = self._record_latency(tick)
            for handler in list(self._handlers):
                try:
                    result = handler(dict(tick))
                    if asyncio.iscoroutine(result):
                        await result
                except Exception:  # noqa: BLE001
                    self._logger.exception(
                        "Tick handler failed", extra={"latency_ms": latency_ms}
                    )

    # ------------------------------------------------------------------
    # Gap detection & backfill
    def _detect_gap(self, symbol: str, timestamp_ms: int) -> None:  # pragma: no cover
        previous = self._last_gap_at.get(symbol)
        if previous is None:
            self._last_gap_at[symbol] = float(timestamp_ms)
            return
        gap_seconds = (timestamp_ms - previous) / 1000.0
        self._last_gap_at[symbol] = float(timestamp_ms)
        if gap_seconds <= self._settings.streamer.gap_seconds:
            return
        self._logger.warning(
            "Tick gap detected",
            extra={"symbol": symbol, "gap_seconds": round(gap_seconds, 3)},
        )
        try:
            self._gaps.inc()
        except Exception:  # pragma: no cover - optional metrics
            pass
        if self._loop is None or self._loop.is_closed():
            return
        task = self._loop.create_task(
            self._request_backfill(symbol, timestamp_ms, gap_seconds)
        )
        self._backfill_tasks.add(task)

        def _cleanup(t: asyncio.Task[None]) -> None:
            self._backfill_tasks.discard(t)

        task.add_done_callback(_cleanup)

    async def _request_backfill(
        self, symbol: str, timestamp_ms: int, gap_seconds: float
    ) -> None:  # pragma: no cover
        try:
            self._backfills.inc()
        except Exception:  # pragma: no cover - optional metrics
            pass
        end = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)
        start = end - self._bounded_gap_delta(gap_seconds)
        candles = await self._loop.run_in_executor(
            None,
            lambda: self._rest.get_ohlc(
                symbol,
                self._settings.streamer.backfill_interval,
                start.isoformat(),
                end.isoformat(),
            ),
        )
        for candle in candles[-self._settings.streamer.max_backfill_candles :]:
            payload = {
                "symbol": symbol,
                "ts_ms": int(self._extract_candle_ts(candle)),
                "price": (
                    candle[-1]
                    if isinstance(candle, (list, tuple)) and len(candle) >= 5
                    else None
                ),
                "source": "backfill",
            }
            if self._queue.full():
                with suppress(QueueEmpty):
                    _ = self._queue.get_nowait()
            try:
                self._queue.put_nowait(payload)
            except QueueFull:  # pragma: no cover - defensive
                with suppress(QueueEmpty):
                    _ = self._queue.get_nowait()
                self._queue.put_nowait(payload)

    def _bounded_gap_delta(self, gap_seconds: float) -> timedelta:  # pragma: no cover
        max_gap = max(gap_seconds, self._settings.streamer.gap_seconds)
        minutes = max_gap / 60.0
        return timedelta(minutes=min(minutes, 60.0))

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_timestamp_ms(tick: dict[str, object]) -> int:  # pragma: no cover
        raw = tick.get("ts_ms") or tick.get("timestamp") or tick.get("last_trade_time")
        if isinstance(raw, datetime):
            return int(raw.timestamp() * 1000)
        if isinstance(raw, (int, float)):
            value = float(raw)
            return int(value if value > 1_000_000_000_000 else value * 1000)
        if isinstance(raw, str):
            with suppress(ValueError):
                dt = datetime.fromisoformat(raw)
                return int(dt.timestamp() * 1000)
        return int(datetime.now(timezone.utc).timestamp() * 1000)

    @staticmethod
    def _extract_price(tick: dict[str, object]) -> float | None:  # pragma: no cover
        for key in ("ltp", "price", "last_price"):
            value = tick.get(key)
            if value is None:
                continue
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                with suppress(ValueError):
                    return float(value)
        return None

    def _record_latency(self, tick: dict[str, object]) -> float:  # pragma: no cover
        timestamp_ms = self._extract_timestamp_ms(tick)
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        latency_ms = max(now_ms - timestamp_ms, 0)
        try:
            self._latency_ms.observe(latency_ms)
        except Exception:  # pragma: no cover - optional metrics
            pass
        symbol = str(tick.get("symbol") or tick.get("tradingsymbol") or "")
        try:
            METRICS.observe_tick(
                symbol=symbol,
                latency_seconds=float(latency_ms) / 1000.0,
                staleness_seconds=None,
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in ResilientStreamer._record_latency: %s",
                exc,
                extra={"event": "stream_latency_metric_error", "symbol": symbol},
            )
        return float(latency_ms)

    @staticmethod
    def _extract_candle_ts(candle: object) -> int:  # pragma: no cover
        if isinstance(candle, (list, tuple)) and candle:
            raw = candle[0]
            if isinstance(raw, str):
                with suppress(ValueError):
                    dt = datetime.fromisoformat(raw)
                    return int(dt.timestamp() * 1000)
        return int(datetime.now(timezone.utc).timestamp() * 1000)


__all__ = ["ResilientStreamer", "StreamMetrics"]
