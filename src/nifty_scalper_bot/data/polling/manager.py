"""Async polling manager coordinating REST-based tick fetches."""

from __future__ import annotations

import asyncio
import contextlib
import time
from datetime import datetime, timezone
from typing import Awaitable, Callable, Iterable, Optional

from nifty_scalper_bot.config import env as cfg
from nifty_scalper_bot.infra.metrics import (
    POLL_ERRORS,
    POLL_HEARTBEAT_SKIPS,
    POLL_LAST_TICK_TS,
    POLL_RECONNECTS,
    POLL_TICK_LAG_MS,
)
from nifty_scalper_bot.utils.backoff import exp_backoff
from nifty_scalper_bot.utils.logging import get_logger
from nifty_scalper_bot.utils.market import is_market_open

LOGGER = get_logger(__name__)

TickFetcher = Callable[[], Awaitable[Iterable[dict[str, object]] | None]]
TickHandler = Callable[[dict[str, object]], None]


class PollingManager:
    """Run an asyncio loop that fetches ticks and emits observability metrics."""

    def __init__(self, fetch_fn: TickFetcher, on_tick: TickHandler) -> None:
        """Initialise the polling manager with fetch and dispatch callbacks.

        Args:
            fetch_fn: Awaitable callable returning an iterable of tick dictionaries.
            on_tick: Callback executed for every tick payload received.

        Returns:
            None.

        Raises:
            ValueError: If ``fetch_fn`` or ``on_tick`` is not callable.
        """

        if not callable(fetch_fn):
            raise ValueError('fetch_fn must be callable')
        if not callable(on_tick):
            raise ValueError('on_tick must be callable')
        self._fetch_fn = fetch_fn
        self._on_tick = on_tick
        self._running = False
        self._poll_task: Optional[asyncio.Task[None]] = None
        self._last_success = time.monotonic()
        self._reconnect_attempt = 0

    async def start(self) -> None:
        """Start the polling loop if it is not already running.

        Args:
            None.

        Returns:
            None.

        Raises:
            RuntimeError: If the polling task fails to start.
        """

        LOGGER.debug('Entered PollingManager.start', extra={'event': 'poll_start'})
        if self._running:
            LOGGER.info('PollingManager.start ignored; already running')
            return
        try:
            self._running = True
            self._last_success = time.monotonic()
            self._reconnect_attempt = 0
            self._poll_task = asyncio.create_task(self._poll_loop(), name='poll.loop')
            LOGGER.info('PollingManager started', extra={'event': 'poll_started'})
        except Exception as exc:  # noqa: BLE001
            LOGGER.error('Failure in PollingManager.start: %s', exc)
            self._running = False
            self._poll_task = None
            raise RuntimeError('Failed to start polling manager') from exc

    async def stop(self) -> None:
        """Stop the polling loop and cancel the background task.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """

        LOGGER.debug('Entered PollingManager.stop', extra={'event': 'poll_stop'})
        self._running = False
        task = self._poll_task
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._poll_task = None
        LOGGER.info('PollingManager stopped', extra={'event': 'poll_stopped'})

    async def _poll_loop(self) -> None:
        """Run the internal polling loop with adaptive intervals and backoff.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """

        LOGGER.debug('Entered PollingManager._poll_loop', extra={'event': 'poll_loop'})
        while self._running:
            market_open = is_market_open()
            interval = (
                cfg.POLL_INTERVAL_SECS_MARKET()
                if market_open
                else cfg.POLL_INTERVAL_SECS_OFFHOURS()
            )
            deadline = max(interval * 3.0, 5.0)
            try:
                await self._tick_once()
                heartbeat_lag = time.monotonic() - self._last_success
                if heartbeat_lag > deadline:
                    phase = 'market' if market_open else 'offhours'
                    LOGGER.warning(
                        'Polling heartbeat missed: %.2fs>%0.2fs phase=%s',
                        heartbeat_lag,
                        deadline,
                        phase,
                        extra={
                            'event': 'poll_heartbeat_miss',
                            'lag': heartbeat_lag,
                            'deadline': deadline,
                            'phase': phase,
                        },
                    )
                    POLL_HEARTBEAT_SKIPS.labels(phase=phase).inc()
                    raise RuntimeError('poll_heartbeat_miss')
                await asyncio.sleep(max(interval, 0.0))
            except asyncio.CancelledError:  # pragma: no cover - cooperative shutdown
                raise
            except Exception as exc:  # noqa: BLE001
                LOGGER.error('Failure in PollingManager._poll_loop: %s', exc)
                POLL_ERRORS.labels(reason=type(exc).__name__).inc()
                POLL_RECONNECTS.labels(reason='error').inc()
                delay = exp_backoff(
                    self._reconnect_attempt,
                    base=0.5,
                    cap=float(cfg.POLL_RECONNECT_CAP_SECS()),
                    jitter=0.2,
                )
                self._reconnect_attempt += 1
                await asyncio.sleep(delay)

    async def _tick_once(self) -> None:
        """Fetch ticks once and dispatch them to the handler.

        Args:
            None.

        Returns:
            None.

        Raises:
            RuntimeError: If the fetch callback fails.
        """

        LOGGER.debug('Entered PollingManager._tick_once', extra={'event': 'poll_tick'})
        try:
            ticks = await self._fetch_fn()
        except Exception as exc:  # noqa: BLE001
            LOGGER.error('Failure in PollingManager._tick_once fetch: %s', exc)
            raise RuntimeError('poll_fetch_failed') from exc
        now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
        POLL_LAST_TICK_TS.set(now_ms)
        self._last_success = time.monotonic()
        self._reconnect_attempt = 0
        if not ticks:
            return
        for tick in ticks:
            exch_ts = tick.get('ts_ms') or tick.get('timestamp_ms')
            if isinstance(exch_ts, (int, float)) and exch_ts > 0:
                lag_ms = max(0, now_ms - int(exch_ts))
                POLL_TICK_LAG_MS.set(lag_ms)
            try:
                self._on_tick(tick)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error('Failure in PollingManager._tick_once on_tick: %s', exc)
                # Continue processing remaining ticks even if one handler fails.
                continue

