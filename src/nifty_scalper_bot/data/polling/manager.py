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

    # Threshold for consecutive empty depth responses before emitting critical log
    _EMPTY_DEPTH_THRESHOLD = 3
    
    # Threshold for consecutive fetch failures before reconnect
    _FAILURE_THRESHOLD = 5

    def __init__(
        self,
        fetch_fn: TickFetcher,
        on_tick: TickHandler,
        interval_ms: int = 3000,
        max_batch_size: int = 50,
    ) -> None:
        """
        Initialize the polling manager.

        Args:
            fetch_fn: Async callable that returns an iterable of tick dicts
            on_tick: Sync callable invoked per tick
            interval_ms: Polling interval in milliseconds
            max_batch_size: Maximum symbols to fetch per batch
        """
        self._fetch = fetch_fn
        self._on_tick = on_tick
        self._interval = interval_ms / 1000.0
        self._max_batch = max_batch_size
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._empty_depth_counts: dict[str, int] = {}
        self._consecutive_failures = 0
        self._last_successful_fetch = time.time()
        self._add_synthetic_depth = cfg.POLL_ADD_SYNTHETIC_DEPTH()

    def start(self) -> None:
        """Start the polling loop in a background task."""
        if self._running:
            LOGGER.warning("PollingManager already running; ignoring start request.")
            return

        self._running = True
        LOGGER.info(
            "Starting PollingManager (interval=%.2fs, batch_size=%d, synthetic_depth=%s)",
            self._interval,
            self._max_batch,
            self._add_synthetic_depth,
            extra={
                'event': 'polling_manager_start',
                'interval_s': self._interval,
                'batch_size': self._max_batch,
                'synthetic_depth': self._add_synthetic_depth,
            },
        )

        loop = asyncio.get_event_loop()
        self._task = loop.create_task(self._loop())

    def stop(self) -> None:
        """Stop the polling loop."""
        if not self._running:
            LOGGER.warning("PollingManager not running; ignoring stop request.")
            return

        self._running = False
        LOGGER.info("Stopping PollingManager", extra={'event': 'polling_manager_stop'})

        if self._task and not self._task.done():
            self._task.cancel()

    async def _loop(self) -> None:
        """Main polling loop with exponential backoff on failures."""
        backoff_gen = exp_backoff(base_ms=500, max_ms=30000)

        while self._running:
            start = time.time()

            try:
                await self._tick_once()
                
                # Reset failure counter on success
                self._consecutive_failures = 0
                self._last_successful_fetch = time.time()
                
                # Sleep for interval
                elapsed = time.time() - start
                sleep_time = max(0, self._interval - elapsed)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                else:
                    LOGGER.debug(
                        "Polling cycle took longer than interval (%.2fs > %.2fs)",
                        elapsed,
                        self._interval,
                        extra={'event': 'poll_cycle_slow', 'elapsed_s': elapsed}
                    )

            except asyncio.CancelledError:
                LOGGER.info("PollingManager loop cancelled", extra={'event': 'poll_loop_cancelled'})
                break

            except Exception as exc:
                self._consecutive_failures += 1
                POLL_ERRORS.labels(reason='fetch_exception').inc()
                
                LOGGER.error(
                    "Polling fetch failed (attempt %d/%d): %s",
                    self._consecutive_failures,
                    self._FAILURE_THRESHOLD,
                    exc,
                    extra={
                        'event': 'poll_fetch_error',
                        'consecutive_failures': self._consecutive_failures,
                        'error': str(exc),
                    },
                    exc_info=True,
                )

                # Apply exponential backoff on repeated failures
                if self._consecutive_failures >= self._FAILURE_THRESHOLD:
                    backoff_ms = next(backoff_gen)
                    LOGGER.warning(
                        "Multiple consecutive failures (%d), backing off for %dms",
                        self._consecutive_failures,
                        backoff_ms,
                        extra={'event': 'poll_backoff', 'backoff_ms': backoff_ms}
                    )
                    POLL_RECONNECTS.inc()
                    await asyncio.sleep(backoff_ms / 1000.0)
                else:
                    # Short pause before retry
                    await asyncio.sleep(1.0)

        LOGGER.info("PollingManager loop exited", extra={'event': 'poll_loop_exit'})

    async def _tick_once(self) -> None:
        """Fetch and process a single batch of ticks."""
        
        # Fetch raw ticks from broker
        try:
            raw_ticks = await self._fetch()
        except Exception as fetch_exc:
            LOGGER.error(
                "Critical: fetch_fn raised exception: %s",
                fetch_exc,
                extra={'event': 'poll_fetch_fn_exception'},
                exc_info=True,
            )
            raise

        # Handle empty or None response
        if not raw_ticks:
            POLL_HEARTBEAT_SKIPS.inc()
            LOGGER.debug(
                "Fetch returned empty/None ticks",
                extra={'event': 'poll_fetch_empty'}
            )
            return

        # Process each tick
        processed_count = 0
        skipped_count = 0
        
        for tick in raw_ticks:
            if not isinstance(tick, dict):
                LOGGER.warning(
                    "Non-dict tick received: %s",
                    type(tick).__name__,
                    extra={'event': 'poll_invalid_tick_type', 'type': type(tick).__name__}
                )
                skipped_count += 1
                continue

            try:
                # Validate and enrich tick
                if not self._validate_tick(tick):
                    skipped_count += 1
                    continue

                # Add synthetic depth if enabled and needed
                if self._add_synthetic_depth:
                    tick = self._enrich_tick_with_synthetic_depth(tick)

                # Dispatch to handler
                self._on_tick(tick)
                processed_count += 1

                # Update metrics
                POLL_LAST_TICK_TS.set(time.time())
                
                # Track tick lag if timestamp available
                tick_ts = tick.get('exchange_timestamp') or tick.get('timestamp')
                if tick_ts:
                    lag_ms = (time.time() - float(tick_ts)) * 1000
                    POLL_TICK_LAG_MS.observe(lag_ms)

            except Exception as exc:
                LOGGER.error(
                    "Failed to process tick: %s",
                    exc,
                    extra={
                        'event': 'poll_tick_processing_error',
                        'tick_sample': _short_tick_repr(tick),
                    },
                    exc_info=True,
                )
                skipped_count += 1
                continue

        # Log batch summary
        if processed_count > 0 or skipped_count > 0:
            LOGGER.debug(
                "Batch processed: %d successful, %d skipped",
                processed_count,
                skipped_count,
                extra={
                    'event': 'poll_batch_complete',
                    'processed': processed_count,
                    'skipped': skipped_count,
                }
            )

    def _validate_tick(self, tick: dict) -> bool:
        """
        Validate tick has minimum required data.
        
        Returns True if tick is valid, False if should be skipped.
        """
        
        # Identify symbol key
        symbol_key = (
            tick.get('tradingsymbol')
            or tick.get('instrument_token')
            or tick.get('symbol')
        )
        
        if not symbol_key:
            LOGGER.debug(
                "Tick missing symbol identifier: %s",
                _short_tick_repr(tick),
                extra={'event': 'poll_tick_no_symbol'}
            )
            return False

        symbol_key = str(symbol_key)

        # Extract price data
        ltp = _safe_float(
            tick.get('ltp')
            or tick.get('last_price')
            or tick.get('close')
            or tick.get('price')
        )

        # Extract depth data
        buy_depth = tick.get('buy') or tick.get('depth', {}).get('buy', [])
        sell_depth = tick.get('sell') or tick.get('depth', {}).get('sell', [])

        # Check if depth is empty
        has_empty_depth = (
            isinstance(buy_depth, (list, tuple)) and len(buy_depth) == 0 and
            isinstance(sell_depth, (list, tuple)) and len(sell_depth) == 0
        )

        # ✅ CRITICAL LOGIC: Only reject if BOTH conditions are true:
        # 1. No LTP/price data (ltp is None)
        # 2. Empty depth arrays
        if ltp is None and has_empty_depth:
            # Completely empty tick - reject
            self._empty_depth_counts[symbol_key] = self._empty_depth_counts.get(symbol_key, 0) + 1
            count = self._empty_depth_counts[symbol_key]
            
            POLL_ERRORS.labels(reason='empty_depth').inc()
            
            if count >= self._EMPTY_DEPTH_THRESHOLD:
                LOGGER.critical(
                    "Persistent empty ticks for %s (count=%d) - likely invalid symbol",
                    symbol_key,
                    count,
                    extra={
                        'event': 'poll_empty_depth_persistent',
                        'symbol': symbol_key,
                        'count': count,
                    }
                )
            else:
                LOGGER.warning(
                    "Empty tick for %s (count=%d): %s",
                    symbol_key,
                    count,
                    _short_tick_repr(tick),
                    extra={
                        'event': 'poll_empty_depth',
                        'symbol': symbol_key,
                        'count': count,
                    }
                )
            
            return False

        # ✅ CRITICAL: If we have LTP but no depth, it's VALID (off-market hours)
        if ltp is not None and has_empty_depth:
            LOGGER.debug(
                "Accepting tick with LTP but no depth for %s (off-market hours)",
                symbol_key,
                extra={
                    'event': 'poll_offhours_tick',
                    'symbol': symbol_key,
                    'ltp': ltp,
                }
            )
            # Reset empty depth counter since we got valid data
            self._empty_depth_counts.pop(symbol_key, None)

        # Reset empty depth counter on valid tick
        if symbol_key in self._empty_depth_counts:
            self._empty_depth_counts.pop(symbol_key, None)

        return True

    def _enrich_tick_with_synthetic_depth(self, tick: dict) -> dict:
        """
        Add synthetic market depth if missing (for off-market testing).
        
        Only adds depth if:
        1. Tick has LTP but no depth
        2. LTP is positive
        """
        
        # Check if depth already exists
        buy_depth = tick.get('buy') or tick.get('depth', {}).get('buy', [])
        sell_depth = tick.get('sell') or tick.get('depth', {}).get('sell', [])
        
        has_depth = (
            isinstance(buy_depth, (list, tuple)) and len(buy_depth) > 0 and
            isinstance(sell_depth, (list, tuple)) and len(sell_depth) > 0 and
            buy_depth[0].get('price', 0) > 0
        )
        
        if has_depth:
            return tick  # Already has real depth

        # Get LTP
        ltp = _safe_float(
            tick.get('ltp')
            or tick.get('last_price')
            or tick.get('close')
            or tick.get('price')
        )
        
        if ltp is None or ltp <= 0:
            return tick  # Can't create synthetic depth

        # Calculate realistic spread
        # For NIFTY options: typically 0.3-0.8% spread
        spread_pct = 0.005  # 0.5%
        spread = max(ltp * spread_pct, 0.05)  # Min 0.05 (1 tick)
        tick_size = 0.05  # NIFTY options tick size

        # Build 5-level depth on each side
        synthetic_buy = []
        synthetic_sell = []
        
        for level in range(1, 6):
            # Decreasing liquidity as you move away from LTP
            quantity = 75 * (6 - level)  # 375, 300, 225, 150, 75
            orders = level + 2  # 3, 4, 5, 6, 7
            
            # Calculate prices (rounded to tick size)
            buy_price = round((ltp - (spread * level)) / tick_size) * tick_size
            sell_price = round((ltp + (spread * level)) / tick_size) * tick_size
            
            synthetic_buy.append({
                "quantity": quantity,
                "price": max(buy_price, tick_size),  # Don't go below tick size
                "orders": orders,
            })
            
            synthetic_sell.append({
                "quantity": quantity,
                "price": sell_price,
                "orders": orders,
            })

        # Update tick with synthetic depth
        tick['buy'] = synthetic_buy
        tick['sell'] = synthetic_sell
        
        # Add best bid/ask for convenience
        tick['bid'] = synthetic_buy[0]['price']
        tick['ask'] = synthetic_sell[0]['price']
        
        # Mark as synthetic for debugging
        tick['_synthetic_depth'] = True
        
        symbol = tick.get('tradingsymbol') or tick.get('instrument_token')
        LOGGER.debug(
            "Added synthetic depth for %s (ltp=%.2f, bid=%.2f, ask=%.2f)",
            symbol,
            ltp,
            tick['bid'],
            tick['ask'],
            extra={
                'event': 'poll_synthetic_depth',
                'symbol': symbol,
                'ltp': ltp,
                'spread_pct': spread_pct,
            }
        )
        
        return tick


def _safe_float(value) -> Optional[float]:
    """Safely convert value to float, returning None on failure."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _short_tick_repr(tick: dict) -> str:
    """Create a short string representation of tick for logging."""
    if not isinstance(tick, dict):
        return str(tick)
    
    # Show only key fields
    symbol = tick.get('tradingsymbol') or tick.get('instrument_token') or tick.get('symbol')
    ltp = tick.get('ltp') or tick.get('last_price')
    
    return f"{{symbol={symbol}, ltp={ltp}, keys={list(tick.keys())[:8]}}}"