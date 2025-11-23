"""Async polling manager coordinating REST-based tick fetches with production-grade resilience."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import defaultdict
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
    """
    Production-grade async polling manager with comprehensive error handling,
    synthetic depth generation, and intelligent tick validation.
    
    Features:
    - Smart empty depth detection (off-hours vs invalid symbols)
    - Synthetic market depth for off-hours testing
    - Exponential backoff on failures
    - Comprehensive metrics and observability
    - Circuit breaker pattern for consecutive failures
    - Tick deduplication and staleness tracking
    """

    # Thresholds
    _EMPTY_DEPTH_THRESHOLD = 3
    _FAILURE_THRESHOLD = 5
    _MAX_TICK_AGE_SECONDS = 300  # Reject ticks older than 5 minutes
    _DEDUP_WINDOW_SECONDS = 2.0  # Deduplication window
    
    # Synthetic depth configuration
    _SYNTHETIC_SPREAD_PCT = 0.005  # 0.5% spread
    _SYNTHETIC_MIN_SPREAD = 0.05  # Minimum 1 tick (₹0.05)
    _TICK_SIZE = 0.05  # NIFTY options tick size

    def __init__(
        self,
        fetch_fn: TickFetcher,
        on_tick: TickHandler,
        interval_ms: int = 3000,
        max_batch_size: int = 50,
    ) -> None:
        """
        Initialize the polling manager with production-grade defaults.

        Args:
            fetch_fn: Async callable returning tick dicts
            on_tick: Sync handler for processed ticks
            interval_ms: Polling interval in milliseconds
            max_batch_size: Maximum symbols per batch
        """
        # Core components
        self._fetch = fetch_fn
        self._on_tick = on_tick
        self._interval = interval_ms / 1000.0
        self._max_batch = max_batch_size
        
        # Task management
        self._task: Optional[asyncio.Task] = None
        self._running = False
        
        # State tracking
        self._empty_depth_counts: dict[str, int] = {}
        self._consecutive_failures = 0
        self._last_successful_fetch = time.time()
        self._tick_history: dict[str, tuple[float, dict]] = {}  # symbol -> (timestamp, tick)
        self._processed_count = 0
        self._skipped_count = 0
        self._synthetic_depth_count = 0
        
        # Configuration
        self._add_synthetic_depth = cfg.POLL_ADD_SYNTHETIC_DEPTH()
        self._enable_deduplication = cfg.POLL_ENABLE_DEDUP() if hasattr(cfg, 'POLL_ENABLE_DEDUP') else True
        self._strict_validation = cfg.POLL_STRICT_VALIDATION() if hasattr(cfg, 'POLL_STRICT_VALIDATION') else False
        
        # Metrics
        self._metrics = {
            'total_processed': 0,
            'total_skipped': 0,
            'total_duplicates': 0,
            'total_synthetic': 0,
            'last_reset': time.time()
        }

    def start(self) -> None:
        """Start the polling loop with comprehensive logging."""
        if self._running:
            LOGGER.warning(
                "PollingManager already running; ignoring start request",
                extra={'event': 'polling_manager_already_running'}
            )
            return

        self._running = True
        
        LOGGER.info(
            "Starting PollingManager (interval=%.2fs, batch=%d, synthetic=%s, dedup=%s)",
            self._interval,
            self._max_batch,
            self._add_synthetic_depth,
            self._enable_deduplication,
            extra={
                'event': 'polling_manager_start',
                'config': {
                    'interval_s': self._interval,
                    'batch_size': self._max_batch,
                    'synthetic_depth': self._add_synthetic_depth,
                    'deduplication': self._enable_deduplication,
                    'strict_validation': self._strict_validation,
                }
            },
        )

        loop = asyncio.get_event_loop()
        self._task = loop.create_task(self._loop())

    def stop(self) -> None:
        """Stop the polling loop and log final metrics."""
        if not self._running:
            LOGGER.warning(
                "PollingManager not running; ignoring stop request",
                extra={'event': 'polling_manager_not_running'}
            )
            return

        self._running = False
        
        # Log final metrics
        uptime = time.time() - self._metrics['last_reset']
        LOGGER.info(
            "Stopping PollingManager (uptime=%.1fs, processed=%d, skipped=%d, synthetic=%d)",
            uptime,
            self._metrics['total_processed'],
            self._metrics['total_skipped'],
            self._metrics['total_synthetic'],
            extra={
                'event': 'polling_manager_stop',
                'metrics': self._metrics,
                'uptime_seconds': uptime
            }
        )

        if self._task and not self._task.done():
            self._task.cancel()

    async def _loop(self) -> None:
        """Main polling loop with circuit breaker and exponential backoff."""
        backoff_gen = exp_backoff(base_ms=500, max_ms=30000)
        last_health_log = time.time()
        health_log_interval = 300  # Log health every 5 minutes

        while self._running:
            start = time.time()

            try:
                await self._tick_once()
                
                # Reset failure counter on success
                if self._consecutive_failures > 0:
                    LOGGER.info(
                        "Polling recovered after %d failures",
                        self._consecutive_failures,
                        extra={'event': 'poll_recovery', 'failures': self._consecutive_failures}
                    )
                    self._consecutive_failures = 0
                
                self._last_successful_fetch = time.time()
                
                # Periodic health logging
                if time.time() - last_health_log >= health_log_interval:
                    self._log_health_status()
                    last_health_log = time.time()
                
                # Calculate sleep time
                elapsed = time.time() - start
                sleep_time = max(0, self._interval - elapsed)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                else:
                    LOGGER.debug(
                        "Polling cycle exceeded interval (%.2fs > %.2fs)",
                        elapsed,
                        self._interval,
                        extra={'event': 'poll_cycle_slow', 'elapsed_s': elapsed, 'target_s': self._interval}
                    )

            except asyncio.CancelledError:
                LOGGER.info(
                    "PollingManager loop cancelled gracefully",
                    extra={'event': 'poll_loop_cancelled'}
                )
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
                        'error_type': type(exc).__name__,
                        'error': str(exc),
                    },
                    exc_info=True,
                )

                # Circuit breaker: exponential backoff on repeated failures
                if self._consecutive_failures >= self._FAILURE_THRESHOLD:
                    backoff_ms = next(backoff_gen)
                    LOGGER.warning(
                        "Circuit breaker triggered: %d consecutive failures, backing off %dms",
                        self._consecutive_failures,
                        backoff_ms,
                        extra={
                            'event': 'poll_circuit_breaker',
                            'failures': self._consecutive_failures,
                            'backoff_ms': backoff_ms
                        }
                    )
                    POLL_RECONNECTS.inc()
                    await asyncio.sleep(backoff_ms / 1000.0)
                else:
                    # Short pause before retry
                    await asyncio.sleep(1.0)

        LOGGER.info(
            "PollingManager loop exited cleanly",
            extra={'event': 'poll_loop_exit', 'final_metrics': self._metrics}
        )

    async def _tick_once(self) -> None:
        """Fetch and process a single batch of ticks with comprehensive error handling."""
        
        # Fetch raw ticks from broker
        try:
            raw_ticks = await self._fetch()
        except Exception as fetch_exc:
            LOGGER.error(
                "Critical: fetch_fn raised exception: %s",
                fetch_exc,
                extra={
                    'event': 'poll_fetch_fn_exception',
                    'error_type': type(fetch_exc).__name__
                },
                exc_info=True,
            )
            raise

        # Handle empty or None response
        if not raw_ticks:
            POLL_HEARTBEAT_SKIPS.inc()
            LOGGER.debug(
                "Fetch returned empty/None ticks (heartbeat)",
                extra={'event': 'poll_fetch_empty'}
            )
            return

        # Process each tick
        batch_start = time.time()
        processed_count = 0
        skipped_count = 0
        duplicate_count = 0
        synthetic_count = 0
        
        for tick in raw_ticks:
            # Type validation
            if not isinstance(tick, dict):
                LOGGER.warning(
                    "Non-dict tick received: %s",
                    type(tick).__name__,
                    extra={
                        'event': 'poll_invalid_tick_type',
                        'type': type(tick).__name__,
                        'value': str(tick)[:100]
                    }
                )
                skipped_count += 1
                continue

            try:
                # Deduplication check
                if self._enable_deduplication and self._is_duplicate(tick):
                    duplicate_count += 1
                    continue

                # Validate tick
                if not self._validate_tick(tick):
                    skipped_count += 1
                    continue

                # Enrich with synthetic depth if needed
                was_enriched = False
                if self._add_synthetic_depth:
                    original_has_depth = self._has_real_depth(tick)
                    tick = self._enrich_tick_with_synthetic_depth(tick)
                    if not original_has_depth and self._has_real_depth(tick):
                        synthetic_count += 1
                        was_enriched = True

                # Dispatch to handler
                self._on_tick(tick)
                processed_count += 1

                # Update tick history for deduplication
                if self._enable_deduplication:
                    self._update_tick_history(tick)

                # Update metrics
                POLL_LAST_TICK_TS.set(time.time())
                
                # Track tick lag if timestamp available
                tick_ts = tick.get('exchange_timestamp') or tick.get('timestamp')
                if tick_ts:
                    try:
                        lag_ms = (time.time() - float(tick_ts)) * 1000
                        if lag_ms >= 0:  # Only track positive lag
                            POLL_TICK_LAG_MS.observe(lag_ms)
                    except (TypeError, ValueError):
                        pass

            except Exception as exc:
                LOGGER.error(
                    "Failed to process tick: %s",
                    exc,
                    extra={
                        'event': 'poll_tick_processing_error',
                        'tick_sample': _short_tick_repr(tick),
                        'error_type': type(exc).__name__,
                    },
                    exc_info=True,
                )
                skipped_count += 1
                continue

        # Update global metrics
        self._metrics['total_processed'] += processed_count
        self._metrics['total_skipped'] += skipped_count
        self._metrics['total_duplicates'] += duplicate_count
        self._metrics['total_synthetic'] += synthetic_count

        # Log batch summary
        batch_duration = time.time() - batch_start
        if processed_count > 0 or skipped_count > 0:
            LOGGER.debug(
                "Batch processed in %.3fs: %d ok, %d skipped, %d dupes, %d synthetic",
                batch_duration,
                processed_count,
                skipped_count,
                duplicate_count,
                synthetic_count,
                extra={
                    'event': 'poll_batch_complete',
                    'processed': processed_count,
                    'skipped': skipped_count,
                    'duplicates': duplicate_count,
                    'synthetic': synthetic_count,
                    'duration_s': batch_duration
                }
            )

    def _validate_tick(self, tick: dict) -> bool:
        """
        Validate tick with production-grade checks.
        
        Returns:
            True if tick is valid and should be processed
            False if tick should be skipped
        """
        
        # Extract symbol identifier
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

        # ✅ CRITICAL LOGIC: Only reject if BOTH conditions true
        # 1. No LTP/price data
        # 2. Empty depth arrays
        if ltp is None and has_empty_depth:
            # Completely empty tick - likely invalid symbol
            self._empty_depth_counts[symbol_key] = self._empty_depth_counts.get(symbol_key, 0) + 1
            count = self._empty_depth_counts[symbol_key]
            
            POLL_ERRORS.labels(reason='empty_depth').inc()
            
            if count >= self._EMPTY_DEPTH_THRESHOLD:
                LOGGER.critical(
                    "Persistent empty ticks for %s (count=%d) - likely invalid symbol or expired contract",
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
                    "Empty tick for %s (count=%d/%d): %s",
                    symbol_key,
                    count,
                    self._EMPTY_DEPTH_THRESHOLD,
                    _short_tick_repr(tick),
                    extra={
                        'event': 'poll_empty_depth',
                        'symbol': symbol_key,
                        'count': count,
                    }
                )
            
            return False

        # ✅ CRITICAL: LTP with no depth is VALID (off-market hours)
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
            # Reset empty depth counter
            self._empty_depth_counts.pop(symbol_key, None)

        # Reset counter on valid tick with depth
        if ltp is not None and not has_empty_depth:
            self._empty_depth_counts.pop(symbol_key, None)

        # Validate LTP is positive and reasonable
        if ltp is not None:
            if ltp <= 0:
                LOGGER.warning(
                    "Invalid LTP (<= 0) for %s: %.4f",
                    symbol_key,
                    ltp,
                    extra={'event': 'poll_invalid_ltp', 'symbol': symbol_key, 'ltp': ltp}
                )
                return False
            
            # Sanity check: NIFTY options typically < ₹5000
            if self._strict_validation and ltp > 10000:
                LOGGER.warning(
                    "Suspiciously high LTP for %s: %.2f",
                    symbol_key,
                    ltp,
                    extra={'event': 'poll_suspicious_ltp', 'symbol': symbol_key, 'ltp': ltp}
                )

        # Validate timestamp if present
        tick_ts = tick.get('exchange_timestamp') or tick.get('timestamp')
        if tick_ts is not None:
            try:
                tick_age = time.time() - float(tick_ts)
                if tick_age < 0:
                    LOGGER.warning(
                        "Tick from future for %s (age=%.1fs)",
                        symbol_key,
                        tick_age,
                        extra={'event': 'poll_future_tick', 'symbol': symbol_key, 'age_s': tick_age}
                    )
                    return False
                
                if tick_age > self._MAX_TICK_AGE_SECONDS:
                    LOGGER.warning(
                        "Stale tick for %s (age=%.1fs > %ds)",
                        symbol_key,
                        tick_age,
                        self._MAX_TICK_AGE_SECONDS,
                        extra={'event': 'poll_stale_tick', 'symbol': symbol_key, 'age_s': tick_age}
                    )
                    return False
            except (TypeError, ValueError):
                pass

        return True

    def _enrich_tick_with_synthetic_depth(self, tick: dict) -> dict:
        """
        Add production-grade synthetic market depth for off-hours testing.
        
        Features:
        - Realistic spread modeling (0.5% for NIFTY options)
        - Proper tick size alignment (₹0.05)
        - Decreasing liquidity away from LTP
        - Best bid/ask calculation
        """
        
        # Check if depth already exists
        if self._has_real_depth(tick):
            return tick

        # Get LTP
        ltp = _safe_float(
            tick.get('ltp')
            or tick.get('last_price')
            or tick.get('close')
            or tick.get('price')
        )
        
        if ltp is None or ltp <= 0:
            return tick

        # Calculate realistic spread
        spread = max(ltp * self._SYNTHETIC_SPREAD_PCT, self._SYNTHETIC_MIN_SPREAD)

        # Build 5-level depth on each side
        synthetic_buy = []
        synthetic_sell = []
        
        for level in range(1, 6):
            # Decreasing liquidity (375 → 75 lots in steps of 75)
            quantity = 75 * (6 - level)
            orders = level + 2  # 3 to 7 orders per level
            
            # Calculate prices aligned to tick size
            buy_price = round((ltp - (spread * level)) / self._TICK_SIZE) * self._TICK_SIZE
            sell_price = round((ltp + (spread * level)) / self._TICK_SIZE) * self._TICK_SIZE
            
            synthetic_buy.append({
                "quantity": quantity,
                "price": max(buy_price, self._TICK_SIZE),  # Floor at tick size
                "orders": orders,
            })
            
            synthetic_sell.append({
                "quantity": quantity,
                "price": sell_price,
                "orders": orders,
            })

        # Update tick
        tick['buy'] = synthetic_buy
        tick['sell'] = synthetic_sell
        tick['bid'] = synthetic_buy[0]['price']
        tick['ask'] = synthetic_sell[0]['price']
        tick['_synthetic_depth'] = True
        tick['_synthetic_timestamp'] = time.time()
        
        symbol = tick.get('tradingsymbol') or tick.get('instrument_token')
        LOGGER.debug(
            "Synthetic depth added for %s (ltp=%.2f, bid=%.2f, ask=%.2f, spread=%.2f%%)",
            symbol,
            ltp,
            tick['bid'],
            tick['ask'],
            (tick['ask'] - tick['bid']) / ltp * 100,
            extra={
                'event': 'poll_synthetic_depth',
                'symbol': symbol,
                'ltp': ltp,
                'bid': tick['bid'],
                'ask': tick['ask'],
                'spread_pct': self._SYNTHETIC_SPREAD_PCT * 100,
            }
        )
        
        return tick

    def _has_real_depth(self, tick: dict) -> bool:
        """Check if tick has valid market depth."""
        buy_depth = tick.get('buy') or tick.get('depth', {}).get('buy', [])
        sell_depth = tick.get('sell') or tick.get('depth', {}).get('sell', [])
        
        return (
            isinstance(buy_depth, (list, tuple)) and len(buy_depth) > 0 and
            isinstance(sell_depth, (list, tuple)) and len(sell_depth) > 0 and
            buy_depth[0].get('price', 0) > 0 and
            sell_depth[0].get('price', 0) > 0
        )

    def _is_duplicate(self, tick: dict) -> bool:
        """Check if tick is a duplicate within deduplication window."""
        symbol = (
            tick.get('tradingsymbol')
            or tick.get('instrument_token')
            or tick.get('symbol')
        )
        
        if not symbol:
            return False
        
        symbol = str(symbol)
        current_time = time.time()
        
        # Check if we've seen this symbol recently
        if symbol in self._tick_history:
            last_ts, last_tick = self._tick_history[symbol]
            
            # Within deduplication window?
            if current_time - last_ts <= self._DEDUP_WINDOW_SECONDS:
                # Compare key fields
                current_ltp = _safe_float(tick.get('ltp') or tick.get('last_price'))
                last_ltp = _safe_float(last_tick.get('ltp') or last_tick.get('last_price'))
                
                if current_ltp == last_ltp:
                    LOGGER.debug(
                        "Duplicate tick for %s (ltp=%.2f, age=%.1fs)",
                        symbol,
                        current_ltp or 0,
                        current_time - last_ts,
                        extra={'event': 'poll_duplicate_tick', 'symbol': symbol}
                    )
                    return True
        
        return False

    def _update_tick_history(self, tick: dict) -> None:
        """Update tick history for deduplication."""
        symbol = (
            tick.get('tradingsymbol')
            or tick.get('instrument_token')
            or tick.get('symbol')
        )
        
        if symbol:
            self._tick_history[str(symbol)] = (time.time(), tick)
            
            # Cleanup old entries (> 10 seconds)
            current_time = time.time()
            expired = [
                s for s, (ts, _) in self._tick_history.items()
                if current_time - ts > 10
            ]
            for s in expired:
                del self._tick_history[s]

    def _log_health_status(self) -> None:
        """Log periodic health status."""
        uptime = time.time() - self._metrics['last_reset']
        ticks_per_sec = self._metrics['total_processed'] / max(uptime, 1)
        
        LOGGER.info(
            "PollingManager health: uptime=%.1fs, tps=%.2f, processed=%d, skipped=%d, synthetic=%d, dupes=%d",
            uptime,
            ticks_per_sec,
            self._metrics['total_processed'],
            self._metrics['total_skipped'],
            self._metrics['total_synthetic'],
            self._metrics['total_duplicates'],
            extra={
                'event': 'poll_health_status',
                'metrics': {
                    **self._metrics,
                    'uptime_seconds': uptime,
                    'ticks_per_second': ticks_per_sec,
                    'tracked_symbols': len(self._tick_history)
                }
            }
        )


def _safe_float(value) -> Optional[float]:
    """Safely convert value to float, returning None on failure."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _short_tick_repr(tick: dict) -> str:
    """Create a concise string representation of tick for logging."""
    if not isinstance(tick, dict):
        return str(tick)[:100]
    
    symbol = tick.get('tradingsymbol') or tick.get('instrument_token') or tick.get('symbol')
    ltp = tick.get('ltp') or tick.get('last_price')
    
    return f"{{symbol={symbol}, ltp={ltp}, keys={list(tick.keys())[:8]}}}"