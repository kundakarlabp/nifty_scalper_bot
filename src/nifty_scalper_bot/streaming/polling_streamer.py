"""Polling-based market data streamer for Zerodha REST APIs."""

from __future__ import annotations

import math
import random
import threading
import time
from contextlib import suppress
from typing import Any, Callable, Iterable, Sequence

# [FIX] Use centralized logging utilities
from nifty_scalper_bot.utils.logging import get_logger, log_throttled
from nifty_scalper_bot.utils.metrics import Counter, Gauge

LOGGER = get_logger(__name__)


class PollingStreamer:
    """Production-safe poller that replaces WebSocket streaming."""

    def __init__(
        self,
        *,
        broker_client: Any,
        on_tick: Callable[[dict[str, Any]], None],
        instrument_resolver: Any,
        poll_interval_ms: int = 700,
        batch_size: int = 200,
        require_depth: bool = False,
        warn_on_rate_limit: bool = True,
    ) -> None:
        self._broker = broker_client
        self._on_tick = on_tick
        self._resolver = instrument_resolver
        self._interval_s = max(0.2, float(poll_interval_ms) / 1000.0)
        self._batch_size = max(1, int(batch_size))
        self._tokens: set[int] = set()
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._require_depth = bool(require_depth)
        self._warn_on_rate_limit = bool(warn_on_rate_limit)
        self._rate_limit_warned = False

        # Metrics
        self._m_poll_ok = Counter("poll_ok_total", "Successful polling rounds")
        self._m_poll_err = Counter("poll_err_total", "Polling rounds with error")
        self._m_tokens = Gauge("poll_tokens", "Tracked tokens for polling")
        self._m_interval = Gauge("poll_interval_seconds", "Polling interval (seconds)")
        self._m_last_tick = Gauge("poll_last_tick_ts", "Epoch ms of last tick")
        self._m_last_success = Gauge("poll_last_success_ts", "Epoch ms of last successful poll")

        with suppress(Exception):
            self._m_interval.set(self._interval_s)

    def start(self) -> None:
        if self.is_running():
            return
        self._stop.clear()
        worker = threading.Thread(
            target=self._run, name="polling-streamer", daemon=True
        )
        self._thread = worker
        worker.start()
        LOGGER.info(
            "Polling streamer started: interval=%.3fs batch_size=%d",
            self._interval_s,
            self._batch_size,
        )

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
            self._thread = None
        LOGGER.info("Polling streamer stopped")

    def is_running(self) -> bool:
        """Return ``True`` when the polling worker thread is active."""
        thread = self._thread
        if thread is None:
            return False
        if self._stop.is_set():
            return False
        return thread.is_alive()

    def ensure_running(self) -> None:
        """Ensure the polling worker thread is running."""
        if not self.is_running():
            self.start()

    def subscribe(self, tokens: Sequence[int]) -> None:
        with self._lock:
            self.ensure_running()
            self._tokens.update(int(token) for token in tokens)
            count = len(self._tokens)
            self._set_token_metric(count)
            LOGGER.debug("Subscribed tokens; now tracking %d tokens", count)
        self._maybe_warn_rate_limits(count)

    def subscribe_tokens(self, tokens: Sequence[int]) -> None:
        """Alias for :meth:`subscribe` used by env-driven bootstrapping."""
        self.subscribe(tokens)

    def unsubscribe(self, tokens: Sequence[int]) -> None:
        with self._lock:
            for token in tokens:
                self._tokens.discard(int(token))
            count = len(self._tokens)
            self._set_token_metric(count)
            LOGGER.debug("Unsubscribed tokens; now tracking %d tokens", count)
        self._maybe_warn_rate_limits(count)

    def tracked_tokens(self) -> list[int]:
        with self._lock:
            return sorted(self._tokens)

    def _set_token_metric(self, count: int) -> None:
        with suppress(Exception):
            self._m_tokens.set(count)

    def _maybe_warn_rate_limits(self, token_count: int) -> None:
        if not self._warn_on_rate_limit:
            return
        if token_count <= 0:
            self._rate_limit_warned = False
            return
        if self._interval_s <= 0:
            return
        polls_per_second = 1.0 / self._interval_s
        batches_per_poll = math.ceil(token_count / self._batch_size)
        estimated_reqs_per_sec = batches_per_poll * polls_per_second
        safe_capacity = self._batch_size * polls_per_second
        if token_count > safe_capacity:
            with suppress(Exception):
                self._m_poll_err.inc()
            
            # Use throttled logging for rate limit warnings to avoid spamming error logs
            log_throttled(
                LOGGER,
                "poll_rate_limit_warn",
                f"[POLL-RATE] Token count {token_count} exceeds safe capacity ({safe_capacity:.2f})",
                level=30, # WARNING
                interval_sec=60.0
            )

            if not self._rate_limit_warned:
                self._rate_limit_warned = True
        elif self._rate_limit_warned:
            LOGGER.info("[POLL-RATE] Token count back within estimated rate limits")
            self._rate_limit_warned = False

    def _chunks(self, payload: list[int], size: int) -> Iterable[list[int]]:
        for idx in range(0, len(payload), size):
            yield payload[idx : idx + size]

    def _run(self) -> None:
        backoff = self._interval_s
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                # Copy token list under lock to avoid holding lock during network calls
                with self._lock:
                    tokens = list(self._tokens)
                if tokens:
                    for batch in self._chunks(tokens, self._batch_size):
                        ticks = self._fetch_ticks(batch)
                        
                        # Alert if persistent empty polling batch (only if we expected data)
                        if not ticks:
                             log_throttled(
                                LOGGER,
                                "poll_empty_batch",
                                f"[POLL-TRACE] Empty ticks returned for batch {batch[:5]}...",
                                level=10, # DEBUG
                                interval_sec=60.0
                            )
                        
                        for tick in ticks:
                            # Validate tick shape
                            if (
                                "instrument_token" not in tick
                                or "last_price" not in tick
                                or "timestamp" not in tick
                            ):
                                LOGGER.error("[POLL-ERR] Invalid tick payload structure: %s", tick)
                                continue
                            with suppress(Exception):
                                self._m_last_tick.set(int(time.time() * 1000))
                            with suppress(Exception):
                                self._on_tick(tick)
                with suppress(Exception):
                    self._m_poll_ok.inc()
                with suppress(Exception):
                    self._m_last_success.set(int(time.time() * 1000))
                # reset backoff to normal when successful
                backoff = self._interval_s
            except Exception:  # noqa: BLE001
                # full stacktrace for observability
                LOGGER.exception("[POLL-ERR] Polling round failed")
                with suppress(Exception):
                    self._m_poll_err.inc()
                # exponential backoff with jitter, bounded
                jitter = random.uniform(-0.2, 0.2) * backoff
                backoff = min(max(backoff * 2.0 + jitter, self._interval_s), 8.0)
            elapsed = max(0.0, time.monotonic() - started)
            sleep_for = max(0.0, backoff - elapsed)
            self._stop.wait(sleep_for)

    def _fetch_ticks(self, batch: list[int]) -> list[dict[str, Any]]:
        """
        Fetch ticks for a batch with strict 3s timeout to prevent 'Zombie Mode'.
        Wraps existing logic in a thread to unblock the main loop if Broker API hangs.
        """
        if not batch:
            return []

        # Container to capture results from the thread
        result_holder = {"ticks": []}

        # 1. Define the logic wrapper (Preserving your EXACT existing flow)
        def _fetch_logic():
            try:
                timestamp_ms = int(time.time() * 1000)
                
                # --- STRATEGY: TRY QUOTE FIRST (Contains Volume + VWAP) ---
                ticks = self._try_quote_bulk(batch, timestamp_ms)
                if ticks:
                    log_throttled(
                        LOGGER, 
                        "quote_fetch_success", 
                        f"✅ QUOTE SUCCESS: Fetched {len(ticks)} ticks with Volume/VWAP data (Throttled 60s)", 
                        interval_sec=60.0
                    )
                    result_holder["ticks"] = ticks
                    return

                # --- FALLBACK: TRY LTP BULK (Price Only) ---
                log_throttled(
                    LOGGER,
                    "poll_fallback_ltp",
                    "[POLL-WARN] Quote fetch failed/empty. Falling back to LTP (NO VOLUME DATA!)",
                    level=30, # WARNING
                    interval_sec=10.0
                )
                
                ticks = self._try_ltp_bulk(batch, timestamp_ms)
                if ticks:
                    result_holder["ticks"] = ticks
                    return

                # --- LAST RESORT: SINGLE QUOTE ---
                get_quote_single = getattr(self._broker, "get_quote_by_token", None)
                ticks = []
                if callable(get_quote_single):
                    for token in batch:
                        try:
                            quote = get_quote_single(int(token))
                        except Exception: 
                            continue
                        
                        lp = float(quote.get("last_price", 0.0) or 0.0)
                        if lp <= 0: continue
                        
                        vol = quote.get("volume", 0)
                        avg_price = quote.get("average_price", 0.0)
                        
                        tick = {
                            "instrument_token": int(token),
                            "last_price": lp,
                            "timestamp": timestamp_ms,
                            "volume": vol,
                            "average_price": avg_price
                        }
                        depth = quote.get("depth")
                        if isinstance(depth, dict):
                            tick["depth"] = depth
                        ticks.append(tick)
                    
                    result_holder["ticks"] = ticks
                    return

                log_throttled(
                    LOGGER,
                    "poll_all_failed",
                    "[POLL-ERR] All polling methods failed for batch",
                    level=40, # ERROR
                    interval_sec=5.0
                )
            except Exception as e:
                # Catch internal logic errors so thread finishes cleanly
                LOGGER.debug(f"[POLL-THREAD] Logic error: {e}")

        # 2. Execute with Timeout
        # Daemon=True ensures this thread doesn't block app shutdown
        t = threading.Thread(target=_fetch_logic, name="poll_safe_fetch", daemon=True)
        t.start()
        
        # ✅ THE FIX: Wait max 3.0 seconds. If it hangs, we move on.
        t.join(timeout=3.0)

        # 3. Check for Hang
        if t.is_alive():
            log_throttled(
                LOGGER,
                "poll_timeout_crit",
                "🚨 CRITICAL: Broker Polling Hung! Timeout enforced (3s). Skipping batch.",
                level=50, # CRITICAL
                interval_sec=5.0
            )
            return []

        return result_holder["ticks"]

    def _try_ltp_bulk(
        self, batch: list[int], timestamp_ms: int
    ) -> list[dict[str, Any]] | None:
        fetch_ltp = getattr(self._broker, "get_ltp_bulk", None)
        if not callable(fetch_ltp):
            return None
        try:
            # Zerodha returns: {'256265': {'instrument_token': 256265, 'last_price': 17000.0}}
            data = fetch_ltp(batch)
        except Exception:  # noqa: BLE001
            return None

        if not data:
            return None

        ticks: list[dict[str, Any]] = []
        for token in batch:
            # Normalize token lookups: try both int and str keys
            key_candidates = (int(token), str(int(token)))
            ltp = 0.0
            
            # [FIX] Correctly parse Nested Dictionary from Zerodha
            for k in key_candidates:
                if k in data:
                    val = data[k]
                    if isinstance(val, dict):
                        # Extract from dict: {'last_price': 123.45}
                        try:
                            ltp = float(val.get("last_price") or val.get("ltp") or 0.0)
                        except Exception:
                            ltp = 0.0
                    else:
                        # Direct value (unlikely but safe fallback)
                        try:
                            ltp = float(val or 0.0)
                        except Exception:
                            ltp = 0.0
                    
                    if ltp > 0:
                        break
            
            if ltp <= 0:
                continue

            ticks.append(
                {
                    "instrument_token": int(token),
                    "last_price": ltp,
                    "timestamp": timestamp_ms,
                    # Fallback values to prevent KeyError in strategies
                    "volume": 0,
                    "average_price": 0.0
                }
            )
        return ticks or None
    def _try_quote_bulk(
        self, batch: list[int], timestamp_ms: int
    ) -> list[dict[str, Any]] | None:
        fetch_quote_bulk = getattr(self._broker, "get_quote_bulk", None)
        if not callable(fetch_quote_bulk):
            return None
        try:
            quote_map = fetch_quote_bulk(batch)
        except Exception as exc:  # noqa: BLE001
            # Throttled debug log for connection issues
            log_throttled(
                LOGGER,
                "poll_quote_exception",
                f"[POLL-WARN] Quote bulk fetch exception: {exc}",
                level=10, # DEBUG
                interval_sec=30.0
            )
            return None
            
        if not quote_map:
             return None

        ticks: list[dict[str, Any]] = []
        for token, quote in quote_map.items():
            try:
                normalized_token = int(token)
            except Exception:
                # skip if token key cannot be normalized
                continue
            try:
                lp = float(quote.get("last_price", 0.0) or 0.0)
            except Exception:
                lp = 0.0
            if lp <= 0:
                continue
            
            tick: dict[str, Any] = {
                "instrument_token": normalized_token,
                "last_price": lp,
                "timestamp": timestamp_ms,
                # --- VITAL DATA FOR STRATEGIES ---
                "volume": quote.get("volume", 0),
                "average_price": quote.get("average_price", 0.0),
            }
            depth = quote.get("depth")
            if isinstance(depth, dict):
                tick["depth"] = depth
            ticks.append(tick)
            
        return ticks or None
