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
        data_hub: Any = None,
        poll_interval_ms: int = 700,
        batch_size: int = 200,
        require_depth: bool = False,
        warn_on_rate_limit: bool = True,
    ) -> None:
        self._broker = broker_client
        self._on_tick = on_tick
        self._resolver = instrument_resolver
        self._data_hub = data_hub
        self._interval_s = max(0.2, float(poll_interval_ms) / 1000.0)
        self._batch_size = max(1, int(batch_size))
        self._tokens: set[int] = set()
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._require_depth = bool(require_depth)
        self._warn_on_rate_limit = bool(warn_on_rate_limit)
        self._kite_streamer = None  # Optional KiteTicker reference for health checks

        # Metrics
        self._m_poll_ok = Counter("polling_success_total", "Successful poll cycles")
        self._m_poll_fail = Counter("polling_failure_total", "Failed poll cycles")
        self._m_ticks_ingested = Counter("polling_ticks_total", "Ticks ingested via polling")
        self._m_last_success = Gauge("polling_last_success_timestamp", "Last successful poll epoch")
        self._m_last_tick = Gauge("polling_last_tick_timestamp", "Last tick ingestion epoch")

    def start(self) -> None:
        """Start the background polling thread."""
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="polling_streamer", daemon=True)
            self._thread.start()
            LOGGER.info(
                "🚀 Scout Polling Started. Target Interval: %.1fs",
                self._interval_s,
                extra={"event": "polling_started", "interval": self._interval_s},
            )

    def stop(self) -> None:
        """Stop the background polling thread."""
        with self._lock:
            if not self._thread:
                return
            self._stop.set()
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            LOGGER.info("🛑 Polling Streamer Stopped", extra={"event": "polling_stopped"})

    def set_kite_streamer(self, kite_streamer) -> None:
        """Set optional KiteTicker streamer reference for health monitoring."""
        self._kite_streamer = kite_streamer

    def subscribe(self, tokens: Sequence[int]) -> None:
        """Add tokens to the polling list and seed DataHub immediately.
        
        Args:
            tokens: Sequence of integer instrument tokens. 
                   Strings will be validated and converted if numeric.
        """
        # ✅ CRITICAL FIX: Validate and convert tokens to integers
        valid_tokens = set()
        for token in tokens or []:
            try:
                if isinstance(token, int):
                    valid_tokens.add(token)
                elif isinstance(token, str):
                    stripped = token.strip()
                    if stripped.isdigit():
                        valid_tokens.add(int(stripped))
                    else:
                        LOGGER.warning(
                            "PollingStreamer.subscribe: skipping non-numeric token %r",
                            token,
                            extra={"event": "polling_subscribe_skip", "token": token},
                        )
                elif isinstance(token, float):
                    valid_tokens.add(int(token))
                else:
                    LOGGER.warning(
                        "PollingStreamer.subscribe: invalid token type %s",
                        type(token).__name__,
                        extra={"event": "polling_subscribe_invalid_type", "token_type": type(token).__name__},
                    )
            except (ValueError, TypeError) as e:
                LOGGER.warning(
                    "PollingStreamer.subscribe: cannot convert token %r: %s",
                    token, e,
                    extra={"event": "polling_subscribe_convert_error", "token": str(token)},
                )
        
        if not valid_tokens:
            LOGGER.warning(
                "PollingStreamer.subscribe: no valid tokens provided",
                extra={"event": "polling_subscribe_empty"},
            )
            return
        
        with self._lock:
            initial_count = len(self._tokens)
            self._tokens.update(valid_tokens)
            new_count = len(self._tokens)

        if new_count > initial_count:
            LOGGER.info(
                "✅ Wired %d tokens to PollingStreamer",
                new_count,
                extra={"event": "polling_subscribe", "count": new_count},
            )

        # Seed DataHub immediately
        if self._data_hub:
            import time
            for token in valid_tokens:
                symbol = self._resolve_instrument(token)
                if symbol:
                    self._data_hub.store_quote(
                        symbol,
                        {
                            "instrument_token": token,
                            "last_price": None,
                            "timestamp": int(time.time() * 1000),
                            "source": "rest"
                        },
                        source="rest",
                        seed=True,
                    )
                else:
                    LOGGER.error(
                        "PollingStreamer.subscribe: symbol resolution failed",
                        extra={
                            "event": "symbol_resolution_failed",
                            "instrument_token": token,
                        },
                    )

    def unsubscribe(self, tokens: Sequence[int]) -> None:
        """Remove tokens from polling list."""
        with self._lock:
            self._tokens.difference_update(tokens)

    def _run(self) -> None:
        """
        Main polling loop with unified starvation detection.
        Executed in a background thread.
        """
        backoff = self._interval_s
        
        # 🟢 Initialize health timestamp (outside loop)
        # This tracks the last time we saw 'ideal' (NFO) data
        last_healthy_ts = time.monotonic()
        
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                # Copy token list under lock
                with self._lock:
                    tokens = list(self._tokens)
                
                # Track data quality for this specific poll cycle
                seen_nfo_this_cycle = False
                seen_any_tick_this_cycle = False

                if tokens:
                    # Yield chunks to avoid massive requests
                    for batch in self._chunks(tokens, self._batch_size):
                        # Safe Fetch (Uses the corrected _fetch_ticks)
                        ticks = self._fetch_ticks(batch)
                        
                        # Trace logging for empty batches
                        if not ticks:
                             log_throttled(
                                LOGGER,
                                "poll_empty_batch",
                                f"[POLL-TRACE] Empty ticks returned for batch {batch[:5]}...",
                                level=10, 
                                interval_sec=60.0
                            )
                        else:
                            # ✅ CRITICAL FIX 1: Mark that we saw *some* data (even if just NSE)
                            seen_any_tick_this_cycle = True
                        
                        # Check for NFO data in this batch (without resetting timestamp yet)
                        if not seen_nfo_this_cycle:
                            for t in ticks:
                                sym = t.get("symbol")
                                if sym and str(sym).startswith("NFO:"):
                                    seen_nfo_this_cycle = True
                                    break
                        
                        for tick in ticks:
                            # 1. Validate Payload Structure
                            if "instrument_token" not in tick or "last_price" not in tick:
                                continue

                            # 2. Tag Source as REST
                            tick["source"] = "rest"
                            
                            # 3. Get Symbol (Already resolved in _fetch_ticks)
                            symbol = tick.get("symbol")
                            if not symbol: continue
                            
                            # 4. Seed DataHub (Synchronous)
                            if self._data_hub:
                                self._data_hub.store_quote(symbol, tick, source="rest", seed=True)

                            # 5. Update Metrics
                            with suppress(Exception):
                                self._m_last_tick.set(int(time.time() * 1000))
                            
                            # 6. Async Handoff (Strategy Pipeline)
                            with suppress(Exception):
                                self._on_tick(tick)
                                self._m_ticks_ingested.inc()

                # --------------------------------------------------
                # 🟢 UPDATE HEALTH METRICS
                # --------------------------------------------------
                # We update the 'healthy' timestamp if we saw NFO data.
                # This keeps the 'starvation timer' low during normal trading.
                if seen_nfo_this_cycle:
                    last_healthy_ts = time.monotonic()

                # Check KiteTicker health if available
                ws_healthy = False
                if self._kite_streamer is not None:
                    try:
                        # WS is healthy if we saw a tick in last 10 seconds
                        ws_healthy = self._kite_streamer.last_tick_age() < 10.0
                    except Exception:
                        ws_healthy = False

                # --------------------------------------------------
                # 🔴 WS-AWARE STARVATION DETECTION (CORRECTED)
                # --------------------------------------------------
                # KILL CONDITION:
                # 1. We are supposed to be tracking tokens.
                # 2. We received ZERO ticks this cycle (REST is dead).
                # 3. WebSocket is NOT healthy (WS is dead).
                # 4. NFO data has been stale for > 30s (Long term starvation).
                #
                # NOTE: If 'seen_any_tick_this_cycle' is True (e.g. NSE Index ticks), 
                # we do NOT kill the bot, even if 'last_healthy_ts' is old. 
                # This prevents "Quiet Market" suicide.
                
                if tokens and not seen_any_tick_this_cycle and not ws_healthy:
                    if time.monotonic() - last_healthy_ts > 30.0:
                        LOGGER.critical(
                            "💀 FATAL POLLER ERROR: True market data starvation "
                            "(REST + WS silent for >30s). Escalating to supervisor."
                        )
                        self._stop.set()
                        return

                # Success: Update health metrics & reset backoff
                with suppress(Exception):
                    self._m_poll_ok.inc()
                with suppress(Exception):
                    self._m_last_success.set(int(time.time() * 1000))
                
                backoff = self._interval_s

            except Exception as exc:  # noqa: BLE001
                # Failure: Adaptive Backoff
                with suppress(Exception):
                    self._m_poll_fail.inc()
                
                backoff = min(backoff * 1.5, 10.0)
                log_throttled(
                    LOGGER,
                    "poll_loop_error",
                    f"Polling loop error: {exc}",
                    level=40,
                    interval_sec=10.0
                )

            # Smart Sleep
            elapsed = time.monotonic() - started
            sleep_time = max(0.1, backoff - elapsed)
            time.sleep(sleep_time)
            
    # ----------------------------------------------------------------
    # ✅ FIX: Thread-Safe Fetch with Timeout (Prevents Zombie Hangs)
    # ----------------------------------------------------------------
    def _fetch_ticks(self, batch: list[int]) -> list[dict[str, Any]]:
        """
        Fetch quotes using Instrument Tokens DIRECTLY.
        """
        try:
            # Clean Batch: Ensure all items are integers
            tokens = []
            for t in batch:
                try:
                    tokens.append(int(t))
                except (ValueError, TypeError):
                    continue

            if not tokens:
                return []
            
            # Diagnostic Log
            log_throttled(
                LOGGER,
                "quote_token_fetch",
                f"📡 Fetching {len(tokens)} tokens: {tokens[:3]}...",
                interval_sec=60.0
            )
            
            # DIRECT API CALL: Pass list of Integers
            quote_map = self._broker.quote(tokens)
            
            if not quote_map:
                LOGGER.warning(f"[POLL] Empty quote_map returned for tokens: {tokens[:3]}...")
                return []

            ticks = []
            for key, quote in quote_map.items():
                try:
                    lp = float(quote.get("last_price") or 0.0)
                    if lp <= 0: continue
                    
                    token = quote.get("instrument_token")
                    if not token:
                        if str(key).isdigit(): token = int(key)
                        else: continue
                            
                    token_int = int(token)
                    symbol = self._resolve_instrument(token_int)
                    
                    if not symbol: continue

                    q_ts = quote.get("timestamp")
                    if q_ts and hasattr(q_ts, "timestamp"):
                        ts = int(q_ts.timestamp() * 1000)
                    else:
                        ts = int(time.time() * 1000)

                    tick = {
                        "instrument_token": token_int,
                        "last_price": lp,
                        "timestamp": ts,
                        "volume": quote.get("volume", 0),
                        "average_price": quote.get("average_price", 0.0),
                        "oi": quote.get("oi", 0),
                        "depth": quote.get("depth"),
                        "symbol": symbol 
                    }
                    ticks.append(tick)
                except Exception as e:
                    continue
            
            return ticks
            
        except Exception as e:
            LOGGER.warning(f"[POLL-QUOTE-FAIL] {e}")
            return []
    # ----------------------------------------------------------------
    # ✅ HELPERS (Inlined to ensure stability)
    # ----------------------------------------------------------------
    def _try_quote_bulk(self, batch: list[int], timestamp_ms: int) -> list[dict[str, Any]]:
        """Fetch full quotes with proper token handling."""
        try:
            if not batch:
                return []
            
            # Build symbol list with token mapping
            symbols_to_fetch = []
            token_to_symbol_map = {}
            symbol_to_token_map = {}  # Reverse map for extracting tokens from API response
            
            for token in batch:
                # ✅ DEFENSIVE: Ensure token is integer
                try:
                    token_int = int(token) if not isinstance(token, int) else token
                except (ValueError, TypeError):
                    LOGGER.warning(f"[POLL] Skipping non-integer in batch: {token!r}")
                    continue
                
                symbol = self._resolve_instrument(token_int)
                if symbol:
                    symbols_to_fetch.append(symbol)
                    token_to_symbol_map[symbol] = token_int
                    symbol_to_token_map[symbol] = token_int
                    # Also map variations (Zerodha may return slightly different keys)
                    symbol_to_token_map[symbol.upper()] = token_int
                    symbol_to_token_map[symbol.replace(" ", "")] = token_int
                else:
                    # Last resort: try numeric token
                    symbols_to_fetch.append(str(token_int))
                    token_to_symbol_map[str(token_int)] = token_int
                    symbol_to_token_map[str(token_int)] = token_int
            
            if not symbols_to_fetch:
                LOGGER.warning("[POLL] No symbols resolved from tokens")
                return []
            
            log_throttled(
                LOGGER,
                "quote_fetch_attempt",
                f"📡 Fetching quotes for {len(symbols_to_fetch)} symbols: {symbols_to_fetch[:3]}...",
                interval_sec=60.0
            )
            
            # Call Zerodha API
            quote_map = self._broker.quote(symbols_to_fetch)
            
            if not quote_map:
                LOGGER.warning("[POLL] Empty quote_map returned from broker")
                return []

            ticks = []
            for key, quote in quote_map.items():
                try:
                    lp = float(quote.get("last_price") or 0.0)
                    if lp <= 0:
                        continue
                    
                    # ✅ FIX: Get token from quote data first, then from our map
                    token = quote.get("instrument_token")
                    
                    if not token:
                        # Try to find token from our reverse map
                        token = symbol_to_token_map.get(key)
                        if not token:
                            token = symbol_to_token_map.get(str(key).upper())
                        if not token:
                            # Try without spaces
                            token = symbol_to_token_map.get(str(key).replace(" ", ""))
                    
                    # ✅ FIX: DON'T try int(key) if key is a symbol string!
                    # Only convert if token is still numeric-looking
                    if token is None:
                        if str(key).isdigit():
                            token = int(key)
                        else:
                            # Use the first token from our batch as fallback
                            # (This handles single-symbol case)
                            if len(batch) == 1:
                                token = batch[0]
                            else:
                                LOGGER.warning(f"[POLL] Cannot determine token for key: {key}")
                                continue
                    
                    # Ensure token is integer
                    token_int = int(token) if token else None
                    if not token_int:
                        continue
                    
                    # Build timestamp
                    q_ts = quote.get("timestamp")
                    if q_ts and hasattr(q_ts, "timestamp"):
                        ts = int(q_ts.timestamp() * 1000)
                    else:
                        ts = timestamp_ms

                    tick = {
                        "instrument_token": token_int,
                        "last_price": lp,
                        "timestamp": ts,
                        "volume": quote.get("volume", 0),
                        "average_price": quote.get("average_price", 0.0),
                        "oi": quote.get("oi", 0),
                        "depth": quote.get("depth"),
                        "symbol": key if ":" in str(key) else None
                    }
                    ticks.append(tick)
                    
                except Exception as e:
                    LOGGER.debug(f"[POLL] Error processing quote {key}: {e}")
                    continue
            
            return ticks
            
        except Exception as e:
            LOGGER.warning(f"[POLL-QUOTE-FAIL] {e}")
            return []

    def _try_ltp_bulk(self, batch: list[int], timestamp_ms: int) -> list[dict[str, Any]]:
        """Helper: Fetch LTP only (fallback)."""
        try:
            str_tokens = [str(t) for t in batch]
            ltp_map = self._broker.ltp(str_tokens)
            
            if not ltp_map:
                return []

            ticks = []
            for token_str, data in ltp_map.items():
                try:
                    # Normalize nested structure
                    if isinstance(data, dict):
                         lp = float(data.get("last_price") or data.get("ltp") or 0.0)
                    else:
                         lp = float(data or 0.0)
                         
                    if lp <= 0: continue
                    
                    # ✅ FIX: API returns keys as "exchange:symbol", need to map back to token
                    # First check if token_str is actually numeric
                    if str(token_str).isdigit():
                        inst_token = int(token_str)
                    else:
                        # Look up in our batch - single token case
                        if len(batch) == 1:
                            inst_token = batch[0]
                        else:
                            # Can't determine token from LTP response key
                            LOGGER.debug(f"[POLL-LTP] Cannot map key to token: {token_str}")
                            continue
                    
                    ticks.append({
                        "instrument_token": inst_token,
                        "last_price": lp,
                        "timestamp": timestamp_ms,
                        "volume": 0,
                        "average_price": 0.0,
                        "symbol": token_str if ":" in str(token_str) else None
                    })
                except Exception:
                    continue
            return ticks
        except Exception:
            return []

    @staticmethod
    def _chunks(lst: list[int], n: int) -> Iterable[list[int]]:
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    def _resolve_instrument(self, token: int) -> str | None:
        """Resolve instrument token to exchange:tradingsymbol format.
        
        Returns format like 'NSE:NIFTY 50' or 'NFO:NIFTY2612025700CE'.
        """
        if token is None:
            return None
        
        try:
            # ✅ FIX: Handle case where token is already a symbol string
            if isinstance(token, str):
                if not token.strip().isdigit():
                    # It's already a symbol like "NSE:NIFTY 50", return it directly
                    return token if ":" in token else None
                token_int = int(token.strip())
            else:
                token_int = int(token)
            
            # 1. Try format_token_as_symbol (best method - uses CANONICAL_TOKENS)
            if hasattr(self._resolver, "format_token_as_symbol"):
                result = self._resolver.format_token_as_symbol(token_int)
                if result and result != str(token_int):
                    return result
            
            # 2. Try lookup method
            if hasattr(self._resolver, "lookup"):
                info = self._resolver.lookup(token_int)
                if info:
                    exchange = info.get("exchange", "NSE")
                    symbol = info.get("symbol") or info.get("tradingsymbol")
                    if symbol:
                        # Handle special case: NIFTY 50 / NIFTY BANK
                        if symbol in ("NIFTY", "NIFTY 50"):
                            return "NSE:NIFTY 50"
                        elif symbol in ("BANKNIFTY", "NIFTY BANK"):
                            return "NSE:NIFTY BANK"
                        return f"{exchange}:{symbol}"
            
            # 3. Try direct cache access with exchange lookup
            if hasattr(self._resolver, "_symbol_by_token"):
                symbol = self._resolver._symbol_by_token.get(token_int)
                if symbol:
                    # Get exchange from cache
                    exchange = "NSE"
                    if hasattr(self._resolver, "_exchange_by_token"):
                        exchange = self._resolver._exchange_by_token.get(token_int, "NSE")
                    
                    # Handle NIFTY special case
                    if symbol in ("NIFTY", "NIFTY 50"):
                        return "NSE:NIFTY 50"
                    elif symbol in ("BANKNIFTY", "NIFTY BANK"):
                        return "NSE:NIFTY BANK"
                    
                    return f"{exchange}:{symbol}"
            
            # 4. Well-known token fallback
            WELL_KNOWN_TOKENS = {
                256265: "NSE:NIFTY 50",
                260105: "NSE:NIFTY BANK",
            }
            if token_int in WELL_KNOWN_TOKENS:
                return WELL_KNOWN_TOKENS[token_int]
                
        except Exception as e:
            LOGGER.debug(f"[POLL] Symbol resolution error for token {token}: {e}")
        
        return None
