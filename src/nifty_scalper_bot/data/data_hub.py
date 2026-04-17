# ===========================
# CORE IMPORTS (UNCHANGED SAFE SET)
# ===========================
from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

LOGGER = logging.getLogger(__name__)

Tick = Dict[str, Any]
TickListener = Callable[[Tick], None]


# ===========================
# DATAHUB (SSOT ENABLED)
# ===========================
class DataHub:
    """
    Market Data Single Source of Truth (SSOT)

    Key principles:
    - Token-first identity
    - WS priority over poll
    - Backward compatibility with symbol APIs
    """

    def __init__(self, market_data_manager: Any, options_only: bool = False, **kwargs):

        self._mdm = market_data_manager

        # ===========================
        # SSOT STORAGE
        # ===========================
        self._ticks: Dict[int, Tick] = {}  # ✅ PRIMARY TRUTH

        # ===========================
        # COMPATIBILITY LAYER
        # ===========================
        self._quotes: Dict[str, Tick] = {}  # ⚠️ legacy support

        self._options_only = options_only
        self._token_by_symbol: Dict[str, int] = {}
        self._symbol_by_token: Dict[int, str] = {}

        # ===========================
        # TIMESTAMP TRACKING
        # ===========================
        self._last_ts: Dict[str, float] = {}
        self._last_arrival: Dict[str, float] = {}

        self._last_ws_arrival: Dict[str, float] = {}
        self._last_global_ws_arrival: float = 0.0

        # ===========================
        # SUBSCRIBERS
        # ===========================
        self._tick_subscribers: Dict[str, set[TickListener]] = defaultdict(set)

        # ===========================
        # LOCK
        # ===========================
        self._lock = threading.RLock()

        # ===========================
        # CONFIG
        # ===========================
        self._poll_block_ms = 800

        # ===========================
        # 🔗 BIND MDM → DATAHUB (CRITICAL FIX)
        # ===========================
        self._bind_to_mdm()

    # =========================================================
    # 🔗 BINDING (CRITICAL)
    # =========================================================
    def _bind_to_mdm(self):

        try:
            attach_cb = getattr(self._mdm, "attach_tick_handler", None)
            if callable(attach_cb):
                attach_cb(self.ingest_tick_sync)

            # HARD BIND (guarantees delivery)
            setattr(self._mdm, "_external_tick_handler", self.ingest_tick_sync)

            LOGGER.info("DataHub successfully bound to MDM")

        except Exception as exc:
            LOGGER.error("Failed binding to MDM: %s", exc, exc_info=exc)

    # =========================================================
    # INGESTION ENTRY
    # =========================================================
    def ingest_tick_sync(self, tick: Tick):

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(self.ingest_tick(tick), loop)
                return
            except Exception:
                pass

        asyncio.run(self.ingest_tick(tick))

    # =========================================================
    # CORE INGESTION ENGINE
    # =========================================================
    async def ingest_tick(self, tick: Tick):

        if not tick:
            return

        # -------------------------
        # FAST NORMALIZATION
        # -------------------------
        symbol = str(tick.get("symbol", "")).upper().strip()
        token = tick.get("instrument_token")

        if not symbol:
            return

        try:
            token = int(token) if token is not None else None
        except Exception:
            token = None

        ts = tick.get("timestamp") or time.time()
        ts_ms = float(ts) * 1000 if ts < 1e12 else float(ts)
        now_ms = time.time() * 1000
        source = tick.get("source", "ws")

        # -------------------------
        # READ STATE (MIN LOCK)
        # -------------------------
        with self._lock:
            last_ts = self._last_ts.get(symbol, 0.0)
            last_arr = self._last_arrival.get(symbol, 0.0)
            last_ws = self._last_ws_arrival.get(symbol, 0.0)

        # -------------------------
        # DEDUP LOGIC
        # -------------------------
        if ts_ms < last_ts:
            return

        if ts_ms == last_ts and now_ms <= last_arr:
            return

        # -------------------------
        # WS PRIORITY
        # -------------------------
        if source == "poll":
            if (now_ms - last_ws) < self._poll_block_ms:
                return

        # -------------------------
        # BUILD CANONICAL TICK
        # -------------------------
        canonical = dict(tick)
        canonical["symbol"] = symbol
        canonical["timestamp"] = ts_ms
        canonical["arrival_time"] = now_ms
        canonical["source"] = source

        # -------------------------
        # WRITE STATE (LOCK)
        # -------------------------
        with self._lock:

            # SSOT WRITE
            if token:
                self._ticks[token] = canonical
                self._symbol_by_token[token] = symbol
                self._token_by_symbol[symbol] = token

            # COMPAT WRITE
            self._quotes[symbol] = canonical

            # TIMESTAMP UPDATE
            self._last_ts[symbol] = ts_ms
            self._last_arrival[symbol] = now_ms

            if source == "ws":
                self._last_ws_arrival[symbol] = now_ms
                self._last_global_ws_arrival = now_ms

        # -------------------------
        # NOTIFY SUBSCRIBERS
        # -------------------------
        subs = list(self._tick_subscribers.get(symbol, []))
        for cb in subs:
            try:
                cb(canonical)
            except Exception as exc:
                LOGGER.error("Subscriber error: %s", exc, exc_info=exc)

    # =========================================================
    # ACCESSORS
    # =========================================================
    def get_quote(self, symbol: str, allow_pull: bool = False) -> Optional[Tick]:

        symbol = str(symbol).upper().strip()

        with self._lock:

            # SSOT READ FIRST
            token = self._token_by_symbol.get(symbol)
            if token and token in self._ticks:
                return dict(self._ticks[token])

            # FALLBACK
            tick = self._quotes.get(symbol)
            if tick:
                return dict(tick)

        return None

    def get_tick_by_token(self, token: int) -> Optional[Tick]:

        with self._lock:
            t = self._ticks.get(token)
            return dict(t) if t else None

    # =========================================================
    # SUBSCRIPTIONS
    # =========================================================
    def subscribe(self, symbol: str, callback: TickListener):

        symbol = str(symbol).upper().strip()

        if callback:
            self._tick_subscribers[symbol].add(callback)

    def unsubscribe(self, symbol: str, callback: TickListener):

        symbol = str(symbol).upper().strip()

        if callback in self._tick_subscribers.get(symbol, set()):
            self._tick_subscribers[symbol].remove(callback)

    # =========================================================
    # HEALTH CHECKS
    # =========================================================
    def is_ws_fresh(self, symbol: str, threshold_sec: float = 2.0) -> bool:

        symbol = str(symbol).upper().strip()

        with self._lock:
            last = self._last_ws_arrival.get(symbol)

        if not last:
            return False

        return (time.time() - (last / 1000)) < threshold_sec

    def stats(self) -> Dict[str, Any]:

        with self._lock:
            return {
                "tokens": len(self._ticks),
                "symbols": len(self._quotes),
                "last_ws_age_sec": time.time()
                - (self._last_global_ws_arrival / 1000),
            }
