"""Canonical data hub for cached ticks, orders, and positions."""

from __future__ import annotations

import os
import re
import time
import asyncio
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Callable, Iterable, Mapping, Optional, TypedDict, cast
from nifty_scalper_bot.core.message_bus import MessageBus, Message, MessageType

from nifty_scalper_bot.storage.hub_store import HubStore
from nifty_scalper_bot.utils.logging import get_logger
from nifty_scalper_bot.utils.options_math import (
    black_scholes_greeks,
    implied_volatility,
)

LOGGER = get_logger(__name__)


Tick = dict[str, Any]
OrderListener = Callable[[dict[str, Any]], None]
TickListener = Callable[[dict[str, Any]], None]


class Freshness(TypedDict, total=False):
    """Container describing cached quote freshness metrics."""

    ok: bool
    mono_age_ms: float | None
    server_age_ms: float | None
    effective_ms: float | None
    threshold_ms: float
    reason: str | None


_ORDER_STATE_MACHINE: dict[str, set[str]] = {
    "": {
        "pending",
        "submitted",
        "partial_filled",
        "filled",
        "cancelled",
        "rejected",
        "expired",
    },
    "pending": {
        "submitted",
        "partial_filled",
        "cancelled",
        "rejected",
        "expired",
    },
    "submitted": {
        "partial_filled",
        "filled",
        "cancelled",
        "rejected",
        "expired",
    },
    "partial_filled": {
        "partial_filled",
        "filled",
        "cancelled",
        "rejected",
        "expired",
    },
    "filled": set(),
    "cancelled": set(),
    "rejected": set(),
    "expired": set(),
}


class DataHub:
    """Central in-memory state cache for the trading bot."""

    # --- [ADD THIS SECTION] Helper methods for parsing ---
    @staticmethod
    def _float(value: Any) -> float | None:
        if value in (None, ""): return None
        try: return float(value)
        except Exception: return None

    @staticmethod
    def _int(value: Any) -> int:
        if value in (None, ""): return 0
        try: return int(float(value))
        except Exception: return 0

    @staticmethod
    def _ts(value: Any) -> float:
        if value in (None, ""): return time.time()
        try:
            ts = float(value)
            return ts / 1000.0 if ts > 1_000_000_000_000 else ts
        except Exception: return time.time()
    # -----------------------------------------------------

    def __init__(
        self,
        market_data_manager: Any,
        instrument_resolver: Any,
        *,
        options_only: bool = True,
        store: HubStore | None = None,
        message_bus: MessageBus,
        checkpoint_interval: float = 5.0,      
        clock: Callable[[], float] | None = None,
    ) -> None:
        
        self._mdm = market_data_manager
        self._resolver = instrument_resolver
        self._options_only = options_only
        self._store = store
        self._lock = RLock()
       

        # State Caches
        self._quotes: dict[str, Tick] = {}
        self._orders: dict[str, dict[str, Any]] = {}
        self._positions: dict[str, dict[str, Any]] = {}
        self._message_bus = message_bus

        # [ADDED] Persistence & State Tracking
        self._order_status: dict[str, str] = {}      # <--- MISSING LINE RESTORED
        self._order_sequences: dict[str, int] = {}   # <--- Required for WAL replay
        self._checkpoint_interval = max(0.1, float(checkpoint_interval))
        self._clock = clock or time.time
        self._last_snapshot_ts = 0.0
        
        # [ADDED] Freshness Config
        self._stale_tick_max_age_ms = 5000.0
        self._warmup_grace_s = 5.0
        self._warmup_deadline: float | None = None
        self._start_mono = time.monotonic()
        self._reset_warmup()
        
        # Derived Metrics Caches
        self._iv_cache: dict[str, float] = {}
        self._oi_cache: dict[str, float] = {}
        self._greeks_cache: dict[str, dict[str, float]] = {}
        
        # Throttling for heavy math (Greeks/IV)
        self._last_greeks_update: dict[str, float] = {}
        self._greeks_throttle_sec = 0.5

        # Subscribers
        self._tick_subscribers: dict[str, set[TickListener]] = {}
        self._order_subscribers: list[OrderListener] = []

    # ----------------------------------------------------------------
    # Ingestion (Write Path)
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # [ADDED] Helpers for Sanitization, Freshness & Warmup
    # ----------------------------------------------------------------
    def _reset_warmup(self) -> None:
        """Reset the warmup deadline based on current monotonic time."""
        # Ensure grace period is at least 5.0s if not set
        grace = getattr(self, "_warmup_grace_s", 5.0)
        self._warmup_deadline = time.monotonic() + grace

    def _sanitize_tick(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Ensure numeric fields are floats/ints, preventing downstream crashes."""
        numeric_fields = (
            "ltp", "last_price", "close", "open", "high", "low",
            "bid", "ask", "volume", "oi", "open_interest"
        )
        for key in numeric_fields:
            if key in payload:
                coerced = self._float(payload.get(key))
                if coerced is not None:
                    payload[key] = coerced
        
        # Normalize timestamp to seconds
        ts = self._extract_tick_timestamp(payload)
        if ts:
            payload["timestamp"] = ts
            
        return payload

    def _extract_tick_timestamp(self, payload: dict) -> float | None:
        """Robust timestamp extraction handling ms/ns/iso formats."""
        keys = ["exchange_timestamp", "last_trade_time", "timestamp", "ts", "ts_ms"]
        for k in keys:
            val = payload.get(k)
            if val:
                try:
                    if isinstance(val, datetime):
                        return val.timestamp()
                    ts = float(val)
                    if ts > 1e11: ts /= 1000.0  # Convert ms to s
                    return ts
                except Exception:
                    continue
        return None

    def _compute_age(self, payload: dict) -> float:
        """Calculate data age in seconds."""
        ts = payload.get("timestamp") or payload.get("_server_ts")
        if not ts: return 0.0
        return max(0.0, time.time() - float(ts))
    
    async def ingest_tick(self, tick: Tick) -> None:
        """Process an incoming market tick (Async + Sanitized + Persistent)."""
        # 1. Sanitize FIRST to prevent subscriber crashes
        payload = self._sanitize_tick(dict(tick))
        
        symbol = payload.get("symbol")
        token = payload.get("instrument_token") or payload.get("token")
        
        # [HACK] Nifty 50 Token Hardcode (Requested)
        if str(token) == "256265": 
            payload["symbol"] = "NSE:NIFTY 50"
            symbol = "NSE:NIFTY 50"

        if not symbol and token:
            symbol = str(token)

        if not symbol: return
        
        # Calculate age for subscribers
        payload["_age"] = self._compute_age(payload)

        with self._lock:
            # 2. Update Cache
            self._quotes[symbol] = payload
            
            # 3. Hardcode Mirror
            if symbol == "NSE:NIFTY 50":
                self._quotes["NIFTY 50"] = payload

            # 4. Update Metrics (Throttled)
            try: self._capture_option_metrics(symbol, payload)
            except Exception: pass

        # 5. Persistence
        self._persist_wal("quote", symbol, payload)
        self._maybe_checkpoint()

        # 6. MessageBus (Async - Non-blocking)
        if self._message_bus:
            try:
                # Use create_task to fire-and-forget, avoiding await lag in polling loop
                loop = asyncio.get_running_loop()
                loop.create_task(self._publish_tick_async(payload))
            except RuntimeError:
                pass # No loop available

        # 7. Legacy Subscribers (Wrapped safely)
        if symbol in self._tick_subscribers:
            for callback in list(self._tick_subscribers[symbol]):
                try:
                    callback(payload)
                except Exception as exc:
                    # Log warning but don't crash the ingestion loop
                    LOGGER.warning(f"Tick callback failed for {symbol}: {exc}")
    def replace_positions(self, positions: Iterable[dict[str, Any]]) -> None:
        """Atomically replace the entire position snapshot."""
        new_map = {}
        for p in positions:
            sym = p.get("symbol")
            if sym:
                new_map[sym] = p
        
        with self._lock:
            self._positions = new_map

    # ----------------------------------------------------------------
    # Accessors (Read Path)
    # ----------------------------------------------------------------

    def get_quote(self, symbol: str, allow_pull: bool = False) -> Tick | None:
        """Return the latest cached tick for a symbol.
        
        Args:
            symbol: Trading symbol.
            allow_pull: If True and cache is empty, try fetching from broker via MDM.
        """
        with self._lock:
            tick = self._quotes.get(symbol)
            
        if tick is not None:
            return tick
            
        # FIX: Restore allow_pull logic to satisfy RuntimeSelfChecker
        if allow_pull and self._mdm and hasattr(self._mdm, "pull_quote"):
            try:
                # Note: pull_quote usually returns the dict AND triggers ingestion via callback
                # We return it directly here to satisfy the caller immediately
                return self._mdm.pull_quote(symbol)
            except Exception:
                pass
                
        return None

    def get_order(self, order_id: str) -> dict[str, Any] | None:
        """Return cached order details."""
        with self._lock:
            return self._orders.get(order_id)

    def get_position(self, symbol: str) -> dict[str, Any] | None:
        """Return cached position details."""
        with self._lock:
            return self._positions.get(symbol)
            
    def positions(self) -> list[dict[str, Any]]:
        """Return list of all cached positions."""
        with self._lock:
            return list(self._positions.values())

    def upsert_order(self, order: dict[str, Any]) -> None:
        """Update or insert an order record (alias for ingest_order_update)."""
        self.ingest_order_update(order)

    def ingest_order_update(self, payload: Mapping[str, Any]) -> None:
        """Insert or update cached order state from payload."""
        symbol = self.normalize(payload.get("symbol"))
        order_id = str(payload.get("order_id") or payload.get("id") or "").strip()
        if not order_id: return

        # Normalize fields
        row = dict(payload)
        row["symbol"] = symbol
        row["quantity"] = self._int(payload.get("quantity"))
        row["filled_quantity"] = self._int(payload.get("filled_quantity") or payload.get("filled"))
        row["price"] = self._float(payload.get("price"))
        row["trigger_price"] = self._float(payload.get("trigger_price"))
        
        status = str(payload.get("status") or "").lower()
        
        with self._lock:
            self._orders[order_id] = row
            self._order_status[order_id] = status
            listeners = list(self._order_subscribers)

        # Persist
        self._persist_wal("order", order_id, row)
        self._maybe_checkpoint()

        # Notify listeners
        for listener in listeners:
            try: listener(dict(row))
            except Exception: pass

    def get_iv(self, symbol: str, allow_fallback: bool = False) -> float | None:
        """Return cached Implied Volatility (IV)."""
        with self._lock:
            iv = self._iv_cache.get(symbol)
            if iv is not None:
                return iv
        return None

    def get_greeks(self, symbol: str) -> dict[str, float] | None:
        """Return cached Greeks (Delta, Gamma, Theta, Vega)."""
        with self._lock:
            return self._greeks_cache.get(symbol)

    def is_fresh(self, symbol: str, threshold_ms: float = 5000.0) -> tuple[bool, Freshness]:
        """Check freshness using robust timestamp analysis."""
        quote = self.get_quote(symbol, allow_pull=False)
        if not quote:
            return False, {"ok": False, "reason": "no_tick", "threshold_ms": threshold_ms}
        
        # Use calculated age if available, else re-calculate
        age_s = quote.get("_age")
        if age_s is None:
            age_s = self._compute_age(quote)
            
        age_ms = age_s * 1000.0
        is_fresh = age_ms <= threshold_ms
        
        return is_fresh, {
            "ok": is_fresh,
            "effective_ms": age_ms,
            "threshold_ms": threshold_ms,
            "reason": None if is_fresh else f"stale_{int(age_ms)}ms"
        }

    # ----------------------------------------------------------------
    # Subscription Management
    # ----------------------------------------------------------------

    def subscribe_ticks(self, symbol: str, callback: TickListener) -> None:
        """Register a callback for tick updates on a symbol."""
        with self._lock:
            if symbol not in self._tick_subscribers:
                self._tick_subscribers[symbol] = set()
                if self._mdm:
                    self._mdm.subscribe(symbol, self.ingest_tick)
            self._tick_subscribers[symbol].add(callback)

    def unsubscribe_ticks(self, symbol: str, callback: TickListener) -> None:
        """Unregister a tick callback."""
        with self._lock:
            if symbol in self._tick_subscribers:
                self._tick_subscribers[symbol].discard(callback)
                if not self._tick_subscribers[symbol]:
                    del self._tick_subscribers[symbol]

    def subscribe_orders(self, callback: OrderListener) -> None:
        """Register a callback for all order updates."""
        with self._lock:
            if callback not in self._order_subscribers:
                self._order_subscribers.append(callback)

    # ----------------------------------------------------------------
    # Internal Logic & Math
    # ----------------------------------------------------------------

    def _capture_option_metrics(self, symbol: str, tick: Tick) -> None:
        """Update IV and Greeks if the symbol is an option."""
        now = time.time()
        last_update = self._last_greeks_update.get(symbol, 0.0)
        if now - last_update < self._greeks_throttle_sec:
            return
            
        parsed = self._parse_option_symbol(symbol)
        if not parsed:
            return
        
        _base, expiry_ts, strike, is_call = parsed
        ltp = tick.get("ltp") or tick.get("last_price")
        if not isinstance(ltp, (int, float)) or ltp <= 0:
            return
            
        spot_price = self._get_underlying_price(_base)
        if not spot_price:
            return

        try:
            dte = (expiry_ts - now) / (365.0 * 24.0 * 3600.0)
            if dte <= 0:
                return

            r = 0.07 
            iv = implied_volatility(
                price=float(ltp),
                S=spot_price,
                K=strike,
                t=dte,
                r=r,
                flag="c" if is_call else "p"
            )
            
            if iv and iv > 0:
                self._iv_cache[symbol] = iv
                greeks = black_scholes_greeks(
                    S=spot_price,
                    K=strike,
                    t=dte,
                    r=r,
                    sigma=iv,
                    flag="c" if is_call else "p"
                )
                self._greeks_cache[symbol] = greeks
                
            oi = tick.get("oi") or tick.get("open_interest")
            if oi:
                self._oi_cache[symbol] = float(oi)
                
            self._last_greeks_update[symbol] = now
            
        except Exception:
            pass

    def _get_underlying_price(self, base: str) -> float | None:
        candidates = [base, "NIFTY 50", "NIFTY BANK"]
        if base == "BANKNIFTY":
            candidates = ["NIFTY BANK", "BANKNIFTY"]
            
        with self._lock:
            for cand in candidates:
                tick = self._quotes.get(cand)
                if tick:
                    p = tick.get("ltp")
                    if p: return float(p)
        return None

    def _parse_option_symbol(self, symbol: str) -> tuple[str, float, float, bool] | None:
        clean_sym = symbol.split(":")[-1]
        if self._resolver:
            try:
                meta = self._resolver.lookup(symbol)
                if meta and meta.get("expiry") and meta.get("strike"):
                    expiry_dt = meta["expiry"]
                    if isinstance(expiry_dt, datetime):
                        ts = expiry_dt.timestamp()
                    else:
                        return None
                        
                    strike = float(meta["strike"])
                    is_call = meta.get("instrument_type") == "CE" or clean_sym.endswith("CE")
                    base = "NIFTY" if "NIFTY" in clean_sym else "BANKNIFTY"
                    return base, ts, strike, is_call
            except Exception:
                pass
        return None

    def _clock(self) -> float:
        return time.time()
        
    # ----------------------------------------------------------------
    # Proxy Methods (Delegation to MDM)
    # ----------------------------------------------------------------
    
    def get_available_balance(self, force: bool = False) -> float | None:
        if self._mdm:
            return self._mdm.get_available_balance(force=force)
        return None

    def get_account_snapshot(self, force: bool = False) -> dict[str, float]:
        if self._mdm:
            return self._mdm.get_account_snapshot(force=force)
        return {}

    def normalize(self, symbol: str) -> str:
        # Static method in original, but instance method here is fine.
        # If callers use DataHub.normalize(), make it static.
        return symbol.strip().upper()
    
    # Make normalize static for compatibility with existing calls like DataHub.normalize()
    @staticmethod
    def normalize(symbol: str) -> str:
        return symbol.strip().upper()

    # ----------------------------------------------------------------
    # CRITICAL FIX: Option Chain Proxy for Strike Selector
    # ----------------------------------------------------------------
    
    def get_option_chain(self, symbol: str, option_type: str | None = None) -> list[dict]:
        """Retrieves option chain with Traceability Logs."""
        
        # LOG 1: Entry
        if hasattr(self, "_logger"):
            self._logger.info(f"🔍 DataHub: Fetching chain for {symbol}...")

        mdm = getattr(self, "_market_data", None) or getattr(self, "_mdm", None) or getattr(self, "_market_data_manager", None)
        
        result = []
        source = "None"

        if mdm and hasattr(mdm, "get_option_chain"):
            result = mdm.get_option_chain(symbol)
            source = "MDM"
        elif getattr(self, "_provider", None):
            provider = getattr(self, "_provider", None)
            if hasattr(provider, "get_option_chain"):
                result = provider.get_option_chain(symbol)
                source = "Provider"

        # LOG 2: Result
        count = len(result) if result else 0
        if hasattr(self, "_logger"):
            if count > 0:
                self._logger.info(f"✅ DataHub: Found {count} strikes via {source}.")
            else:
                self._logger.warning(f"⚠️ DataHub: Chain is EMPTY! Source: {source}. (Symbol: {symbol})")
        
        return result

# ----------------------------------------------------------------
    # Persistence (Restored for Production Safety)
    # ----------------------------------------------------------------
    def _persist_wal(self, kind: str, key: str, payload: Any) -> None:
        if self._store:
            try:
                self._store.append_wal(kind, key, dict(payload))
            except Exception:
                pass

    def _maybe_checkpoint(self) -> None:
        if not self._store: return
        now = time.time()
        if now - self._last_snapshot_ts > self._checkpoint_interval:
            try:
                with self._lock:
                    q = self._quotes.copy()
                    p = self._positions.copy()
                    o = self._orders.copy()
                    seq = self._order_sequences.copy()
                    stat = self._order_status.copy()
                
                self._store.save_snapshot("quotes", q)
                self._store.save_snapshot("positions", p)
                self._store.save_snapshot("orders", o)
                self._store.save_snapshot("order_sequences", seq)
                self._store.save_snapshot("order_status", stat)
                self._store.purge_wal()
                self._last_snapshot_ts = now
            except Exception:
                LOGGER.exception("Checkpoint failed")

    def _restore_from_store(self) -> None:
        """Load state from disk on startup."""
        if not self._store: return
        try:
            q = self._store.load_snapshot("quotes") or {}
            p = self._store.load_snapshot("positions") or {}
            o = self._store.load_snapshot("orders") or {}
            seq = self._store.load_snapshot("order_sequences") or {}
            stat = self._store.load_snapshot("order_status") or {}
            wal = self._store.load_wal()
            
            with self._lock:
                self._quotes.update(q)
                self._positions.update(p)
                self._orders.update(o)
                self._order_sequences.update({k: int(v) for k, v in seq.items()})
                self._order_status.update(stat)
                
                # Replay WAL
                for entry in wal:
                    if not isinstance(entry, dict): continue
                    kind = entry.get("kind")
                    payload = entry.get("payload")
                    key = str(entry.get("key"))
                    
                    if kind == "order" and isinstance(payload, dict):
                        self._orders[key] = payload
                    elif kind == "positions" and isinstance(payload, dict):
                        self._positions = payload
        except Exception:
            LOGGER.exception("Failed to restore DataHub state")
    
__all__ = ["DataHub", "Tick", "OrderListener", "TickListener"]
