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

    def __init__(
        self,
        market_data_manager: Any,
        instrument_resolver: Any,
        *,
        options_only: bool = True,
        store: HubStore | None = None,
        message_bus: MessageBus
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

    # data_hub.py, around line 89: Find ingest_tick
    def ingest_tick(self, tick: Tick) -> None:
        """Process an incoming market tick."""
        symbol = tick.get("symbol")
        if not symbol:
            return

        with self._lock:
            # Update quote cache
            self._quotes[symbol] = tick
            
            # Update derived metrics (Throttled)
            self._capture_option_metrics(symbol, tick)
            
            # Notify subscribers
            if symbol in self._tick_subscribers:
                for callback in list(self._tick_subscribers[symbol]):
                    try:
                        callback(tick)
                    except Exception as exc:
                        LOGGER.error(
                            "Tick subscriber failed for %s: %s", 
                            symbol, exc, exc_info=True
                        )
    
    def ingest_tick(self, tick: Tick) -> None:
        """Process an incoming market tick."""
        symbol = tick.get("symbol")
        if not symbol:
            return

        with self._lock:
            # Update quote cache
            self._quotes[symbol] = tick
            
            # Update derived metrics (Throttled)
            self._capture_option_metrics(symbol, tick)
            
            # ✅ FIX: PUBLISH TICK MESSAGE TO THE BUS FIRST
            if self._message_bus:
                try:
                    # Note: We use asyncio.create_task to run the async publish 
                    # from this synchronous thread safely.
                    asyncio.create_task(
                        self._message_bus.publish(
                            Message(
                                type=MessageType.TICK,
                                timestamp=datetime.now(timezone.utc), # Use UTC for consistency
                                data=tick,
                                source="data_hub"
                            )
                        )
                    )
                except Exception as exc:
                    # Catch loop not running or task creation failures
                    LOGGER.warning("Failed to publish tick to MessageBus: %s", exc)

            # Notify old synchronous subscribers for compatibility
            if symbol in self._tick_subscribers:
                for callback in list(self._tick_subscribers[symbol]):
                    try:
                        callback(tick)
                    except Exception as exc:
                        LOGGER.error(
                            "Tick subscriber failed for %s: %s", 
                            symbol, exc, exc_info=True
                        )

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

    def is_fresh(self, symbol: str, threshold_ms: float = 2000.0) -> tuple[bool, Freshness]:
        """Check if the quote for a symbol is fresh."""
        quote = self.get_quote(symbol)
        if not quote:
            return False, {"ok": False, "reason": "no_quote", "threshold_ms": threshold_ms}

        now = self._clock() * 1000.0
        ts = quote.get("timestamp")
        
        if isinstance(ts, datetime):
            ts_ms = ts.timestamp() * 1000.0
        elif isinstance(ts, (int, float)):
            ts_ms = float(ts) * (1000.0 if ts < 1e11 else 1.0)
        else:
            return False, {"ok": False, "reason": "invalid_ts", "threshold_ms": threshold_ms}

        age = max(0.0, now - ts_ms)
        is_fresh = age <= threshold_ms
        
        return is_fresh, {
            "ok": is_fresh,
            "effective_ms": age,
            "threshold_ms": threshold_ms,
            "reason": None if is_fresh else "stale"
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

__all__ = ["DataHub", "Tick", "OrderListener", "TickListener"]
