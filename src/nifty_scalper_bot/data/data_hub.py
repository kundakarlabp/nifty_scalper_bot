"""Canonical data hub for cached ticks, orders, and positions."""

from __future__ import annotations

import os
import re
import time
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Callable, Iterable, Mapping, Optional, TypedDict, cast

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
    """Central in-memory state cache for the trading bot.
    
    Aggregates data from MarketDataManager (ticks) and OrderManager (orders),
    providing a unified, thread-safe interface for Strategies and Risk.
    """

    def __init__(
        self,
        market_data_manager: Any,
        instrument_resolver: Any,
        *,
        options_only: bool = True,
        store: HubStore | None = None,
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
        
        # Derived Metrics Caches
        self._iv_cache: dict[str, float] = {}
        self._oi_cache: dict[str, float] = {}
        self._greeks_cache: dict[str, dict[str, float]] = {}
        
        # Throttling for heavy math (Greeks/IV)
        self._last_greeks_update: dict[str, float] = {}
        self._greeks_throttle_sec = 0.5  # Calculate max once per 500ms

        # Subscribers
        self._tick_subscribers: dict[str, set[TickListener]] = {}
        self._order_subscribers: list[OrderListener] = []

    # ----------------------------------------------------------------
    # Ingestion (Write Path)
    # ----------------------------------------------------------------

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

    def ingest_order_update(self, order: dict[str, Any]) -> None:
        """Process an order status update."""
        order_id = order.get("order_id")
        if not order_id:
            return

        with self._lock:
            # State machine validation (optional, for safety)
            current = self._orders.get(order_id, {})
            old_status = current.get("status", "")
            new_status = order.get("status", "")
            
            # Simple upsert logic for now, can add strict transition checks here
            self._orders[order_id] = order

            # Notify subscribers
            for callback in self._order_subscribers:
                try:
                    callback(order)
                except Exception as exc:
                    LOGGER.error("Order subscriber failed: %s", exc, exc_info=True)

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

    def get_quote(self, symbol: str) -> Tick | None:
        """Return the latest cached tick for a symbol."""
        with self._lock:
            return self._quotes.get(symbol)

    def get_order(self, order_id: str) -> dict[str, Any] | None:
        """Return cached order details."""
        with self._lock:
            return self._orders.get(order_id)

    def get_position(self, symbol: str) -> dict[str, Any] | None:
        """Return cached position details."""
        with self._lock:
            return self._positions.get(symbol)

    def get_iv(self, symbol: str, allow_fallback: bool = False) -> float | None:
        """Return cached Implied Volatility (IV)."""
        with self._lock:
            iv = self._iv_cache.get(symbol)
            if iv is not None:
                return iv
            
        if allow_fallback:
            return self._atm_iv(self.get_quote(symbol))
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
        
        # Handle datetime objects or timestamps
        if isinstance(ts, datetime):
            ts_ms = ts.timestamp() * 1000.0
        elif isinstance(ts, (int, float)):
            ts_ms = float(ts) * (1000.0 if ts < 1e11 else 1.0) # Auto-detect sec/ms
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
                # Ensure the underlying MDM is also subscribed
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
        # 1. Throttle Check
        now = time.time()
        last_update = self._last_greeks_update.get(symbol, 0.0)
        if now - last_update < self._greeks_throttle_sec:
            return
            
        # 2. Parse Symbol
        parsed = self._parse_option_symbol(symbol)
        if not parsed:
            return
        
        _base, expiry_ts, strike, is_call = parsed
        
        # 3. Extract Price
        ltp = tick.get("ltp") or tick.get("last_price")
        if not isinstance(ltp, (int, float)) or ltp <= 0:
            return
            
        # 4. Get Underlying Price (Spot)
        # Assuming NIFTY options, underlying is usually "NIFTY 50" or similar token
        # This requires looking up the spot price. For simple scalping, we might skip
        # rigorous spot lookup if we don't have it, or use the Future price if mapped.
        # Fallback: Use 'NIFTY' token if available in quotes
        spot_price = self._get_underlying_price(_base)
        if not spot_price:
            return

        # 5. Calculate Math
        try:
            dte = (expiry_ts - now) / (365.0 * 24.0 * 3600.0) # Years to expiry
            if dte <= 0:
                return

            # Risk-free rate (approx)
            r = 0.07 
            
            # Calculate IV
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
                
                # Calculate Greeks
                greeks = black_scholes_greeks(
                    S=spot_price,
                    K=strike,
                    t=dte,
                    r=r,
                    sigma=iv,
                    flag="c" if is_call else "p"
                )
                self._greeks_cache[symbol] = greeks
                
            # Capture OI
            oi = tick.get("oi") or tick.get("open_interest")
            if oi:
                self._oi_cache[symbol] = float(oi)
                
            self._last_greeks_update[symbol] = now
            
        except Exception as exc:
            # Math errors shouldn't crash the hub
            # LOGGER.debug("Math error for %s: %s", symbol, exc) 
            pass

    def _get_underlying_price(self, base: str) -> float | None:
        """Resolve spot price for the underlying."""
        # Heuristic: Try direct symbol "NIFTY" or "NIFTY 50"
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
        """Parse NIFTY23OCT19500CE -> ('NIFTY', expiry_ts, 19500.0, True)."""
        # Clean NFO: prefix
        clean_sym = symbol.split(":")[-1]
        
        # Regex for standard Nifty symbols: NIFTY + YY + MMM + STRIKE + CE/PE
        # e.g. NIFTY23OCT19500CE
        # This is a simplified parser. For robust parsing, use the InstrumentResolver data.
        # Relying on regex for all formats is fragile. 
        # Prefer using cached instrument data if available.
        
        if self._resolver:
            try:
                meta = self._resolver.lookup(symbol)
                if meta and meta.get("expiry") and meta.get("strike"):
                    expiry_dt = meta["expiry"]
                    # Convert to timestamp
                    if isinstance(expiry_dt, datetime):
                        ts = expiry_dt.timestamp()
                    elif isinstance(expiry_dt, str):
                         # Quick parse fallback
                         return None 
                    else:
                        return None
                        
                    strike = float(meta["strike"])
                    is_call = meta["instrument_type"] == "CE" or clean_sym.endswith("CE")
                    base = "NIFTY" if "NIFTY" in clean_sym else "BANKNIFTY" # Simplified
                    return base, ts, strike, is_call
            except Exception:
                pass
        
        return None

    def _atm_iv(self, quote: Tick | None) -> float | None:
        """Estimate ATM IV fallback."""
        # Placeholder: In a real system, average IV of nearby strikes
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
        return symbol.strip().upper()

__all__ = ["DataHub", "Tick", "OrderListener", "TickListener"]
