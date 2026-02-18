"""Canonical data hub for cached ticks, orders, and positions."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import os
import re
from collections import defaultdict
from threading import RLock
import time
from typing import Any, Callable, Iterable, Mapping, TypedDict, cast

from nifty_scalper_bot.core.message_bus import Message, MessageBus, MessageType
from nifty_scalper_bot.storage.hub_store import HubStore
from nifty_scalper_bot.utils.logging import get_logger
from nifty_scalper_bot.utils.market_hours import MarketState, get_market_state
from nifty_scalper_bot.utils.options_math import (
    black_scholes_greeks,
    implied_volatility,
)
from nifty_scalper_bot.utils.symbols import (
    canonical,
    enforce_canonical,
    normalize_symbol,
)

LOGGER = get_logger(__name__)


Tick = dict[str, Any]
OrderListener = Callable[[dict[str, Any]], None]
TickListener = Callable[[dict[str, Any]], None]


class EventBus:
    """Generic in-process event bus.

    Args: none; Returns: none; Raises: none.
    """

    def __init__(self) -> None:
        """Args: none; Returns: none; Raises: none."""
        self._handlers: dict[str, list[Callable[[dict[str, Any]], None]]] = defaultdict(
            list
        )
        self._lock = RLock()

    def subscribe(self, event_name: str, handler: Callable[[dict[str, Any]], None]) -> None:
        """Args: event_name, handler; Returns: none; Raises: none."""
        with self._lock:
            self._handlers[event_name].append(handler)

    def publish(self, event_name: str, payload: dict[str, Any]) -> None:
        """Args: event_name, payload; Returns: none; Raises: none."""
        with self._lock:
            handlers = list(self._handlers.get(event_name, ()))
        for handler in handlers:
            try:
                handler(payload)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("Failure in EventBus.publish: %s", exc, exc_info=exc)


class TickBus:
    """Backward-compatible tick bus wrapper around :class:`EventBus`."""

    def __init__(self) -> None:
        """Args: none; Returns: none; Raises: none."""
        self._event_bus = EventBus()

    def subscribe(self, handler: TickListener) -> None:
        """Args: handler; Returns: none; Raises: none."""
        self._event_bus.subscribe("tick", handler)

    def publish(self, tick: dict[str, Any]) -> None:
        """Args: tick; Returns: none; Raises: none."""
        self._event_bus.publish("tick", tick)

    def subscribe_event(
        self,
        event_name: str,
        handler: Callable[[dict[str, Any]], None],
    ) -> None:
        """Args: event_name, handler; Returns: none; Raises: none."""
        self._event_bus.subscribe(event_name, handler)

    def publish_event(self, event_name: str, payload: dict[str, Any]) -> None:
        """Args: event_name, payload; Returns: none; Raises: none."""
        self._event_bus.publish(event_name, payload)


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


@dataclass(frozen=True, slots=True)
class _HistoryCacheKey:
    """Key for option-chain-aware historical cache entries."""

    symbol_root: str
    expiry: str
    interval: str


@dataclass(slots=True)
class _HistoryCacheEntry:
    """Cached history rows and fetch timestamp."""

    rows: list[dict[str, Any]]
    fetched_at: float

    def is_fresh(self, max_age_seconds: float, now: float) -> bool:
        """Return True when this cache entry is still fresh."""
        return (now - self.fetched_at) <= max_age_seconds


@dataclass(frozen=True, slots=True)
class _HistoryFetchContext:
    """Input metadata for a history fetch request."""

    symbol: str
    root: str
    expiry: str
    interval: str
    now: float


class DataHub:
    """Central in-memory state cache for the trading bot."""

    def __init__(
        self,
        market_data_manager: Any,
        instrument_resolver: Any,
        *,
        options_only: bool = True,
        store: HubStore | None = None,
        message_bus: MessageBus | None = None,
        event_bus: MessageBus | None = None,
    ) -> None:

        self._mdm = market_data_manager
        self._resolver = instrument_resolver
        self._options_only = options_only
        self._store = store
        self._lock = RLock()
        self.tick_bus = TickBus()
        self.tick_bus.subscribe(self.ingest_tick_sync)
        attach_tick_bus = getattr(self._mdm, "attach_tick_bus", None)
        if callable(attach_tick_bus):
            try:
                attach_tick_bus(self.tick_bus)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("Failure in DataHub.__init__: %s", exc, exc_info=exc)

        # State Caches
        self._quotes: dict[str, Tick] = {}
        self._orders: dict[str, dict[str, Any]] = {}
        self._positions: dict[str, dict[str, Any]] = {}
        self._message_bus = message_bus or event_bus
        LOGGER.debug("DataHub using MessageBus id=%s", id(self._message_bus))

        # Derived Metrics Caches
        self._iv_cache: dict[str, float] = {}
        self._oi_cache: dict[str, float] = {}
        self._greeks_cache: dict[str, dict[str, float]] = {}

        # Throttling for heavy math (Greeks/IV)
        self._last_greeks_update: dict[str, float] = {}
        self._greeks_throttle_sec = 0.5

        # Subscribers
        self._tick_subscribers: dict[str, set[TickListener]] = {}
        self._subscribed_symbols: set[str] = set()
        self._order_subscribers: list[OrderListener] = []
        self._history_cache_lock = RLock()
        self._history_cache: dict[_HistoryCacheKey, _HistoryCacheEntry] = {}
        self._history_cache_ttl_seconds = float(
            os.getenv("HISTORY_CACHE_TTL_SECONDS", "120")
        )
        self._history_freshness_max_age_seconds = float(
            os.getenv("HISTORY_FRESHNESS_MAX_AGE_SECONDS", "120")
        )

    # ----------------------------------------------------------------
    # Ingestion (Write Path)
    # ----------------------------------------------------------------

    def ingest_tick_sync(self, tick: dict) -> None:
        """Synchronous bridge to schedule ingest_tick on the running loop."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.ingest_tick(tick))
        except RuntimeError:
            pass

    async def ingest_tick(self, tick: Tick) -> None:
        """Process an incoming market tick."""
        symbol = tick.get("symbol")
        if not symbol:
            return
        normalized_symbol = enforce_canonical(canonical(str(symbol)))

        if self._message_bus is None:
            LOGGER.error("DataHub has no event_bus — cannot emit tick")
            return

        with self._lock:
            # 1. Update Cache
            self._quotes[normalized_symbol] = tick

            # 2. Update Metrics (Throttled)
            try:
                self._capture_option_metrics(normalized_symbol, tick)
            except Exception:
                pass  # Don't let math errors kill the tick

            # 3. Publish to MessageBus (The Critical Fix)
            if self._message_bus:
                try:
                    await self._message_bus.publish(
                        Message(
                            type=MessageType.TICK,
                            timestamp=datetime.now(timezone.utc),
                            data=tick,
                            source="data_hub",
                        )
                    )
                except Exception as exc:
                    LOGGER.error("Failure in DataHub.ingest_tick: %s", exc, exc_info=exc)

            # 4. Notify Legacy Subscribers (Backward Compatibility)
            if normalized_symbol in self._tick_subscribers:
                for callback in list(self._tick_subscribers[normalized_symbol]):
                    try:
                        callback(tick)
                    except Exception as exc:
                        # This is likely where the "Tick callback failed" log comes from
                        LOGGER.error(
                            f"Tick subscriber failed for {symbol}: {exc}",
                            exc_info=True,  # Prints full traceback to help debug
                        )

    def store_quote(
        self,
        symbol: str,
        quote_data: dict[str, Any],
        source: str = "ws",
        seed: bool = False,
    ) -> None:
        """
        Universal entry point.
        Redirects legacy/polling calls to the ACTIVE ingestion pipeline.
        """
        # 1. Defensive Copy & Normalization
        # Prevents reference bugs if the caller reuses the dict
        payload = dict(quote_data)

        # 2. Enforce Metadata
        payload["source"] = source
        payload["seed"] = bool(seed)

        # Ensure symbol presence
        canonical_symbol = enforce_canonical(canonical(symbol))
        assert canonical_symbol.count(":") == 1
        payload["symbol"] = canonical_symbol

        # Ensure timestamp (critical for freshness checks)
        if "timestamp" not in payload:
            import time

            payload["timestamp"] = int(time.time() * 1000)

        # 3. 🔥 CRITICAL REDIRECT 🔥
        # Send to the async pipeline to trigger MessageBus, Strategies, and Greeks.
        # This makes Polling/Rest/Scout sources "Alive".
        self.ingest_tick_sync(payload)

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
        lookup_symbol = canonical(symbol)
        with self._lock:
            tick = self._quotes.get(lookup_symbol)

        if tick is not None:
            return tick

        if allow_pull and self._mdm and hasattr(self._mdm, "pull_quote"):
            try:
                transport_status = getattr(self._mdm, "transport_status", None)
                if callable(transport_status):
                    state = transport_status()
                    poll_enabled = bool((state or {}).get("polling"))
                    ws_connected = bool((state or {}).get("websocket"))
                    if ws_connected and not poll_enabled:
                        return None
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

    def is_fresh(
        self,
        symbol: str,
        threshold_ms: float = 5000.0,
    ) -> tuple[bool, dict[str, Any]]:
        """Check freshness with market-state and source-aware TTL handling."""
        market_state = get_market_state()
        if market_state != MarketState.OPEN:
            return True, {
                "ok": True,
                "reason": "market_closed",
                "market_state": market_state.value,
                "threshold_ms": threshold_ms,
            }

        transport_status = getattr(self._mdm, "transport_status", None)
        if callable(transport_status):
            status = transport_status() or {}
            ws_state = str(status.get("ws_state") or "").lower()
            if ws_state != "connected":
                return True, {
                    "ok": True,
                    "reason": "ws_not_connected",
                    "market_state": market_state.value,
                    "threshold_ms": threshold_ms,
                }

        quote = self.get_quote(symbol)
        if not quote:
            return False, {
                "ok": False,
                "reason": "no_quote",
                "threshold_ms": threshold_ms,
            }

        if quote.get("seed"):
            return True, {
                "ok": True,
                "reason": "seed_warmup",
                "source": quote.get("source"),
                "threshold_ms": threshold_ms,
            }

        now = (self._clock() if hasattr(self, "_clock") else time.time()) * 1000.0
        ts = quote.get("timestamp")
        if isinstance(ts, datetime):
            ts_ms = ts.timestamp() * 1000.0
        elif isinstance(ts, (int, float)):
            ts_ms = float(ts) * (1000.0 if ts < 1e11 else 1.0)
        else:
            return False, {
                "ok": False,
                "reason": "invalid_ts",
                "threshold_ms": threshold_ms,
            }

        age = max(0.0, now - ts_ms)
        source = str(quote.get("source", "ws")).lower()
        if source == "ws":
            ttl_ms = 5_000.0
        elif source == "rest":
            ttl_ms = 90_000.0
        elif source == "historical":
            ttl_ms = float("inf")
        else:
            ttl_ms = threshold_ms
        effective_threshold = ttl_ms
        is_fresh = age < effective_threshold

        return is_fresh, {
            "ok": is_fresh,
            "effective_ms": age,
            "threshold_ms": effective_threshold,
            "source": source,
            "reason": None if is_fresh else "stale",
        }

    # ----------------------------------------------------------------
    # Subscription Management
    # ----------------------------------------------------------------

    def subscribe_ticks(self, symbol: str, callback: TickListener) -> None:
        """Register a callback for tick updates on a symbol."""
        normalized = enforce_canonical(canonical(symbol))
        assert normalized.count(":") == 1
        with self._lock:
            if normalized not in self._tick_subscribers:
                self._tick_subscribers[normalized] = set()
            self._tick_subscribers[normalized].add(callback)
            if normalized in self._subscribed_symbols:
                return
        if self._mdm:
            try:
                self._mdm.subscribe(normalized, self.ingest_tick_sync)
                with self._lock:
                    self._subscribed_symbols.add(normalized)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("Failure in DataHub.subscribe_ticks: %s", exc)

    def unsubscribe_ticks(self, symbol: str, callback: TickListener) -> None:
        """Unregister a tick callback."""
        normalized = enforce_canonical(canonical(symbol))
        with self._lock:
            if normalized in self._tick_subscribers:
                self._tick_subscribers[normalized].discard(callback)
                if not self._tick_subscribers[normalized]:
                    del self._tick_subscribers[normalized]
                    self._subscribed_symbols.discard(normalized)

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
                flag="c" if is_call else "p",
            )

            if iv and iv > 0:
                self._iv_cache[symbol] = iv
                greeks = black_scholes_greeks(
                    S=spot_price,
                    K=strike,
                    t=dte,
                    r=r,
                    sigma=iv,
                    flag="c" if is_call else "p",
                )
                self._greeks_cache[symbol] = greeks

            oi = tick.get("oi") or tick.get("open_interest")
            if oi:
                self._oi_cache[symbol] = float(oi)

            self._last_greeks_update[symbol] = now

        except Exception:
            pass

    def _get_underlying_price(self, base: str) -> float | None:
        candidates = [canonical(base)]
        if base == "BANKNIFTY":
            candidates = [canonical("BANKNIFTY")]

        with self._lock:
            for cand in candidates:
                tick = self._quotes.get(cand)
                if tick:
                    p = tick.get("ltp")
                    if p:
                        return float(p)
        return None

    def _parse_option_symbol(
        self, symbol: str
    ) -> tuple[str, float, float, bool] | None:
        clean_sym = symbol
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
                    is_call = meta.get("instrument_type") == "CE" or clean_sym.endswith(
                        "CE"
                    )
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

    @staticmethod
    def normalize(symbol: str) -> str:
        """Normalize symbol. Args: symbol. Returns: canonical symbol. Raises: None."""
        return normalize_symbol(symbol)

    # ----------------------------------------------------------------
    # CRITICAL FIX: Option Chain Proxy for Strike Selector
    # ----------------------------------------------------------------

    def get_option_chain(
        self, symbol: str, option_type: str | None = None
    ) -> list[dict]:
        """Retrieves option chain with Traceability Logs."""

        # LOG 1: Entry
        if hasattr(self, "_logger"):
            self._logger.info(f"🔍 DataHub: Fetching chain for {symbol}...")

        mdm = (
            getattr(self, "_market_data", None)
            or getattr(self, "_mdm", None)
            or getattr(self, "_market_data_manager", None)
        )

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
                self._logger.warning(
                    f"⚠️ DataHub: Chain is EMPTY! Source: {source}. (Symbol: {symbol})"
                )

        return result

    # [INSERT THIS AT THE BOTTOM OF THE DataHub CLASS, BEFORE __all__]

    # ----------------------------------------------------------------
    # Historical Data Proxy (CRITICAL FOR BACKFILL)
    # ----------------------------------------------------------------

    async def fetch_history(
        self, symbol: str, interval: str, days: int = 3
    ) -> list[dict]:
        """Fetch closed candles. Args: inputs. Returns: candles. Raises: None."""
        context = self._build_history_context(symbol=symbol, interval=interval)
        cache_key = _HistoryCacheKey(
            symbol_root=context.root,
            expiry=context.expiry,
            interval=context.interval,
        )

        with self._history_cache_lock:
            entry = self._history_cache.get(cache_key)
            if entry and entry.is_fresh(self._history_cache_ttl_seconds, context.now):
                LOGGER.debug(
                    "Condition met: history_cache_hit",
                    extra={
                        "event": "history_cache_hit",
                        "symbol": context.symbol,
                        "root": context.root,
                        "expiry": context.expiry,
                        "interval": context.interval,
                        "age_seconds": round(context.now - entry.fetched_at, 3),
                    },
                )
                return [dict(row) for row in entry.rows]

        mdm = getattr(self, "_mdm", None) or getattr(self, "_market_data", None)
        if not (mdm and hasattr(mdm, "fetch_history")):
            LOGGER.warning(
                "DataHub: Underlying MDM missing fetch_history",
                extra={"event": "history_fetcher_missing", "symbol": context.symbol},
            )
            return []

        try:
            rows = await mdm.fetch_history(context.symbol, context.interval, days)
        except Exception as e:  # noqa: BLE001
            LOGGER.error(
                "DataHub Proxy Error: fetch_history failed for %s: %s",
                context.symbol,
                e,
                extra={"event": "history_fetch_failed", "symbol": context.symbol},
                exc_info=True,
            )
            return self._load_cached_history(cache_key)

        normalized_rows = self._normalize_history_rows(rows)
        if not normalized_rows:
            return self._load_cached_history(cache_key)

        with self._history_cache_lock:
            self._history_cache[cache_key] = _HistoryCacheEntry(
                rows=normalized_rows,
                fetched_at=context.now,
            )

        return [dict(row) for row in normalized_rows]

    def _build_history_context(
        self, symbol: str, interval: str
    ) -> _HistoryFetchContext:
        """Build fetch context. Args: inputs. Returns: context. Raises: None."""
        root, expiry = self._extract_history_group(symbol)
        return _HistoryFetchContext(
            symbol=(symbol or "").strip().upper(),
            root=root,
            expiry=expiry,
            interval=(interval or "").strip().lower() or "minute",
            now=time.time(),
        )

    def _load_cached_history(self, cache_key: _HistoryCacheKey) -> list[dict[str, Any]]:
        """Load stale cache on failure. Args: cache_key. Returns: rows. Raises: None."""
        with self._history_cache_lock:
            entry = self._history_cache.get(cache_key)
            if entry:
                LOGGER.warning(
                    "Condition met: history_fetch_using_stale_cache",
                    extra={
                        "event": "history_fetch_using_stale_cache",
                        "symbol_root": cache_key.symbol_root,
                        "expiry": cache_key.expiry,
                        "interval": cache_key.interval,
                        "rows": len(entry.rows),
                    },
                )
                return [dict(row) for row in entry.rows]
        return []

    def _normalize_history_rows(self, rows: Any) -> list[dict[str, Any]]:
        """Normalize rows. Args: rows. Returns: clean rows. Raises: None."""
        if not isinstance(rows, Iterable):
            return []

        now_utc = datetime.now(timezone.utc)
        cutoff = now_utc.replace(second=0, microsecond=0)
        normalized: list[dict[str, Any]] = []
        seen_timestamps: set[datetime] = set()
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            timestamp = self._coerce_history_timestamp(
                row.get("timestamp") or row.get("date")
            )
            if timestamp is None:
                continue
            candle_ts = timestamp.astimezone(timezone.utc).replace(
                second=0, microsecond=0
            )
            if candle_ts >= cutoff:
                continue
            if candle_ts in seen_timestamps:
                continue
            seen_timestamps.add(candle_ts)
            normalized.append(
                {
                    "timestamp": candle_ts,
                    "open": float(row.get("open", 0.0) or 0.0),
                    "high": float(row.get("high", 0.0) or 0.0),
                    "low": float(row.get("low", 0.0) or 0.0),
                    "close": float(row.get("close", 0.0) or 0.0),
                    "volume": float(row.get("volume", 0.0) or 0.0),
                }
            )
        normalized.sort(key=lambda item: cast(datetime, item["timestamp"]))
        return normalized

    @staticmethod
    def _coerce_history_timestamp(value: Any) -> datetime | None:
        """Parse timestamp. Args: value. Returns: UTC datetime. Raises: None."""
        if isinstance(value, datetime):
            ts = value
        elif isinstance(value, str):
            try:
                ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
        elif isinstance(value, (int, float)):
            numeric = float(value)
            if numeric > 1e12:
                numeric /= 1000.0
            ts = datetime.fromtimestamp(numeric, tz=timezone.utc)
        else:
            return None

        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts

    def history_freshness(
        self,
        symbol: str,
        interval: str,
        *,
        max_age_seconds: float | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """Return freshness diagnostics for historical cache groups."""
        root, expiry = self._extract_history_group(symbol)
        cache_key = _HistoryCacheKey(symbol_root=root, expiry=expiry, interval=interval)
        now = time.time()
        age_limit = (
            float(max_age_seconds)
            if max_age_seconds is not None
            else self._history_freshness_max_age_seconds
        )
        with self._history_cache_lock:
            entry = self._history_cache.get(cache_key)

        if entry is None:
            return False, {
                "ok": False,
                "reason": "missing",
                "symbol_root": root,
                "expiry": expiry,
                "interval": interval,
                "max_age_seconds": age_limit,
            }

        age_seconds = max(0.0, now - entry.fetched_at)
        is_fresh = age_seconds <= age_limit
        return is_fresh, {
            "ok": is_fresh,
            "reason": None if is_fresh else "stale",
            "symbol_root": root,
            "expiry": expiry,
            "interval": interval,
            "age_seconds": round(age_seconds, 3),
            "max_age_seconds": age_limit,
            "candles": len(entry.rows),
        }

    @staticmethod
    def _extract_history_group(symbol: str) -> tuple[str, str]:
        """Return ``(symbol_root, expiry)`` grouping keys for history caching."""
        normalized = (symbol or "").strip().upper()
        clean = normalized
        match = re.match(
            r"^(?P<root>[A-Z]+)(?P<expiry>(?:\d{2}[A-Z]\d{2}|\d{2}[A-Z]{3}|\d{2}[A-Z]{3}\d{2}))\d+(?:CE|PE)$",
            clean,
        )
        if match:
            return match.group("root"), match.group("expiry")
        return clean, "SPOT"


__all__ = ["DataHub", "Tick", "OrderListener", "TickListener"]
