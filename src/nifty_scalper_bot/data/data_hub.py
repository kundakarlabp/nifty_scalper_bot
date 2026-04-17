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
        self._positions: Dict[int, dict] = {}

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

        self._store = kwargs.get("store")
        self._event_bus = kwargs.get("event_bus")

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


    def replace_positions(self, positions: list[dict]) -> None:
        """
        Replace entire position state (used during restore/startup sync).
        This makes DataHub the SSOT for positions.
        """
        if not hasattr(self, "_positions"):
            self._positions = {}

        # Normalize → dict keyed by instrument/token
        new_positions = {}
        for pos in positions:
            token = pos.get("instrument_token")
            if token is None:
                continue

            key = int(token)
            if key is None:
                continue
            new_positions[key] = pos

        with self._lock:
            self._positions = new_positions
        # Optional: persist if store exists
        if self._store and hasattr(self._store, "save_positions"):
            try:
                self._store.save_positions(self._positions)
            except Exception as e:
                logger.warning(f"⚠️ Failed to persist positions: {e}")

        LOGGER.info(f"✅ Positions replaced in DataHub | count={len(self._positions)}")
    def get_positions(self) -> dict:
        with self._lock:
            return dict(self._positions)

    def update_position(self, position: dict) -> None:
        if not hasattr(self, "_positions"):
            self._positions = {}

        token = position.get("instrument_token")
        if token is None:
            return

        try:
            key = int(token)
        except (TypeError, ValueError):
            return

        with self._lock:
            self._positions[key] = position

    def clear_positions(self) -> None:
        with self._lock:
            self._positions = {}

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
    def subscribe(self, symbol: str, callback: Optional[TickListener] = None):
        """Register a tick listener for *symbol*.

        When ``callback`` is omitted the call is forwarded to MDM so existing
        MDM-only subscribers (REST tracking) still work.
        """

        symbol = str(symbol).upper().strip()

        if callback is not None:
            self._tick_subscribers[symbol].add(callback)

        mdm_sub = getattr(self._mdm, "subscribe", None)
        if callable(mdm_sub):
            try:
                if callback is None:
                    mdm_sub(symbol)
                else:
                    mdm_sub(symbol, callback)
            except TypeError:
                # MDM.subscribe may require callback; ignore signature mismatch
                pass
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("MDM subscribe delegation failed for %s: %s", symbol, exc)

    # Aliases used by some callers (runner/lifecycle_manager)
    subscribe_ticks = subscribe

    def unsubscribe(self, symbol: str, callback: Optional[TickListener] = None):

        symbol = str(symbol).upper().strip()

        if callback is not None and callback in self._tick_subscribers.get(symbol, set()):
            self._tick_subscribers[symbol].remove(callback)

        mdm_unsub = getattr(self._mdm, "unsubscribe", None)
        if callable(mdm_unsub):
            try:
                if callback is None:
                    mdm_unsub(symbol)
                else:
                    mdm_unsub(symbol, callback)
            except TypeError:
                pass
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("MDM unsubscribe delegation failed for %s: %s", symbol, exc)

    unsubscribe_ticks = unsubscribe

    # =========================================================
    # FACADE: read-through to MDM (SSOT for strategies/execution)
    # =========================================================
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Return last traded price for *symbol* preferring SSOT cache."""

        tick = self.get_quote(symbol)
        if tick:
            for key in ("ltp", "last_price", "price", "close"):
                value = tick.get(key)
                if value is not None:
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        continue

        mdm_fn = getattr(self._mdm, "get_latest_price", None)
        if callable(mdm_fn):
            try:
                return mdm_fn(symbol)
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("get_latest_price delegate failed for %s: %s", symbol, exc)
        return None

    def pull_quote(self, symbol: str) -> Dict[str, Any]:
        """Force a REST-backed refresh via MDM."""

        mdm_fn = getattr(self._mdm, "pull_quote", None)
        if callable(mdm_fn):
            try:
                return mdm_fn(symbol) or {}
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("pull_quote delegate failed for %s: %s", symbol, exc)
        return self.get_quote(symbol) or {}

    def probe_quote(self, symbol: str) -> Dict[str, Any]:
        mdm_fn = getattr(self._mdm, "probe_quote", None)
        if callable(mdm_fn):
            try:
                return mdm_fn(symbol) or {}
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("probe_quote delegate failed for %s: %s", symbol, exc)
        return self.get_quote(symbol) or {}

    def get_ohlc_bars(
        self, symbol: str, *, limit: Optional[int] = None
    ) -> list:
        mdm_fn = getattr(self._mdm, "get_ohlc_bars", None)
        if callable(mdm_fn):
            try:
                return mdm_fn(symbol, limit=limit) if limit is not None else mdm_fn(symbol)
            except TypeError:
                try:
                    return mdm_fn(symbol)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.debug("get_ohlc_bars fallback failed for %s: %s", symbol, exc)
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("get_ohlc_bars delegate failed for %s: %s", symbol, exc)
        return []

    def ensure_tracking(self, symbol: str, *, seed: bool = True) -> bool:
        mdm_fn = getattr(self._mdm, "ensure_tracking", None)
        if callable(mdm_fn):
            try:
                return bool(mdm_fn(symbol, seed=seed))
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("ensure_tracking delegate failed for %s: %s", symbol, exc)
        return False

    def is_symbol_ready(self, symbol: str) -> bool:
        mdm_fn = getattr(self._mdm, "is_symbol_ready", None)
        if callable(mdm_fn):
            try:
                return bool(mdm_fn(symbol))
            except Exception:  # noqa: BLE001
                return False
        return self.get_quote(symbol) is not None

    def resolve_symbol_token(self, symbol: str) -> Optional[int]:
        symbol = str(symbol).upper().strip()
        with self._lock:
            token = self._token_by_symbol.get(symbol)
        if token:
            return token
        mdm_fn = getattr(self._mdm, "resolve_symbol_token", None)
        if callable(mdm_fn):
            try:
                return mdm_fn(symbol)
            except Exception:  # noqa: BLE001
                return None
        return None

    def get_option_chain(self, *args, **kwargs):
        mdm_fn = getattr(self._mdm, "get_option_chain", None)
        if callable(mdm_fn):
            return mdm_fn(*args, **kwargs)
        return None

    def process_ticks(self, ticks) -> None:
        """Forward a batch of raw WS ticks into MDM (which re-emits to us)."""
        mdm_fn = getattr(self._mdm, "process_ticks", None)
        if callable(mdm_fn):
            try:
                mdm_fn(ticks)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("process_ticks delegate failed: %s", exc, exc_info=exc)

    # =========================================================
    # ACCOUNT / BALANCE FACADE
    # =========================================================
    def get_available_balance(self, *, force: bool = False) -> Optional[float]:
        """Return latest available margin balance from MDM's cache.

        Safe to call from any thread / event loop. Returns ``None`` when the
        underlying broker has not populated a snapshot yet.
        """

        mdm_fn = getattr(self._mdm, "get_available_balance", None)
        if callable(mdm_fn):
            try:
                return mdm_fn(force=force)
            except TypeError:
                try:
                    return mdm_fn()
                except Exception as exc:  # noqa: BLE001
                    LOGGER.debug("get_available_balance fallback failed: %s", exc)
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("get_available_balance delegate failed: %s", exc)

        # Fallback: read cached snapshot dict directly
        snapshot = self.get_account_snapshot(force=False)
        if snapshot:
            for key in ("available", "net", "cash"):
                value = snapshot.get(key)
                if value is not None:
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        continue
        return None

    def get_account_snapshot(self, *, force: bool = False) -> Dict[str, Any]:
        """Return the broker margin snapshot synchronously.

        MDM exposes this coroutine, so we read the cached dict directly. When
        ``force=True`` we best-effort schedule an async refresh without blocking
        the caller.
        """

        mdm = self._mdm
        cached = getattr(mdm, "_account_snapshot", None)
        snapshot: Dict[str, Any] = dict(cached) if isinstance(cached, dict) else {}

        if force:
            refresher = getattr(mdm, "get_account_snapshot", None)
            if callable(refresher):
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                if loop is not None and loop.is_running():
                    try:
                        asyncio.run_coroutine_threadsafe(refresher(force=True), loop)
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.debug("account_snapshot refresh schedule failed: %s", exc)
                # if no running loop we simply return the cached value;
                # a blocking asyncio.run from a non-async thread is too risky.

        return snapshot

    # =========================================================
    # READINESS / FRESHNESS
    # =========================================================
    def is_ready(self, symbol: Optional[Any] = None) -> bool:
        """Delegate to MDM's readiness check.

        Accepts an optional symbol/token for callers that used the richer
        DataHub contract; MDM's ``is_ready`` ignores the argument.
        """

        mdm_fn = getattr(self._mdm, "is_ready", None)
        if callable(mdm_fn):
            try:
                return bool(mdm_fn())
            except Exception:  # noqa: BLE001
                return False
        return bool(self._ticks)

    def is_fresh(
        self, symbol: str, *, threshold_ms: Optional[float] = None
    ) -> tuple[bool, Dict[str, Any]]:
        """Return (fresh?, meta) using WS arrival timestamps tracked in SSOT."""

        sym = str(symbol or "").upper().strip()
        threshold = float(threshold_ms) if threshold_ms is not None else 2000.0

        with self._lock:
            last_ws = self._last_ws_arrival.get(sym)
            last_any = self._last_arrival.get(sym)

        last = last_ws or last_any
        if not last:
            return False, {"reason": "no_tick", "symbol": sym}

        age_ms = (time.time() * 1000) - last
        fresh = age_ms <= threshold
        return fresh, {"age_ms": age_ms, "threshold_ms": threshold, "symbol": sym}

    # =========================================================
    # OPTIONAL DELEGATES (return safe defaults when absent)
    # =========================================================
    def get_iv(self, symbol: str) -> Optional[float]:
        mdm_fn = getattr(self._mdm, "get_iv", None)
        if callable(mdm_fn):
            try:
                return mdm_fn(symbol)
            except Exception:  # noqa: BLE001
                return None
        tick = self.get_quote(symbol) or {}
        value = tick.get("iv") or tick.get("implied_volatility")
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def get_indicator(self, symbol: str, name: str) -> Optional[float]:
        mdm_fn = getattr(self._mdm, "get_indicator", None)
        if callable(mdm_fn):
            try:
                return mdm_fn(symbol, name)
            except Exception:  # noqa: BLE001
                return None
        return None

    def get_position_state(self, symbol: str) -> Optional[Dict[str, Any]]:
        mdm_fn = getattr(self._mdm, "get_position_state", None)
        if callable(mdm_fn):
            try:
                return mdm_fn(symbol)
            except Exception:  # noqa: BLE001
                return None
        return None

    def get_data(self, token: Any):
        """Return (candles, indicators) for *token* when available."""
        mdm_fn = getattr(self._mdm, "get_data", None)
        if callable(mdm_fn):
            try:
                return mdm_fn(token)
            except Exception:  # noqa: BLE001
                return None, None
        return None, None

    def fetch_history(self, *args, **kwargs):
        mdm_fn = getattr(self._mdm, "fetch_history", None)
        if callable(mdm_fn):
            return mdm_fn(*args, **kwargs)
        return None

    def get_stats(self) -> Dict[str, Any]:
        return self.stats()

    # =========================================================
    # ORDER / LIFECYCLE SUBSCRIPTIONS
    # =========================================================
    def subscribe_orders(self, callback) -> None:
        """Register an order-state listener; safe no-op when MDM lacks one."""

        if callback is None:
            return
        if not hasattr(self, "_order_subscribers"):
            self._order_subscribers: set = set()
        self._order_subscribers.add(callback)
        mdm_fn = getattr(self._mdm, "subscribe_orders", None)
        if callable(mdm_fn):
            try:
                mdm_fn(callback)
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("subscribe_orders delegate failed: %s", exc)

    def unsubscribe_orders(self, callback) -> None:
        if callback is None:
            return
        subs = getattr(self, "_order_subscribers", None)
        if subs is not None:
            subs.discard(callback)
        mdm_fn = getattr(self._mdm, "unsubscribe_orders", None)
        if callable(mdm_fn):
            try:
                mdm_fn(callback)
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("unsubscribe_orders delegate failed: %s", exc)

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
