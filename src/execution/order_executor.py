# Path: src/execution/order_executor.py
from __future__ import annotations

import logging
import math
import os
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple

from src.logs.journal import Journal
from src.utils.broker_errors import (
    AUTH,
    SUBSCRIPTION,
    THROTTLE,
    UNKNOWN,
    classify_broker_error,
)
from src.utils.circuit_breaker import CircuitBreaker
from src.utils.reliability import RateLimiter

from .order_state import (
    LegType,
    OrderLeg,
    OrderSide,
    OrderState,
    TradeFSM,
)

# --- Optional broker SDK (graceful fallback in paper mode) ---
try:
    from kiteconnect import KiteConnect  # type: ignore
    from kiteconnect.exceptions import (  # type: ignore
        InputException,
        NetworkException,
        TokenException,
    )
except ImportError:
    KiteConnect = None  # type: ignore
    NetworkException = TokenException = InputException = Exception  # type: ignore

# --- Local imports (settings is optional for tests) ---
try:
    from src.config import settings
except ImportError:  # pragma: no cover
    settings = None  # type: ignore

log = logging.getLogger(__name__)

__all__ = ["OrderExecutor", "OrderReconciler", "micro_ok", "fetch_quote_with_depth"]

BROKER_EXCEPTIONS = (NetworkException, TokenException, InputException)


# ------------------- small helpers -------------------
def _retry_call(fn, *args, tries: int = 3, base_delay: float = 0.25, **kwargs):
    """Retry transient Kite errors with exponential backoff."""
    delay = base_delay
    for i in range(tries):
        try:
            return fn(*args, **kwargs)
        except (NetworkException, TokenException, InputException) as e:
            if i == tries - 1:
                raise
            log.warning(
                "Transient broker error in %s (try %d/%d): %s",
                getattr(fn, "__name__", "call"),
                i + 1,
                tries,
                e,
            )
            time.sleep(delay + random.uniform(0, 0.05))
            delay *= 2.0


def _round_to_tick(x: float, tick: float) -> float:
    try:
        return round(float(x) / tick) * tick if tick > 0 else float(x)
    except (TypeError, ValueError):
        return float(x)


def _round_to_step(qty: int, step: int) -> int:
    try:
        return int(math.floor(qty / step) * step) if step > 0 else int(qty)
    except (TypeError, ValueError):
        return int(qty)


def _chunks(total: int, chunk: int) -> List[int]:
    """Split a quantity into exchange-compliant child orders."""
    if total <= 0:
        return []
    if chunk <= 0 or chunk >= total:
        return [total]
    out, left = [], total
    while left > 0:
        take = min(chunk, left)
        out.append(take)
        left -= take
    return out


_QUOTE_CACHE: Dict[str, Dict[str, Any]] = {}
_CACHE_LOCK = threading.Lock()
_QUOTE_ERR_RL = RateLimiter(5)
_AUTH_WARNED = False


def fetch_quote_with_depth(
    kite: Optional[KiteConnect], tsym: str, cb: Optional[CircuitBreaker] = None
) -> Dict[str, Any]:
    """Return quote with depth for the given trading symbol."""
    if not kite or not tsym:
        return {
            "ltp": 0.0,
            "bid": 0.0,
            "ask": 0.0,
            "bid_qty": 0,
            "ask_qty": 0,
            "bid5_qty": 0,
            "ask5_qty": 0,
            "oi": None,
            "timestamp": None,
            "source": "none",
        }
    bid = ask = 0.0
    bid_qty = ask_qty = 0
    bid5 = ask5 = 0
    ltp = 0.0
    oi = ts = None
    for attempt in range(2):
        try:
            data = _retry_call(kite.quote, [f"NFO:{tsym}"], tries=2)
            info = data.get(f"NFO:{tsym}", {}) if isinstance(data, dict) else {}
            depth = info.get("depth", {}) if isinstance(info, dict) else {}
            bids = depth.get("buy", []) if isinstance(depth, dict) else []
            asks = depth.get("sell", []) if isinstance(depth, dict) else []
            bid = float(bids[0]["price"]) if bids else 0.0
            ask = float(asks[0]["price"]) if asks else 0.0
            bid_qty = int(bids[0].get("quantity", 0)) if bids else 0
            ask_qty = int(asks[0].get("quantity", 0)) if asks else 0
            bid5 = sum(int(b.get("quantity", 0)) for b in bids[:5]) if bids else 0
            ask5 = sum(int(a.get("quantity", 0)) for a in asks[:5]) if asks else 0
            ltp = float(info.get("last_price") or 0.0)
            oi = info.get("oi")
            ts = info.get("timestamp") or info.get("last_trade_time")
            if bid > 0 and ask > 0 and bid_qty > 0 and ask_qty > 0:
                break
            time.sleep(0.25)
        except BROKER_EXCEPTIONS as e:  # pragma: no cover - broker errors
            kind = classify_broker_error(e)
            if kind == THROTTLE and attempt == 0:
                time.sleep(min(2.0, 0.5))
                continue
            if kind == AUTH and classify_broker_error(e, getattr(e, "status", None)) == AUTH:
                global _AUTH_WARNED
                if cb is not None:
                    cooldown = int(os.getenv("BREAKER_AUTH_COOLDOWN_S", "60"))
                    cb.force_open(cooldown)
                if not _AUTH_WARNED:
                    log.warning("broker auth error: %s", e)
                    _AUTH_WARNED = True
            elif kind == UNKNOWN:
                pass
            elif kind not in (SUBSCRIPTION,):
                if _QUOTE_ERR_RL.allow():
                    log.warning("broker error: %s", e)
            else:
                if _QUOTE_ERR_RL.allow():
                    log.warning("broker subscription error: %s", e)
            break
    if bid <= 0 or ask <= 0 or bid_qty <= 0 or ask_qty <= 0:
        with _CACHE_LOCK:
            cached = _QUOTE_CACHE.get(tsym)
        if cached:
            out = cached.copy()
            out["source"] = "cache"
            out.setdefault("timestamp", datetime.utcnow().isoformat())
            out.setdefault("bid_qty", out.get("bid5_qty", 0))
            out.setdefault("ask_qty", out.get("ask5_qty", 0))
            return out
        ltp = 0.0
        try:
            data = _retry_call(kite.ltp, [f"NFO:{tsym}"], tries=2)
            ltp = float((data or {}).get(f"NFO:{tsym}", {}).get("last_price", 0.0))
        except BROKER_EXCEPTIONS as e:
            log.warning("ltp() failed for %s: %s", tsym, e)
        if ltp <= 0.0 and yf is not None:  # pragma: no cover - best effort
            try:
                tkr = yf.Ticker(f"{tsym}.NS")
                hist = tkr.history(period="1d", interval="1m")
                if not hist.empty:
                    ltp = float(hist["Close"].iloc[-1])
            except (OSError, ValueError) as e:
                log.warning("yfinance fetch failed for %s: %s", tsym, e)
                ltp = 0.0
        if ltp <= 0.0:
            ltp = 1.0
        est_spread_pct = getattr(settings.executor, "default_spread_pct_est", 0.25)
        spr = ltp * (est_spread_pct / 100.0)
        if spr <= 0:
            spr = max(ltp * 0.001, 0.05)
        now_iso = datetime.utcnow().isoformat()
        quote = {
            "ltp": ltp,
            "bid": ltp - spr / 2,
            "ask": ltp + spr / 2,
            "bid_qty": 0,
            "ask_qty": 0,
            "bid5_qty": 0,
            "ask5_qty": 0,
            "oi": None,
            "timestamp": now_iso,
            "source": "ltp_fallback",
        }
        with _CACHE_LOCK:
            _QUOTE_CACHE[tsym] = quote
        return quote
    quote = {
        "ltp": ltp,
        "bid": bid,
        "ask": ask,
        "bid_qty": bid_qty,
        "ask_qty": ask_qty,
        "bid5_qty": bid5,
        "ask5_qty": ask5,
        "oi": oi,
        "timestamp": ts,
        "source": "quote",
    }
    if ltp > 0:
        with _CACHE_LOCK:
            _QUOTE_CACHE[tsym] = quote
    return quote


class KiteOrderConnector:
    """Thin wrapper exposing create/modify/cancel/reconcile operations."""

    def __init__(self, kite: Optional[KiteConnect]) -> None:
        self.kite = kite

    def create_order(self, **payload: Any) -> Any:
        if not self.kite:
            return None
        return _retry_call(self.kite.place_order, **payload)

    def modify(self, order_id: str, **kwargs: Any) -> Any:
        if not self.kite:
            return None
        return _retry_call(self.kite.modify_order, order_id, **kwargs)

    def cancel(self, order_id: str) -> Any:
        if not self.kite:
            return None
        return _retry_call(self.kite.cancel_order, order_id)

    def reconcile(self) -> Any:
        if not self.kite:
            return []
        return _retry_call(self.kite.orders)


def micro_ok(
    quote: Dict[str, Any],
    qty_lots: int,
    lot_size: int,
    max_spread_pct: float,
    depth_mult: int,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Returns (ok: bool, meta: dict). max_spread_pct accepts 0.35 (percent) or 0.0035 (fraction).
    meta: {'spread_pct': float, 'depth_ok': bool, 'bid5': int, 'ask5': int}
    """
    ms = float(max_spread_pct)
    max_spread_frac = ms / 100.0 if ms > 1.0 else ms
    bid = float(getattr(quote, "bid", 0.0) or quote.get("bid", 0.0))
    ask = float(getattr(quote, "ask", 0.0) or quote.get("ask", 0.0))
    ltp = float(getattr(quote, "ltp", 0.0) or quote.get("ltp", 0.0))
    # depth: use top-5 cumulative quantities if available; fall back to top-1
    bid5 = int(quote.get("bid5_qty", quote.get("bid_qty", 0)))
    ask5 = int(quote.get("ask5_qty", quote.get("ask_qty", 0)))
    if bid <= 0 or ask <= 0:
        if ltp <= 0.0:
            return False, None
        spr_pct = getattr(settings.executor, "default_spread_pct_est", 0.25)
        spr = ltp * spr_pct / 100.0
        if bid <= 0:
            bid = ltp - spr / 2
        if ask <= 0:
            ask = ltp + spr / 2
    mid = (bid + ask) / 2.0
    spread_pct = (
        ((ask - bid) / mid * 100.0)
        if (ask > 0 and bid > 0 and mid > 0)
        else getattr(
            settings.executor,
            "default_spread_pct_est",
            0.25,
        )
    )
    qty_contracts = qty_lots * lot_size
    need = depth_mult * qty_contracts
    if (bid5 == 0 or ask5 == 0) and ltp > 0:
        depth_ok = True
    else:
        depth_ok = bid5 >= need and ask5 >= need
    ok = spread_pct <= max_spread_frac * 100.0 and depth_ok
    return ok, {
        "spread_pct": round(spread_pct, 2),
        "depth_ok": depth_ok,
        "bid5": bid5,
        "ask5": ask5,
        "bid": bid,
        "ask": ask,
        "ltp": ltp,
        "source": quote.get("source"),
    }


# ------------------- in-memory order record -------------------
@dataclass
class _OrderRecord:
    order_id: str
    instrument_token: int
    symbol: str
    side: str  # BUY/SELL
    quantity: int
    entry_price: float
    tick_size: float = 0.05

    is_open: bool = True
    filled_qty: int = 0

    # SL GTT
    sl_gtt_id: Optional[int] = None
    sl_price: Optional[float] = None

    # TPs (optional split)
    tp1_order_id: Optional[str] = None
    tp2_order_id: Optional[str] = None
    tp_price: Optional[float] = None
    tp1_done: bool = False

    # broker params
    exchange: str = "NFO"
    product: str = "NRML"
    variety: str = "regular"
    entry_order_type: str = "LIMIT"
    freeze_qty: int = 900
    use_slm_exit: bool = True

    # trailing / partials
    partial_enabled: bool = False
    tp1_ratio: float = 0.5  # 0..1
    breakeven_ticks: int = 2
    trailing_enabled: bool = True
    trailing_mult: float = 1.5
    trail_atr_mult: float = 1.5

    # risk metrics
    r_value: float = 0.0

    # children
    child_order_ids: List[str] = field(default_factory=list)

    @property
    def record_id(self) -> str:
        return self.order_id

    @property
    def remaining_qty(self) -> int:
        return max(0, self.quantity - self.filled_qty)

    def side_sign(self) -> int:
        return 1 if self.side.upper() == "BUY" else -1


# ------------------- executor -------------------
@dataclass
class _QueueItem:
    leg_id: str
    symbol: str
    qty: int
    bid: float
    ask: float
    mid: float
    spr: float
    step_idx: int = 0
    placed_order_id: Optional[str] = None
    placed_at_ms: Optional[int] = None
    acked_at_ms: Optional[int] = None
    retries: int = 0
    plan_ts_iso: Optional[str] = None
    idempotency_key: Optional[str] = None

    def ladder_prices(self) -> List[float]:
        # Never cross; step0 = mid + 0.25*spr ; step1 = mid + 0.40*spr
        return [self.mid + 0.25 * self.spr, self.mid + 0.40 * self.spr]


class OrderExecutor:
    """
    Thin, robust wrapper around Kite order placement/management.
    Matches the runner/telegram expectations:
      - place_order(payload) -> bool/str (record_id/"PAPER"/None)
      - setup_gtt_orders(record_id, sl_price, tp_price)
      - update_trailing_stop(...)
      - sync_and_enforce_oco() -> fills list
      - get_active_orders(), get_positions_kite(), cancel_all_orders()
      - health_check(), shutdown()

    Extra:
      - set_live_broker(kite) / set_kite(kite) to flip live/paper mode safely
      - togglers for trailing/partial/breakeven/tp1_ratio
      - last_error for diag
    """

    def __init__(
        self,
        kite: Optional[KiteConnect] | Any,
        telegram_controller: Any = None,
        on_trade_closed: Optional[Callable[[float], None]] = None,
        journal: Optional[Journal] = None,
    ) -> None:
        """Create a new executor.

        The first argument can either be a ``KiteConnect`` instance or a plain
        settings object containing attributes like ``ACK_TIMEOUT_MS`` used to
        tweak router behaviour.  This keeps the constructor lightweight for unit
        tests which often pass a dummy object instead of a real broker client.
        """

        cfg_local = None
        if kite is not None and not (
            hasattr(kite, "place_order") or hasattr(kite, "orders")
        ):
            cfg_local = kite
            kite = None

        self._lock = threading.Lock()
        self.kite = kite
        self._live = kite is not None
        self.telegram = telegram_controller
        self._active: Dict[str, _OrderRecord] = {}
        self.last_error: Optional[str] = None

        # execution config (default-safe if settings missing)
        ex = getattr(settings, "executor", object()) if settings else object()
        ins = getattr(settings, "instruments", object()) if settings else object()

        self.exchange = getattr(ex, "exchange", "NFO")
        self.product = getattr(ex, "order_product", "NRML")
        self.variety = getattr(ex, "order_variety", "regular")
        self.entry_order_type = getattr(ex, "entry_order_type", "LIMIT")
        self.tick_size = float(getattr(ex, "tick_size", 0.05))
        self.freeze_qty = int(getattr(ex, "exchange_freeze_qty", 900))
        self.lot_size = int(getattr(ins, "nifty_lot_size", 75))  # NIFTY lot

        # exits / risk
        self.partial_enable = bool(getattr(ex, "partial_tp_enable", False))
        self.tp1_ratio = float(getattr(ex, "tp1_qty_ratio", 0.5))
        self.breakeven_ticks = int(getattr(ex, "breakeven_ticks", 2))
        self.enable_trailing = bool(getattr(ex, "enable_trailing", True))
        self.trailing_mult = float(getattr(ex, "trailing_atr_multiplier", 1.5))
        self.use_slm_exit = bool(getattr(ex, "use_slm_exit", True))

        # microstructure constraints
        self.max_spread_pct = float(getattr(ex, "max_spread_pct", 0.0035))
        self.depth_multiplier = float(getattr(ex, "depth_multiplier", 5.0))
        self.micro_retry_limit = int(getattr(ex, "micro_retry_limit", 3))

        # track last notification to throttle duplicates
        self._last_notification: Tuple[str, float] = ("", 0.0)
        self.logger = log

        # FSM + router queueing
        cfg = cfg_local or settings or object()
        self.router_ack_timeout_ms = int(
            getattr(cfg, "ACK_TIMEOUT_MS", getattr(settings, "ACK_TIMEOUT_MS", 1500))
        )
        self.router_fill_timeout_ms = int(
            getattr(cfg, "FILL_TIMEOUT_MS", getattr(settings, "FILL_TIMEOUT_MS", 10000))
        )
        self.router_backoff_ms = int(
            getattr(cfg, "RETRY_BACKOFF_MS", getattr(settings, "RETRY_BACKOFF_MS", 200))
        )
        self.router_max_place_retries = int(
            getattr(cfg, "MAX_PLACE_RETRIES", getattr(settings, "MAX_PLACE_RETRIES", 2))
        )
        self.router_max_modify_retries = int(
            getattr(
                cfg, "MAX_MODIFY_RETRIES", getattr(settings, "MAX_MODIFY_RETRIES", 2)
            )
        )
        self.router_max_inflight_per_symbol = 1
        self.router_plan_stale_sec = 20

        # Back-compat attributes for existing callbacks
        self.max_place_retries = self.router_max_place_retries
        self.max_modify_retries = self.router_max_modify_retries
        self.retry_backoff_ms = self.router_backoff_ms

        self.cb_orders = CircuitBreaker("orders")
        self.cb_modify = CircuitBreaker("modify")
        self._fsms: Dict[str, TradeFSM] = {}
        self._idemp_map: Dict[str, str] = {}
        self._queues: Dict[str, Deque["_QueueItem"]] = {}
        self._inflight_symbols: Dict[str, int] = {}
        self._ack_lat_ms: List[int] = []
        self._order_ts: Deque[float] = deque()
        self.max_orders_per_min = int(
            getattr(
                cfg, "MAX_ORDERS_PER_MIN", getattr(settings, "MAX_ORDERS_PER_MIN", 20)
            )
        )
        self.on_trade_closed = on_trade_closed
        self._closed_trades: Set[str] = set()
        self.journal = journal

    # ----------- router helpers ----------
    def _now_ms(self) -> int:
        return int(time.monotonic() * 1000)

    def _round_to_tick(self, px: float, tick: float = 0.05) -> float:
        return round(math.floor(px / tick + 1e-9) * tick, 2)

    @staticmethod
    def _mid_spread(bid: float, ask: float) -> Tuple[Optional[float], Optional[float]]:
        if not bid or not ask or bid <= 0 or ask <= 0 or ask <= bid:
            return None, None
        mid = (bid + ask) / 2.0
        spr = ask - bid
        return mid, spr

    def _record_ack_latency(self, ms: int) -> None:
        self._ack_lat_ms.append(ms)
        if len(self._ack_lat_ms) > 500:
            self._ack_lat_ms = self._ack_lat_ms[-500:]

    @staticmethod
    def _p95(arr: List[int]) -> Optional[int]:
        if not arr:
            return None
        a = sorted(arr)
        k = int(0.95 * (len(a) - 1))
        return a[k]

    # ----------- live/paper control ----------
    def set_live_broker(self, kite: Optional[KiteConnect]) -> None:
        """Hot-swap Kite session (None => paper)."""
        with self._lock:
            self.kite = kite
            self._live = kite is not None

    # Back-compat alias used by some runners
    def set_kite(self, kite: Optional[KiteConnect]) -> None:
        self.set_live_broker(kite)

    # ----------- public info ----------
    def get_active_orders(self) -> List[_OrderRecord]:
        """Return a snapshot of open records; touch broker for fresh state."""
        self.last_error = None
        with self._lock:
            recs = list(self._active.values())
        if not self.kite:
            return recs
        try:
            _retry_call(self.kite.orders, tries=2)
        except BROKER_EXCEPTIONS as e:
            self.last_error = f"orders: {e}"
            log.error("orders() failed: %s", e)
        return recs

    @property
    def open_count(self) -> int:
        with self._lock:
            return sum(1 for r in self._active.values() if r.is_open)

    def get_positions_kite(self) -> Dict[str, Any]:
        self.last_error = None
        if not self.kite:
            return {}
        try:
            data = _retry_call(self.kite.positions, tries=2) or {}
            return {p.get("tradingsymbol"): p for p in data.get("day", [])}
        except BROKER_EXCEPTIONS as e:
            self.last_error = f"positions: {e}"
            log.error("positions() failed: %s", e)
            return {}

    # ----------- quotes diagnostics ----------
    def _fetch_quote(self, token: int) -> Dict[str, Any]:
        if not self.kite:
            return {
                "ltp": 0.0,
                "bid": 0.0,
                "ask": 0.0,
                "bid_qty": 0,
                "ask_qty": 0,
                "bid_qty_top5": 0,
                "ask_qty_top5": 0,
                "oi": 0,
            }
        try:
            data = _retry_call(self.kite.quote, [token], tries=2)
            info = data.get(str(token), {}) if isinstance(data, dict) else {}
            depth = info.get("depth", {}) if isinstance(info, dict) else {}
            bids = depth.get("buy", []) if isinstance(depth, dict) else []
            asks = depth.get("sell", []) if isinstance(depth, dict) else []
            bid = float(bids[0]["price"]) if bids else 0.0
            ask = float(asks[0]["price"]) if asks else 0.0
            bid_qty = int(bids[0]["quantity"]) if bids else 0
            ask_qty = int(asks[0]["quantity"]) if asks else 0
            bid5 = sum(int(b.get("quantity", 0)) for b in bids[:5]) if bids else 0
            ask5 = sum(int(a.get("quantity", 0)) for a in asks[:5]) if asks else 0
            ltp = float(info.get("last_price") or 0.0)
            oi = info.get("oi")
            return {
                "ltp": ltp,
                "bid": bid,
                "ask": ask,
                "bid_qty": bid_qty,
                "ask_qty": ask_qty,
                "bid_qty_top5": bid5,
                "ask_qty_top5": ask5,
                "oi": oi,
            }
        except BROKER_EXCEPTIONS as e:
            self.last_error = f"quote: {e}"
            return {
                "ltp": 0.0,
                "bid": 0.0,
                "ask": 0.0,
                "bid_qty": 0,
                "ask_qty": 0,
                "bid_qty_top5": 0,
                "ask_qty_top5": 0,
                "oi": 0,
            }

    def quote_diagnostics(
        self,
        opt: str = "both",
        qty_lots: int = 1,
        *,
        runner: Any | None = None,
    ) -> str:
        """Return quote and depth info for current ATM options."""

        from src.strategies.runner import StrategyRunner
        from src.strategies.scalping_strategy import _token_to_symbol_and_lot

        runner = runner or StrategyRunner.get_singleton()
        ds = getattr(runner, "data_source", None) if runner else None
        tokens = getattr(ds, "atm_tokens", (None, None)) if ds else (None, None)
        strike = getattr(ds, "current_atm_strike", None) if ds else None
        if not tokens or None in tokens or strike is None:
            return "ATM not ready"

        ce_token, pe_token = tokens
        token_map = {"ce": ce_token, "pe": pe_token}
        types = [opt.lower()] if opt.lower() in ("ce", "pe") else ["ce", "pe"]
        lines: List[str] = [f"ATM strike {int(strike)}"]
        for t in types:
            token = token_map.get(t)
            if not token:
                lines.append(f"{t.upper()}: token_missing")
                continue
            info = _token_to_symbol_and_lot(self.kite, token)
            if not info:
                lines.append(f"{t.upper()}: tsym_missing")
                continue
            tsym, lot = info
            q = fetch_quote_with_depth(self.kite, tsym, self.cb_orders)
            bid = float(q.get("bid") or 0.0)
            ask = float(q.get("ask") or 0.0)
            spread_pct = (
                ((ask - bid) / ((ask + bid) / 2) * 100.0)
                if bid > 0 and ask > 0
                else None
            )
            bid5 = int(q.get("bid5_qty") or 0)
            ask5 = int(q.get("ask5_qty") or 0)
            depth_ok = min(bid5, ask5) >= int(self.depth_multiplier) * qty_lots * lot
            ok = (
                spread_pct is not None
                and spread_pct <= self.max_spread_pct * 100
                and depth_ok
            )
            lines.append(
                f"{t.upper()} tsym={tsym} src={q.get('source')} ltp={q.get('ltp')} bid={bid} ask={ask} "
                f"spread%={None if spread_pct is None else round(spread_pct,2)} bid5={bid5} ask5={ask5} "
                f"oi={q.get('oi')} micro={'OK' if ok else 'FAIL'}"
            )
        return "\n".join(lines)

    def selftest(self, opt: str = "ce") -> str:
        from src.utils.strike_selector import (
            _fetch_instruments_nfo,
            _get_spot_ltp,
            resolve_weekly_atm,
        )

        inst_dump = _fetch_instruments_nfo(self.kite) or []
        spot = _get_spot_ltp(
            self.kite,
            getattr(getattr(settings, "instruments", object()), "spot_symbol", ""),
        )
        atm = resolve_weekly_atm(spot or 0.0, inst_dump)
        if not atm:
            return "instrument resolution failed"
        info = atm.get(opt.lower())
        if not info:
            return "token missing"
        tsym, lot = info
        q = fetch_quote_with_depth(self.kite, tsym, self.cb_orders)
        bid = float(q.get("bid") or 0.0)
        ask = float(q.get("ask") or 0.0)
        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0.0
        spread = ask - bid
        ok, meta = micro_ok(
            {
                "bid": bid,
                "ask": ask,
                "bid5_qty": q.get("bid5_qty", 0),
                "ask5_qty": q.get("ask5_qty", 0),
            },
            qty_lots=1,
            lot_size=lot,
            max_spread_pct=self.max_spread_pct,
            depth_mult=int(self.depth_multiplier),
        )
        steps = []
        if mid and spread:
            steps = [round(mid + 0.25 * spread, 2), round(mid + 0.40 * spread, 2)]
        result = (
            "WOULD_PLACE"
            if ok
            else f"WOULD_BLOCK(spread%={meta.get('spread_pct')} depth_ok={meta.get('depth_ok')})"
        )
        return f"mid={mid:.2f} spread%={meta.get('spread_pct')} depth_ok={meta.get('depth_ok')} steps={steps} {result}"

    # ----------- entry API (runner calls place_order with payload) ----------
    def place_order(self, payload: Dict[str, Any]) -> Optional[str]:
        """
        payload = {
          action: 'BUY'|'SELL',
          quantity: int (in units, not lots),
          entry_price: float,
          stop_loss: float,
          take_profit: float,
          option_type: 'CE'|'PE',
          strike: float,
          (symbol|instrument_token) optional if you resolve elsewhere
        }
        """
        self.last_error = None

        action = str(payload.get("action", "")).upper()
        qty = int(payload.get("quantity", 0))
        price = float(payload.get("entry_price", 0))
        bid = payload.get("bid")
        ask = payload.get("ask")
        depth = payload.get("depth")
        refresh_cb = payload.get("refresh_market")
        symbol = payload.get("symbol") or self._infer_symbol(payload)
        token = int(payload.get("instrument_token") or 0)

        if action not in ("BUY", "SELL") or qty <= 0 or price <= 0 or not symbol:
            self.last_error = "invalid_payload"
            log.warning("Invalid order payload: %s", payload)
            return None

        # round down to lot step
        qty = _round_to_step(qty, self.lot_size)
        if qty <= 0:
            self.last_error = "qty_rounded_to_zero"
            return None

        if self.kite:
            try:
                qd = fetch_quote_with_depth(self.kite, symbol, self.cb_orders)
                bid5 = int(qd.get("bid5_qty") or qd.get("bid_qty") or 0)
                ask5 = int(qd.get("ask5_qty") or qd.get("ask_qty") or 0)
                avail = int(min(bid5, ask5) * 0.8)
                qty = min(qty, _round_to_step(avail, self.lot_size))
            except BROKER_EXCEPTIONS as e:
                log.warning("depth fetch failed: %s", e)

        # prevent duplicate symbol entries (case-insensitive)
        norm_symbol = str(symbol).upper()
        with self._lock:
            if any(
                rec.symbol.upper() == norm_symbol and rec.is_open
                for rec in self._active.values()
            ):
                self.last_error = "duplicate_symbol_open"
                log.warning("Open record exists on %s; skip new entry.", norm_symbol)
                return None

        # microstructure gates + mid execution
        if bid is not None and ask is not None:
            tries = 0
            while tries < self.micro_retry_limit:
                mid = (float(bid) + float(ask)) / 2.0
                spread = float(ask) - float(bid)
                spread_pct = spread / mid * 100.0 if mid else float("inf")
                max_sp = self.max_spread_pct
                if max_sp < 1:
                    max_sp *= 100.0
                depth_ok = True
                if depth is not None:
                    try:
                        if isinstance(depth, (tuple, list)):
                            depth_val = min(float(depth[0]), float(depth[1]))
                        else:
                            depth_val = float(depth)
                        depth_ok = depth_val >= self.depth_multiplier * qty
                    except (TypeError, ValueError):
                        depth_ok = True
                if spread_pct <= max_sp and depth_ok:
                    price = min(float(ask), mid + 0.15 * spread)
                    price = _round_to_tick(price, self.tick_size)
                    break
                tries += 1
                if callable(refresh_cb):
                    bid, ask, depth = refresh_cb()
                time.sleep(0.2 + 0.1 * tries)
                if tries >= self.micro_retry_limit:
                    self.last_error = "microstructure_block"
                    return None

        if not self.kite:
            # PAPER MODE: create a synthetic record so the rest of the flow works
            rid = f"PAPER-{int(time.time() * 1000)}"
            rec = _OrderRecord(
                order_id=rid,
                instrument_token=token,
                symbol=symbol,
                side=action,
                quantity=qty,
                entry_price=float(price),
                tick_size=self.tick_size,
                exchange=self.exchange,
                product=self.product,
                variety=self.variety,
                entry_order_type=self.entry_order_type,
                freeze_qty=self.freeze_qty,
                use_slm_exit=self.use_slm_exit,
                partial_enabled=self.partial_enable,
                tp1_ratio=self.tp1_ratio,
                breakeven_ticks=self.breakeven_ticks,
                trailing_enabled=self.enable_trailing,
                trailing_mult=self.trailing_mult,
                trail_atr_mult=float(payload.get("trail_atr_mult", self.trailing_mult)),
                r_value=abs(float(price) - float(payload.get("stop_loss", 0.0))),
                child_order_ids=[],
            )
            with self._lock:
                self._active[rec.record_id] = rec
            # Notify (best-effort)
            self._notify(f"🧪 PAPER entry: {symbol} {action} qty={qty} @ {price}")
            log.info(
                "Paper entry recorded: %s %s qty=%d @%s", symbol, action, qty, price
            )
            return rec.record_id

        # LIVE MODE
        parts = _chunks(qty, self.freeze_qty if self.exchange.upper() == "NFO" else qty)
        child_ids: List[str] = []
        record_id: Optional[str] = None

        for q in parts:
            params = dict(
                variety=self.variety,
                exchange=self.exchange,
                tradingsymbol=symbol,
                transaction_type=action,
                quantity=int(q),
                product=self.product,
                order_type=self.entry_order_type.upper(),
            )
            if self.entry_order_type.upper() == "LIMIT":
                params["price"] = _round_to_tick(price, self.tick_size)

            res = self._place_with_cb(params)
            if not res.get("ok"):
                self.last_error = f"place_order: {res.get('reason')}"
                log.error("place_order failed: %s", res.get("reason"))
                continue
            oid = res.get("order_id")
            if not oid:
                continue
            child_ids.append(oid)
            record_id = record_id or oid
            log.info(
                "Entry child placed: %s %s qty=%d price=%s -> %s",
                symbol,
                action,
                q,
                params.get("price"),
                oid,
            )

        if not record_id:
            return None

        rec = _OrderRecord(
            order_id=record_id,
            instrument_token=token,
            symbol=symbol,
            side=action,
            quantity=qty,
            entry_price=float(price),
            tick_size=self.tick_size,
            exchange=self.exchange,
            product=self.product,
            variety=self.variety,
            entry_order_type=self.entry_order_type,
            freeze_qty=self.freeze_qty,
            use_slm_exit=self.use_slm_exit,
            partial_enabled=self.partial_enable,
            tp1_ratio=self.tp1_ratio,
            breakeven_ticks=self.breakeven_ticks,
            trailing_enabled=self.enable_trailing,
            trailing_mult=self.trailing_mult,
            trail_atr_mult=float(payload.get("trail_atr_mult", self.trailing_mult)),
            r_value=abs(float(price) - float(payload.get("stop_loss", 0.0))),
            child_order_ids=child_ids,
        )
        with self._lock:
            self._active[rec.record_id] = rec

        self._notify(
            f"✅ LIVE entry: {symbol} {action} qty={qty} @ {price} (rid={rec.record_id})"
        )
        return rec.record_id

    # ----------- SL/TP orchestration ----------
    def setup_gtt_orders(
        self, record_id: str, sl_price: float, tp_price: float
    ) -> None:
        """Place/refresh SL-GTT and TP limit orders for an active record."""
        with self._lock:
            rec = self._active.get(record_id)
        if not rec or not rec.is_open:
            return

        qty = rec.remaining_qty or rec.quantity
        if qty <= 0:
            return

        exit_side = "SELL" if rec.side == "BUY" else "BUY"
        tp_price = _round_to_tick(tp_price, rec.tick_size)
        sl_price = _round_to_tick(sl_price, rec.tick_size)

        # cancel old TPs if any
        for old in (rec.tp1_order_id, rec.tp2_order_id):
            if old:
                res_cancel = self._cancel_with_cb(old, variety=rec.variety)
                if not res_cancel.get("ok"):
                    log.debug(
                        "Failed to cancel TP order %s: %s",
                        old,
                        res_cancel.get("reason"),
                    )
        rec.tp1_order_id = rec.tp2_order_id = None

        # partial split
        if rec.partial_enabled:
            q_tp1 = _round_to_step(
                int(round(qty * max(0, min(1, rec.tp1_ratio)))), self.lot_size
            )
            q_tp1 = min(max(q_tp1, self.lot_size), qty)
            q_tp2 = qty - q_tp1
        else:
            q_tp1, q_tp2 = 0, qty

        if q_tp1 > 0:
            res1 = self._place_with_cb(
                {
                    "variety": rec.variety,
                    "exchange": rec.exchange,
                    "tradingsymbol": rec.symbol,
                    "transaction_type": exit_side,
                    "quantity": q_tp1,
                    "product": rec.product,
                    "order_type": "LIMIT",
                    "price": tp_price,
                }
            )
            if res1.get("ok"):
                rec.tp1_order_id = res1.get("order_id")
            else:
                self.last_error = f"tp1: {res1.get('reason')}"
                log.error("TP1 failed: %s", res1.get("reason"))
        if q_tp2 > 0:
            res2 = self._place_with_cb(
                {
                    "variety": rec.variety,
                    "exchange": rec.exchange,
                    "tradingsymbol": rec.symbol,
                    "transaction_type": exit_side,
                    "quantity": q_tp2,
                    "product": rec.product,
                    "order_type": "LIMIT",
                    "price": tp_price,
                }
            )
            if res2.get("ok"):
                rec.tp2_order_id = res2.get("order_id")
            else:
                self.last_error = f"tp2: {res2.get('reason')}"
                log.error("TP2 failed: %s", res2.get("reason"))
        rec.tp_price = tp_price

        self._refresh_sl_gtt(rec, sl_price=sl_price, qty=qty)

    def _refresh_sl_gtt(self, rec: _OrderRecord, *, sl_price: float, qty: int) -> None:
        """(Re)create SL GTT as single-leg order."""
        if not self.kite or qty <= 0:
            return
        # cancel existing GTT
        if rec.sl_gtt_id:
            try:
                _retry_call(self.kite.cancel_gtt, rec.sl_gtt_id, tries=2)
            except BROKER_EXCEPTIONS as e:
                log.debug(
                    "Failed to cancel existing SL GTT %s: %s",
                    rec.sl_gtt_id,
                    e,
                    exc_info=True,
                )
            rec.sl_gtt_id = None

        exit_side = "SELL" if rec.side == "BUY" else "BUY"
        sl_leg = {
            "exchange": rec.exchange,
            "tradingsymbol": rec.symbol,
            "transaction_type": exit_side,
            "quantity": int(qty),
            "order_type": "SL-M" if rec.use_slm_exit else "SL",
            "product": rec.product,
            "price": None if rec.use_slm_exit else sl_price,
            "trigger_price": sl_price,
        }
        try:
            # Note: Real Kite API expects trigger_values list etc.
            # We keep current signature as per your existing integration.
            res = _retry_call(
                self.kite.place_gtt,
                trigger_type="single",
                tradingsymbol=rec.symbol,
                exchange=rec.exchange,
                trigger_price=sl_price,
                last_price=None,
                orders=[sl_leg],
                tries=2,
            )
            gid = res["id"] if isinstance(res, dict) else int(res)
            rec.sl_gtt_id, rec.sl_price = gid, sl_price
        except BROKER_EXCEPTIONS as e:
            self.last_error = f"sl_gtt: {e}"
            log.error("SL GTT failed: %s", e)

    # ----------- trailing stop maintenance ----------
    def update_trailing_stop(
        self,
        record_id: str,
        *,
        current_price: float,
        atr: float,
        atr_multiplier: Optional[float] = None,
    ) -> None:
        with self._lock:
            rec = self._active.get(record_id)
        if not rec or not rec.is_open:
            return
        if not rec.trailing_enabled or atr <= 0 or current_price <= 0:
            return

        m = float(atr_multiplier or rec.trailing_mult)
        if rec.side == "BUY":
            proposed = _round_to_tick(current_price - m * atr, rec.tick_size)
            if rec.tp1_done and rec.sl_price is not None:
                proposed = max(proposed, rec.sl_price)
        else:
            proposed = _round_to_tick(current_price + m * atr, rec.tick_size)
            if rec.tp1_done and rec.sl_price is not None:
                proposed = min(proposed, rec.sl_price)

        rec.sl_price = proposed
        self._refresh_sl_gtt(
            rec, sl_price=proposed, qty=rec.remaining_qty or rec.quantity
        )

    def handle_tp1_fill(self, record_id: str) -> None:
        """Internal helper used to apply TP1 partial exit effects."""
        with self._lock:
            rec = self._active.get(record_id)
        if not rec or rec.tp1_done:
            return
        tp_qty = int(round(rec.quantity * rec.tp1_ratio)) if rec.partial_enabled else 0
        rec.tp1_done = True
        if tp_qty > 0:
            rec.quantity -= tp_qty
            if rec.quantity < 0:
                rec.quantity = 0
        new_sl = rec.entry_price + rec.side_sign() * 0.1 * rec.r_value
        rec.sl_price = _round_to_tick(new_sl, rec.tick_size)
        rec.trailing_mult = rec.trail_atr_mult or rec.trailing_mult
        self._refresh_sl_gtt(
            rec, sl_price=rec.sl_price, qty=rec.remaining_qty or rec.quantity
        )

    # ----------- polling / OCO enforcement ----------
    def sync_and_enforce_oco(self) -> List[Tuple[str, float]]:
        """Refresh order state and enforce OCO-style exits.

        If the broker lacks native one-cancels-other support, this method
        acts as a watcher: it polls open orders and cancels siblings when one
        leg fills. It returns a list of newly filled legs as ``[(record_id, px)]``
        tuples.
        """
        if not self.kite:
            return []
        fills: List[Tuple[str, float]] = []
        try:
            omap = {
                o.get("order_id"): o
                for o in _retry_call(self.kite.orders, tries=2) or []
            }
        except BROKER_EXCEPTIONS as e:
            self.last_error = f"orders: {e}"
            log.error("orders() failed: %s", e)
            omap = {}
        try:
            gmap = {}
            for g in _retry_call(self.kite.gtts, tries=2) or []:
                gid = g.get("id") or g.get("gtt_id")
                if gid:
                    gmap[int(gid)] = g
        except BROKER_EXCEPTIONS as e:
            self.last_error = f"gtts: {e}"
            log.error("gtts() failed: %s", e)
            gmap = {}

        with self._lock:
            recs = list(self._active.values())
        for rec in recs:
            if not rec.is_open:
                continue

            # entry fills (aggregate child orders)
            total_filled = sum(
                int(omap.get(oid, {}).get("filled_quantity", 0))
                for oid in rec.child_order_ids
                if (omap.get(oid, {}).get("status", "").upper() == "COMPLETE")
            )
            if total_filled > rec.filled_qty:
                rec.filled_qty = total_filled

            # TP fills / breakeven bump after TP1
            for tid in ("tp1_order_id", "tp2_order_id"):
                oid = getattr(rec, tid)
                if not oid:
                    continue
                o = omap.get(oid, {})
                if o.get("status", "").upper() == "COMPLETE":
                    if (
                        tid == "tp1_order_id"
                        and rec.partial_enabled
                        and not rec.tp1_done
                    ):
                        self.handle_tp1_fill(rec.record_id)

            # SL GTT state
            if rec.sl_gtt_id:
                g = gmap.get(rec.sl_gtt_id) or {}
                s = (g.get("status") or g.get("state") or "").lower()
                if s in ("triggered", "cancelled", "deleted", "expired", "completed"):
                    rec.is_open = False
                    fills.append((rec.record_id, float(rec.sl_price or 0.0)))
                    continue

            # All TPs done?
            tp1_done = (not rec.tp1_order_id) or (
                omap.get(rec.tp1_order_id, {}).get("status", "").upper() == "COMPLETE"
            )
            tp2_done = (
                rec.tp2_order_id
                and omap.get(rec.tp2_order_id, {}).get("status", "").upper()
                == "COMPLETE"
            )
            if tp1_done and tp2_done:
                rec.is_open = False
                fills.append((rec.record_id, float(rec.tp_price or 0.0)))
                if rec.sl_gtt_id:
                    try:
                        _retry_call(self.kite.cancel_gtt, rec.sl_gtt_id, tries=2)
                    except BROKER_EXCEPTIONS as e:
                        log.debug(
                            "Failed to cancel SL GTT %s during sweep: %s",
                            rec.sl_gtt_id,
                            e,
                            exc_info=True,
                        )

        if fills:
            with self._lock:
                self._active = {k: v for k, v in self._active.items() if v.is_open}
            # Notify (best-effort)
            try:
                if self.telegram and hasattr(self.telegram, "notify_fills"):
                    self.telegram.notify_fills(fills)
                else:
                    self._notify(
                        "📬 Fills: " + ", ".join([f"{rid}@{px}" for rid, px in fills])
                    )
            except Exception as e:
                log.debug("Failed to send fills notification: %s", e, exc_info=True)

        return fills

    # ----------- admin ----------
    def cancel_all_orders(self) -> None:
        """Best-effort cancel of all live child orders + SL GTTs for our records."""
        if not self.kite:
            with self._lock:
                self._active.clear()
            log.info("Paper mode: cleared in-memory orders.")
            return

        with self._lock:
            recs = list(self._active.values())

        for rec in recs:
            for oid in (rec.tp1_order_id, rec.tp2_order_id, *rec.child_order_ids):
                if not oid:
                    continue
                try:
                    _retry_call(
                        self.kite.cancel_order,
                        variety=rec.variety,
                        order_id=oid,
                        tries=2,
                    )
                except BROKER_EXCEPTIONS as e:
                    log.debug("Failed to cancel order %s: %s", oid, e, exc_info=True)
            if rec.sl_gtt_id:
                try:
                    _retry_call(self.kite.cancel_gtt, rec.sl_gtt_id, tries=2)
                except BROKER_EXCEPTIONS as e:
                    log.debug(
                        "Failed to cancel SL GTT %s: %s",
                        rec.sl_gtt_id,
                        e,
                        exc_info=True,
                    )
            rec.is_open = False

        with self._lock:
            self._active.clear()
        log.info("All open executor orders cancelled.")

    def health_check(self) -> None:
        """No-op placeholder (runner calls this periodically)."""
        try:
            if not self.kite:
                return
            # optionally touch lightweight endpoints
            # _retry_call(self.kite.profile, tries=1)
        except BROKER_EXCEPTIONS as e:
            log.warning("OrderExecutor health warning: %s", e)

    def shutdown(self) -> None:
        """Graceful teardown: cancel orders and journal shutdown."""
        try:
            self.cancel_all_orders()
        except Exception as e:
            log.debug("Error during executor shutdown: %s", e, exc_info=True)
        if self.journal:
            try:
                self.journal.append_event(
                    ts=datetime.utcnow().isoformat(),
                    trade_id="SYSTEM",
                    leg_id="SHUTDOWN",
                    etype="SHUTDOWN",
                )
            except Exception as e:
                log.debug("Error journalling shutdown: %s", e, exc_info=True)

    # ----------- telegram-wired mutators ----------
    def set_trailing_enabled(self, on: bool) -> None:
        self.enable_trailing = bool(on)
        with self._lock:
            for r in self._active.values():
                r.trailing_enabled = self.enable_trailing

    def set_trailing_mult(self, x: float) -> None:
        if x <= 0:
            return
        self.trailing_mult = float(x)
        with self._lock:
            for r in self._active.values():
                r.trailing_mult = self.trailing_mult

    def set_partial_enabled(self, on: bool) -> None:
        self.partial_enable = bool(on)
        with self._lock:
            for r in self._active.values():
                r.partial_enabled = self.partial_enable

    def set_tp1_ratio(self, pct: float) -> None:
        # accepts 0..100 or 0..1
        v = float(pct)
        v = v / 100.0 if v > 1.0 else v
        v = max(0.0, min(1.0, v))
        self.tp1_ratio = v
        with self._lock:
            for r in self._active.values():
                r.tp1_ratio = self.tp1_ratio

    def set_breakeven_ticks(self, ticks: int) -> None:
        t = int(max(0, ticks))
        self.breakeven_ticks = t
        with self._lock:
            for r in self._active.values():
                r.breakeven_ticks = self.breakeven_ticks

    # ----------- FSM/queue helpers ----------
    def create_trade_fsm(self, plan: Dict[str, Any]) -> TradeFSM:
        """Build a TradeFSM from a strategy plan."""
        trade_id = str(plan.get("trade_id") or int(time.time() * 1000))
        side = (
            OrderSide.BUY
            if str(plan.get("action", "BUY")).upper() == "BUY"
            else OrderSide.SELL
        )
        symbol = str(plan.get("symbol") or plan.get("tradingsymbol") or "")
        qty = int(plan.get("qty") or plan.get("quantity") or 0)
        price = plan.get("limit_price") or plan.get("entry")
        leg_id = f"{trade_id}:ENTRY"
        idemp = f"{trade_id}:ENTRY:{int(time.time() * 1000)}"
        leg = OrderLeg(
            trade_id=trade_id,
            leg_id=leg_id,
            leg_type=LegType.ENTRY,
            side=side,
            symbol=symbol,
            qty=qty,
            limit_price=price,
            state=OrderState.NEW,
            idempotency_key=idemp,
            created_at=datetime.utcnow(),
            expires_at=None,
            reason=None,
        )
        fsm = TradeFSM(trade_id=trade_id, legs={leg_id: leg})
        # store basics for PnL estimation later
        try:
            fsm.entry_price = float(plan.get("entry") or 0.0)  # type: ignore[attr-defined]
            fsm.stop_loss = float(plan.get("sl") or 0.0)  # type: ignore[attr-defined]
        except (TypeError, ValueError):
            fsm.entry_price = float(price or 0.0)  # type: ignore[attr-defined]
            fsm.stop_loss = 0.0  # type: ignore[attr-defined]
        self._fsms[trade_id] = fsm
        return fsm

    def place_trade(self, fsm: TradeFSM) -> None:
        for leg in fsm.open_legs():
            self.enqueue_leg(leg)

    def enqueue_leg(self, leg: OrderLeg) -> None:
        if self.journal:
            try:
                self.journal.append_event(
                    ts=datetime.utcnow().isoformat(),
                    trade_id=leg.trade_id,
                    leg_id=leg.leg_id,
                    etype="NEW",
                    payload={
                        "side": leg.side.name,
                        "symbol": leg.symbol,
                        "qty": leg.qty,
                        "limit_price": leg.limit_price,
                    },
                )
            except Exception as e:
                log.debug("Journal append failed: %s", e)
        if leg.idempotency_key in self._idemp_map:
            return
        self._idemp_map[leg.idempotency_key] = leg.leg_id
        fsm = self._fsms.setdefault(leg.trade_id, TradeFSM(leg.trade_id, legs={}))
        fsm.legs[leg.leg_id] = leg
        quote = {
            "bid": float(leg.limit_price or 100.0),
            "ask": float(leg.limit_price or 100.0) + 0.4,
        }
        self._enqueue_router(leg, quote, None)

    def _enqueue_router(
        self, leg: OrderLeg, quote: Dict[str, float], plan_ts_iso: Optional[str]
    ) -> bool:
        bid = quote.get("bid", 0.0)
        ask = quote.get("ask", 0.0)
        mid, spr = self._mid_spread(bid, ask)
        if mid is None:
            leg.on_reject("no_book")
            return False
        qi = _QueueItem(
            leg_id=leg.leg_id,
            symbol=leg.symbol,
            qty=leg.qty,
            bid=bid,
            ask=ask,
            mid=mid,
            spr=spr,
            plan_ts_iso=plan_ts_iso,
        )
        q = self._queues.setdefault(leg.symbol, deque())
        q.append(qi)
        self.logger.info(
            f"ROUTER enqueue {leg.symbol} leg={leg.leg_id} mid={mid:.2f} spr={spr:.2f}"
        )
        return True

    def open_trades(self) -> List[TradeFSM]:
        return [fsm for fsm in self._fsms.values() if fsm.status == "OPEN"]

    def get_or_create_fsm(self, trade_id: str) -> TradeFSM:
        """Return existing FSM for ``trade_id`` or create a new one."""
        fsm = self._fsms.get(trade_id)
        if fsm is None:
            fsm = TradeFSM(trade_id=trade_id, legs={})
            self._fsms[trade_id] = fsm
        return fsm

    def attach_leg_from_journal(self, fsm: TradeFSM, leg_dict: Dict[str, Any]) -> None:
        """Attach a leg restored from journal without placing a new order."""
        state_str = str(leg_dict.get("state", "NEW"))
        if state_str == "ACK":
            state = OrderState.PENDING
        else:
            try:
                state = OrderState(state_str)
            except ValueError:
                state = OrderState.NEW
        side = OrderSide[str(leg_dict.get("side", "BUY"))]
        leg_type_str = str(leg_dict.get("leg_id", "")).split(":")[-1]
        try:
            leg_type = LegType[leg_type_str]
        except KeyError:
            leg_type = LegType.ENTRY
        leg = OrderLeg(
            trade_id=leg_dict["trade_id"],
            leg_id=leg_dict["leg_id"],
            leg_type=leg_type,
            side=side,
            symbol=leg_dict.get("symbol", ""),
            qty=int(leg_dict.get("qty", 0)),
            limit_price=leg_dict.get("limit_price"),
            state=state,
            filled_qty=int(leg_dict.get("filled_qty", 0)),
            avg_price=float(leg_dict.get("avg_price", 0.0)),
            broker_order_id=leg_dict.get("broker_order_id"),
            idempotency_key=leg_dict.get("idempotency_key", ""),
            created_at=datetime.utcnow(),
            expires_at=None,
            reason=None,
        )
        fsm.legs[leg.leg_id] = leg
        if leg.idempotency_key:
            self._idemp_map[leg.idempotency_key] = leg.leg_id

    def open_legs_snapshot(self) -> List[Dict[str, Any]]:
        """Return a compact snapshot of open legs for checkpoints."""
        snap: List[Dict[str, Any]] = []
        for fsm in self._fsms.values():
            for leg in fsm.open_legs():
                snap.append(
                    {
                        "trade": leg.trade_id,
                        "leg": leg.leg_id,
                        "sym": leg.symbol,
                        "state": leg.state.name,
                        "qty": leg.qty,
                        "filled": leg.filled_qty,
                        "avg": leg.avg_price,
                    }
                )
        return snap

    def _compute_pnl_R(self, fsm: TradeFSM) -> float:
        """Estimate trade PnL in R units for the given FSM."""

        entry_leg = next(
            (leg for leg in fsm.legs.values() if leg.leg_type is LegType.ENTRY), None
        )
        if not entry_leg:
            return 0.0
        exit_legs = [
            leg for leg in fsm.legs.values() if leg.leg_type is not LegType.ENTRY
        ]
        if not exit_legs:
            return 0.0
        pnl_rupees = 0.0
        for leg in exit_legs:
            if leg.state is not OrderState.FILLED:
                continue
            if entry_leg.side is OrderSide.BUY:
                pnl_rupees += (leg.avg_price - entry_leg.avg_price) * leg.qty
            else:
                pnl_rupees += (entry_leg.avg_price - leg.avg_price) * leg.qty
        risk_rupees = (
            abs(
                getattr(fsm, "entry_price", entry_leg.avg_price)
                - getattr(fsm, "stop_loss", entry_leg.avg_price)
            )
            * entry_leg.qty
        )
        if risk_rupees <= 0:
            return 0.0
        return pnl_rupees / risk_rupees

    def step_queue(self, now: Optional[datetime] = None) -> None:
        now_ms = self._now_ms()
        for sym, q in list(self._queues.items()):
            if (
                self._inflight_symbols.get(sym, 0)
                >= self.router_max_inflight_per_symbol
            ):
                continue
            if not q:
                continue
            qi = q[0]

            targets = qi.ladder_prices()
            tgt = self._round_to_tick(targets[qi.step_idx])

            idemp = f"{qi.leg_id}:{qi.step_idx}:{now_ms}"
            req = self._build_place_request(
                symbol=sym, qty=qi.qty, price=tgt, tag=idemp
            )

            leg_obj: Optional[OrderLeg] = None
            for fsm in self._fsms.values():
                leg_obj = fsm.legs.get(qi.leg_id)
                if leg_obj:
                    break
            if self.journal and leg_obj:
                try:
                    self.journal.append_event(
                        ts=datetime.utcnow().isoformat(),
                        trade_id=leg_obj.trade_id,
                        leg_id=leg_obj.leg_id,
                        etype="IDEMP",
                        idempotency_key=idemp,
                        payload={},
                    )
                except Exception as e:
                    log.debug("Journal append failed: %s", e)

            t0 = self._now_ms()
            res = self._place_with_cb(req)
            if not res.get("ok"):
                delay = self.router_backoff_ms * (1 + qi.retries) + random.randint(
                    0, 25
                )
                time.sleep(delay / 1000.0)
                qi.retries += 1
                if qi.retries > self.router_max_place_retries:
                    self.logger.warning(
                        f"ROUTER drop {sym} leg={qi.leg_id} reason={res.get('reason')}"
                    )
                    q.popleft()
                continue

            oid = res.get("order_id") or f"PAPER-{t0}"
            qi.placed_order_id = oid
            qi.placed_at_ms = t0
            qi.idempotency_key = idemp
            self._idemp_map[idemp] = qi.leg_id
            self._inflight_symbols[sym] = self._inflight_symbols.get(sym, 0) + 1
            self.logger.info(
                f"ROUTER placed {sym} leg={qi.leg_id} step={qi.step_idx} px={tgt:.2f} oid={qi.placed_order_id}"
            )

            # update leg state
            if leg_obj and qi.placed_order_id:
                leg_obj.mark_acked(qi.placed_order_id)
                if self.journal:
                    try:
                        self.journal.append_event(
                            ts=datetime.utcnow().isoformat(),
                            trade_id=leg_obj.trade_id,
                            leg_id=leg_obj.leg_id,
                            etype="ACK",
                            broker_order_id=qi.placed_order_id,
                            idempotency_key=idemp,
                            payload={"price": tgt},
                        )
                    except Exception as e:
                        log.debug("Journal append failed: %s", e)

    def on_order_acked(self, symbol: str, leg_id: str) -> None:
        q = self._queues.get(symbol)
        if not q:
            return
        qi = q[0]
        if qi.leg_id != leg_id:
            return
        ack_ms = self._now_ms()
        if qi.placed_at_ms:
            qi.acked_at_ms = ack_ms
            self._record_ack_latency(ack_ms - qi.placed_at_ms)

    def on_order_timeout_check(self) -> None:
        now_ms = self._now_ms()
        for sym, q in list(self._queues.items()):
            if not q:
                continue
            qi = q[0]

            if (
                qi.placed_at_ms
                and not qi.acked_at_ms
                and (now_ms - qi.placed_at_ms) > self.router_ack_timeout_ms
            ):
                if qi.step_idx == 0:
                    targets = qi.ladder_prices()
                    new_px = self._round_to_tick(targets[1])
                    mod = self._modify_with_cb(qi.placed_order_id, price=new_px)
                    self.logger.info(
                        f"ROUTER modify {sym} leg={qi.leg_id} step=1 px={new_px:.2f} res={mod.get('ok')}"
                    )
                    qi.step_idx = 1
                    qi.placed_at_ms = self._now_ms()
                else:
                    self._cancel_with_cb(qi.placed_order_id)
                    self.logger.info(
                        f"ROUTER cancel {sym} leg={qi.leg_id} reason=ack_timeout"
                    )
                    q.popleft()
                    self._inflight_symbols[sym] = max(
                        0, self._inflight_symbols.get(sym, 0) - 1
                    )
                    continue

            if (
                qi.acked_at_ms
                and (now_ms - qi.acked_at_ms) > self.router_fill_timeout_ms
            ):
                self._cancel_with_cb(qi.placed_order_id)
                self.logger.info(
                    f"ROUTER cancel {sym} leg={qi.leg_id} reason=fill_timeout"
                )
                q.popleft()
                self._inflight_symbols[sym] = max(
                    0, self._inflight_symbols.get(sym, 0) - 1
                )

    def router_health(self) -> dict:
        return {
            "ack_p95_ms": self._p95(self._ack_lat_ms),
            "queues": {k: len(v) for k, v in self._queues.items()},
            "inflight": dict(self._inflight_symbols),
        }

    def _place_leg(self, leg: OrderLeg) -> None:
        order_id: Optional[str] = None
        if self.kite and hasattr(self.kite, "create_order"):
            try:
                resp = self.kite.create_order(leg, idempotency_key=leg.idempotency_key)
                order_id = getattr(resp, "order_id", None) or str(resp)
            except BROKER_EXCEPTIONS as e:  # pragma: no cover - broker errors
                log.debug("place_order failed: %s", e)
        else:
            order_id = f"PAPER-{int(time.time() * 1000)}"
        if not order_id:
            leg.on_reject("place_failed")
            return
        leg.mark_acked(str(order_id))
        self._inflight_symbols[leg.symbol] = (
            self._inflight_symbols.get(leg.symbol, 0) + 1
        )
        if self.journal:
            try:
                self.journal.append_event(
                    ts=datetime.utcnow().isoformat(),
                    trade_id=leg.trade_id,
                    leg_id=leg.leg_id,
                    etype="ACK",
                    broker_order_id=str(order_id),
                    idempotency_key=leg.idempotency_key,
                    payload={"price": leg.limit_price},
                )
            except Exception as e:
                log.debug("Journal append failed: %s", e)

    # ----------- circuit breaker wrapped calls ----------

    @staticmethod
    def _is_transient(reason: str) -> bool:
        """Return True if error reason looks transient."""
        r = reason.lower()
        return any(
            x in r
            for x in [
                "timeout",
                "timed out",
                "502",
                "503",
                "504",
                "network",
                "429",
                "rate",
                "throttle",
            ]
        )

    def _build_place_request(
        self, *, symbol: str, qty: int, price: float, tag: str
    ) -> Dict[str, Any]:
        return {
            "variety": self.variety,
            "exchange": self.exchange,
            "tradingsymbol": symbol,
            "transaction_type": "BUY",
            "quantity": int(qty),
            "product": self.product,
            "order_type": "LIMIT",
            "price": price,
            "tag": tag,
        }

    def _place_with_cb(self, req: Dict[str, Any]) -> Dict[str, Any]:
        """Place order via broker with circuit breaker and retries."""

        def single() -> Dict[str, Any]:
            if not self.cb_orders.allow():
                return {
                    "ok": False,
                    "reason": "api_breaker_open",
                    "cb": self.cb_orders.health(),
                }
            now = time.monotonic()
            while self._order_ts and now - self._order_ts[0] > 60:
                self._order_ts.popleft()
            if len(self._order_ts) >= self.max_orders_per_min:
                return {"ok": False, "reason": "rate_limited"}
            self._order_ts.append(now)
            t0 = time.monotonic()
            try:
                oid = self.kite.place_order(**req) if self.kite else None
                lat = int((time.monotonic() - t0) * 1000)
                self.cb_orders.record_success(lat)
                return {"ok": True, "order_id": oid, "lat_ms": lat}
            except BROKER_EXCEPTIONS as e:  # pragma: no cover - broker errors
                lat = int((time.monotonic() - t0) * 1000)
                self.cb_orders.record_failure(lat, reason=str(e))
                return {"ok": False, "reason": str(e), "lat_ms": lat}

        res = single()
        attempt = 0
        while (
            not res.get("ok")
            and self._is_transient(str(res.get("reason", "")))
            and attempt < self.max_place_retries
        ):
            delay = (self.retry_backoff_ms * (2**attempt)) / 1000.0
            time.sleep(delay + random.uniform(0, 0.025))
            attempt += 1
            res = single()
        return res

    def _modify_with_cb(self, order_id: str, **kwargs: Any) -> Dict[str, Any]:
        """Modify order via broker with breaker and retries."""

        def single() -> Dict[str, Any]:
            if not self.cb_modify.allow():
                return {
                    "ok": False,
                    "reason": "api_breaker_open",
                    "cb": self.cb_modify.health(),
                }
            t0 = time.monotonic()
            try:
                resp = self.kite.modify_order(order_id, **kwargs) if self.kite else None
                lat = int((time.monotonic() - t0) * 1000)
                self.cb_modify.record_success(lat)
                return {"ok": True, "resp": resp, "lat_ms": lat}
            except BROKER_EXCEPTIONS as e:  # pragma: no cover
                lat = int((time.monotonic() - t0) * 1000)
                self.cb_modify.record_failure(lat, reason=str(e))
                return {"ok": False, "reason": str(e), "lat_ms": lat}

        res = single()
        attempt = 0
        while (
            not res.get("ok")
            and self._is_transient(str(res.get("reason", "")))
            and attempt < self.max_modify_retries
        ):
            delay = (self.retry_backoff_ms * (2**attempt)) / 1000.0
            time.sleep(delay + random.uniform(0, 0.025))
            attempt += 1
            res = single()
        return res

    def _cancel_with_cb(self, order_id: str, **kwargs: Any) -> Dict[str, Any]:
        """Cancel order via broker with breaker and retries."""

        def single() -> Dict[str, Any]:
            if not self.cb_modify.allow():
                return {
                    "ok": False,
                    "reason": "api_breaker_open",
                    "cb": self.cb_modify.health(),
                }
            t0 = time.monotonic()
            try:
                resp = self.kite.cancel_order(order_id, **kwargs) if self.kite else None
                lat = int((time.monotonic() - t0) * 1000)
                self.cb_modify.record_success(lat)
                return {"ok": True, "resp": resp, "lat_ms": lat}
            except BROKER_EXCEPTIONS as e:  # pragma: no cover
                lat = int((time.monotonic() - t0) * 1000)
                self.cb_modify.record_failure(lat, reason=str(e))
                return {"ok": False, "reason": str(e), "lat_ms": lat}

        res = single()
        attempt = 0
        while (
            not res.get("ok")
            and self._is_transient(str(res.get("reason", "")))
            and attempt < self.max_modify_retries
        ):
            delay = (self.retry_backoff_ms * (2**attempt)) / 1000.0
            time.sleep(delay + random.uniform(0, 0.025))
            attempt += 1
            res = single()
        return res

    def api_health(self) -> Dict[str, Dict[str, object]]:
        """Return current API breaker health."""
        return {"orders": self.cb_orders.health(), "modify": self.cb_modify.health()}

    # ----------- util ----------
    @staticmethod
    def _infer_symbol(payload: Dict[str, Any]) -> Optional[str]:
        """
        If not provided, compose a reasonable NFO symbol from strike & option_type.
        Example placeholder: NIFTY24500CE  (expiry not encoded here).
        """
        try:
            base = (
                getattr(settings.instruments, "trade_symbol", "NIFTY")
                if settings
                else "NIFTY"
            )
        except AttributeError:
            base = "NIFTY"
        strike = payload.get("strike")
        opt = str(payload.get("option_type", "")).upper()  # CE/PE
        if strike is None or opt not in ("CE", "PE"):
            return None
        try:
            s = int(float(strike))
        except (TypeError, ValueError):
            return None
        return f"{base}{s}{opt}"

    # ----------- internal notify helper ----------
    def _notify(self, text: str) -> None:
        now = time.time()
        last_msg, last_ts = self._last_notification
        if text == last_msg and (now - last_ts) < 300:
            return
        self._last_notification = (text, now)
        try:
            if self.telegram:
                if hasattr(self.telegram, "send_message"):
                    self.telegram.send_message(text)
                elif hasattr(self.telegram, "notify"):
                    self.telegram.notify(text)  # legacy fallback
        except Exception as e:
            log.debug("Failed to send notification: %s", e, exc_info=True)


class OrderReconciler:
    """Poll broker orders and reconcile with local FSM legs."""

    def __init__(
        self, kite: Any, fsm_store: OrderExecutor, logger: logging.Logger
    ) -> None:
        self.kite = kite
        self.store = fsm_store
        self.log = logger

    def step(self, now: datetime) -> int:
        if not self.kite:
            return 0
        try:
            orders = self.kite.orders() or []
        except BROKER_EXCEPTIONS as e:  # pragma: no cover - network errors
            self.log.debug("reconcile orders failed: %s", e)
            return 0

        updated = 0
        for o in orders:
            tag = str(o.get("tag") or "")
            oid = o.get("order_id")
            leg: Optional[OrderLeg] = None
            # match by order id
            if oid:
                for fsm in self.store._fsms.values():
                    for leg_obj in fsm.legs.values():
                        if leg_obj.broker_order_id == oid:
                            leg = leg_obj
                            break
                    if leg:
                        break
            # match by idempotency tag
            if leg is None and tag:
                leg_id = self.store._idemp_map.get(tag)
                if leg_id:
                    for fsm in self.store._fsms.values():
                        leg = fsm.legs.get(leg_id)
                        if leg:
                            break
            if leg is None:
                continue

            status = str(o.get("status") or "").upper()
            filled_qty = int(o.get("filled_quantity") or 0)
            avg_price = float(o.get("average_price") or 0.0)
            if leg.state is OrderState.NEW and oid:
                leg.mark_acked(oid)
            self.store.on_order_acked(leg.symbol, leg.leg_id)
            event_type = None
            payload: Dict[str, Any] = {}
            if status in {"COMPLETE", "FILLED"} or filled_qty >= leg.qty:
                leg.on_fill(avg_price)
                event_type = "FILLED"
                payload = {"filled_qty": leg.filled_qty, "avg_price": leg.avg_price}
            elif status in {"PARTIALLY_FILLED"} or 0 < filled_qty < leg.qty:
                leg.on_partial(filled_qty, avg_price)
                event_type = "PARTIAL"
                payload = {"filled_qty": leg.filled_qty, "avg_price": leg.avg_price}
            elif status == "REJECTED":
                leg.on_reject(o.get("status_message", ""))
                event_type = "REJECTED"
                payload = {"reason": leg.reason}
            elif status == "CANCELLED":
                leg.on_cancel(o.get("status_message", ""))
                event_type = "CANCELLED"
                payload = {"reason": leg.reason}
            else:
                continue

            if event_type and self.store.journal:
                try:
                    self.store.journal.append_event(
                        ts=datetime.utcnow().isoformat(),
                        trade_id=leg.trade_id,
                        leg_id=leg.leg_id,
                        etype=event_type,
                        broker_order_id=leg.broker_order_id,
                        payload=payload,
                    )
                except Exception as e:
                    self.log.debug("journal append failed: %s", e)

            if leg.state in {
                OrderState.FILLED,
                OrderState.CANCELLED,
                OrderState.REJECTED,
            }:
                q = self.store._queues.get(leg.symbol)
                if q and q and q[0].leg_id == leg.leg_id:
                    q.popleft()
                self.store._inflight_symbols[leg.symbol] = max(
                    0, self.store._inflight_symbols.get(leg.symbol, 0) - 1
                )
            fsm = self.store._fsms.get(leg.trade_id)
            if fsm:
                before = fsm.status
                fsm.close_if_done()
                if (
                    fsm.status == "CLOSED"
                    and before != "CLOSED"
                    and fsm.trade_id not in self.store._closed_trades
                ):
                    pnl_R = self.store._compute_pnl_R(fsm)
                    entry_leg = next(
                        (l for l in fsm.legs.values() if l.leg_type is LegType.ENTRY),
                        None,
                    )
                    exit_price = leg.avg_price
                    exit_reason = leg.reason
                    if entry_leg and self.store.journal:
                        try:
                            risk = (
                                abs(
                                    getattr(fsm, "entry_price", entry_leg.avg_price)
                                    - getattr(fsm, "stop_loss", entry_leg.avg_price)
                                )
                                * entry_leg.qty
                            )
                            pnl_rupees = (
                                (exit_price - entry_leg.avg_price) * entry_leg.qty
                                if entry_leg.side is OrderSide.BUY
                                else (entry_leg.avg_price - exit_price) * entry_leg.qty
                            )
                            trade_rec = {
                                "ts_entry": entry_leg.created_at.isoformat(),
                                "ts_exit": datetime.utcnow().isoformat(),
                                "trade_id": fsm.trade_id,
                                "side": entry_leg.side.name,
                                "symbol": entry_leg.symbol,
                                "qty": entry_leg.qty,
                                "entry": entry_leg.avg_price,
                                "exit": exit_price,
                                "exit_reason": exit_reason,
                                "R": round(risk, 2),
                                "pnl_R": round(pnl_R, 2),
                                "pnl_rupees": round(pnl_rupees, 2),
                            }
                            self.store.journal.append_trade(trade_rec)
                        except Exception as e:
                            self.log.debug("append_trade failed: %s", e)
                    if self.store.on_trade_closed:
                        try:
                            self.store.on_trade_closed(pnl_R)
                        except Exception as e:
                            self.log.debug(
                                "on_trade_closed callback failed: %s", e, exc_info=True
                            )
                    self.store._closed_trades.add(fsm.trade_id)

            updated += 1

        return updated
