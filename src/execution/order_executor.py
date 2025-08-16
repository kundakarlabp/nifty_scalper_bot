# src/execution/order_executor.py
"""
Order execution (LIVE + SIM) with:
- Bracket-like exits (SL + TP), optional GTT OCO
- Lot-aware partial profit taking (TP1/TP2) when enabled
- Breakeven hop after TP1 (tighten-only)
- ATR-based trailing SL updates with cooldown and floor
- Hard-stop failsafe if SL is breached but broker modify fails
- Exchange freeze-quantity chunking for entries/exits
- Utility quotes: best bid/ask, mid, last

Public API (used by RealTimeTrader):
- place_entry_order(...)
- setup_gtt_orders(...)      # also handles REGULAR partial exits
- update_trailing_stop(order_id, current_price, atr)
- exit_order(order_id, exit_reason="manual")
- get_active_orders()
- get_positions()
- cancel_all_orders()
- get_last_price(symbol)
- get_best_bid_ask(symbol)
- get_mid_price(symbol)
- get_tick_size() -> float
- sync_and_enforce_oco() -> list[(entry_order_id, fill_price)]
"""

from __future__ import annotations

import logging
import math
import random
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Dict, Optional, Any, List, Tuple

from src.config import Config

logger = logging.getLogger(__name__)


# ---------------------------- config helpers ---------------------------- #

def _cfg(name: str, default: Any) -> Any:
    return getattr(Config, name, default)


# ---------------------------- rounding helpers -------------------------- #

def _round_to_tick(x: float, tick: float) -> float:
    if tick <= 0:
        return float(x)
    return round(float(x) / tick) * tick


def _round_down_to_tick(x: float, tick: float) -> float:
    if tick <= 0:
        return float(x)
    return math.floor(float(x) / tick) * tick


def _round_up_to_tick(x: float, tick: float) -> float:
    if tick <= 0:
        return float(x)
    return math.ceil(float(x) / tick) * tick


# ---------------------------- lot helpers ------------------------------- #

def _lots_from_qty(qty: int, lot_size: int) -> int:
    if lot_size <= 0:
        return qty
    return max(0, int(qty // lot_size))


def _qty_from_lots(lots: int, lot_size: int) -> int:
    return max(0, int(lots * max(1, lot_size)))


# ---------------------------- retry helper ------------------------------ #

def _with_retries(fn, *args, tries: int = 3, backoff: float = 0.5, jitter: float = 0.25, **kwargs):
    """Retry wrapper for idempotent broker calls."""
    last_exc = None
    for i in range(tries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if i == tries - 1:
                break
            sleep_s = backoff * (2 ** i) + random.random() * jitter
            logger.warning("Transient error on %s (attempt %d/%d): %s → retrying in %.2fs",
                           getattr(fn, "__name__", "call"), i + 1, tries, exc, sleep_s)
            time.sleep(sleep_s)
    raise last_exc


# ---------------------------- model ------------------------------------- #

@dataclass
class OrderRecord:
    """Internal representation of one managed trade (entry with exits)."""
    order_id: str
    symbol: str
    exchange: str
    transaction_type: str  # BUY/SELL (entry side)
    quantity: int          # contracts (NOT lots)
    entry_price: float
    stop_loss: float
    target: float

    # exit linkage (broker order IDs or GTT)
    use_gtt: bool = False
    gtt_id: Optional[int] = None
    sl_order_id: Optional[str] = None
    tp_order_id: Optional[str] = None  # single-TP mode

    exit_side: Optional[str] = None  # SELL if entry BUY, else BUY
    is_open: bool = True

    # --- Partial profits (REGULAR only) ---
    partial_enabled: bool = False
    tp1_price: Optional[float] = None
    tp1_order_id: Optional[str] = None
    tp1_qty: int = 0
    tp1_filled: bool = False

    tp2_price: Optional[float] = None
    tp2_order_id: Optional[str] = None
    tp2_qty: int = 0
    tp2_filled: bool = False

    # track SL qty (so we can recreate/resize it after TP1)
    sl_qty: int = 0

    # --- Hard stop bookkeeping ---
    breach_ts: Optional[float] = None  # first time SL breach observed

    # close info
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None

    # trailing
    last_trail_ts: float = 0.0
    trailing_step_atr_multiplier: float = 1.5  # can be overridden via Config


# ---------------------------- executor ---------------------------------- #

class OrderExecutor:
    """Thin wrapper around order placement and exit management.

    If `kite` is None: simulated mode (no network calls).
    """

    def __init__(self, kite: Optional[Any] = None) -> None:
        self.kite = kite
        self._lock = threading.RLock()
        self.orders: Dict[str, OrderRecord] = {}  # entry_order_id -> record

        # Defaults (safe fallbacks if Config misses anything)
        self.default_product = _cfg("DEFAULT_PRODUCT", "MIS")
        self.default_order_type = _cfg("DEFAULT_ORDER_TYPE", "MARKET")
        self.default_validity = _cfg("DEFAULT_VALIDITY", "DAY")
        self.atr_mult = float(_cfg("ATR_SL_MULTIPLIER", 1.5))
        self.tick_size = float(_cfg("TICK_SIZE", 0.05))
        self.trail_cooldown = float(_cfg("TRAIL_COOLDOWN_SEC", 12.0))
        self.preferred_exit_mode = str(_cfg("PREFERRED_EXIT_MODE", "AUTO")).upper()  # AUTO | GTT | REGULAR

        # Partial profits
        self.partial_enable = bool(_cfg("PARTIAL_TP_ENABLE", True))
        self.partial_ratio = float(_cfg("PARTIAL_TP_RATIO", 0.5))  # fraction of qty at TP1 (by lots)
        self.partial_use_midpoint = bool(_cfg("PARTIAL_TP_USE_MIDPOINT", True))
        self.partial_tp2_r_mult = float(_cfg("PARTIAL_TP2_R_MULT", 2.0))  # if not using midpoint

        # Breakeven hop right after TP1
        self.breakeven_after_tp1 = bool(_cfg("BREAKEVEN_AFTER_TP1_ENABLE", True))
        # move SL to entry ± N ticks
        self.breakeven_offset_ticks = int(_cfg("BREAKEVEN_OFFSET_TICKS", 1))

        # Hard stop (failsafe)
        self.hard_stop_enable = bool(_cfg("HARD_STOP_ENABLE", True))
        self.hard_stop_grace_sec = float(_cfg("HARD_STOP_GRACE_SEC", 3.0))
        self.hard_stop_slip_bps = float(_cfg("HARD_STOP_SLIPPAGE_BPS", _cfg("SLIPPAGE_BPS", 5.0)))

        # SL management
        self.use_slm_exit = bool(_cfg("USE_SLM_EXIT", True))
        self.sl_limit_offset_ticks = int(_cfg("SL_LIMIT_OFFSET_TICKS", 2))

        # Exchange freeze qty (per order)
        self.freeze_qty = int(_cfg("NFO_FREEZE_QTY", 1800))

        # Lot size guard
        self.lot_size = int(_cfg("NIFTY_LOT_SIZE", 75))

        # Trailing floor (points)
        self.trail_min_points = float(getattr(Config, "TRAIL_MIN_POINTS", 1.0))

        # best-effort last price cache for SIM / slow APIs
        self._last_price_cache: Dict[str, float] = {}

    # ------------------------------- helpers -------------------------------- #

    def get_tick_size(self) -> float:
        return float(self.tick_size)

    def _generate_order_id(self) -> str:
        return str(uuid.uuid4())

    def _safe_float(self, v: Any, default: float = 0.0) -> float:
        try:
            f = float(v)
            return f if f == f else default  # filter NaN
        except Exception:
            return default

    def _place_in_chunks(
        self, *, tradingsymbol: str, exchange: str, side: str,
        quantity: int, product: str, order_type: str, validity: str,
        price: Optional[float] = None, trigger_price: Optional[float] = None,
        variety: Optional[str] = None
    ) -> Optional[str]:
        """Respect exchange freeze quantity by splitting into chunks; return first order_id anchor."""
        remain = int(quantity)
        if remain <= 0:
            return None
        anchor_id = None
        while remain > 0:
            q = min(remain, self.freeze_qty if self.freeze_qty > 0 else remain)
            oid = self._kite_place_order(
                tradingsymbol=tradingsymbol,
                exchange=exchange,
                transaction_type=side,
                quantity=q,
                product=product,
                order_type=order_type,
                validity=validity,
                price=price,
                trigger_price=trigger_price,
                variety=variety,
            )
            if not oid:
                return None
            anchor_id = anchor_id or oid
            remain -= q
        return anchor_id

    # ------------------------------- live API -------------------------------- #

    def _kite_place_order(
        self,
        *,
        tradingsymbol: str,
        exchange: str,
        transaction_type: str,
        quantity: int,
        product: str,
        order_type: str,
        validity: str,
        price: Optional[float] = None,
        trigger_price: Optional[float] = None,
        variety: Optional[str] = None,
    ) -> Optional[str]:
        """Place order via Kite; return order_id or None."""
        if not self.kite:
            oid = self._generate_order_id()
            logger.info("🧪 SIM order: %s %s x%d %s price=%s trig=%s",
                        transaction_type, tradingsymbol, quantity, order_type, price, trigger_price)
            return oid
        try:
            def _place():
                params = dict(
                    tradingsymbol=tradingsymbol,
                    exchange=exchange,
                    transaction_type=transaction_type,
                    quantity=int(quantity),
                    product=product,
                    order_type=order_type,
                    validity=validity,
                    variety=variety or getattr(self.kite, "VARIETY_REGULAR", "regular"),
                )
                if price is not None:
                    params["price"] = float(price)
                if trigger_price is not None:
                    params["trigger_price"] = float(trigger_price)
                resp = self.kite.place_order(**params)
                if isinstance(resp, dict):
                    order_id = resp.get("order_id") or resp.get("data", {}).get("order_id")
                else:
                    order_id = resp
                return str(order_id)

            order_id = _with_retries(_place, tries=_cfg("ORDER_RETRY_LIMIT", 2))
            logger.info("✅ Kite order placed: id=%s %s %s x%d (%s)", order_id, transaction_type, tradingsymbol, quantity, order_type)
            return order_id
        except Exception as exc:
            logger.error("💥 place_order failed: %s", exc, exc_info=True)
            return None

    def _kite_place_gtt_oco(
        self,
        *,
        tradingsymbol: str,
        exchange: str,
        last_price: float,
        exit_side: str,
        quantity: int,
        sl_price: float,
        tp_price: float,
    ) -> Optional[int]:
        """Place GTT OCO; return trigger_id or None."""
        if not self.kite:
            logger.info("🧪 SIM GTT OCO created for %s (sl=%.2f, tp=%.2f)", tradingsymbol, sl_price, tp_price)
            return None
        try:
            def _place_gtt():
                legs: List[Dict[str, Any]] = [
                    {
                        "transaction_type": exit_side,
                        "quantity": int(quantity),
                        "product": self.default_product,
                        "order_type": _cfg("DEFAULT_ORDER_TYPE_EXIT", "LIMIT"),
                        "price": float(sl_price),
                    },
                    {
                        "transaction_type": exit_side,
                        "quantity": int(quantity),
                        "product": self.default_product,
                        "order_type": _cfg("DEFAULT_ORDER_TYPE_EXIT", "LIMIT"),
                        "price": float(tp_price),
                    },
                ]
                trigger_vals = [float(sl_price), float(tp_price)]
                trigger_type = getattr(self.kite, "GTT_TYPE_OCO", "two-leg")
                try:
                    resp = self.kite.place_gtt(
                        trigger_type=trigger_type,
                        tradingsymbol=tradingsymbol,
                        exchange=exchange,
                        trigger_values=trigger_vals,
                        last_price=float(last_price),
                        orders=legs,
                    )
                except TypeError:
                    resp = self.kite.place_gtt(
                        trigger_type=trigger_type,
                        tradingsymbol=tradingsymbol,
                        exchange=exchange,
                        trigger_values=trigger_vals,
                        orders=legs,
                    )
                if isinstance(resp, dict):
                    trig_id = resp.get("trigger_id") or resp.get("data", {}).get("trigger_id")
                else:
                    trig_id = resp
                return int(trig_id) if trig_id else None

            trig_id = _with_retries(_place_gtt, tries=_cfg("ORDER_RETRY_LIMIT", 2))
            if trig_id:
                logger.info("✅ GTT OCO created: trigger_id=%s for %s", trig_id, tradingsymbol)
                return trig_id
            logger.info("ℹ️ GTT OCO placed but no trigger_id returned; broker may accept silently.")
            return None
        except Exception as exc:
            logger.info("ℹ️ GTT OCO rejected/unavailable, fallback to REGULAR: %s", exc)
            return None

    def _kite_modify_sl(self, order_id: str, new_stop: float) -> bool:
        if not self.kite:
            logger.info("🧪 SIM modify SL(%s) → %.2f", order_id, new_stop)
            return True
        try:
            def _modify():
                ordertype = getattr(self.kite, "ORDER_TYPE_SLM", "SL-M") if self.use_slm_exit \
                            else getattr(self.kite, "ORDER_TYPE_SL", "SL")
                return self.kite.modify_order(
                    variety=getattr(self.kite, "VARIETY_REGULAR", "regular"),
                    order_id=order_id,
                    order_type=ordertype,
                    price=(None if ordertype in ("SLM", "SL-M") else float(new_stop)),
                    trigger_price=float(new_stop),
                    validity=self.default_validity,
                )
            _with_retries(_modify, tries=_cfg("ORDER_RETRY_LIMIT", 2))
            logger.info("✏️  Modified SL %s → trig %.2f", order_id, new_stop)
            return True
        except Exception as exc:
            logger.error("💥 modify_order (SL) failed: %s", exc, exc_info=True)
            return False

    def _kite_cancel(self, order_id: str) -> bool:
        if not self.kite:
            logger.info("🧪 SIM cancel %s", order_id)
            return True
        try:
            def _cancel():
                return self.kite.cancel_order(
                    variety=getattr(self.kite, "VARIETY_REGULAR", "regular"),
                    order_id=order_id,
                )
            _with_retries(_cancel, tries=_cfg("ORDER_RETRY_LIMIT", 2))
            logger.info("🧹 Cancelled order: %s", order_id)
            return True
        except Exception as exc:
            logger.error("💥 cancel_order failed: %s", exc, exc_info=True)
            return False

    def _order_status_map(self) -> Dict[str, Dict[str, Any]]:
        """Return {order_id: order_dict} for quick lookups."""
        if not self.kite:
            return {}
        try:
            mp: Dict[str, Dict[str, Any]] = {}
            for o in (self.kite.orders() or []):
                oid = str(o.get("order_id") or o.get("id") or "")
                if oid:
                    mp[oid] = o
            return mp
        except Exception:
            return {}

    # ------------------------------- public API ------------------------------ #

    def place_entry_order(
        self,
        symbol: str,
        direction: Optional[str] = None,          # BUY/SELL (preferred name from trader)
        transaction_type: Optional[str] = None,   # BUY/SELL (alt name)
        quantity: int = 0,
        price: Optional[float] = None,            # ignored for MARKET
        product: Optional[str] = None,
        order_type: Optional[str] = None,
        validity: Optional[str] = None,
        live: bool = False,
        exchange: Optional[str] = None,
    ) -> Optional[str]:
        """
        Place the initial entry order. Returns order_id or None.
        Quantity must be contracts. (If you size by lots, multiply before calling.)
        """
        if quantity <= 0:
            logger.warning("Attempted to place order with non-positive quantity: %s", quantity)
            return None

        side = (direction or transaction_type or "").upper()
        if side not in ("BUY", "SELL"):
            logger.warning("place_entry_order: invalid side %r", side)
            return None

        # lot integrity guard
        L = int(self.lot_size)
        if L > 0 and quantity % L != 0:
            adj = (quantity // L) * L
            logger.warning("Quantity %d not a multiple of lot size %d; adjusting down to %d.", quantity, L, adj)
            quantity = adj
            if quantity <= 0:
                return None

        product = product or self.default_product
        order_type = order_type or self.default_order_type
        validity = validity or self.default_validity
        exchange = exchange or _cfg("TRADE_EXCHANGE", "NFO")

        oid = self._place_in_chunks(
            tradingsymbol=symbol,
            exchange=exchange,
            side=side,
            quantity=int(quantity),
            product=product,
            order_type=order_type,
            validity=validity,
        )
        return oid

    def setup_gtt_orders(
        self,
        entry_order_id: str,
        symbol: Optional[str] = None,
        direction: Optional[str] = None,
        transaction_type: Optional[str] = None,
        entry_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        stop_loss_price: Optional[float] = None,
        target: Optional[float] = None,
        target_price: Optional[float] = None,
        lot_size: Optional[int] = None,
        quantity: Optional[int] = None,   # alternative to lot_size for direct contracts
        live: bool = False,
        exchange: Optional[str] = None,
    ) -> bool:
        """
        Create exits. If partial profits are enabled -> REGULAR with two TPs;
        else try GTT OCO, fallback to REGULAR.
        Always records an internal OrderRecord.

        Accepts both naming styles (direction/transaction_type, stop_loss/stop_loss_price, target/target_price).
        """
        try:
            symbol = str(symbol or "")
            if not symbol:
                logger.warning("setup_gtt_orders: missing symbol")
                return False

            side = (direction or transaction_type or "").upper()
            if side not in ("BUY", "SELL"):
                logger.warning("setup_gtt_orders: invalid side %r", side)
                return False

            exchange = exchange or _cfg("TRADE_EXCHANGE", "NFO")

            entry_price = self._safe_float(entry_price, 0.0)
            sl = self._safe_float(stop_loss if stop_loss is not None else stop_loss_price, 0.0)
            tp = self._safe_float(target if target is not None else target_price, 0.0)

            if entry_price <= 0 or sl <= 0 or tp <= 0:
                logger.warning("⚠️ Invalid SL/TP/entry for exits setup.")
                return False

            # SL sanity vs side
            if side == "BUY" and sl >= entry_price:
                sl = max(0.0, entry_price - max(self.tick_size * 4, abs(entry_price - tp) * 0.2))
            elif side == "SELL" and sl <= entry_price:
                sl = entry_price + max(self.tick_size * 4, abs(entry_price - tp) * 0.2)

            exit_side = "SELL" if side == "BUY" else "BUY"

            # quantity (contracts) from lot_size if given
            if quantity is None:
                L = int(lot_size or self.lot_size)
                quantity = max(0, L)

            # Decide partials
            want_partial = self.partial_enable and quantity >= 2  # need at least 2 contracts
            use_gtt = False
            gtt_id = None
            sl_id = None
            tp_id = None

            tp1 = None
            tp2 = None
            tp1_qty = 0
            tp2_qty = 0
            tp1_id = None
            tp2_id = None

            if want_partial:
                # Compute 1R
                r = abs(entry_price - sl)
                if self.partial_use_midpoint:
                    # TP2 = strategy TP; TP1 = midpoint between entry and TP2
                    tp2 = tp
                    tp1 = entry_price + (tp2 - entry_price) / 2.0 if side == "BUY" else entry_price - (entry_price - tp2) / 2.0
                else:
                    # R-multiples
                    if side == "BUY":
                        tp1 = entry_price + r * 1.0
                        tp2 = entry_price + r * max(1.0, float(self.partial_tp2_r_mult))
                    else:
                        tp1 = entry_price - r * 1.0
                        tp2 = entry_price - r * max(1.0, float(self.partial_tp2_r_mult))

                # Quantities by lots (not raw contracts)
                L = max(1, self.lot_size)
                total_lots = _lots_from_qty(quantity, L)

                tp1_lots = max(1, int(round(total_lots * self.partial_ratio)))
                tp2_lots = max(0, int(total_lots - tp1_lots))
                if tp2_lots == 0 and tp1_lots > 1:
                    tp1_lots -= 1
                    tp2_lots = 1

                tp1_qty = _qty_from_lots(tp1_lots, L)
                tp2_qty = _qty_from_lots(tp2_lots, L)

                # Place SL against full qty, TP legs separate (REGULAR)
                sl_ord = self._place_in_chunks(
                    tradingsymbol=symbol, exchange=exchange, side=exit_side,
                    quantity=int(quantity), product=self.default_product,
                    order_type=("SL-M" if self.use_slm_exit else "SL"),
                    validity=self.default_validity,
                    price=(None if self.use_slm_exit else float(sl)),
                    trigger_price=float(sl),
                )
                tp1_ord = self._place_in_chunks(
                    tradingsymbol=symbol, exchange=exchange, side=exit_side,
                    quantity=int(tp1_qty), product=self.default_product,
                    order_type="LIMIT", validity=self.default_validity,
                    price=float(_round_to_tick(tp1, self.tick_size)),
                )
                tp2_ord = self._place_in_chunks(
                    tradingsymbol=symbol, exchange=exchange, side=exit_side,
                    quantity=int(tp2_qty), product=self.default_product,
                    order_type="LIMIT", validity=self.default_validity,
                    price=float(_round_to_tick(tp2, self.tick_size)),
                )

                sl_id, tp1_id, tp2_id = sl_ord, tp1_ord, tp2_ord

            else:
                # Try GTT OCO if AUTO/GTT; fallback to REGULAR bracket
                mode = self.preferred_exit_mode
                if mode in ("AUTO", "GTT"):
                    ltp = self.get_last_price(symbol) or entry_price
                    gtt_id = self._kite_place_gtt_oco(
                        tradingsymbol=symbol, exchange=exchange, last_price=float(ltp),
                        exit_side=exit_side, quantity=int(quantity),
                        sl_price=float(_round_to_tick(sl, self.tick_size)),
                        tp_price=float(_round_to_tick(tp, self.tick_size)),
                    )
                    use_gtt = bool(gtt_id is not None)

                if not use_gtt or mode == "REGULAR":
                    sl_ord = self._place_in_chunks(
                        tradingsymbol=symbol, exchange=exchange, side=exit_side,
                        quantity=int(quantity), product=self.default_product,
                        order_type=("SL-M" if self.use_slm_exit else "SL"),
                        validity=self.default_validity,
                        price=(None if self.use_slm_exit else float(_round_to_tick(sl, self.tick_size))),
                        trigger_price=float(_round_to_tick(sl, self.tick_size)),
                    )
                    tp_ord = self._place_in_chunks(
                        tradingsymbol=symbol, exchange=exchange, side=exit_side,
                        quantity=int(quantity), product=self.default_product,
                        order_type="LIMIT", validity=self.default_validity,
                        price=float(_round_to_tick(tp, self.tick_size)),
                    )
                    sl_id, tp_id = sl_ord, tp_ord

            # Record
            rec = OrderRecord(
                order_id=str(entry_order_id),
                symbol=symbol,
                exchange=exchange,
                transaction_type=side,
                quantity=int(quantity),
                entry_price=float(entry_price),
                stop_loss=float(_round_to_tick(sl, self.tick_size)),
                target=float(_round_to_tick(tp, self.tick_size)),
                use_gtt=use_gtt,
                gtt_id=gtt_id,
                sl_order_id=sl_id,
                tp_order_id=tp_id,
                exit_side=exit_side,
                partial_enabled=bool(want_partial),
                tp1_price=(float(_round_to_tick(tp1, self.tick_size)) if tp1 is not None else None),
                tp1_order_id=tp1_id,
                tp1_qty=int(tp1_qty),
                tp2_price=(float(_round_to_tick(tp2, self.tick_size)) if tp2 is not None else None),
                tp2_order_id=tp2_id,
                tp2_qty=int(tp2_qty),
                sl_qty=int(quantity),
                trailing_step_atr_multiplier=float(self.atr_mult),
            )
            with self._lock:
                self.orders[str(entry_order_id)] = rec

            logger.info("📦 Exits setup for %s (partial=%s, gtt=%s)", entry_order_id, want_partial, use_gtt)
            return True

        except Exception as e:
            logger.error("setup_gtt_orders failed: %s", e, exc_info=True)
            return False

    # --- trailing / maintenance ---

    def update_trailing_stop(self, entry_order_id: str, current_price: float, atr: float) -> bool:
        """Tighten SL in the direction of profit only, observing cooldown & floor."""
        try:
            with self._lock:
                rec = self.orders.get(str(entry_order_id))
            if not rec or not rec.is_open:
                return False
            now = time.time()
            if (now - rec.last_trail_ts) < float(self.trail_cooldown):
                return False

            if atr <= 0 or current_price <= 0:
                return False

            # desired new stop based on ATR trail
            step_mult = float(rec.trailing_step_atr_multiplier or self.atr_mult)
            move = max(self.trail_min_points, float(atr) * step_mult)

            new_stop = rec.stop_loss
            if rec.transaction_type == "BUY":
                # trail upwards
                candidate = current_price - move
                if candidate > rec.stop_loss:
                    new_stop = _round_to_tick(candidate, self.tick_size)
            else:
                # SELL entry → price falling profits, trail downwards
                candidate = current_price + move
                if candidate < rec.stop_loss:
                    new_stop = _round_to_tick(candidate, self.tick_size)

            # tighten-only
            if new_stop != rec.stop_loss:
                ok = self._kite_modify_sl(rec.sl_order_id or "", new_stop)
                if ok:
                    rec.stop_loss = float(new_stop)
                    rec.last_trail_ts = now
                    logger.info("📈 Trailed SL for %s → %.2f", entry_order_id, new_stop)
                    return True
            return False
        except Exception as e:
            logger.debug("update_trailing_stop error: %s", e)
            return False

    def sync_and_enforce_oco(self) -> List[Tuple[str, float]]:
        """Detect filled exits (TP/SL/partials) and return closed entries list[(entry_id, exit_price)]."""
        filled: List[Tuple[str, float]] = []
        try:
            with self._lock:
                items = list(self.orders.items())

            live_map = self._order_status_map() if self.kite else {}

            for entry_id, rec in items:
                if not rec.is_open:
                    continue

                # LIVE: infer from broker order status
                if self.kite and live_map:
                    # if any exit is complete, infer an exit price
                    to_check = [rec.sl_order_id, rec.tp_order_id, rec.tp1_order_id, rec.tp2_order_id]
                    any_complete = False
                    exit_px = None
                    for oid in [x for x in to_check if x]:
                        od = live_map.get(str(oid), {})
                        st = (od.get("status") or "").upper()
                        if st in ("COMPLETE", "FILLED", "EXECUTED"):
                            any_complete = True
                            try:
                                exit_px = float(od.get("average_price") or od.get("price") or rec.stop_loss)
                            except Exception:
                                exit_px = rec.stop_loss
                            break
                    if any_complete:
                        rec.is_open = False
                        rec.exit_price = float(exit_px or rec.stop_loss)
                        rec.exit_reason = "exit-filled"
                        filled.append((entry_id, rec.exit_price))
                        continue

                # SIM: simple price-cross logic vs LTP
                if not self.kite:
                    ltp = self.get_last_price(rec.symbol)
                    if not ltp:
                        continue
                    # partial TP handling (best-effort)
                    if rec.partial_enabled and not rec.tp1_filled and rec.tp1_price:
                        if (rec.transaction_type == "BUY" and ltp >= rec.tp1_price) or \
                           (rec.transaction_type == "SELL" and ltp <= rec.tp1_price):
                            rec.tp1_filled = True
                            rec.sl_qty = max(0, rec.sl_qty - rec.tp1_qty)
                            # breakeven hop
                            if self.breakeven_after_tp1:
                                be = _round_to_tick(
                                    rec.entry_price + (self.breakeven_offset_ticks * self.tick_size) * (1 if rec.transaction_type == "BUY" else -1),
                                    self.tick_size
                                )
                                if (rec.transaction_type == "BUY" and be > rec.stop_loss) or \
                                   (rec.transaction_type == "SELL" and be < rec.stop_loss):
                                    rec.stop_loss = be
                            # if TP2 also hit, mark close
                            if rec.tp2_price:
                                if (rec.transaction_type == "BUY" and ltp >= rec.tp2_price) or \
                                   (rec.transaction_type == "SELL" and ltp <= rec.tp2_price):
                                    rec.is_open = False
                                    rec.exit_price = float(rec.tp2_price)
                                    rec.exit_reason = "tp2"
                                    filled.append((entry_id, rec.exit_price))
                                    continue

                    # single TP/SL
                    if (rec.transaction_type == "BUY" and ltp <= rec.stop_loss) or \
                       (rec.transaction_type == "SELL" and ltp >= rec.stop_loss):
                        rec.is_open = False
                        rec.exit_price = float(rec.stop_loss)
                        rec.exit_reason = "sl"
                        filled.append((entry_id, rec.exit_price))
                        continue
                    if rec.tp_order_id and rec.target:
                        if (rec.transaction_type == "BUY" and ltp >= rec.target) or \
                           (rec.transaction_type == "SELL" and ltp <= rec.target):
                            rec.is_open = False
                            rec.exit_price = float(rec.target)
                            rec.exit_reason = "tp"
                            filled.append((entry_id, rec.exit_price))
                            continue

        except Exception as e:
            logger.debug("sync_and_enforce_oco error: %s", e)
        return filled

    # --- exits / cancels / positions ---

    def exit_order(self, entry_order_id: str, exit_reason: str = "manual") -> bool:
        """Market-out best effort: cancel exits and place opposing MARKET for full remaining qty."""
        try:
            with self._lock:
                rec = self.orders.get(str(entry_order_id))
            if not rec or not rec.is_open:
                return False

            # cancel existing exits
            for oid in [rec.sl_order_id, rec.tp_order_id, rec.tp1_order_id, rec.tp2_order_id]:
                if oid:
                    try:
                        self._kite_cancel(oid)
                    except Exception:
                        pass

            # place market exit
            side = rec.exit_side or ("SELL" if rec.transaction_type == "BUY" else "BUY")
            oid = self._place_in_chunks(
                tradingsymbol=rec.symbol, exchange=rec.exchange, side=side,
                quantity=int(rec.quantity), product=self.default_product,
                order_type="MARKET", validity=self.default_validity
            )
            ltp = self.get_last_price(rec.symbol) or rec.stop_loss
            rec.is_open = False
            rec.exit_price = float(ltp)
            rec.exit_reason = exit_reason or "manual"
            logger.info("🚪 Manual exit placed for %s @ ~%.2f", entry_order_id, ltp)
            return True
        except Exception as e:
            logger.error("exit_order failed: %s", e, exc_info=True)
            return False

    def get_active_orders(self):
        """Return either {entry_id: OrderRecord} or list of open records; both are supported upstream."""
        with self._lock:
            open_map = {k: v for k, v in self.orders.items() if v.is_open}
        return open_map

    def get_positions(self):
        """Best-effort position snapshot (count open records)."""
        with self._lock:
            return [v for v in self.orders.values() if v.is_open]

    def cancel_all_orders(self) -> bool:
        ok = True
        try:
            with self._lock:
                ids = list(self.orders.keys())
            for eid in ids:
                try:
                    self.exit_order(eid, exit_reason="cancel_all")
                except Exception:
                    ok = False
        except Exception:
            ok = False
        return ok

    # --- quotes helpers used by spread guard and status ---

    def _format_quote_token(self, symbol: str, exchange: Optional[str] = None) -> str:
        if ":" in symbol:
            return symbol
        ex = exchange or _cfg("TRADE_EXCHANGE", "NFO")
        return f"{ex}:{symbol}"

    def get_last_price(self, symbol: str) -> Optional[float]:
        try:
            if self.kite:
                tok = self._format_quote_token(symbol)
                l = self.kite.ltp([tok]) or {}
                lp = None
                try:
                    lp = (l.get(tok) or {}).get("last_price")
                except Exception:
                    pass
                if lp:
                    self._last_price_cache[symbol] = float(lp)
                    return float(lp)
            # fallback to cache
            return self._last_price_cache.get(symbol)
        except Exception:
            return self._last_price_cache.get(symbol)

    def get_best_bid_ask(self, symbol: str) -> Optional[Dict[str, float]]:
        if not self.kite:
            return None
        try:
            tok = self._format_quote_token(symbol)
            q = self.kite.quote([tok]) or {}
            dd = (q.get(tok) or {}).get("depth") or {}
            bids = (dd.get("buy") or [])
            asks = (dd.get("sell") or [])
            bid = float(bids[0]["price"]) if bids else None
            ask = float(asks[0]["price"]) if asks else None
            if bid and ask and ask >= bid:
                return {"bid": bid, "ask": ask}
        except Exception:
            pass
        return None

    def get_mid_price(self, symbol: str) -> Optional[float]:
        bb = self.get_best_bid_ask(symbol)
        if bb:
            return 0.5 * (bb["bid"] + bb["ask"])
        lp = self.get_last_price(symbol)
        return float(lp) if lp else None

    # --------------------------- END CLASS ----------------------------------- #