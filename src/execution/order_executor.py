# src/execution/order_executor.py
"""
Order execution module (live + simulation) with bracket-like behavior,
lot-aware partial profit-taking, breakeven hop after TP1, trailing-SL support,
hard-stop failsafe, and exchange freeze-quantity chunking.

Public API:
- place_entry_order(...)
- setup_gtt_orders(...)
- update_trailing_stop(order_id, current_price, atr)
- exit_order(order_id, exit_reason="manual")
- get_active_orders()
- get_positions()
- cancel_all_orders()
- get_last_price(symbol)
- get_tick_size() -> float
- sync_and_enforce_oco() -> list[(entry_order_id, fill_price)]

Notes:
- If PARTIAL_TP_ENABLE=true, exits are managed with REGULAR orders
  (GTT OCO cannot split partials cleanly).
- After TP1 fill we:
    * shrink SL to remaining quantity
    * optionally hop SL to breakeven ± ticks (tighten-only)
- If USE_SLM_EXIT=true, SL uses market trigger (SL-M/SLM). Otherwise SL (limit)
  with a protective offset.
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
        self.trail_cooldown = float(_cfg("TRAIL_COOLDOWN_SEC", 8.0))
        self.preferred_exit_mode = str(_cfg("PREFERRED_EXIT_MODE", "REGULAR")).upper()  # AUTO | GTT | REGULAR

        # Partial profits
        self.partial_enable = bool(_cfg("PARTIAL_TP_ENABLE", True))
        self.partial_ratio = float(_cfg("PARTIAL_TP_RATIO", 0.5))  # fraction of qty at TP1 (by lots)
        self.partial_use_midpoint = bool(_cfg("PARTIAL_TP_USE_MIDPOINT", True))
        self.partial_tp2_r_mult = float(_cfg("PARTIAL_TP2_R_MULT", 2.0))  # if not using midpoint

        # Breakeven hop right after TP1
        self.breakeven_after_tp1 = bool(_cfg("BREAKEVEN_AFTER_TP1_ENABLE", True))
        # move SL to entry ± N ticks
        self.breakeven_offset_ticks = int(_cfg("BREAKEVEN_OFFSET_TICKS", 2))

        # Hard stop (failsafe)
        self.hard_stop_enable = bool(_cfg("HARD_STOP_ENABLE", True))
        self.hard_stop_grace_sec = float(_cfg("HARD_STOP_GRACE_SEC", 3.0))
        self.hard_stop_slip_bps = float(_cfg("HARD_STOP_SLIPPAGE_BPS", _cfg("SLIPPAGE_BPS", 6.0)))

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
        exchange: str,
        transaction_type: str,
        quantity: int,
        product: str = None,
        order_type: str = None,
        validity: str = None,
    ) -> Optional[str]:
        """
        Place the initial entry order. Returns order_id or None.
        Quantity must be contracts. (If you size by lots, multiply before calling.)
        """
        if quantity <= 0:
            logger.warning("Attempted to place order with non-positive quantity: %s", quantity)
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

        oid = self._place_in_chunks(
            tradingsymbol=symbol,
            exchange=exchange,
            side=transaction_type.upper(),
            quantity=int(quantity),
            product=product,
            order_type=order_type,
            validity=validity,
        )
        return oid

    def setup_gtt_orders(
        self,
        entry_order_id: str,
        entry_price: float,
        stop_loss_price: float,
        target_price: float,
        symbol: str,
        exchange: str,
        quantity: int,
        transaction_type: str,
    ) -> bool:
        """
        Create exits. If partial profits are enabled -> REGULAR with two TPs;
        else try GTT OCO, fallback to REGULAR.
        Always records an internal OrderRecord.
        """
        try:
            entry_price = self._safe_float(entry_price)
            sl = self._safe_float(stop_loss_price)
            tp = self._safe_float(target_price)

            if entry_price <= 0 or sl <= 0 or tp <= 0:
                logger.warning("⚠️ Invalid SL/TP/entry for exits setup.")
                return False

            # SL sanity vs side
            side = transaction_type.upper()
            if side == "BUY" and sl >= entry_price:
                sl = max(0.0, entry_price - max(self.tick_size * 4, abs(entry_price - tp) * 0.2))
            elif side == "SELL" and sl <= entry_price:
                sl = entry_price + max(self.tick_size * 4, abs(entry_price - tp) * 0.2)

            exit_side = "SELL" if side == "BUY" else "BUY"

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

            mode = self.preferred_exit_mode  # AUTO | GTT | REGULAR
            try_gtt = (mode in ("AUTO", "GTT")) and not want_partial

            if try_gtt:
                gtt_id = self._kite_place_gtt_oco(
                    tradingsymbol=symbol,
                    exchange=exchange,
                    last_price=entry_price,
                    exit_side=exit_side,
                    quantity=int(quantity),
                    sl_price=_round_to_tick(sl, self.tick_size),
                    tp_price=_round_to_tick(tp, self.tick_size),
                )
                use_gtt = gtt_id is not None
                if not use_gtt:
                    logger.info("ℹ️ Falling back to REGULAR exits (GTT not available / rejected).")

            if not use_gtt:
                # REGULAR exits
                # --- Stop Loss ---
                sl_trig = _round_to_tick(sl, self.tick_size)

                if self.use_slm_exit:
                    sl_ordertype = getattr(self.kite, "ORDER_TYPE_SLM", "SL-M")
                    sl_price = None
                else:
                    sl_ordertype = getattr(self.kite, "ORDER_TYPE_SL", "SL")
                    off = max(0, int(self.sl_limit_offset_ticks))
                    if exit_side == "SELL":
                        sl_price = _round_down_to_tick(sl_trig - off * self.tick_size, self.tick_size)
                    else:
                        sl_price = _round_up_to_tick(sl_trig + off * self.tick_size, self.tick_size)

                sl_id = self._place_in_chunks(
                    tradingsymbol=symbol,
                    exchange=exchange,
                    side=exit_side,
                    quantity=int(quantity),
                    product=self.default_product,
                    order_type=sl_ordertype if self.kite else ("SL-M" if self.use_slm_exit else "SL"),
                    validity=self.default_validity,
                    price=(None if self.use_slm_exit else float(sl_price)),
                    trigger_price=float(sl_trig),
                )

                if want_partial:
                    # --- TP1 ---
                    tp1_id = self._place_in_chunks(
                        tradingsymbol=symbol,
                        exchange=exchange,
                        side=exit_side,
                        quantity=int(tp1_qty),
                        product=self.default_product,
                        order_type=getattr(self.kite, "ORDER_TYPE_LIMIT", "LIMIT"),
                        validity=self.default_validity,
                        price=_round_to_tick(tp1, self.tick_size),
                    )
                    # --- TP2 ---
                    tp2_id = self._place_in_chunks(
                        tradingsymbol=symbol,
                        exchange=exchange,
                        side=exit_side,
                        quantity=int(tp2_qty),
                        product=self.default_product,
                        order_type=getattr(self.kite, "ORDER_TYPE_LIMIT", "LIMIT"),
                        validity=self.default_validity,
                        price=_round_to_tick(tp2, self.tick_size),
                    )
                else:
                    # --- single TP ---
                    tp_id = self._place_in_chunks(
                        tradingsymbol=symbol,
                        exchange=exchange,
                        side=exit_side,
                        quantity=int(quantity),
                        product=self.default_product,
                        order_type=getattr(self.kite, "ORDER_TYPE_LIMIT", "LIMIT"),
                        validity=self.default_validity,
                        price=_round_to_tick(tp, self.tick_size),
                    )

            # record
            rec = OrderRecord(
                order_id=entry_order_id,
                symbol=symbol,
                exchange=exchange,
                transaction_type=side,
                quantity=int(quantity),
                entry_price=float(entry_price),
                stop_loss=float(sl),
                target=float(tp),
                use_gtt=bool(use_gtt),
                gtt_id=(int(gtt_id) if gtt_id else None),
                sl_order_id=sl_id,
                tp_order_id=tp_id,
                exit_side=exit_side,
                partial_enabled=bool(want_partial),
                tp1_price=(float(tp1) if tp1 else None),
                tp1_order_id=tp1_id,
                tp1_qty=int(tp1_qty),
                tp1_filled=False,
                tp2_price=(float(tp2) if tp2 else None),
                tp2_order_id=tp2_id,
                tp2_qty=int(tp2_qty),
                tp2_filled=False,
                sl_qty=int(quantity),
                trailing_step_atr_multiplier=float(self.atr_mult),
            )

            with self._lock:
                self.orders[entry_order_id] = rec

            self._log_state(rec, "exits_setup")
            return True

        except Exception as e:
            logger.error(f"💥 setup_gtt_orders failed: {e}", exc_info=True)
            return False

    def update_trailing_stop(self, entry_order_id: str, current_price: float, atr: float) -> None:
        """Tighten-only trailing stop based on ATR multiplier + minimum floor."""
        if not entry_order_id:
            return

        with self._lock:
            rec = self.orders.get(entry_order_id)
        if not rec or not rec.is_open or rec.use_gtt:
            # trailing with GTT is not supported here
            return
        if not rec.sl_order_id:
            return

        now = time.time()
        if now - rec.last_trail_ts < float(self.trail_cooldown):
            return

        atr = self._safe_float(atr)
        if atr <= 0:
            return

        step = max(self.trail_min_points, float(rec.trailing_step_atr_multiplier) * float(atr))
        tick = self.tick_size

        # Compute candidate
        if rec.transaction_type.upper() == "BUY":
            # for longs: raise SL upwards
            new_sl = _round_to_tick(max(rec.stop_loss, current_price - step), tick)
            if new_sl <= rec.stop_loss + tick * 0.5:
                return
            new_sl = max(rec.stop_loss, new_sl)
        else:
            # for shorts: lower SL downwards
            new_sl = _round_to_tick(min(rec.stop_loss, current_price + step), tick)
            if new_sl >= rec.stop_loss - tick * 0.5:
                return
            new_sl = min(rec.stop_loss, new_sl)

        if self._kite_modify_sl(rec.sl_order_id, new_sl):
            rec.stop_loss = float(new_sl)
            rec.last_trail_ts = now
            self._log_state(rec, "trail_update")

    def exit_order(self, entry_order_id: str, exit_reason: str = "manual") -> bool:
        """Exit remaining position at market; cancel peers; close record."""
        with self._lock:
            rec = self.orders.get(entry_order_id)
        if not rec or not rec.is_open:
            return False

        # Cancel TPs
        for oid in [rec.tp_order_id, rec.tp1_order_id, rec.tp2_order_id]:
            if oid:
                self._kite_cancel(oid)
        # Cancel SL to avoid double fill
        if rec.sl_order_id:
            self._kite_cancel(rec.sl_order_id)

        # Market-out remaining
        rem_qty = int(rec.sl_qty) if rec.partial_enabled else int(rec.quantity)
        if rem_qty > 0:
            self._place_in_chunks(
                tradingsymbol=rec.symbol,
                exchange=rec.exchange,
                side=rec.exit_side or ("SELL" if rec.transaction_type == "BUY" else "BUY"),
                quantity=rem_qty,
                product=self.default_product,
                order_type=getattr(self.kite, "ORDER_TYPE_MARKET", "MARKET"),
                validity=self.default_validity,
            )

        rec.exit_reason = exit_reason
        rec.exit_price = None  # Realized exit price best retrieved from broker book by the caller if needed
        rec.is_open = False
        self._log_state(rec, "manual_exit")
        return True

    def get_active_orders(self) -> Dict[str, OrderRecord]:
        with self._lock:
            return {k: v for k, v in self.orders.items() if v.is_open}

    def get_positions(self) -> List[Dict[str, Any]]:
        with self._lock:
            out = []
            for rec in self.orders.values():
                if not rec.is_open:
                    continue
                out.append({
                    "order_id": rec.order_id,
                    "symbol": rec.symbol,
                    "side": rec.transaction_type,
                    "qty": rec.quantity,
                    "sl": rec.stop_loss,
                    "tp": rec.target,
                    "partial": rec.partial_enabled,
                })
            return out

    def cancel_all_orders(self) -> None:
        with self._lock:
            ids = list(self.orders.keys())
        for oid in ids:
            try:
                self.exit_order(oid, exit_reason="cancel_all")
            except Exception:
                pass

    def get_last_price(self, symbol: str) -> Optional[float]:
        """Best-effort last price; SIM returns cache if any."""
        if not symbol:
            return None
        if not self.kite:
            return self._last_price_cache.get(symbol)
        try:
            exch = None
            # try to infer exchange from current orders
            with self._lock:
                for rec in self.orders.values():
                    if rec.symbol == symbol:
                        exch = rec.exchange
                        break
            exch = exch or "NFO"
            key = f"{exch}:{symbol}"
            data = self.kite.ltp([key])
            px = float(data[key]["last_price"])
            # cache for SIM-ish usage
            self._last_price_cache[symbol] = px
            return px
        except Exception:
            return None

    # ------------------------------------------------------------------------- #
    # OCO housekeeping + hard stop + TP1 breakeven hop
    # ------------------------------------------------------------------------- #

    def sync_and_enforce_oco(self) -> List[Tuple[str, float]]:
        """
        Poll broker orders and enforce OCO semantics. Returns a list of
        (entry_order_id, fill_price) for trades that fully closed.
        - Detect TP1 fill → shrink SL, optional breakeven hop
        - If SL or TP/TP2 fully filled → cancel peers and close
        - Hard-stop: if SL is crossed and not filled after grace, market-out
        """
        fills: List[Tuple[str, float]] = []

        status_map = self._order_status_map() if self.kite else {}

        with self._lock:
            entries = list(self.orders.items())

        for entry_id, rec in entries:
            if not rec.is_open:
                continue

            # --- live status fetch ---
            def _is_complete(oid: Optional[str]) -> bool:
                if not oid:
                    return False
                if not status_map:
                    return False
                st = status_map.get(oid, {})
                return (st.get("status") or st.get("order_status") or "").upper() in ("COMPLETE", "FILLED", "EXECUTED")

            def _avg_price(oid: Optional[str]) -> Optional[float]:
                if not oid or not status_map:
                    return None
                st = status_map.get(oid, {})
                v = st.get("average_price") or st.get("avg_price") or st.get("filled_price")
                try:
                    return float(v) if v is not None else None
                except Exception:
                    return None

            # --- PARTIAL management: TP1 fill path ---
            if rec.partial_enabled and rec.tp1_order_id and not rec.tp1_filled and _is_complete(rec.tp1_order_id):
                rec.tp1_filled = True
                rec.sl_qty = int(rec.tp2_qty)  # remaining managed by SL + TP2
                self._log_state(rec, "tp1_filled")

                if self.breakeven_after_tp1 and rec.sl_order_id:
                    # Compute breakeven ± ticks (tighten-only)
                    offset = max(0, int(self.breakeven_offset_ticks))
                    if rec.transaction_type.upper() == "BUY":
                        target_sl = _round_to_tick(rec.entry_price + offset * self.tick_size, self.tick_size)
                        if target_sl > rec.stop_loss:  # tighten-only
                            if self._kite_modify_sl(rec.sl_order_id, target_sl):
                                rec.stop_loss = float(target_sl)
                                self._log_state(rec, "breakeven_hop")
                    else:
                        target_sl = _round_to_tick(rec.entry_price - offset * self.tick_size, self.tick_size)
                        if target_sl < rec.stop_loss:  # tighten-only
                            if self._kite_modify_sl(rec.sl_order_id, target_sl):
                                rec.stop_loss = float(target_sl)
                                self._log_state(rec, "breakeven_hop")

            # --- Hard stop detection (price breached but SL not filled) ---
            if self.hard_stop_enable and rec.sl_order_id and not _is_complete(rec.sl_order_id):
                ltp = self.get_last_price(rec.symbol) or rec.entry_price
                breached = False
                if rec.transaction_type.upper() == "BUY":
                    breached = ltp <= rec.stop_loss - 1e-9
                else:
                    breached = ltp >= rec.stop_loss + 1e-9
                if breached:
                    if rec.breach_ts is None:
                        rec.breach_ts = time.time()
                        self._log_state(rec, "hard_stop_breach_detected")
                    elif time.time() - rec.breach_ts >= float(self.hard_stop_grace_sec):
                        # Market-out remaining quantity
                        rem_qty = int(rec.sl_qty) if rec.partial_enabled else int(rec.quantity)
                        if rem_qty > 0:
                            self._place_in_chunks(
                                tradingsymbol=rec.symbol,
                                exchange=rec.exchange,
                                side=rec.exit_side or ("SELL" if rec.transaction_type == "BUY" else "BUY"),
                                quantity=rem_qty,
                                product=self.default_product,
                                order_type=getattr(self.kite, "ORDER_TYPE_MARKET", "MARKET"),
                                validity=self.default_validity,
                            )
                        # Cancel peers
                        for oid in [rec.tp_order_id, rec.tp1_order_id, rec.tp2_order_id]:
                            if oid:
                                self._kite_cancel(oid)
                        # Close record
                        rec.is_open = False
                        rec.exit_reason = "hard_stop"
                        rec.exit_price = ltp
                        fills.append((entry_id, float(ltp)))
                        self._log_state(rec, "hard_stop_executed")
                        continue  # next trade
                else:
                    rec.breach_ts = None  # reset if back inside

            # --- Close paths ---
            # Single-TP OCO: any one complete -> cancel the other and close
            if not rec.partial_enabled and (rec.sl_order_id or rec.tp_order_id):
                sl_done = _is_complete(rec.sl_order_id)
                tp_done = _is_complete(rec.tp_order_id)
                if sl_done or tp_done:
                    # cancel peer
                    if sl_done and rec.tp_order_id:
                        self._kite_cancel(rec.tp_order_id)
                        px = _avg_price(rec.sl_order_id) or rec.stop_loss
                        rec.exit_reason = "stop_loss"
                        rec.exit_price = px
                    elif tp_done and rec.sl_order_id:
                        self._kite_cancel(rec.sl_order_id)
                        px = _avg_price(rec.tp_order_id) or rec.target
                        rec.exit_reason = "target"
                        rec.exit_price = px
                    rec.is_open = False
                    fills.append((entry_id, float(rec.exit_price or rec.target)))
                    self._log_state(rec, "oco_close")
                    continue

            # Partial mode: TP2 or SL completes => close & cancel peers
            if rec.partial_enabled:
                sl_done = _is_complete(rec.sl_order_id)
                tp2_done = _is_complete(rec.tp2_order_id) if rec.tp2_order_id else False
                if sl_done or tp2_done:
                    # cancel any remaining peers
                    for oid in [rec.tp1_order_id, rec.tp2_order_id, rec.tp_order_id]:
                        if oid and not (sl_done and oid == rec.sl_order_id):
                            self._kite_cancel(oid)
                    if sl_done:
                        px = _avg_price(rec.sl_order_id) or rec.stop_loss
                        rec.exit_reason = "stop_loss"
                        rec.exit_price = px
                    else:
                        px = _avg_price(rec.tp2_order_id) or rec.tp2_price or rec.target
                        rec.exit_reason = "target"
                        rec.exit_price = px
                    rec.is_open = False
                    fills.append((entry_id, float(rec.exit_price or rec.target)))
                    self._log_state(rec, "partial_close")
                    continue

        return fills

    # ------------------------------- logging --------------------------------- #

    def _log_state(self, rec: OrderRecord, tag: str) -> None:
        try:
            info = {
                "id": rec.order_id,
                "sym": rec.symbol,
                "side": rec.transaction_type,
                "qty": rec.quantity,
                "sl_qty": rec.sl_qty,
                "sl": round(rec.stop_loss, 2),
                "tp": round(rec.target, 2),
                "tp1": (round(rec.tp1_price, 2) if rec.tp1_price else None),
                "tp2": (round(rec.tp2_price, 2) if rec.tp2_price else None),
                "tp1_filled": rec.tp1_filled,
                "tp2_filled": rec.tp2_filled,
                "use_gtt": rec.use_gtt,
                "is_open": rec.is_open,
                "reason": rec.exit_reason,
            }
            logger.debug("EXECUTOR[%s] %s", tag, info)
        except Exception:
            pass