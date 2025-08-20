# src/execution/order_executor.py
"""
Order execution module (live + simulation) with bracket-like behavior,
lot-aware partial profit-taking, breakeven hop after TP1, trailing-SL support,
hard-stop failsafe, and exchange freeze-quantity chunking.

Public API:
- place_entry_order(...)
- setup_gtt_orders(...)
- update_trailing_stop(order_id, current_price, atr, atr_multiplier)
- exit_order(order_id, exit_reason="manual")
- get_active_orders()
- get_positions()
- cancel_all_orders()
- get_last_price(symbol)
- get_tick_size() -> float
- sync_and_enforce_oco() -> list[(order_id, fill_price)]

Notes:
- If PARTIAL_TP_ENABLE=true, exits are managed with REGULAR orders (GTT OCO cannot split partials cleanly).
- After TP1 fill we:
    * shrink SL to remaining quantity
    * optionally hop SL to breakeven ± ticks (tighten-only)
- If USE_SLM_EXIT=true, SL uses market trigger (SL-M/SLM). Otherwise SL (limit) with a protective offset.
"""

from __future__ import annotations

import logging
import math
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Dict, Optional, Any, List, Tuple

from src.config import ExecutorConfig
from src.utils.retry import retry

logger = logging.getLogger(__name__)


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
    tp_order_id: Optional[str] = None
    exit_side: Optional[str] = None  # SELL if entry BUY, else BUY

    is_open: bool = True
    last_trail_ts: float = 0.0
    trailing_step_atr_multiplier: float = 1.5  # can be overridden via Config

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

    # --- Hard stop bookkeeping (hook for future use) ---
    breach_ts: Optional[float] = None

    # close info
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None


# ---------------------------- executor ---------------------------------- #

class OrderExecutor:
    """
    Thin wrapper around order placement and exit management.
    If `kite` is None, it runs in simulation mode.
    """

    def __init__(self, config: ExecutorConfig, kite: Optional[Any] = None):
        if not isinstance(config, ExecutorConfig):
            raise TypeError("A valid ExecutorConfig instance is required.")
        self.config = config
        self.kite = kite
        self._lock = threading.RLock()
        self.orders: Dict[str, OrderRecord] = {}

    # ------------------------------- helpers -------------------------------- #

    def get_tick_size(self) -> float:
        return float(self.config.tick_size)

    @property
    def default_product(self) -> str:
        return self.config.default_product

    @property
    def default_order_type(self) -> str:
        return self.config.default_order_type

    @property
    def default_validity(self) -> str:
        return self.config.default_validity

    @property
    def use_slm_exit(self) -> bool:
        return self.config.use_slm_exit

    @property
    def breakeven_after_tp1(self) -> bool:
        return bool(self.config.breakeven_after_tp1_enable)

    @property
    def sl_limit_offset_ticks(self) -> int:
        return int(self.config.sl_limit_offset_ticks)

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
        freeze_qty = int(self.config.nfo_freeze_qty or 0)
        while remain > 0:
            q = min(remain, freeze_qty if freeze_qty > 0 else remain)
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

    @retry(tries=3, delay=1, backoff=2)
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
            logger.info(
                "🧪 SIM order: %s %s x%d %s price=%s trig=%s",
                transaction_type, tradingsymbol, quantity, order_type, price, trigger_price
            )
            return oid
        try:
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

            order_id = self.kite.place_order(**params)
            if isinstance(order_id, dict):
                order_id = order_id.get("order_id") or order_id.get("data", {}).get("order_id")
            order_id = str(order_id)
            logger.info(
                "✅ Kite order placed: id=%s %s %s x%d (%s)",
                order_id, transaction_type, tradingsymbol, quantity, order_type
            )
            return order_id
        except Exception as exc:
            logger.error("💥 place_order failed: %s", exc, exc_info=True)
            return None

    @retry(tries=3, delay=1, backoff=2)
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
            logger.info(
                "🧪 SIM GTT OCO created for %s (sl=%.2f, tp=%.2f)",
                tradingsymbol, sl_price, tp_price
            )
            return None
        try:
            legs: List[Dict[str, Any]] = [
                {
                    "transaction_type": exit_side,
                    "quantity": int(quantity),
                    "product": self.config.default_product,
                    "order_type": "LIMIT",  # GTT legs are limit
                    "price": float(sl_price),
                },
                {
                    "transaction_type": exit_side,
                    "quantity": int(quantity),
                    "product": self.config.default_product,
                    "order_type": "LIMIT",
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
                # some SDK variants don't require last_price
                resp = self.kite.place_gtt(
                    trigger_type=trigger_type,
                    tradingsymbol=tradingsymbol,
                    exchange=exchange,
                    trigger_values=trigger_vals,
                    orders=legs,
                )

            trig_id = None
            if isinstance(resp, dict):
                trig_id = resp.get("trigger_id") or resp.get("data", {}).get("trigger_id")
            if trig_id:
                logger.info("✅ GTT OCO created: trigger_id=%s for %s", trig_id, tradingsymbol)
                return int(trig_id)
            logger.info("ℹ️ GTT OCO placed but no trigger_id returned; broker may accept silently.")
            return None
        except Exception as exc:
            logger.info("ℹ️ GTT OCO rejected/unavailable, falling back to REGULAR: %s", exc)
            return None

    @retry(tries=3, delay=1, backoff=2)
    def _kite_modify_sl(self, order_id: str, new_stop: float) -> bool:
        if not self.kite:
            logger.info("🧪 SIM modify SL(%s) → %.2f", order_id, new_stop)
            return True
        try:
            ordertype = getattr(self.kite, "ORDER_TYPE_SLM", "SL-M") if self.config.use_slm_exit \
                        else getattr(self.kite, "ORDER_TYPE_SL", "SL")
            self.kite.modify_order(
                variety=getattr(self.kite, "VARIETY_REGULAR", "regular"),
                order_id=order_id,
                order_type=ordertype,
                price=(None if ordertype in ("SLM", "SL-M") else float(new_stop)),
                trigger_price=float(new_stop),
                validity=self.config.default_validity,
            )
            logger.info("✏️  Modified SL %s → trig %.2f (%s)", order_id, new_stop, ordertype)
            return True
        except Exception as exc:
            logger.error("💥 modify_order (SL) failed: %s", exc, exc_info=True)
            return False

    @retry(tries=3, delay=1, backoff=2)
    def _kite_cancel(self, order_id: str) -> bool:
        if not self.kite:
            logger.info("🧪 SIM cancel %s", order_id)
            return True
        try:
            self.kite.cancel_order(
                variety=getattr(self.kite, "VARIETY_REGULAR", "regular"),
                order_id=order_id,
            )
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
            for o in self.kite.orders() or []:
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
    ) -> Optional[str]:
        """
        Place the initial entry order. Returns order_id or None.
        Quantity must be contracts.
        """
        if quantity <= 0:
            logger.warning("Attempted to place order with non-positive quantity: %s", quantity)
            return None

        # Lot size integrity guard
        lot_size = int(self.config.nifty_lot_size or 0)
        if lot_size > 0 and quantity % lot_size != 0:
            adj_qty = (quantity // lot_size) * lot_size
            logger.warning(
                "Quantity %d not a multiple of lot size %d; adjusting down to %d.",
                quantity, lot_size, adj_qty
            )
            quantity = adj_qty
            if quantity <= 0:
                return None

        oid = self._place_in_chunks(
            tradingsymbol=symbol,
            exchange=exchange,
            side=transaction_type.upper(),
            quantity=int(quantity),
            product=self.config.default_product,
            order_type=self.config.default_order_type,
            validity=self.config.default_validity,
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

            exit_side = "SELL" if transaction_type.upper() == "BUY" else "BUY"

            # Decide partials
            want_partial = bool(self.config.partial_tp_enable) and quantity >= int(self.config.nifty_lot_size or 2)
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
                # R-multiples for TP calculation
                r = abs(entry_price - sl)
                if transaction_type.upper() == "BUY":
                    tp1 = entry_price + r * 1.0
                    tp2 = entry_price + r * max(1.0, float(self.config.partial_tp2_r_mult))
                else:
                    tp1 = entry_price - r * 1.0
                    tp2 = entry_price - r * max(1.0, float(self.config.partial_tp2_r_mult))

                # Quantities by lots
                lot_size = int(self.config.nifty_lot_size or 0)
                total_lots = _lots_from_qty(quantity, lot_size)
                tp1_lots = max(1, int(round(total_lots * float(self.config.partial_tp_ratio))))
                tp2_lots = max(0, int(total_lots - tp1_lots))
                if tp2_lots == 0 and tp1_lots > 1:
                    tp1_lots -= 1
                    tp2_lots = 1

                tp1_qty = _qty_from_lots(tp1_lots, lot_size)
                tp2_qty = _qty_from_lots(tp2_lots, lot_size)

            mode = self.config.preferred_exit_mode
            try_gtt = (mode in ("AUTO", "GTT")) and not want_partial

            tick = self.get_tick_size()

            if try_gtt:
                gtt_id = self._kite_place_gtt_oco(
                    tradingsymbol=symbol,
                    exchange=exchange,
                    last_price=entry_price,
                    exit_side=exit_side,
                    quantity=int(quantity),
                    sl_price=_round_to_tick(sl, tick),
                    tp_price=_round_to_tick(tp, tick),
                )
                use_gtt = gtt_id is not None
                if not use_gtt:
                    logger.info("ℹ️ Falling back to REGULAR exits (GTT not available / rejected).")

            if not use_gtt:
                # REGULAR exits
                sl_trig = _round_to_tick(sl, tick)

                if self.use_slm_exit:
                    sl_ordertype = "SL-M"
                    sl_price = None
                else:
                    sl_ordertype = "SL"
                    off = self.sl_limit_offset_ticks
                    if exit_side == "SELL":
                        sl_price = _round_down_to_tick(sl_trig - off * tick, tick)
                    else:
                        sl_price = _round_up_to_tick(sl_trig + off * tick, tick)

                sl_id = self._place_in_chunks(
                    tradingsymbol=symbol,
                    exchange=exchange,
                    side=exit_side,
                    quantity=int(quantity),
                    product=self.default_product,
                    order_type=sl_ordertype,
                    validity=self.default_validity,
                    price=(None if self.use_slm_exit else float(sl_price)),
                    trigger_price=float(sl_trig),
                )

                if want_partial:
                    tp1_rounded = _round_to_tick(float(tp1), tick)
                    tp2_rounded = _round_to_tick(float(tp2), tick)
                    tp1_id = self._place_in_chunks(
                        tradingsymbol=symbol, exchange=exchange, side=exit_side,
                        quantity=int(tp1_qty), product=self.default_product,
                        order_type="LIMIT", validity=self.default_validity, price=float(tp1_rounded),
                    )
                    tp2_id = self._place_in_chunks(
                        tradingsymbol=symbol, exchange=exchange, side=exit_side,
                        quantity=int(tp2_qty), product=self.default_product,
                        order_type="LIMIT", validity=self.default_validity, price=float(tp2_rounded),
                    )
                else:
                    tp_rounded = _round_to_tick(tp, tick)
                    tp_id = self._place_in_chunks(
                        tradingsymbol=symbol, exchange=exchange, side=exit_side,
                        quantity=int(quantity), product=self.default_product,
                        order_type="LIMIT", validity=self.default_validity, price=float(tp_rounded),
                    )

            with self._lock:
                rec = OrderRecord(
                    order_id=entry_order_id, symbol=symbol, exchange=exchange,
                    transaction_type=transaction_type.upper(), quantity=int(quantity),
                    entry_price=entry_price, stop_loss=sl, target=tp,
                    use_gtt=use_gtt, gtt_id=gtt_id, sl_order_id=sl_id,
                    tp_order_id=tp_id, exit_side=exit_side,
                    partial_enabled=bool(want_partial),
                    tp1_price=(float(tp1) if want_partial else None),
                    tp1_order_id=(tp1_id if want_partial else None),
                    tp1_qty=(int(tp1_qty) if want_partial else 0),
                    tp2_price=(float(tp2) if want_partial else None),
                    tp2_order_id=(tp2_id if want_partial else None),
                    tp2_qty=(int(tp2_qty) if want_partial else 0),
                    sl_qty=int(quantity),
                )
                self.orders[entry_order_id] = rec

            logger.info(
                "🔧 Exits set for %s | mode=%s sl_id=%s tp_id=%s gtt_id=%s partial=%s",
                entry_order_id, ("GTT" if use_gtt else "REGULAR"), sl_id, tp_id, gtt_id, want_partial
            )
            return True

        except Exception as exc:
            logger.error("💥 setup_gtt_orders failed for %s: %s", entry_order_id, exc, exc_info=True)
            # Attempt to cancel the entry order if exits failed
            self.cancel_order(entry_order_id)
            return False

    def cancel_order(self, order_id: str) -> bool:
        """Cancels a single order by its ID."""
        return self._kite_cancel(order_id)

    # ------------------------------- trailing & exit -------------------------- #

    def _breakeven_price(self, rec: OrderRecord) -> float:
        off = float(self.config.breakeven_offset_ticks) * self.get_tick_size()
        if rec.transaction_type.upper() == "BUY":
            return rec.entry_price + off
        return rec.entry_price - off

    def update_trailing_stop(self, order_id: str, current_price: float, atr: float, atr_multiplier: float) -> None:
        """
        Trailing SL:
        - New SL = price ± (ATR * multiplier), rounded to tick; tighten only.
        - Updates internal record.
        - If using REGULAR exits, modifies the live SL order (price & trigger).
        - If using GTT, log only (cancel & recreate not done automatically here).
        """
        with self._lock:
            rec = self.orders.get(order_id)

        if not rec or not rec.is_open:
            return

        now = time.time()
        if now - rec.last_trail_ts < float(self.config.trail_cooldown_sec):
            return

        if atr is None or atr <= 0 or current_price <= 0:
            return

        tick = self.get_tick_size()

        if rec.transaction_type.upper() == "BUY":
            raw = current_price - atr_multiplier * atr
            new_sl = _round_up_to_tick(raw, tick)
            better = new_sl > rec.stop_loss
        else:
            raw = current_price + atr_multiplier * atr
            new_sl = _round_down_to_tick(raw, tick)
            better = new_sl < rec.stop_loss

        if not better:
            return

        rec.stop_loss = float(new_sl)
        rec.last_trail_ts = now

        if rec.use_gtt:
            logger.info("🧭 GTT trailing noted for %s → %.2f (no auto-modify).", order_id, new_sl)
            return

        if rec.sl_order_id:
            self._kite_modify_sl(rec.sl_order_id, float(new_sl))

    def exit_order(self, order_id: str, exit_reason: str = "manual") -> bool:
        """Market out whatever remains; cancel peer orders."""
        with self._lock:
            rec = self.orders.get(order_id)
        if not rec or not rec.is_open:
            return False

        try:
            qty = int(rec.sl_qty if rec.partial_enabled else rec.quantity)
            if qty <= 0:
                qty = int(rec.quantity)

            side = rec.exit_side or ("SELL" if rec.transaction_type.upper() == "BUY" else "BUY")
            oid = self._place_in_chunks(
                tradingsymbol=rec.symbol,
                exchange=rec.exchange,
                side=side,
                quantity=qty,
                product=self.default_product,
                order_type="MARKET",
                validity=self.default_validity,
            )
            # best-effort cancel peers
            for x in (rec.sl_order_id, rec.tp_order_id, rec.tp1_order_id, rec.tp2_order_id):
                if x:
                    self._kite_cancel(x)

            with self._lock:
                rec.is_open = False
                rec.exit_reason = exit_reason
                rec.exit_price = None  # not guaranteed from here

            logger.info("🏁 Forced exit %s (%s) x%d for %s (oid=%s)", side, exit_reason, qty, rec.symbol, oid)
            return True
        except Exception as exc:
            logger.error("💥 exit_order failed: %s", exc, exc_info=True)
            return False

    # ------------------------------- queries & maintenance -------------------- #

    def get_active_orders(self):
        with self._lock:
            return {k: v for k, v in self.orders.items() if v.is_open}

    def get_positions(self):
        if not self.kite:
            return []
        try:
            pos = self.kite.positions() or {}
            return pos
        except Exception:
            return []

    def cancel_all_orders(self) -> None:
        with self._lock:
            ids = list(self.orders.keys())
        for oid in ids:
            try:
                rec = self.orders.get(oid)
                if not rec:
                    continue
                for x in (rec.sl_order_id, rec.tp_order_id, rec.tp1_order_id, rec.tp2_order_id):
                    if x:
                        self._kite_cancel(x)
                if rec.gtt_id and self.kite:
                    try:
                        self.kite.delete_gtt(rec.gtt_id)
                    except Exception:
                        pass
                rec.is_open = False
            except Exception:
                pass
        logger.info("🧹 cancel_all_orders completed.")

    def get_last_price(self, symbol: str) -> Optional[float]:
        if not self.kite:
            return None
        try:
            q = self.kite.ltp([symbol]) or {}
            lp = q.get(symbol, {}).get("last_price")
            return float(lp) if lp is not None else None
        except Exception:
            return None

    def sync_and_enforce_oco(self) -> List[Tuple[str, float]]:
        """
        Best-effort check to see if any exits imply the pair should be closed or resized.
        - Detect TP1 fill -> shrink SL to remaining qty and hop to breakeven (optional).
        - Detect final TP/SL fill -> mark closed and cancel siblings.
        Return list of (entry_order_id, exit_fill_price).
        """
        fills: List[Tuple[str, float]] = []
        if not self.kite:
            return fills

        status = self._order_status_map()

        with self._lock:
            records = list(self.orders.values())

        for rec in records:
            if not rec.is_open:
                continue

            # --- TP1 filled handling (partial mode) ---
            if rec.partial_enabled and rec.tp1_order_id and not rec.tp1_filled:
                s = status.get(rec.tp1_order_id, {})
                if str(s.get("status", "")).upper() == "COMPLETE":
                    rec.tp1_filled = True
                    remain_qty = max(0, int(rec.quantity - rec.tp1_qty))
                    rec.sl_qty = remain_qty

                    # Cancel old SL and recreate for remaining qty
                    if rec.sl_order_id:
                        self._kite_cancel(rec.sl_order_id)
                        rec.sl_order_id = None

                    new_sl = rec.stop_loss
                    if self.breakeven_after_tp1 and remain_qty > 0:
                        new_sl = self._breakeven_price(rec)

                    tick = self.get_tick_size()
                    trig = _round_to_tick(new_sl, tick)
                    if self.use_slm_exit:
                        ordtype = "SL-M"
                        px = None
                    else:
                        ordtype = "SL"
                        off = max(0, int(self.sl_limit_offset_ticks))
                        if rec.exit_side == "SELL":
                            px = _round_down_to_tick(trig - off * tick, tick)
                        else:
                            px = _round_up_to_tick(trig + off * tick, tick)

                    if remain_qty > 0:
                        rec.sl_order_id = self._place_in_chunks(
                            tradingsymbol=rec.symbol,
                            exchange=rec.exchange,
                            side=rec.exit_side or ("SELL" if rec.transaction_type.upper() == "BUY" else "BUY"),
                            quantity=int(remain_qty),
                            product=self.default_product,
                            order_type=ordtype,
                            validity=self.default_validity,
                            price=(None if self.use_slm_exit else float(px)),
                            trigger_price=float(trig),
                        )
                    logger.info(
                        "♻️  TP1 filled for %s. SL resized to %d and moved to %.2f",
                        rec.order_id, remain_qty, trig
                    )

            # --- Close detection: TP2 complete, (single) TP complete, or SL complete ---
            closed = False
            exit_px: Optional[float] = None
            # prefer SL/TP2 completion as closure signal
            candidate_ids = [rec.tp2_order_id, rec.tp_order_id, rec.sl_order_id]
            # if no partial, tp1_order_id may be the actual TP
            if not rec.partial_enabled and rec.tp1_order_id:
                candidate_ids.insert(0, rec.tp1_order_id)

            for oid in filter(None, candidate_ids):
                s = status.get(oid, {})
                if str(s.get("status", "")).upper() == "COMPLETE":
                    closed = True
                    px = s.get("average_price") or s.get("price")
                    if px:
                        exit_px = float(px)
                    break

            if closed:
                with self._lock:
                    rec.is_open = False
                # best-effort cancel peers
                for x in (rec.sl_order_id, rec.tp_order_id, rec.tp1_order_id, rec.tp2_order_id):
                    if x:
                        self._kite_cancel(x)
                if exit_px is not None:
                    fills.append((rec.order_id, exit_px))

        return fills
