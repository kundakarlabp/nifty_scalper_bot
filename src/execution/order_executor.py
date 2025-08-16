# src/execution/order_executor.py
"""
Order execution module (live + simulation) with:
- Lot-aware partial profit taking (TP1 / TP2) using REGULAR orders
- Optional adaptive "runner" after TP1 (ATR trailing + profit lock)
- Breakeven hop after TP1 (tighten-only, tick-aware)
- Trailing-SL update API (ATR-driven, with profit-lock + step)
- SL-M (default) or SL (limit with protective offset)
- Exchange freeze-quantity chunking
- Safe fallbacks and clear logging in simulation

Public API (used by RealTimeTrader):
- place_entry_order(...)
- setup_gtt_orders(...)            # manages REGULAR exits; GTT used only if partials disabled and allowed
- update_trailing_stop(order_id, current_price, atr)
- exit_order(order_id, exit_reason="manual")
- get_active_orders()
- get_positions()
- cancel_all_orders()
- get_last_price(symbol)
- get_tick_size() -> float
- sync_and_enforce_oco() -> list[(entry_order_id, fill_price)]   # polling fallback (no-op by default)

Notes:
- If PARTIAL_TP_ENABLE=true, exits are managed with REGULAR orders (GTT OCO cannot split partials cleanly).
- After TP1 fill, a typical flow is:
    * shrink SL to remaining quantity
    * optionally move SL to breakeven +/- ticks (tighten-only)
    * if ADAPTIVE_TP_ENABLE=true, TP2 can act as a "runner" (very far cap) while trailing SL controls profit-taking
- This executor does not listen to broker fills. Your RealTimeTrader should correlate fills or use webhooks; the
  provided `sync_and_enforce_oco()` is a best-effort polling hook you can extend if needed.
"""

from __future__ import annotations

import logging
import math
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Dict, Optional, Any, List, Tuple, Callable

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

def _with_retries(fn: Callable, *args, _attempts: int = 2, _sleep: float = 0.6, **kwargs):
    """Very light retry wrapper for broker hiccups."""
    for i in range(max(1, _attempts)):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if i == _attempts - 1:
                raise
            logger.debug("retrying after error: %s", exc)
            time.sleep(_sleep)


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
    tp_order_id: Optional[str] = None      # used only when NOT partials
    exit_side: Optional[str] = None        # SELL if entry BUY, else BUY

    is_open: bool = True
    last_trail_ts: float = 0.0

    # ATR trailing (step & profit lock used by update_trailing_stop)
    trailing_step_atr_multiplier: float = 1.5
    trailing_step_ticks: int = 2
    trailing_profit_lock_bps: float = 30.0  # 30 bps = 0.30%

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
    tp2_is_runner_cap: bool = False        # if True, TP2 acts as far cap while SL trails

    # track SL qty (so we can recreate/resize it after TP1)
    sl_qty: int = 0

    # --- Hard stop bookkeeping (placeholder for higher level enforcement) ---
    breach_ts: Optional[float] = None

    # close info
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None


# ---------------------------- executor ---------------------------------- #

class OrderExecutor:
    """Thin wrapper around order placement and exit management.

    If `kite` is None: simulated mode (no network calls).
    """

    def __init__(self, kite: Optional[Any] = None) -> None:
        self.kite = kite
        self._lock = threading.RLock()
        self.orders: Dict[str, OrderRecord] = {}

        # Defaults (safe fallbacks if Config misses anything)
        self.default_product = _cfg("DEFAULT_PRODUCT", "MIS")
        self.default_order_type = _cfg("DEFAULT_ORDER_TYPE", "MARKET")
        self.default_validity = _cfg("DEFAULT_VALIDITY", "DAY")
        self.atr_mult = float(_cfg("ATR_SL_MULTIPLIER", 1.5))
        self.tick_size = float(_cfg("TICK_SIZE", 0.05))
        self.trail_cooldown = float(_cfg("TRAIL_COOLDOWN_SEC", 10.0))
        self.preferred_exit_mode = str(_cfg("PREFERRED_EXIT_MODE", "REGULAR")).upper()  # AUTO|GTT|REGULAR

        # Partial profits
        self.partial_enable = bool(_cfg("PARTIAL_TP_ENABLE", True))
        self.partial_ratio = float(_cfg("PARTIAL_TP_RATIO", 0.5))  # fraction of qty at TP1 (by lots)
        self.partial_use_midpoint = bool(_cfg("PARTIAL_TP_USE_MIDPOINT", True))
        self.partial_tp2_r_mult = float(_cfg("PARTIAL_TP2_R_MULT", 2.0))  # if not midpoint

        # Adaptive runner after TP1
        self.adaptive_tp_enable = bool(_cfg("ADAPTIVE_TP_ENABLE", True))
        self.runner_cap_r = float(_cfg("RUNNER_TP_CAP_R", 8.0))  # far cap at N*R when using runner
        self.trail_profit_lock_bps = float(_cfg("TRAIL_PROFIT_LOCK_BPS", 30.0))  # 30 bps
        self.trail_step_ticks = int(_cfg("TRAIL_STEP_TICKS", 2))

        # Breakeven hop right after TP1
        self.breakeven_after_tp1 = bool(_cfg("BREAKEVEN_AFTER_TP1_ENABLE", True))
        self.breakeven_offset_ticks = int(_cfg("BREAKEVEN_OFFSET_TICKS", 1))

        # Hard stop (failsafe) – executor exposes hooks; enforcement is at a higher level
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
            def _do():
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
                return str(order_id)

            order_id = _with_retries(_do, _attempts=2, _sleep=0.5)
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
            def _do():
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
                trig_id = None
                if isinstance(resp, dict):
                    trig_id = resp.get("trigger_id") or resp.get("data", {}).get("trigger_id")
                return int(trig_id) if trig_id is not None else None

            trig_id = _with_retries(_do, _attempts=2, _sleep=0.6)
            if trig_id:
                logger.info("✅ GTT OCO created: trigger_id=%s for %s", trig_id, tradingsymbol)
                return trig_id
            logger.info("ℹ️ GTT OCO placed but no trigger_id returned; broker may accept silently.")
            return None
        except Exception as exc:
            logger.info("ℹ️ GTT OCO rejected/unavailable, falling back to REGULAR: %s", exc)
            return None

    def _kite_modify_sl(self, order_id: str, new_stop: float) -> bool:
        if not self.kite:
            logger.info("🧪 SIM modify SL(%s) → %.2f", order_id, new_stop)
            return True
        try:
            def _do():
                ordertype = getattr(self.kite, "ORDER_TYPE_SLM", "SL-M") if self.use_slm_exit \
                            else getattr(self.kite, "ORDER_TYPE_SL", "SL")
                self.kite.modify_order(
                    variety=getattr(self.kite, "VARIETY_REGULAR", "regular"),
                    order_id=order_id,
                    order_type=ordertype,
                    price=(None if ordertype in ("SLM", "SL-M") else float(new_stop)),
                    trigger_price=float(new_stop),
                    validity=self.default_validity,
                )
                return True
            _with_retries(_do, _attempts=2, _sleep=0.5)
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
            def _do():
                self.kite.cancel_order(
                    variety=getattr(self.kite, "VARIETY_REGULAR", "regular"),
                    order_id=order_id,
                )
                return True
            _with_retries(_do, _attempts=2, _sleep=0.4)
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
        Create exits. If partial profits are enabled -> REGULAR with TP1/TP2 (or runner);
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

            side_entry = transaction_type.upper()
            exit_side = "SELL" if side_entry == "BUY" else "BUY"

            # Decide partials
            want_partial = self.partial_enable and quantity >= 2 * max(1, self.lot_size)  # need at least 2 lots

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
            tp2_is_runner_cap = False

            if want_partial:
                # Compute 1R
                r = abs(entry_price - sl)
                if self.partial_use_midpoint:
                    tp2 = tp
                    tp1 = entry_price + (tp2 - entry_price) / 2.0
                else:
                    if side_entry == "BUY":
                        tp1 = entry_price + r * 1.0
                        tp2 = entry_price + r * max(1.0, float(self.partial_tp2_r_mult))
                    else:
                        tp1 = entry_price - r * 1.0
                        tp2 = entry_price - r * max(1.0, float(self.partial_tp2_r_mult))

                # Quantities by lots
                L = max(1, self.lot_size)
                total_lots = _lots_from_qty(quantity, L)
                tp1_lots = max(1, int(round(total_lots * self.partial_ratio)))
                tp2_lots = max(0, int(total_lots - tp1_lots))
                if tp2_lots == 0 and tp1_lots > 1:
                    tp1_lots -= 1
                    tp2_lots = 1
                tp1_qty = _qty_from_lots(tp1_lots, L)
                tp2_qty = _qty_from_lots(tp2_lots, L)

                # Adaptive runner mode: place TP2 as a far cap (e.g., runner_cap_r * R)
                if self.adaptive_tp_enable and tp2_qty > 0:
                    R = max(1e-6, r)
                    runner_cap = (entry_price + self.runner_cap_r * R) if side_entry == "BUY" \
                                 else (entry_price - self.runner_cap_r * R)
                    tp2 = runner_cap
                    tp2_is_runner_cap = True

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
                    sl_price_lim = None
                else:
                    sl_ordertype = getattr(self.kite, "ORDER_TYPE_SL", "SL")
                    off = max(0, int(self.sl_limit_offset_ticks))
                    if exit_side == "SELL":
                        sl_price_lim = _round_down_to_tick(sl_trig - off * self.tick_size, self.tick_size)
                    else:
                        sl_price_lim = _round_up_to_tick(sl_trig + off * self.tick_size, self.tick_size)

                sl_id = self._place_in_chunks(
                    tradingsymbol=symbol,
                    exchange=exchange,
                    side=exit_side,
                    quantity=int(quantity),
                    product=self.default_product,
                    order_type=sl_ordertype if self.kite else ("SL-M" if self.use_slm_exit else "SL"),
                    validity=self.default_validity,
                    price=(None if self.use_slm_exit else float(sl_price_lim)),
                    trigger_price=float(sl_trig),
                )

                # --- Take Profits ---
                ordtype_lim = getattr(self.kite, "ORDER_TYPE_LIMIT", "LIMIT") if self.kite else "LIMIT"

                if want_partial:
                    # TP1
                    tp1_rounded = (_round_down_to_tick(tp1, self.tick_size) if exit_side == "BUY"
                                   else _round_up_to_tick(tp1, self.tick_size))
                    tp1_id = self._place_in_chunks(
                        tradingsymbol=symbol,
                        exchange=exchange,
                        side=exit_side,
                        quantity=int(tp1_qty),
                        product=self.default_product,
                        order_type=ordtype_lim,
                        validity=self.default_validity,
                        price=float(tp1_rounded),
                    )

                    # TP2 (either classic or runner cap)
                    if tp2_qty > 0:
                        tp2_rounded = (_round_down_to_tick(tp2, self.tick_size) if exit_side == "BUY"
                                       else _round_up_to_tick(tp2, self.tick_size))
                        tp2_id = self._place_in_chunks(
                            tradingsymbol=symbol,
                            exchange=exchange,
                            side=exit_side,
                            quantity=int(tp2_qty),
                            product=self.default_product,
                            order_type=ordtype_lim,
                            validity=self.default_validity,
                            price=float(tp2_rounded),
                        )
                else:
                    # Single TP
                    tp_rounded = (_round_down_to_tick(tp, self.tick_size) if exit_side == "BUY"
                                  else _round_up_to_tick(tp, self.tick_size))
                    tp_id = self._place_in_chunks(
                        tradingsymbol=symbol,
                        exchange=exchange,
                        side=exit_side,
                        quantity=int(quantity),
                        product=self.default_product,
                        order_type=ordtype_lim,
                        validity=self.default_validity,
                        price=float(tp_rounded),
                    )

            # record internal
            with self._lock:
                rec = OrderRecord(
                    order_id=entry_order_id,
                    symbol=symbol,
                    exchange=exchange,
                    transaction_type=side_entry,
                    quantity=int(quantity),
                    entry_price=entry_price,
                    stop_loss=sl,
                    target=tp,
                    use_gtt=use_gtt,
                    gtt_id=gtt_id,
                    sl_order_id=sl_id,
                    tp_order_id=tp_id,
                    exit_side=exit_side,
                    trailing_step_atr_multiplier=float(_cfg("ATR_SL_MULTIPLIER", 1.5)),
                    trailing_step_ticks=int(self.trail_step_ticks),
                    trailing_profit_lock_bps=float(self.trail_profit_lock_bps),
                    partial_enabled=bool(want_partial),
                    tp1_price=(float(tp1) if want_partial else None),
                    tp1_order_id=(tp1_id if want_partial else None),
                    tp1_qty=(int(tp1_qty) if want_partial else 0),
                    tp2_price=(float(tp2) if want_partial else None),
                    tp2_order_id=(tp2_id if want_partial else None),
                    tp2_qty=(int(tp2_qty) if want_partial else 0),
                    tp2_is_runner_cap=bool(tp2_is_runner_cap),
                    sl_qty=int(quantity),
                )
                self.orders[entry_order_id] = rec

            logger.info(
                "🔧 Exits set for %s | mode=%s sl_id=%s tp_id=%s tp1=%s tp2=%s partial=%s runner=%s",
                entry_order_id, ("GTT" if use_gtt else "REGULAR"), sl_id, tp_id,
                (tp1_id if want_partial else None), (tp2_id if want_partial else None),
                want_partial, (tp2_is_runner_cap if want_partial else False)
            )
            return True

        except Exception as exc:
            logger.error("💥 setup_gtt_orders failed for %s: %s", entry_order_id, exc, exc_info=True)
            return False

    def update_trailing_stop(self, order_id: str, current_price: float, atr: float) -> None:
        """
        Trailing SL (tighten-only):
        - Base trail = price ± (ATR * multiplier), rounded to tick
        - Profit lock (bps): ensure SL does not give back more than configured fraction
        - Step ticks: move only if SL improves by at least N ticks
        - If using GTT, we only log (no auto modify)
        """
        with self._lock:
            rec = self.orders.get(order_id)

        if not rec or not rec.is_open:
            return

        now = time.time()
        if now - rec.last_trail_ts < self.trail_cooldown:
            return

        mult = float(rec.trailing_step_atr_multiplier or self.atr_mult)
        if atr is None or atr <= 0 or current_price <= 0:
            return

        # Base ATR trail
        if rec.transaction_type == "BUY":
            raw = current_price - mult * atr
            new_sl = _round_up_to_tick(raw, self.tick_size)
            better = new_sl > rec.stop_loss  # tighten only
        else:
            raw = current_price + mult * atr
            new_sl = _round_down_to_tick(raw, self.tick_size)
            better = new_sl < rec.stop_loss

        if not better:
            return

        # Profit lock enforcement (basis points of price)
        lock_bps = float(rec.trailing_profit_lock_bps or 0.0)
        if lock_bps > 0:
            lock_frac = lock_bps / 10000.0
            if rec.transaction_type == "BUY":
                min_lock_sl = current_price * (1.0 - lock_frac)
                new_sl = max(new_sl, _round_up_to_tick(min_lock_sl, self.tick_size))
            else:
                max_lock_sl = current_price * (1.0 + lock_frac)
                new_sl = min(new_sl, _round_down_to_tick(max_lock_sl, self.tick_size))

        # Step filter in ticks
        min_step = int(rec.trailing_step_ticks or 0)
        if min_step > 0:
            step_val = min_step * self.tick_size
            if rec.transaction_type == "BUY":
                if new_sl < rec.stop_loss + step_val:
                    return
            else:
                if new_sl > rec.stop_loss - step_val:
                    return

        # apply
        rec.stop_loss = float(new_sl)
        rec.last_trail_ts = now

        if rec.use_gtt:
            logger.info("🧭 GTT trailing noted for %s → %.2f (no auto-modify).", order_id, new_sl)
            return

        if rec.sl_order_id:
            self._kite_modify_sl(rec.sl_order_id, float(new_sl))

    def exit_order(self, order_id: str, exit_reason: str = "manual") -> bool:
        """
        Market out whatever remains; cancel peer orders.
        """
        with self._lock:
            rec = self.orders.get(order_id)
        if not rec or not rec.is_open:
            return False

        try:
            qty = int(rec.sl_qty if rec.partial_enabled else rec.quantity)
            if qty <= 0:
                qty = int(rec.quantity)

            # send market order on exit side
            side = rec.exit_side or ("SELL" if rec.transaction_type == "BUY" else "BUY")
            oid = self._place_in_chunks(
                tradingsymbol=rec.symbol,
                exchange=rec.exchange,
                side=side,
                quantity=qty,
                product=self.default_product,
                order_type=getattr(self.kite, "ORDER_TYPE_MARKET", "MARKET") if self.kite else "MARKET",
                validity=self.default_validity,
            )
            # best-effort cancel others
            for x in (rec.sl_order_id, rec.tp_order_id, rec.tp1_order_id, rec.tp2_order_id):
                if x:
                    self._kite_cancel(x)

            with self._lock:
                rec.is_open = False
                rec.exit_reason = exit_reason
                rec.exit_price = None  # executor not guaranteed to know fill px here

            logger.info("🏁 Forced exit %s (%s) x%d for %s", side, exit_reason, qty, rec.symbol)
            return True
        except Exception as exc:
            logger.error("💥 exit_order failed: %s", exc, exc_info=True)
            return False

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
        Polling fallback: override or extend this if you wire order webhooks.
        Return list of (entry_order_id, exit_fill_price) when you detect a fill.
        Default: no-op in SIM; rely on RealTimeTrader logic to finalize trades.
        """
        return []