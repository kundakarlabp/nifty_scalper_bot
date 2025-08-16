# src/execution/order_executor.py
"""
Order execution (LIVE + SIM) with:
- Entry placement with freeze-qty chunking
- Bracket-like exits:
    • GTT OCO when allowed (no partials)
    • REGULAR exits with partial TP1/TP2, breakeven hop after TP1
- Trailing stop modes (tighten-only):
    • ATR step (default)
    • CHAND (Chandelier) using ATR multiple off high/low water mark
- Time-based trade expiry (auto close after MAX_TRADE_DURATION_MIN)
- TP2 ratchet (optional): nudge TP2 as trend extends
- Mid-price improvement for limit TPs
- OCO enforcement + fill syncing
- Tight lot integrity & tick rounding

Config keys used:
- DEFAULT_PRODUCT, DEFAULT_ORDER_TYPE, DEFAULT_VALIDITY
- ATR_SL_MULTIPLIER, TICK_SIZE, TRAIL_COOLDOWN_SEC
- PREFERRED_EXIT_MODE  (AUTO | GTT | REGULAR)
- USE_SLM_EXIT, SL_LIMIT_OFFSET_TICKS
- PARTIAL_TP_ENABLE, PARTIAL_TP_RATIO, PARTIAL_TP_USE_MIDPOINT, PARTIAL_TP2_R_MULT
- BREAKEVEN_AFTER_TP1_ENABLE, BREAKEVEN_OFFSET_TICKS
- HARD_STOP_ENABLE, HARD_STOP_GRACE_SEC, HARD_STOP_SLIPPAGE_BPS
- NFO_FREEZE_QTY, NIFTY_LOT_SIZE (75 as per your current setup)
- TRAIL_MODE                (ATR | CHAND)  [VWAP reserved for runner-managed mode]
- TRAIL_CHAND_ATR_MULT      (default 3.0)
- MAX_TRADE_DURATION_MIN    (0 disables)
- EXIT_ON_TIME_EXPIRE       (bool)
- TP2_RATCHET_ENABLE        (bool)
- TP2_RATCHET_STEP_R        (0.5 means every +0.5R)
- MIDPRICE_IMPROVE_TICKS    (ticks to bias TP limit for faster fills)
"""

from __future__ import annotations

import logging
import math
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
    tp_order_id: Optional[str] = None  # used when not partial
    exit_side: Optional[str] = None  # SELL if entry BUY, else BUY

    is_open: bool = True
    open_ts: float = 0.0               # for time-based expiry
    last_trail_ts: float = 0.0
    trailing_step_atr_multiplier: float = 1.5  # override via Config

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

    # --- Trend tracking for trailing / ratchet ---
    high_water: float = 0.0
    low_water: float = 0.0
    one_r: float = 0.0                 # abs(entry - stop) at start
    ratchet_steps_done: int = 0        # how many +kR bumps already applied

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
        self.trail_cooldown = float(_cfg("TRAIL_COOLDOWN_SEC", 8.0))
        self.preferred_exit_mode = str(_cfg("PREFERRED_EXIT_MODE", "REGULAR")).upper()  # AUTO | GTT | REGULAR

        # Trailing modes
        self.trail_mode = str(_cfg("TRAIL_MODE", "ATR")).upper()       # ATR | CHAND
        self.chand_mult = float(_cfg("TRAIL_CHAND_ATR_MULT", 3.0))

        # Time expiry
        self.max_trade_minutes = int(_cfg("MAX_TRADE_DURATION_MIN", 0))  # 0 = disabled
        self.exit_on_time_expire = bool(_cfg("EXIT_ON_TIME_EXPIRE", True))

        # Partial profits
        self.partial_enable = bool(_cfg("PARTIAL_TP_ENABLE", True))
        self.partial_ratio = float(_cfg("PARTIAL_TP_RATIO", 0.5))  # fraction of qty at TP1 (by lots)
        self.partial_use_midpoint = bool(_cfg("PARTIAL_TP_USE_MIDPOINT", True))
        self.partial_tp2_r_mult = float(_cfg("PARTIAL_TP2_R_MULT", 2.0))  # if not using midpoint

        # TP2 ratchet
        self.tp2_ratchet_enable = bool(_cfg("TP2_RATCHET_ENABLE", True))
        self.tp2_ratchet_step_r = float(_cfg("TP2_RATCHET_STEP_R", 0.5))  # every +0.5R advance -> nudge TP2
        self.mid_improve_ticks = int(_cfg("MIDPRICE_IMPROVE_TICKS", 1))

        # Breakeven hop right after TP1
        self.breakeven_after_tp1 = bool(_cfg("BREAKEVEN_AFTER_TP1_ENABLE", True))
        self.breakeven_offset_ticks = int(_cfg("BREAKEVEN_OFFSET_TICKS", 1))

        # Hard stop (failsafe hooks — enforcement stays outside)
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
            if trig_id:
                logger.info("✅ GTT OCO created: trigger_id=%s for %s", trig_id, tradingsymbol)
                return int(trig_id)
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
            logger.info("✏️  Modified SL %s → trig %.2f (%s)", order_id, new_stop, ordertype)
            return True
        except Exception as exc:
            logger.error("💥 modify_order (SL) failed: %s", exc, exc_info=True)
            return False

    def _kite_modify_limit(self, order_id: Optional[str], new_price: float) -> bool:
        """Modify a LIMIT order price (used for TP ratchet)."""
        if not order_id:
            return False
        if not self.kite:
            logger.info("🧪 SIM modify LIMIT(%s) → %.2f", order_id, new_price)
            return True
        try:
            self.kite.modify_order(
                variety=getattr(self.kite, "VARIETY_REGULAR", "regular"),
                order_id=order_id,
                order_type=_cfg("DEFAULT_ORDER_TYPE_EXIT", "LIMIT"),
                price=float(new_price),
                validity=self.default_validity,
            )
            logger.info("✏️  Modified LIMIT %s → %.2f", order_id, new_price)
            return True
        except Exception as exc:
            logger.error("💥 modify_order (LIMIT) failed: %s", exc, exc_info=True)
            return False

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

        # lot integrity guard (lot size 75 by your current env)
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

            exit_side = "SELL" if transaction_type.upper() == "BUY" else "BUY"

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

            # compute 1R
            one_r = abs(entry_price - sl)

            if want_partial:
                # TP2 = strategy TP; TP1 midpoint OR R-multiple based on setting
                if self.partial_use_midpoint:
                    tp2 = tp
                    tp1 = entry_price + (tp2 - entry_price) / 2.0
                else:
                    if transaction_type.upper() == "BUY":
                        tp1 = entry_price + one_r * 1.0
                        tp2 = entry_price + one_r * max(1.0, float(self.partial_tp2_r_mult))
                    else:
                        tp1 = entry_price - one_r * 1.0
                        tp2 = entry_price - one_r * max(1.0, float(self.partial_tp2_r_mult))

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

            # Mid-price improvement for limit TPs
            def _improve_tp(side: str, price: float) -> float:
                if self.mid_improve_ticks <= 0:
                    return _round_to_tick(price, self.tick_size)
                if side.upper() == "SELL":
                    # easier fill: a touch below target
                    return _round_down_to_tick(price - self.mid_improve_ticks * self.tick_size, self.tick_size)
                # BUY exit (for shorts)
                return _round_up_to_tick(price + self.mid_improve_ticks * self.tick_size, self.tick_size)

            if not use_gtt:
                # REGULAR exits
                # --- Stop Loss (full qty to start) ---
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

                # --- Targets ---
                if want_partial and tp1_qty > 0:
                    # TP1
                    tp1_id = self._place_in_chunks(
                        tradingsymbol=symbol,
                        exchange=exchange,
                        side=exit_side,
                        quantity=int(tp1_qty),
                        product=self.default_product,
                        order_type=_cfg("DEFAULT_ORDER_TYPE_EXIT", "LIMIT"),
                        validity=self.default_validity,
                        price=_improve_tp(exit_side, float(tp1)),
                    )
                    # TP2
                    tp2_id = self._place_in_chunks(
                        tradingsymbol=symbol,
                        exchange=exchange,
                        side=exit_side,
                        quantity=int(tp2_qty),
                        product=self.default_product,
                        order_type=_cfg("DEFAULT_ORDER_TYPE_EXIT", "LIMIT"),
                        validity=self.default_validity,
                        price=_improve_tp(exit_side, float(tp2)),
                    )
                else:
                    tp_id = self._place_in_chunks(
                        tradingsymbol=symbol,
                        exchange=exchange,
                        side=exit_side,
                        quantity=int(quantity),
                        product=self.default_product,
                        order_type=_cfg("DEFAULT_ORDER_TYPE_EXIT", "LIMIT"),
                        validity=self.default_validity,
                        price=_improve_tp(exit_side, float(tp)),
                    )

            # record internal
            now = time.time()
            with self._lock:
                rec = OrderRecord(
                    order_id=str(entry_order_id),
                    symbol=symbol,
                    exchange=exchange,
                    transaction_type=transaction_type.upper(),
                    quantity=int(quantity),
                    entry_price=float(entry_price),
                    stop_loss=float(sl),
                    target=float(tp),
                    use_gtt=bool(use_gtt),
                    gtt_id=(int(gtt_id) if gtt_id else None),
                    sl_order_id=sl_id,
                    tp_order_id=tp_id,
                    exit_side=exit_side,
                    trailing_step_atr_multiplier=float(self.atr_mult),
                    partial_enabled=bool(want_partial),
                    tp1_price=(float(tp1) if want_partial else None),
                    tp1_order_id=tp1_id,
                    tp1_qty=int(tp1_qty),
                    tp1_filled=False if want_partial else False,
                    tp2_price=(float(tp2) if want_partial else None),
                    tp2_order_id=tp2_id,
                    tp2_qty=int(tp2_qty),
                    tp2_filled=False if want_partial else False,
                    sl_qty=int(quantity),
                    open_ts=now,
                    high_water=float(entry_price),
                    low_water=float(entry_price),
                    one_r=float(one_r),
                )
                self.orders[str(entry_order_id)] = rec
            return True

        except Exception as exc:
            logger.error("💥 setup_gtt_orders failed: %s", exc, exc_info=True)
            return False

    # ------------------------------- trailing ------------------------------ #

    def update_trailing_stop(self, entry_id: str, current_price: float, atr: float) -> bool:
        """Tighten SL using selected trailing mode; never loosens."""
        with self._lock:
            rec = self.orders.get(str(entry_id))
        if not rec or not rec.is_open:
            return False

        # cooldown
        now = time.time()
        if now - rec.last_trail_ts < max(1.0, float(self.trail_cooldown)):
            return False

        direction_buy = rec.transaction_type == "BUY"
        tick = float(self.tick_size)

        # keep watermarks
        if direction_buy:
            if current_price > rec.high_water:
                rec.high_water = float(current_price)
        else:
            if current_price < rec.low_water:
                rec.low_water = float(current_price)

        # Guard ATR
        if atr is None or atr <= 0:
            atr = abs(rec.entry_price - rec.stop_loss)
            if atr <= 0:
                return False

        new_sl = rec.stop_loss

        if self.trail_mode == "CHAND":
            # Chandelier stop uses an ATR multiple from extreme
            mult = float(self.chand_mult)
            if direction_buy:
                base = rec.high_water - mult * atr
                base = _round_to_tick(base, tick)
                if base > rec.stop_loss:
                    new_sl = base
            else:
                base = rec.low_water + mult * atr
                base = _round_to_tick(base, tick)
                if base < rec.stop_loss:
                    new_sl = base
        else:
            # ATR step trail from current price (standard)
            step = float(self.atr_mult) * float(atr)
            if direction_buy:
                candidate = max(rec.stop_loss, current_price - step)
                candidate = _round_to_tick(candidate, tick)
                if candidate > rec.stop_loss:
                    new_sl = candidate
            else:
                candidate = min(rec.stop_loss, current_price + step)
                candidate = _round_to_tick(candidate, tick)
                if candidate < rec.stop_loss:
                    new_sl = candidate

        if new_sl == rec.stop_loss:
            return False  # no tighten

        # apply to broker SL order (if REGULAR)
        ok = True
        if not rec.use_gtt and rec.sl_order_id:
            ok = self._kite_modify_sl(rec.sl_order_id, new_sl)

        if ok:
            rec.stop_loss = new_sl
            rec.last_trail_ts = now
            logger.info("🔧 Trailed SL for %s → %.2f", entry_id, new_sl)
            return True
        return False

    # --------------------------- manual exit ------------------------------ #

    def exit_order(self, entry_id: str, exit_reason: str = "manual") -> bool:
        """Close the managed position immediately at market; cancel exits."""
        with self._lock:
            rec = self.orders.get(str(entry_id))
        if not rec or not rec.is_open:
            return False

        # cancel exits
        if rec.use_gtt and rec.gtt_id and self.kite:
            try:
                self.kite.delete_gtt(rec.gtt_id)
            except Exception:
                pass
        for oid in [rec.sl_order_id, rec.tp_order_id, rec.tp1_order_id, rec.tp2_order_id]:
            if oid:
                try:
                    self._kite_cancel(oid)
                except Exception:
                    pass

        # market close
        side = "SELL" if rec.transaction_type == "BUY" else "BUY"
        oid = self._place_in_chunks(
            tradingsymbol=rec.symbol,
            exchange=rec.exchange,
            side=side,
            quantity=int(rec.quantity),
            product=self.default_product,
            order_type="MARKET",
            validity=self.default_validity,
        )
        # mark locally
        with self._lock:
            rec.is_open = False
            rec.exit_reason = exit_reason
            # best-effort last price
            lp = self.get_last_price(f"{rec.exchange}:{rec.symbol}") or rec.target or rec.entry_price
            rec.exit_price = float(lp)
        return True

    # ---------------------------- views ---------------------------------- #

    def get_active_orders(self) -> Dict[str, OrderRecord]:
        with self._lock:
            return {k: v for k, v in self.orders.items() if v.is_open}

    def get_positions(self) -> List[Dict[str, Any]]:
        with self._lock:
            pos = []
            for k, rec in self.orders.items():
                if rec.is_open:
                    pos.append(
                        dict(symbol=rec.symbol, side=rec.transaction_type, qty=rec.quantity, entry=rec.entry_price)
                    )
            return pos

    def cancel_all_orders(self) -> None:
        with self._lock:
            ids = list(self.orders.keys())
        for k in ids:
            try:
                self.exit_order(k, exit_reason="cancel_all")
            except Exception:
                pass

    def get_last_price(self, symbol: str) -> Optional[float]:
        """symbol may be 'NFO:XYZ' or 'NSE:NIFTY 50'."""
        if not self.kite:
            return None
        try:
            data = self.kite.ltp([symbol])
            return float(data[symbol]["last_price"])
        except Exception:
            return None

    # --------------------------- OCO enforcement ---------------------------- #

    def _time_expired(self, rec: OrderRecord) -> bool:
        if self.max_trade_minutes <= 0 or not self.exit_on_time_expire:
            return False
        if rec.open_ts <= 0:
            return False
        return (time.time() - rec.open_ts) >= (self.max_trade_minutes * 60)

    def _nudge_tp2_if_needed(self, rec: OrderRecord, current_price: float) -> None:
        """Optional TP2 ratchet as move extends in your favor by +kR steps."""
        if not (rec.partial_enabled and self.tp2_ratchet_enable and rec.tp2_order_id and rec.one_r > 0):
            return
        # how many steps of +kR are in the bag?
        direction_buy = rec.transaction_type == "BUY"
        move = (current_price - rec.entry_price) if direction_buy else (rec.entry_price - current_price)
        if move <= 0:
            return
        steps = int(move / max(1e-9, self.tp2_ratchet_step_r * rec.one_r))
        if steps <= rec.ratchet_steps_done:
            return

        # Move TP2 closer by a small fraction of one step (keep it realistic)
        tick = float(self.tick_size)
        if direction_buy:
            # pull TP2 up to (current - 2 ticks), but never below entry+1R
            new_tp2 = max(rec.entry_price + rec.one_r, current_price - 2 * tick)
            new_tp2 = _round_to_tick(new_tp2, tick)
            if rec.tp2_price and new_tp2 > rec.tp2_price:
                if self._kite_modify_limit(rec.tp2_order_id, new_tp2):
                    rec.tp2_price = new_tp2
                    rec.ratchet_steps_done = steps
                    logger.info("🚀 TP2 ratchet up for %s → %.2f", rec.order_id, new_tp2)
        else:
            # pull TP2 down to (current + 2 ticks), but never above entry-1R
            new_tp2 = min(rec.entry_price - rec.one_r, current_price + 2 * tick)
            new_tp2 = _round_to_tick(new_tp2, tick)
            if rec.tp2_price and new_tp2 < rec.tp2_price:
                if self._kite_modify_limit(rec.tp2_order_id, new_tp2):
                    rec.tp2_price = new_tp2
                    rec.ratchet_steps_done = steps
                    logger.info("🚀 TP2 ratchet down for %s → %.2f", rec.order_id, new_tp2)

    def sync_and_enforce_oco(self) -> List[Tuple[str, float]]:
        """
        Best-effort reconciliation:
        - Time expiry check → optional market exit
        - If TP1 filled: resize SL to remaining qty; optional breakeven hop
        - TP2 ratchet (optional)
        - If TP (no partials) or TP2 filled: close & return (entry_id, fill_px)
        - If SL filled: close & return
        Returns list of closures (entry_id, fill_price)
        """
        closures: List[Tuple[str, float]] = []
        mp = self._order_status_map()

        with self._lock:
            items = list(self.orders.items())

        for entry_id, rec in items:
            if not rec.is_open:
                continue

            # SIM mode: we can't infer fills; runner handles disappearance fallback
            if not self.kite:
                continue

            # Helper to check if an order_id is complete and get price
            def _is_complete(oid: Optional[str]) -> Tuple[bool, Optional[float]]:
                if not oid:
                    return (False, None)
                o = mp.get(str(oid), {})
                st = (o.get("status") or "").upper()
                if st == "COMPLETE":
                    pr = o.get("average_price") or o.get("price") or o.get("filled_price")
                    try:
                        return (True, float(pr))
                    except Exception:
                        return (True, None)
                return (False, None)

            # 0) Time-based expiry (market exit)
            if self._time_expired(rec):
                logger.info("⏳ Time expiry for %s → market exit.", entry_id)
                self.exit_order(entry_id, exit_reason="time_expire")
                continue  # exit_order already closes & records; trader will finalize

            # 1) TP1 logic (REGULAR only)
            if not rec.use_gtt and rec.partial_enabled and rec.tp1_order_id and not rec.tp1_filled:
                ok, px = _is_complete(rec.tp1_order_id)
                if ok:
                    rec.tp1_filled = True
                    # shrink SL qty and possibly breakeven hop
                    rec.sl_qty = max(0, int(rec.quantity - rec.tp1_qty))
                    if rec.sl_order_id and rec.sl_qty > 0:
                        # Cancel & recreate SL for remaining qty
                        try:
                            self._kite_cancel(rec.sl_order_id)
                        except Exception:
                            pass
                        sl_trig = _round_to_tick(rec.stop_loss, self.tick_size)
                        if self.use_slm_exit:
                            sl_ordertype = getattr(self.kite, "ORDER_TYPE_SLM", "SL-M")
                            sl_price = None
                        else:
                            sl_ordertype = getattr(self.kite, "ORDER_TYPE_SL", "SL")
                            off = max(0, int(self.sl_limit_offset_ticks))
                            if rec.exit_side == "SELL":
                                sl_price = _round_down_to_tick(sl_trig - off * self.tick_size, self.tick_size)
                            else:
                                sl_price = _round_up_to_tick(sl_trig + off * self.tick_size, self.tick_size)
                        rec.sl_order_id = self._place_in_chunks(
                            tradingsymbol=rec.symbol,
                            exchange=rec.exchange,
                            side=rec.exit_side,
                            quantity=int(rec.sl_qty),
                            product=self.default_product,
                            order_type=sl_ordertype,
                            validity=self.default_validity,
                            price=(None if self.use_slm_exit else float(sl_price)),
                            trigger_price=float(sl_trig),
                        )

                    # Breakeven hop right after TP1 (tighten-only)
                    if self.breakeven_after_tp1 and rec.sl_order_id:
                        be = _round_to_tick(
                            rec.entry_price + (self.breakeven_offset_ticks * self.tick_size if rec.transaction_type == "BUY"
                                               else -self.breakeven_offset_ticks * self.tick_size),
                            self.tick_size,
                        )
                        if (rec.transaction_type == "BUY" and be > rec.stop_loss) or \
                           (rec.transaction_type != "BUY" and be < rec.stop_loss):
                            if self._kite_modify_sl(rec.sl_order_id, be):
                                rec.stop_loss = be
                                logger.info("↔️  Breakeven hop applied for %s → %.2f", entry_id, be)

            # 2) Optional TP2 ratchet (needs current LTP)
            # Try to get current ltp best-effort
            ltp = self.get_last_price(f"{rec.exchange}:{rec.symbol}")
            if ltp is not None:
                self._nudge_tp2_if_needed(rec, float(ltp))

            # 3) TP2 / Full TP checks
            tp_done = False
            fill_px = None
            if not rec.use_gtt:
                if rec.partial_enabled and rec.tp2_order_id:
                    ok, px = _is_complete(rec.tp2_order_id)
                    if ok:
                        tp_done = True
                        fill_px = px
                elif rec.tp_order_id:
                    ok, px = _is_complete(rec.tp_order_id)
                    if ok:
                        tp_done = True
                        fill_px = px
            # (GTT: we can't inspect legs; runner will treat disappearance as filled)

            if tp_done:
                with self._lock:
                    r = self.orders.get(entry_id)
                    if r:
                        r.is_open = False
                        r.exit_price = float(fill_px or r.target)
                        r.exit_reason = "target"
                # cancel SL (and TP1 if pending)
                for oid in [rec.sl_order_id, rec.tp1_order_id]:
                    if oid:
                        try:
                            self._kite_cancel(oid)
                        except Exception:
                            pass
                closures.append((entry_id, float(fill_px or rec.target)))
                continue

            # 4) SL check
            if not rec.use_gtt and rec.sl_order_id:
                sl_ok, sl_px = _is_complete(rec.sl_order_id)
                if sl_ok:
                    with self._lock:
                        r = self.orders.get(entry_id)
                        if r:
                            r.is_open = False
                            r.exit_price = float(sl_px or r.stop_loss)
                            r.exit_reason = "stop"
                    # cancel any TP legs
                    for oid in [rec.tp_order_id, rec.tp1_order_id, rec.tp2_order_id]:
                        if oid:
                            try:
                                self._kite_cancel(oid)
                            except Exception:
                                pass
                    closures.append((entry_id, float(sl_px or rec.stop_loss)))
                    continue

        return closures