# src/execution/order_executor.py
"""
Order execution module (live + simulation) with bracket-like behavior,
partial profit-taking, breakeven hop after TP1, trailing-SL support, and a hard-stop failsafe.

Public API:
- place_entry_order(...)
- setup_gtt_orders(...)
- update_trailing_stop(order_id, current_price, atr)
- exit_order(order_id, exit_reason="manual")
- get_active_orders()
- get_positions()
- cancel_all_orders()
- get_last_price(symbol)             # for trailing worker
- get_tick_size() -> float           # broker tick size
- sync_and_enforce_oco() -> list     # best-effort peer cancel & fill scan: [(order_id, fill_px)]

Notes:
- If PARTIAL_TP_ENABLE=true, we ALWAYS use REGULAR exits (two TPs + one SL). GTT OCO can’t manage partials cleanly.
- After TP1 fill, we:
    1) Recreate SL for the remaining quantity
    2) Optionally move SL to breakeven (+offset) immediately (tighten-only)
- Trailing uses update_trailing_stop() and modifies the REGULAR SL order for remaining qty.
- Hard stop: If price breaches SL and no fill occurs within a grace window, we force a market exit of remaining qty.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Dict, Optional, Any, List, Tuple

from src.config import Config

logger = logging.getLogger(__name__)


def _cfg(name: str, default: Any) -> Any:
    return getattr(Config, name, default)


def _round_to_tick(x: float, tick: float) -> float:
    if tick <= 0:
        return float(x)
    return round(float(x) / tick) * tick


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

    # --- Hard stop bookkeeping ---
    breach_ts: Optional[float] = None  # first time price seen beyond SL without SL fill


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
        self.trail_cooldown = float(_cfg("TRAIL_COOLDOWN_SEC", 12.0))
        self.preferred_exit_mode = str(_cfg("PREFERRED_EXIT_MODE", "AUTO")).upper()  # AUTO | GTT | REGULAR

        # Partial profits
        self.partial_enable = bool(_cfg("PARTIAL_TP_ENABLE", True))
        self.partial_ratio = float(_cfg("PARTIAL_TP_RATIO", 0.5))  # fraction of qty at TP1
        self.partial_use_midpoint = bool(_cfg("PARTIAL_TP_USE_MIDPOINT", True))
        self.partial_tp2_r_mult = float(_cfg("PARTIAL_TP2_R_MULT", 2.0))  # if not using midpoint

        # Breakeven hop right after TP1
        self.breakeven_after_tp1 = bool(_cfg("BREAKEVEN_AFTER_TP1_ENABLE", True))
        # move SL to entry + N ticks (BUY) or entry - N ticks (SELL)
        self.breakeven_offset_ticks = int(_cfg("BREAKEVEN_OFFSET_TICKS", 1))

        # Hard stop (failsafe)
        self.hard_stop_enable = bool(_cfg("HARD_STOP_ENABLE", True))
        self.hard_stop_grace_sec = float(_cfg("HARD_STOP_GRACE_SEC", 3.0))
        self.hard_stop_slip_bps = float(_cfg("HARD_STOP_SLIPPAGE_BPS", _cfg("SLIPPAGE_BPS", 5.0)))

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
            logger.info("🧪 SIM order: %s %s x%d @ %s (price=%s trig=%s)",
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
                    "quantity": quantity,
                    "product": self.default_product,
                    "order_type": _cfg("DEFAULT_ORDER_TYPE_EXIT", "LIMIT"),
                    "price": float(sl_price),
                },
                {
                    "transaction_type": exit_side,
                    "quantity": quantity,
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
            logger.info("✅ GTT OCO placed (no trigger_id returned).")
            return None
        except Exception as exc:
            logger.warning("GTT OCO failed (fallback to REGULAR exits): %s", exc)
            return None

    def _kite_modify_sl(self, order_id: str, new_stop: float) -> bool:
        if not self.kite:
            logger.info("🧪 SIM modify SL(%s) → %.2f", order_id, new_stop)
            return True
        try:
            self.kite.modify_order(
                variety=getattr(self.kite, "VARIETY_REGULAR", "regular"),
                order_id=order_id,
                order_type=getattr(self.kite, "ORDER_TYPE_SL", "SL"),
                price=float(new_stop),
                trigger_price=float(new_stop),
                validity=self.default_validity,
            )
            logger.info("✏️  Modified SL %s → %.2f", order_id, new_stop)
            return True
        except Exception as exc:
            logger.error("💥 modify_order (SL) failed: %s", exc, exc_info=True)
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

    def _kite_orders(self) -> List[Dict[str, Any]]:
        if not self.kite:
            return []
        try:
            return self.kite.orders() or []
        except Exception:
            return []

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

        product = product or self.default_product
        order_type = order_type or self.default_order_type
        validity = validity or self.default_validity

        oid = self._kite_place_order(
            tradingsymbol=symbol,
            exchange=exchange,
            transaction_type=transaction_type.upper(),
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

            if want_partial:
                # Compute 1R
                r = abs(entry_price - sl)
                if self.partial_use_midpoint:
                    # TP2 is strategy TP; TP1 is midpoint between entry and TP2
                    # Handles BUY/SELL symmetrically
                    tp2 = tp
                    tp1 = entry_price + (tp2 - entry_price) / 2.0
                else:
                    # R-multiples
                    if transaction_type.upper() == "BUY":
                        tp1 = entry_price + r * 1.0
                        tp2 = entry_price + r * max(1.0, float(self.partial_tp2_r_mult))
                    else:
                        tp1 = entry_price - r * 1.0
                        tp2 = entry_price - r * max(1.0, float(self.partial_tp2_r_mult))

                # Quantities
                tp1_qty = max(1, int(round(quantity * self.partial_ratio)))
                tp2_qty = max(0, int(quantity - tp1_qty))
                if tp2_qty == 0 and tp1_qty > 1:
                    tp1_qty -= 1
                    tp2_qty = 1

            mode = self.preferred_exit_mode  # AUTO | GTT | REGULAR
            try_gtt = (mode in ("AUTO", "GTT")) and not want_partial

            if try_gtt:
                gtt_id = self._kite_place_gtt_oco(
                    tradingsymbol=symbol,
                    exchange=exchange,
                    last_price=entry_price,
                    exit_side=exit_side,
                    quantity=int(quantity),
                    sl_price=sl,
                    tp_price=tp,
                )
                use_gtt = gtt_id is not None

            if not use_gtt:
                # REGULAR exits
                # SL (initially for full qty; will be resized after TP1)
                sl_rounded = _round_to_tick(sl, self.tick_size)
                sl_id = self._kite_place_order(
                    tradingsymbol=symbol,
                    exchange=exchange,
                    transaction_type=exit_side,
                    quantity=int(quantity),
                    product=self.default_product,
                    order_type=getattr(self.kite, "ORDER_TYPE_SL", "SL") if self.kite else "SL",
                    validity=self.default_validity,
                    price=sl_rounded,
                    trigger_price=sl_rounded,
                )

                if want_partial:
                    tp1_rounded = _round_to_tick(float(tp1), self.tick_size)
                    tp2_rounded = _round_to_tick(float(tp2), self.tick_size)

                    tp1_id = self._kite_place_order(
                        tradingsymbol=symbol,
                        exchange=exchange,
                        transaction_type=exit_side,
                        quantity=int(tp1_qty),
                        product=self.default_product,
                        order_type=getattr(self.kite, "ORDER_TYPE_LIMIT", "LIMIT") if self.kite else "LIMIT",
                        validity=self.default_validity,
                        price=tp1_rounded,
                    )
                    tp2_id = self._kite_place_order(
                        tradingsymbol=symbol,
                        exchange=exchange,
                        transaction_type=exit_side,
                        quantity=int(tp2_qty),
                        product=self.default_product,
                        order_type=getattr(self.kite, "ORDER_TYPE_LIMIT", "LIMIT") if self.kite else "LIMIT",
                        validity=self.default_validity,
                        price=tp2_rounded,
                    )
                else:
                    # Single TP
                    tp_rounded = _round_to_tick(tp, self.tick_size)
                    tp_id = self._kite_place_order(
                        tradingsymbol=symbol,
                        exchange=exchange,
                        transaction_type=exit_side,
                        quantity=int(quantity),
                        product=self.default_product,
                        order_type=getattr(self.kite, "ORDER_TYPE_LIMIT", "LIMIT") if self.kite else "LIMIT",
                        validity=self.default_validity,
                        price=tp_rounded,
                    )

            # record internal
            with self._lock:
                rec = OrderRecord(
                    order_id=entry_order_id,
                    symbol=symbol,
                    exchange=exchange,
                    transaction_type=transaction_type.upper(),
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
            return False

    def update_trailing_stop(self, order_id: str, current_price: float, atr: float) -> None:
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
        if now - rec.last_trail_ts < self.trail_cooldown:
            return  # cooldown

        try:
            step = max(0.0, float(atr)) * float(rec.trailing_step_atr_multiplier)
            if step <= 0:
                return

            if rec.transaction_type == "BUY":
                desired_sl = _round_to_tick(float(current_price) - step, self.tick_size)
                if desired_sl <= rec.stop_loss:
                    return
            else:
                desired_sl = _round_to_tick(float(current_price) + step, self.tick_size)
                if desired_sl >= rec.stop_loss:
                    return

            before = rec.stop_loss
            rec.stop_loss = desired_sl
            rec.last_trail_ts = now
            with self._lock:
                self.orders[order_id] = rec

            if not rec.use_gtt and rec.sl_order_id:
                ok = self._kite_modify_sl(rec.sl_order_id, desired_sl)
                if not ok:
                    logger.warning("Trailing: failed to modify live SL (%s)", rec.sl_order_id)
            else:
                logger.debug("Trailing (GTT mode): internal SL %.2f → %.2f (no live modify).", before, desired_sl)

            logger.info("Trailing SL %s: %.2f → %.2f (Px=%.2f, ATR=%.2f, mode=%s)",
                        order_id, before, desired_sl, float(current_price), float(atr),
                        "GTT" if rec.use_gtt else "REGULAR")

        except Exception as exc:
            logger.error("💥 trailing update failed for %s: %s", order_id, exc, exc_info=True)

    def exit_order(self, order_id: str, exit_reason: str = "manual") -> None:
        """Mark order as closed and best-effort cancel open exits (SL/TP or GTT)."""
        with self._lock:
            rec = self.orders.get(order_id)
        if not rec:
            logger.warning("⚠️ exit_order on unknown ID: %s", order_id)
            return
        if not rec.is_open:
            return

        try:
            if rec.use_gtt and rec.gtt_id is not None and self.kite:
                try:
                    self.kite.delete_gtt(rec.gtt_id)
                    logger.info("🧹 Deleted GTT trigger_id=%s for %s", rec.gtt_id, rec.symbol)
                except TypeError:
                    self.kite.delete_gtt(trigger_id=rec.gtt_id)
            else:
                for ex_id in (rec.sl_order_id, rec.tp_order_id, rec.tp1_order_id, rec.tp2_order_id):
                    if ex_id:
                        self._kite_cancel(ex_id)
        except Exception as exc:
            logger.warning("exit_order: broker cleanup warning: %s", exc)

        with self._lock:
            rec.is_open = False
            self.orders[order_id] = rec

        logger.info("⏹️ Order exited (%s). ID=%s %s %s x%d",
                    exit_reason, order_id, rec.transaction_type, rec.symbol, rec.quantity)

    def get_active_orders(self) -> Dict[str, OrderRecord]:
        with self._lock:
            return {oid: o for oid, o in self.orders.items() if o.is_open}

    def get_positions(self) -> List[Dict[str, Any]]:
        if self.kite:
            try:
                pos = self.kite.positions()
                net = pos.get("net", []) if isinstance(pos, dict) else []
                results: List[Dict[str, Any]] = []
                for p in net:
                    try:
                        results.append(
                            {
                                "tradingsymbol": p.get("tradingsymbol"),
                                "quantity": int(p.get("quantity", 0)),
                                "average_price": float(p.get("average_price", 0) or 0),
                                "last_price": float(p.get("last_price", 0) or 0),
                                "pnl": float(p.get("pnl", 0) or 0),
                            }
                        )
                    except Exception:
                        continue
                return results
            except Exception as exc:
                logger.warning("Positions fetch failed, using simulated view: %s", exc)

        with self._lock:
            res: List[Dict[str, Any]] = []
            for rec in self.orders.values():
                if not rec.is_open:
                    continue
                res.append(
                    {
                        "tradingsymbol": rec.symbol,
                        "quantity": rec.quantity if rec.transaction_type == "BUY" else -rec.quantity,
                        "average_price": rec.entry_price,
                        "last_price": rec.entry_price,
                        "pnl": 0.0,
                    }
                )
            return res

    def cancel_all_orders(self) -> int:
        """Best-effort cancel of all open exit orders/triggers for tracked trades."""
        cancelled = 0
        with self._lock:
            items = list(self.orders.items())

        for oid, rec in items:
            if not rec.is_open:
                continue

            if rec.use_gtt:
                if rec.gtt_id is not None:
                    if not self.kite:
                        cancelled += 1
                        rec.gtt_id = None
                        with self._lock:
                            self.orders[oid] = rec
                        continue
                    try:
                        self.kite.delete_gtt(rec.gtt_id)
                        cancelled += 1
                        rec.gtt_id = None
                        with self._lock:
                            self.orders[oid] = rec
                    except Exception as exc:
                        logger.debug("delete_gtt %s failed: %s", rec.gtt_id, exc)
                continue

            # REGULAR exits
            for ex in ("sl_order_id", "tp_order_id", "tp1_order_id", "tp2_order_id"):
                ex_id = getattr(rec, ex)
                if not ex_id:
                    continue
                if not self.kite:
                    cancelled += 1
                    setattr(rec, ex, None)
                    with self._lock:
                        self.orders[oid] = rec
                    continue
                try:
                    self._kite_cancel(ex_id)
                    cancelled += 1
                    setattr(rec, ex, None)
                    with self._lock:
                        self.orders[oid] = rec
                except Exception as exc:
                    logger.debug("cancel exit %s failed: %s", ex_id, exc)

        logger.info("Cancelled %d exit orders/triggers.", cancelled)
        return cancelled

    # ---------- helpers for trailing/oco workers & hard stop ---------- #

    def get_last_price(self, symbol: str) -> Optional[float]:
        """Return LTP for 'symbol' (None if not available)."""
        if not symbol:
            return None
        if not self.kite:
            return None
        try:
            data = self.kite.ltp([symbol])
            p = data.get(symbol, {}).get("last_price")
            return float(p) if p is not None else None
        except Exception:
            return None

    def _force_market_exit(self, rec: OrderRecord, reason: str) -> None:
        """Place market exit for remaining qty, cancel exits, and close."""
        remaining = rec.quantity
        if rec.partial_enabled:
            if rec.tp1_filled:
                remaining -= rec.tp1_qty
            if rec.tp2_filled:
                remaining -= rec.tp2_qty
        if remaining <= 0:
            remaining = rec.quantity  # conservative fallback

        logger.warning("⚠️ Hard stop: forcing market exit of %d on %s (%s)", remaining, rec.symbol, reason)
        self._kite_place_order(
            tradingsymbol=rec.symbol,
            exchange=rec.exchange,
            transaction_type=rec.exit_side,
            quantity=int(remaining),
            product=self.default_product,
            order_type=getattr(self.kite, "ORDER_TYPE_MARKET", "MARKET") if self.kite else "MARKET",
            validity=self.default_validity,
        )
        # Best-effort cleanup
        for ex_id in (rec.sl_order_id, rec.tp_order_id, rec.tp1_order_id, rec.tp2_order_id):
            if ex_id:
                try:
                    self._kite_cancel(ex_id)
                except Exception:
                    pass
        rec.is_open = False

    def _apply_breakeven_after_tp1(self, rec: OrderRecord) -> None:
        """Move SL to breakeven (+/- offset) immediately after TP1 fill (tighten-only)."""
        if not self.breakeven_after_tp1 or not rec.sl_order_id:
            return

        offset = self.breakeven_offset_ticks * self.tick_size
        if rec.transaction_type == "BUY":
            be = _round_to_tick(rec.entry_price + offset, self.tick_size)
            if be > rec.stop_loss:
                if self._kite_modify_sl(rec.sl_order_id, be):
                    logger.info("🔒 Breakeven hop (BUY): SL %.2f → %.2f", rec.stop_loss, be)
                    rec.stop_loss = be
        else:
            be = _round_to_tick(rec.entry_price - offset, self.tick_size)
            if be < rec.stop_loss:
                if self._kite_modify_sl(rec.sl_order_id, be):
                    logger.info("🔒 Breakeven hop (SELL): SL %.2f → %.2f", rec.stop_loss, be)
                    rec.stop_loss = be

    def sync_and_enforce_oco(self) -> List[Tuple[str, float]]:
        """
        Best-effort detection of fills, peer-cancel enforcement, partial TP handling,
        breakeven hop after TP1, and hard-stop if SL is breached without fill.

        Returns a list of finalized entries as [(entry_order_id, fill_price)].
        """
        if not self.kite:
            return []

        try:
            broker_orders = self._kite_orders()
            closed: List[Tuple[str, float]] = []

            with self._lock:
                for entry_id, rec in list(self.orders.items()):
                    if not rec.is_open:
                        continue

                    # Build order status map for fast lookups
                    status_map: Dict[str, Dict[str, Any]] = {}
                    for bo in broker_orders:
                        oid = str(bo.get("order_id", ""))
                        status_map[oid] = bo

                    def _is_complete(oid: Optional[str]) -> Tuple[bool, Optional[float]]:
                        if not oid:
                            return (False, None)
                        bo = status_map.get(str(oid), {})
                        status = (bo.get("status") or "").lower()
                        avg = bo.get("average_price", None)
                        return (status == "complete", (float(avg) if avg else None))

                    # ---------- REGULAR path ----------
                    if not rec.use_gtt:
                        sl_done, sl_px = _is_complete(rec.sl_order_id)

                        # Single-TP mode
                        if not rec.partial_enabled:
                            tp_done, tp_px = _is_complete(rec.tp_order_id)
                            if sl_done or tp_done:
                                try:
                                    if sl_done and rec.tp_order_id:
                                        self._kite_cancel(rec.tp_order_id)
                                    if tp_done and rec.sl_order_id:
                                        self._kite_cancel(rec.sl_order_id)
                                except Exception:
                                    pass
                                rec.is_open = False
                                self.orders[entry_id] = rec
                                closed.append((entry_id, float(tp_px if tp_done and tp_px else rec.stop_loss)))
                                continue

                        # Partial-TP mode
                        else:
                            tp1_done, tp1_px = _is_complete(rec.tp1_order_id)
                            tp2_done, tp2_px = _is_complete(rec.tp2_order_id)

                            # TP1 first completion
                            if tp1_done and not rec.tp1_filled:
                                rec.tp1_filled = True
                                # Cancel old full-qty SL, create new SL for remaining
                                try:
                                    if rec.sl_order_id:
                                        self._kite_cancel(rec.sl_order_id)
                                except Exception:
                                    pass
                                remaining = max(0, rec.quantity - rec.tp1_qty)
                                if remaining > 0:
                                    sl_rounded = _round_to_tick(rec.stop_loss, self.tick_size)
                                    rec.sl_order_id = self._kite_place_order(
                                        tradingsymbol=rec.symbol,
                                        exchange=rec.exchange,
                                        transaction_type=rec.exit_side,
                                        quantity=int(remaining),
                                        product=self.default_product,
                                        order_type=getattr(self.kite, "ORDER_TYPE_SL", "SL"),
                                        validity=self.default_validity,
                                        price=sl_rounded,
                                        trigger_price=sl_rounded,
                                    )
                                    rec.sl_qty = int(remaining)

                                    # Immediate breakeven hop (tighten-only)
                                    self._apply_breakeven_after_tp1(rec)

                                self.orders[entry_id] = rec

                            # If SL completes -> cancel remaining TPs and close
                            if sl_done:
                                try:
                                    for ex in (rec.tp1_order_id, rec.tp2_order_id):
                                        if ex:
                                            self._kite_cancel(ex)
                                except Exception:
                                    pass
                                rec.is_open = False
                                self.orders[entry_id] = rec
                                closed.append((entry_id, float(sl_px) if sl_px else rec.stop_loss))
                                continue

                            # If TP2 completes -> cancel SL and any other TP, then close
                            if tp2_done and not rec.tp2_filled:
                                rec.tp2_filled = True
                                try:
                                    if rec.sl_order_id:
                                        self._kite_cancel(rec.sl_order_id)
                                    if rec.tp1_order_id and not rec.tp1_filled:
                                        self._kite_cancel(rec.tp1_order_id)
                                except Exception:
                                    pass
                                rec.is_open = False
                                self.orders[entry_id] = rec
                                closed.append((entry_id, float(tp2_px) if tp2_px else rec.target))
                                continue

                    # ---------- HARD STOP (failsafe) ----------
                    if self.hard_stop_enable and rec.is_open and not rec.use_gtt:
                        try:
                            ltp = self.get_last_price(rec.symbol)
                        except Exception:
                            ltp = None
                        if ltp is not None and rec.sl_order_id:
                            breached = False
                            if rec.transaction_type == "BUY":
                                breached = ltp <= rec.stop_loss
                            else:
                                breached = ltp >= rec.stop_loss
                            if breached:
                                if rec.breach_ts is None:
                                    rec.breach_ts = time.time()
                                elif (time.time() - rec.breach_ts) >= self.hard_stop_grace_sec:
                                    self._force_market_exit(rec, "SL breach no fill")
                                    self.orders[entry_id] = rec
                                    closed.append((entry_id, float(ltp)))
                                    continue
                            else:
                                rec.breach_ts = None
                                self.orders[entry_id] = rec

            return closed

        except Exception as exc:
            logger.debug("sync_and_enforce_oco error: %s", exc)
            return []
