# src/execution/order_executor.py
"""
Order execution module (live + simulation) with bracket-like behavior.

Public API preserved:
- place_entry_order(...)
- setup_gtt_orders(...)
- update_trailing_stop(order_id, current_price, atr)
- exit_order(order_id, exit_reason="manual")
- get_active_orders()
- get_positions()
- cancel_all_orders()

New:
- sync_and_enforce_oco(): polls broker orders; if one exit fills/cancels, cancels the peer
  and closes the internal record. Keeps your book clean even without GTT.

Behavior:
- Tries GTT OCO exits first (if allowed). If not supported/rejected, falls back to
  REGULAR exit orders: SL (ORDER_TYPE_SL) + TP (LIMIT). Your trader can call
  sync_and_enforce_oco periodically for OCO.
- update_trailing_stop() modifies the live SL order (REGULAR path). For GTT, it only
  updates internal state (cancel & recreate would be heavy).
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Dict, Optional, Any, List

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
    quantity: int
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
    trailing_step_atr_multiplier: float = 1.5  # default; can be overridden via Config

    # best-effort status snapshot
    sl_status: str = "PENDING"  # PENDING/FILLED/CANCELLED/REJECTED
    tp_status: str = "PENDING"
    entry_status: str = "PLACED"


class OrderExecutor:
    """
    Thin wrapper around order placement and exit management.

    If `kite` is None: simulated mode (no network calls).
    """

    def __init__(self, kite: Optional[Any] = None) -> None:
        self.kite = kite
        self._lock = threading.RLock()
        self.orders: Dict[str, OrderRecord] = {}

        # Defaults
        self.default_product = _cfg("DEFAULT_PRODUCT", "MIS")
        self.default_order_type = _cfg("DEFAULT_ORDER_TYPE", "MARKET")
        self.default_validity = _cfg("DEFAULT_VALIDITY", "DAY")
        self.atr_mult = float(_cfg("ATR_SL_MULTIPLIER", 1.5))
        self.tick_size = float(_cfg("TICK_SIZE", 0.05))
        self.trail_cooldown = float(_cfg("TRAIL_COOLDOWN_SEC", 12.0))
        self.preferred_exit_mode = str(_cfg("PREFERRED_EXIT_MODE", "AUTO")).upper()  # AUTO | GTT | REGULAR

    # ------------------------------- helpers -------------------------------- #

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
            # Simulate
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
        (Market by default. For limit, pass order_type='LIMIT' and price via your trader path.)
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
        Create exits (TP & SL). Tries GTT OCO first (if preferred/supported), else REGULAR
        exits (TP LIMIT + SL order). Always records an internal OrderRecord.
        """
        try:
            entry_price = self._safe_float(entry_price)
            sl = self._safe_float(stop_loss_price)
            tp = self._safe_float(target_price)

            if entry_price <= 0 or sl <= 0 or tp <= 0:
                logger.warning("⚠️ Invalid SL/TP/entry for exits setup.")
                return False

            exit_side = "SELL" if transaction_type.upper() == "BUY" else "BUY"

            use_gtt = False
            gtt_id = None
            sl_id = None
            tp_id = None

            # choose mode
            mode = self.preferred_exit_mode  # AUTO | GTT | REGULAR
            try_gtt = (mode in ("AUTO", "GTT"))

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
                # SL order (ORDER_TYPE_SL): set both price & trigger
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
                # TP order (LIMIT)
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
                self.orders[entry_order_id] = OrderRecord(
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
                )

            logger.info(
                "🔧 Exits set for %s | mode=%s sl_id=%s tp_id=%s gtt_id=%s",
                entry_order_id, ("GTT" if use_gtt else "REGULAR"), sl_id, tp_id, gtt_id
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

    def sync_and_enforce_oco(self) -> None:
        """
        Check broker order book and enforce OCO for REGULAR exits.
        - If SL filled/cancelled/rejected, cancel TP and close record (and vice versa).
        - If both exits are gone, close record.
        Safe to call periodically (e.g., every 15–30s) from the trader loop.
        """
        with self._lock:
            open_items = [(oid, o) for oid, o in self.orders.items() if o.is_open]

        if not open_items:
            return

        book = self._kite_orders()

        # Build a quick status map for fast lookups
        status_by_id: Dict[str, str] = {}
        if isinstance(book, list):
            for od in book:
                try:
                    oid = str(od.get("order_id") or "")
                    status = (od.get("status") or "").upper()
                    if oid:
                        status_by_id[oid] = status
                except Exception:
                    continue

        for oid, rec in open_items:
            if rec.use_gtt:
                # GTT is OCO at broker; nothing to enforce here.
                continue

            # Probe states
            sl_state = status_by_id.get(rec.sl_order_id or "", "") if rec.sl_order_id else ""
            tp_state = status_by_id.get(rec.tp_order_id or "", "") if rec.tp_order_id else ""

            def _is_inactive(st: str) -> bool:
                st = st.upper()
                return (not st) or (st in ("CANCELLED", "REJECTED", "COMPLETE", "TRIGGER PENDING" and False))

            def _is_filled(st: str) -> bool:
                return st.upper() in ("COMPLETE", "FILLED")

            # If one side filled, cancel the peer
            try:
                if rec.sl_order_id and _is_filled(sl_state):
                    if rec.tp_order_id:
                        self._kite_cancel(rec.tp_order_id)
                        rec.tp_order_id = None
                    rec.sl_status = "FILLED"
                    rec.is_open = False
                    with self._lock:
                        self.orders[oid] = rec
                    logger.info("OCO: SL filled for %s; TP cancelled; record closed.", oid)
                    continue

                if rec.tp_order_id and _is_filled(tp_state):
                    if rec.sl_order_id:
                        self._kite_cancel(rec.sl_order_id)
                        rec.sl_order_id = None
                    rec.tp_status = "FILLED"
                    rec.is_open = False
                    with self._lock:
                        self.orders[oid] = rec
                    logger.info("OCO: TP filled for %s; SL cancelled; record closed.", oid)
                    continue

                # If both are inactive (cancelled/rejected/not present), close
                if (not rec.sl_order_id or _is_inactive(sl_state)) and (not rec.tp_order_id or _is_inactive(tp_state)):
                    rec.sl_status = "CANCELLED" if rec.sl_order_id else rec.sl_status
                    rec.tp_status = "CANCELLED" if rec.tp_order_id else rec.tp_status
                    rec.sl_order_id = None
                    rec.tp_order_id = None
                    rec.is_open = False
                    with self._lock:
                        self.orders[oid] = rec
                    logger.info("OCO: both exits inactive for %s; record closed.", oid)
                    continue
            except Exception as exc:
                logger.debug("OCO sync warning for %s: %s", oid, exc)

    def exit_order(self, order_id: str, exit_reason: str = "manual") -> None:
        """
        Mark order as closed and best-effort cancel open exits (SL/TP or GTT).
        """
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
                if rec.sl_order_id:
                    self._kite_cancel(rec.sl_order_id)
                if rec.tp_order_id:
                    self._kite_cancel(rec.tp_order_id)
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
        """
        Best-effort cancel of all open exit orders/triggers for tracked trades.
        Returns count of cancels attempted.
        """
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
            for ex in ("sl_order_id", "tp_order_id"):
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
