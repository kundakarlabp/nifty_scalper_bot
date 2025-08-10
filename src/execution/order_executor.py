# src/execution/order_executor.py
"""
Order execution module.

- Places entry orders (live or simulated)
- Sets up exits using GTT OCO when supported, else falls back to REGULAR SL+TP orders
- Maintains internal order registry for trailing-stop updates & status
- Provides helpers to cancel/exit and inspect active orders

Public API (unchanged):
    place_entry_order(...)
    setup_gtt_orders(...)
    update_trailing_stop(order_id, current_price, atr)
    exit_order(order_id, exit_reason="manual")
    get_active_orders()
    get_positions()                # passthrough (sim returns [])
    cancel_all_orders()            # cancels live regular exits or simulated orders
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List, Tuple

from src.config import Config

logger = logging.getLogger(__name__)


# ---------- Datamodel ----------

@dataclass
class ExitIds:
    """Holds IDs for the two exit legs when REGULAR exits are used."""
    sl_order_id: Optional[str] = None
    tp_order_id: Optional[str] = None
    gtt_trigger_id: Optional[str] = None  # when GTT OCO is used


@dataclass
class OrderRecord:
    """Internal representation of an open trade."""
    order_id: str
    symbol: str
    exchange: str
    transaction_type: str  # BUY or SELL
    quantity: int
    entry_price: float
    stop_loss: float
    target: float
    trailing_step_atr_multiplier: float
    is_open: bool = True
    exit_ids: ExitIds = field(default_factory=ExitIds)
    exit_mode: str = "AUTO"  # "GTT", "REGULAR", or "AUTO"
    last_trail_ts: float = 0.0


# ---------- Executor ----------

class OrderExecutor:
    """
    If `kite` is None => simulation mode.
    Otherwise, uses the provided KiteConnect client.
    """

    def __init__(self, kite: Optional[Any] = None) -> None:
        self.kite = kite
        self.orders: Dict[str, OrderRecord] = {}

        # Config knobs (with safe defaults)
        self._tick = getattr(Config, "TICK_SIZE", 0.05)
        self._trail_cooldown = getattr(Config, "TRAIL_COOLDOWN_SEC", 5)
        self._preferred_exit_mode = getattr(Config, "PREFERRED_EXIT_MODE", "AUTO").upper()
        self._product = getattr(Config, "DEFAULT_PRODUCT", "MIS")
        self._entry_type = getattr(Config, "DEFAULT_ORDER_TYPE", "MARKET")
        self._validity = getattr(Config, "DEFAULT_VALIDITY", "DAY")
        self._atr_mult = getattr(Config, "ATR_SL_MULTIPLIER", 1.5)

    # ---------- utils ----------

    def _now(self) -> float:
        return time.time()

    def _generate_id(self) -> str:
        return str(uuid.uuid4())

    def _round_to_tick(self, price: float) -> float:
        if self._tick <= 0:
            return float(price)
        # round to nearest tick (exchange increments)
        ticks = round(price / self._tick)
        return float(ticks * self._tick)

    def _cooldown_ok(self, rec: OrderRecord) -> bool:
        return (self._now() - rec.last_trail_ts) >= self._trail_cooldown

    def _reverse_side(self, side: str) -> str:
        return "SELL" if side.upper() == "BUY" else "BUY"

    # ---------- public API ----------

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
        Place the initial entry order. Returns order_id (string) or None.
        In sim mode returns a UUID.
        """
        try:
            if quantity <= 0:
                logger.warning("Attempt to place entry with non-positive quantity: %s", quantity)
                return None

            product = product or self._product
            order_type = order_type or self._entry_type
            validity = validity or self._validity

            if self.kite:
                # PLACE LIVE ORDER
                live_id = self.kite.place_order(
                    tradingsymbol=symbol,
                    exchange=exchange,
                    transaction_type=transaction_type,
                    quantity=quantity,
                    product=product,
                    order_type=order_type,
                    variety="regular",
                    validity=validity,
                )
                logger.info("✅ Live entry placed: %s %s x%d (%s)", transaction_type, symbol, quantity, live_id)
                return str(live_id)
            else:
                # SIMULATED
                sim_id = self._generate_id()
                logger.info("🧪 Sim entry placed: %s %s x%d (%s)", transaction_type, symbol, quantity, sim_id)
                return sim_id

        except Exception as exc:
            logger.error("💥 Entry placement failed for %s: %s", symbol, exc, exc_info=True)
            return None

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
        Create OCO exits. Try GTT first (if supported), else create two REGULAR exit orders.
        Also records the order internally for trailing.
        """
        try:
            # normalize prices to tick
            sl = self._round_to_tick(float(stop_loss_price))
            tp = self._round_to_tick(float(target_price))

            # record first (so we can trail even if exit creation partially fails)
            rec = OrderRecord(
                order_id=entry_order_id,
                symbol=symbol,
                exchange=exchange,
                transaction_type=transaction_type.upper(),
                quantity=int(quantity),
                entry_price=float(entry_price),
                stop_loss=sl,
                target=tp,
                trailing_step_atr_multiplier=float(self._atr_mult),
                exit_mode=self._preferred_exit_mode,
            )
            self.orders[entry_order_id] = rec

            # nothing to do in simulation—just keep internal record:
            if not self.kite:
                logger.info("🧪 Sim exits recorded for %s: SL %.2f | TP %.2f", entry_order_id, sl, tp)
                return True

            # LIVE: choose exit mode
            chosen_mode = self._decide_exit_mode()
            if chosen_mode == "GTT":
                placed = self._try_place_gtt_oco(rec)
                if placed:
                    rec.exit_mode = "GTT"
                    return True
                # fallback to REGULAR if GTT fails
                logger.warning("Falling back to REGULAR exits for %s (GTT failed).", entry_order_id)

            # REGULAR exits
            placed = self._place_regular_exits(rec)
            rec.exit_mode = "REGULAR" if placed else rec.exit_mode
            return placed

        except Exception as exc:
            logger.error("💥 setup_gtt_orders error for %s: %s", entry_order_id, exc, exc_info=True)
            return False

    def update_trailing_stop(self, order_id: str, current_price: float, atr: float) -> None:
        """
        Move SL closer to price if in profit by ATR * multiplier steps.
        - For BUY: new SL = current_price - step
        - For SELL: new SL = current_price + step
        Respects cooldown and tick size. For live REGULAR-exit mode, modifies SL order.
        For GTT mode we only update internal record (modifying GTT legs depends on broker support).
        """
        rec = self.orders.get(order_id)
        if not rec or not rec.is_open:
            return

        if atr is None or atr <= 0:
            return

        if not self._cooldown_ok(rec):
            return

        step = float(atr) * float(rec.trailing_step_atr_multiplier)
        if step <= 0:
            return

        new_sl = rec.stop_loss
        side = rec.transaction_type.upper()

        if side == "BUY":
            cand = self._round_to_tick(float(current_price) - step)
            if cand > rec.stop_loss:
                new_sl = cand
        else:  # SELL
            cand = self._round_to_tick(float(current_price) + step)
            if cand < rec.stop_loss:
                new_sl = cand

        if new_sl == rec.stop_loss:
            return

        # Apply
        old_sl = rec.stop_loss
        rec.stop_loss = new_sl
        rec.last_trail_ts = self._now()

        logger.info("↗️ Trailing SL for %s: %.2f → %.2f (mode=%s)",
                    order_id, old_sl, new_sl, rec.exit_mode)

        if not self.kite:
            # simulation only logs
            return

        if rec.exit_mode == "REGULAR" and rec.exit_ids.sl_order_id:
            # modify SL order price/trigger
            try:
                # Zerodha: SL orders typically need trigger + limit or MARKET
                # Use modify_order with relevant fields
                self.kite.modify_order(
                    variety="regular",
                    order_id=rec.exit_ids.sl_order_id,
                    price=new_sl,
                    trigger_price=new_sl,  # keep equal for simplicity
                )
                logger.debug("Live SL modified for %s to %.2f", order_id, new_sl)
            except Exception as exc:
                logger.warning("Could not modify live SL for %s: %s", order_id, exc)

        # For GTT OCO, modifying leg prices can be done by cancel + re-create,
        # but that risks losing protection in-flight. We keep it conservative.

    def exit_order(self, order_id: str, exit_reason: str = "manual") -> None:
        """
        Mark an order as closed and attempt to cancel existing exits.
        For REGULAR exits, cancel both; for GTT, delete the trigger if we stored it.
        """
        rec = self.orders.get(order_id)
        if not rec or not rec.is_open:
            return

        rec.is_open = False
        logger.info("⏹️ Order %s marked exited (%s).", order_id, exit_reason)

        if not self.kite:
            return

        try:
            if rec.exit_mode == "REGULAR":
                for oid in [rec.exit_ids.sl_order_id, rec.exit_ids.tp_order_id]:
                    if not oid:
                        continue
                    try:
                        self.kite.cancel_order(variety="regular", order_id=oid)
                    except Exception as exc:
                        logger.debug("Cancel regular exit %s failed: %s", oid, exc)
            elif rec.exit_mode == "GTT" and rec.exit_ids.gtt_trigger_id:
                try:
                    self.kite.delete_gtt(rec.exit_ids.gtt_trigger_id)
                except Exception as exc:
                    logger.debug("Delete GTT %s failed: %s", rec.exit_ids.gtt_trigger_id, exc)
        except Exception as exc:
            logger.warning("Cleanup exits for %s encountered issues: %s", order_id, exc)

    def get_active_orders(self) -> Dict[str, OrderRecord]:
        """Return dict of still-open orders (internal registry)."""
        return {oid: o for oid, o in self.orders.items() if o.is_open}

    # ------- Convenience wrappers over Kite (safe in sim) -------

    def get_positions(self) -> List[Dict[str, Any]]:
        if not self.kite:
            return []
        try:
            data = self.kite.positions()
            # Normalize to a flat list of net positions if present
            if isinstance(data, dict) and "net" in data:
                return data["net"]
            return data if isinstance(data, list) else []
        except Exception as exc:
            logger.debug("get_positions failed: %s", exc)
            return []

    def cancel_all_orders(self) -> int:
        """
        Best-effort cancel of all open regular exit orders for tracked trades.
        Returns count of cancels attempted.
        """
        cancelled = 0
        if not self.kite:
            # In sim, just mark exits as cancelled
            for rec in self.orders.values():
                if rec.is_open and rec.exit_mode == "REGULAR":
                    rec.exit_ids.sl_order_id = None
                    rec.exit_ids.tp_order_id = None
                    cancelled += 2
            return cancelled

        for rec in self.orders.values():
            if not rec.is_open:
                continue
            if rec.exit_mode == "REGULAR":
                for oid in [rec.exit_ids.sl_order_id, rec.exit_ids.tp_order_id]:
                    if not oid:
                        continue
                    try:
                        self.kite.cancel_order(variety="regular", order_id=oid)
                        cancelled += 1
                    except Exception as exc:
                        logger.debug("cancel order %s failed: %s", oid, exc)
            elif rec.exit_mode == "GTT" and rec.exit_ids.gtt_trigger_id:
                try:
                    self.kite.delete_gtt(rec.exit_ids.gtt_trigger_id)
                    cancelled += 1
                except Exception as exc:
                    logger.debug("delete gtt %s failed: %s", rec.exit_ids.gtt_trigger_id, exc)

        logger.info("Cancelled %d exit orders/triggers.", cancelled)
        return cancelled

    # ---------- internals for exit placement ----------

    def _decide_exit_mode(self) -> str:
        """
        Decide exit mode based on preference & capabilities.
        Returns "GTT" or "REGULAR".
        """
        pref = self._preferred_exit_mode
        if not self.kite:
            return "REGULAR"  # sim has no real GTT

        if pref == "GTT":
            return "GTT"
        if pref == "REGULAR":
            return "REGULAR"
        # AUTO: try GTT, else fallback
        return "GTT"

    def _try_place_gtt_oco(self, rec: OrderRecord) -> bool:
        """
        Attempt to place Zerodha GTT OCO trigger.
        """
        try:
            # Build order legs for GTT
            exit_side = self._reverse_side(rec.transaction_type)
            orders: List[Dict[str, Any]] = [
                {
                    "transaction_type": exit_side,
                    "quantity": rec.quantity,
                    "product": self._product,
                    "order_type": self._entry_type,  # usually LIMIT or MARKET; many use LIMIT
                    "price": rec.stop_loss,
                },
                {
                    "transaction_type": exit_side,
                    "quantity": rec.quantity,
                    "product": self._product,
                    "order_type": self._entry_type,
                    "price": rec.target,
                },
            ]

            trig = self.kite.place_gtt(
                trigger_type=self.kite.GTT_TYPE_OCO,
                tradingsymbol=rec.symbol,
                exchange=rec.exchange,
                trigger_values=[rec.stop_loss, rec.target],
                last_price=rec.entry_price,
                orders=orders,
            )
            trig_id = None
            if isinstance(trig, dict):
                trig_id = trig.get("trigger_id")

            if trig_id is None:
                logger.warning("GTT OCO returned without trigger_id for %s", rec.order_id)
                return False

            rec.exit_ids.gtt_trigger_id = str(trig_id)
            logger.info("✅ GTT OCO placed for %s (trigger %s)", rec.order_id, rec.exit_ids.gtt_trigger_id)
            return True

        except AttributeError:
            logger.warning("This Kite client has no GTT API methods.")
            return False
        except Exception as exc:
            logger.warning("GTT OCO placement failed for %s: %s", rec.order_id, exc)
            return False

    def _place_regular_exits(self, rec: OrderRecord) -> bool:
        """
        Place two regular exit orders (SL + TP). Caller enforces OCO logic on fills.
        """
        try:
            exit_side = self._reverse_side(rec.transaction_type)

            # SL order: use SL/SL-M (trigger) behavior. With kite, set trigger_price (& price for SL LIMIT).
            sl_id = self.kite.place_order(
                tradingsymbol=rec.symbol,
                exchange=rec.exchange,
                transaction_type=exit_side,
                quantity=rec.quantity,
                product=self._product,
                order_type="SL",           # SL-M sometimes deprecated; SL with same price/trigger acts like market-ish
                trigger_price=rec.stop_loss,
                price=rec.stop_loss,
                variety="regular",
                validity=self._validity,
            )

            # TP order: simple LIMIT
            tp_id = self.kite.place_order(
                tradingsymbol=rec.symbol,
                exchange=rec.exchange,
                transaction_type=exit_side,
                quantity=rec.quantity,
                product=self._product,
                order_type="LIMIT",
                price=rec.target,
                variety="regular",
                validity=self._validity,
            )

            rec.exit_ids.sl_order_id = str(sl_id)
            rec.exit_ids.tp_order_id = str(tp_id)
            logger.info("✅ REGULAR exits placed for %s (SL:%s, TP:%s)", rec.order_id, sl_id, tp_id)
            return True

        except Exception as exc:
            logger.error("REGULAR exits placement failed for %s: %s", rec.order_id, exc, exc_info=True)
            # best effort rollback
            try:
                if rec.exit_ids.sl_order_id:
                    self.kite.cancel_order(variety="regular", order_id=rec.exit_ids.sl_order_id)
            except Exception:
                pass
            try:
                if rec.exit_ids.tp_order_id:
                    self.kite.cancel_order(variety="regular", order_id=rec.exit_ids.tp_order_id)
            except Exception:
                pass
            rec.exit_ids = ExitIds()
            return False