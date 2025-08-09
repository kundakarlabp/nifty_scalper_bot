# src/execution/order_executor.py
"""
Order execution module.

Public API preserved:
- place_entry_order(...)
- setup_gtt_orders(...)
- update_trailing_stop(order_id, current_price, atr)
- exit_order(order_id, exit_reason="manual")
- get_active_orders()
- get_positions()
- cancel_all_orders()

Live mode: uses Zerodha KiteConnect (if provided).
Sim mode: generates UUID order IDs and tracks them in-memory.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Any, List

from src.config import Config

logger = logging.getLogger(__name__)


@dataclass
class OrderRecord:
    """Internal representation of a trade (entry + managed exits)."""
    order_id: str
    symbol: str
    exchange: str
    transaction_type: str  # BUY or SELL
    quantity: int
    entry_price: float
    stop_loss: float
    target: float
    trailing_step_atr_multiplier: float  # Multiplier to convert ATR → step size
    is_open: bool = True
    gtt_id: Optional[int] = None  # Stored when Kite returns one
    notes: str = ""


class OrderExecutor:
    """
    Thin wrapper around order placement and GTT management.

    If `kite` is None: all actions are simulated, no network calls.
    """

    def __init__(self, kite: Optional[Any] = None) -> None:
        self.kite = kite
        self._lock = threading.RLock()
        self.orders: Dict[str, OrderRecord] = {}

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
        symbol: str,
        exchange: str,
        transaction_type: str,
        quantity: int,
        product: str,
        order_type: str,
        validity: str,
    ) -> Optional[str]:
        """Place a market order via Kite; return order_id or None."""
        try:
            order_id = self.kite.place_order(
                tradingsymbol=symbol,
                exchange=exchange,
                transaction_type=transaction_type.upper(),
                quantity=int(quantity),
                product=product,
                order_type=order_type,
                variety="regular",
                validity=validity,
            )
            # Some Kite versions return dicts; normalize to str
            if isinstance(order_id, dict):
                order_id = order_id.get("order_id") or order_id.get("data", {}).get("order_id")
            order_id = str(order_id)
            logger.info(
                "✅ Order placed via Kite. ID=%s, %s %s x%d @ %s",
                order_id, transaction_type.upper(), symbol, quantity, order_type
            )
            return order_id
        except Exception as exc:
            logger.error("💥 Kite place_order failed: %s", exc, exc_info=True)
            return None

    def _kite_place_gtt_oco(
        self,
        symbol: str,
        exchange: str,
        entry_price: float,
        stop_loss_price: float,
        target_price: float,
        exit_transaction_type: str,
        quantity: int,
    ) -> Optional[int]:
        """
        Place OCO GTT with best-effort compatibility across Kite versions.
        Return gtt trigger_id when available, else None.
        """
        try:
            # Build legs
            legs: List[Dict[str, Any]] = [
                {
                    "transaction_type": exit_transaction_type,
                    "quantity": quantity,
                    "product": Config.DEFAULT_PRODUCT,
                    "order_type": Config.DEFAULT_ORDER_TYPE,
                    "price": stop_loss_price,
                },
                {
                    "transaction_type": exit_transaction_type,
                    "quantity": quantity,
                    "product": Config.DEFAULT_PRODUCT,
                    "order_type": Config.DEFAULT_ORDER_TYPE,
                    "price": target_price,
                },
            ]

            trigger_vals = [stop_loss_price, target_price]

            # Some SDKs expose constants; otherwise fall back to string
            trigger_type = getattr(self.kite, "GTT_TYPE_OCO", "two-leg")

            # Try the canonical signature
            try:
                resp = self.kite.place_gtt(
                    trigger_type=trigger_type,
                    tradingsymbol=symbol,
                    exchange=exchange,
                    trigger_values=trigger_vals,
                    last_price=entry_price,
                    orders=legs,
                )
            except TypeError:
                # Older/newer signature variants
                resp = self.kite.place_gtt(
                    trigger_type=trigger_type,
                    tradingsymbol=symbol,
                    exchange=exchange,
                    trigger_values=trigger_vals,
                    orders=legs,
                )

            # Extract trigger_id if present
            gtt_id = None
            if isinstance(resp, dict):
                gtt_id = resp.get("trigger_id") or resp.get("data", {}).get("trigger_id")
            if gtt_id is not None:
                logger.info("✅ GTT OCO placed. trigger_id=%s for %s", gtt_id, symbol)
            else:
                logger.info("✅ GTT OCO placed (no trigger_id returned).")

            return int(gtt_id) if gtt_id is not None else None

        except Exception as exc:
            logger.error("💥 Kite place_gtt failed: %s", exc, exc_info=True)
            return None

    # ------------------------------- public API ------------------------------ #

    def place_entry_order(
        self,
        symbol: str,
        exchange: str,
        transaction_type: str,
        quantity: int,
        product: str = Config.DEFAULT_PRODUCT,
        order_type: str = Config.DEFAULT_ORDER_TYPE,
        validity: str = Config.DEFAULT_VALIDITY,
    ) -> Optional[str]:
        """
        Place the initial entry order. Returns order_id or None.
        """
        if quantity <= 0:
            logger.warning("Attempted to place order with non-positive quantity: %s", quantity)
            return None

        if self.kite:
            order_id = self._kite_place_order(
                symbol, exchange, transaction_type, quantity, product, order_type, validity
            )
            if order_id:
                return order_id
            # fall through to simulation on failure

        # Simulated entry
        order_id = self._generate_order_id()
        logger.info(
            "🧪 Simulated entry placed. ID=%s, %s %s x%d",
            order_id, transaction_type.upper(), symbol, quantity
        )
        return order_id

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
        Create OCO exits (SL & TP). Always records an internal OrderRecord.
        Returns True if internal bookkeeping succeeded (even if live GTT failed).
        """
        try:
            entry_price = self._safe_float(entry_price)
            sl = self._safe_float(stop_loss_price)
            tp = self._safe_float(target_price)

            if entry_price <= 0 or sl <= 0 or tp <= 0:
                logger.warning("⚠️ Invalid SL/TP/entry for GTT setup.")
                return False

            exit_type = "SELL" if transaction_type.upper() == "BUY" else "BUY"

            gtt_id: Optional[int] = None
            if self.kite:
                gtt_id = self._kite_place_gtt_oco(
                    symbol=symbol,
                    exchange=exchange,
                    entry_price=entry_price,
                    stop_loss_price=sl,
                    target_price=tp,
                    exit_transaction_type=exit_type,
                    quantity=quantity,
                )

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
                    trailing_step_atr_multiplier=getattr(Config, "ATR_SL_MULTIPLIER", 1.5),
                    gtt_id=gtt_id,
                )

            logger.debug("📝 internal order recorded: %s", entry_order_id)
            return True

        except Exception as exc:
            logger.error("💥 setup_gtt_orders failed for %s: %s", entry_order_id, exc, exc_info=True)
            return False

    def update_trailing_stop(self, order_id: str, current_price: float, atr: float) -> None:
        """
        Move SL in the direction of profit using ATR * multiplier.
        Only updates in-memory state (and logs); live GTT modification is not attempted here.
        """
        with self._lock:
            order = self.orders.get(order_id)

        if not order or not order.is_open:
            logger.debug("Trailing skip: order %s not found/open.", order_id)
            return

        try:
            step = float(atr) * float(order.trailing_step_atr_multiplier)
            if step <= 0:
                logger.debug("Trailing skip: step <= 0 (ATR=%s, mult=%s).", atr, order.trailing_step_atr_multiplier)
                return

            new_sl = order.stop_loss
            if order.transaction_type == "BUY":
                candidate = float(current_price) - step
                if candidate > order.stop_loss:
                    new_sl = candidate
            else:  # SELL
                candidate = float(current_price) + step
                if candidate < order.stop_loss:
                    new_sl = candidate

            if new_sl != order.stop_loss:
                before = order.stop_loss
                order.stop_loss = new_sl
                with self._lock:
                    self.orders[order_id] = order
                direction = "↗️ (BUY)" if order.transaction_type == "BUY" else "↘️ (SELL)"
                logger.info(
                    "Trailing SL %s %s: %.2f → %.2f (Px=%.2f, ATR=%.2f)",
                    direction, order_id, before, new_sl, float(current_price), float(atr)
                )
            else:
                logger.debug("Trailing checked (no change): %s SL=%.2f", order_id, order.stop_loss)

        except Exception as exc:
            logger.error("💥 trailing update failed for %s: %s", order_id, exc, exc_info=True)

    def exit_order(self, order_id: str, exit_reason: str = "manual") -> None:
        """
        Mark order as closed in the internal registry. Does not place live exit.
        """
        with self._lock:
            order = self.orders.get(order_id)
            if not order:
                logger.warning("⚠️ exit_order on unknown ID: %s", order_id)
                return
            if not order.is_open:
                logger.debug("exit_order: already closed: %s", order_id)
                return
            order.is_open = False
            self.orders[order_id] = order

        logger.info(
            "⏹️ Order exited (%s). ID=%s %s %s x%d",
            exit_reason, order_id, order.transaction_type, order.symbol, order.quantity
        )

    def get_active_orders(self) -> Dict[str, OrderRecord]:
        """Return active orders (internal tracking)."""
        with self._lock:
            active = {oid: o for oid, o in self.orders.items() if o.is_open}
        logger.debug("📊 Active orders: %d", len(active))
        return active

    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Try live positions via Kite; fallback to internal simulated view.
        Returns a list of dicts with keys used by RealTimeTrader (tradingsymbol, quantity, average_price, last_price, pnl).
        """
        # Live
        if self.kite:
            try:
                pos = self.kite.positions()
                # Normalize Zerodha structure: {"day": [...], "net": [...]}
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

        # Simulated fallback
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
                        "last_price": rec.entry_price,  # unknown in sim; caller may enrich
                        "pnl": 0.0,
                    }
                )
            return res

    def cancel_all_orders(self) -> int:
        """
        Attempt to cancel all live GTTs (best-effort) and mark internal orders closed.
        Returns count of orders marked cancelled internally.
        """
        cancelled = 0
        with self._lock:
            order_ids = list(self.orders.keys())

        for oid in order_ids:
            with self._lock:
                rec = self.orders.get(oid)
            if not rec or not rec.is_open:
                continue

            # Try cancel GTT in live mode if we stored an id
            if self.kite and rec.gtt_id is not None:
                try:
                    # Some SDKs need tradingsymbol/exchange too; try simple first
                    try:
                        self.kite.delete_gtt(rec.gtt_id)
                    except TypeError:
                        # Signature variant
                        self.kite.delete_gtt(trigger_id=rec.gtt_id)
                    logger.info("🧹 Deleted GTT trigger_id=%s for %s", rec.gtt_id, rec.symbol)
                except Exception as exc:
                    logger.warning("Failed to delete GTT %s: %s", rec.gtt_id, exc)

            # Mark closed internally
            with self._lock:
                rec.is_open = False
                self.orders[oid] = rec
                cancelled += 1

        logger.info("🧹 cancel_all_orders: closed %d orders.", cancelled)
        return cancelled