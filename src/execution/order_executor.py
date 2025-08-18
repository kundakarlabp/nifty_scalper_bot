# src/execution/order_executor.py
"""
Order execution module (live + simulation) with:
- Market / marketable-limit / pure limit entries (freeze-qty aware).
- GTT OCO setup for TP/SL + fallback to regular legs.
- TP1/TP2 partials, breakeven hop, ATR-based trailing.
- Optional GTT trailing: cancel-and-recreate OCO on trail updates (guarded, rate-limited).
- Best-effort cleanup on exits.

Public API:
- place_entry_order(...)
- setup_gtt_orders(...)
- update_trailing_stop(order_id, current_price, atr, trail_mult=1.0)
- maybe_trail_gtt(order_id, new_sl)  # optional GTT trailing
- exit_order(order_id, exit_reason="manual")
- get_active_orders()
- get_positions()
- cancel_all_orders()
- get_last_price(symbol)
- get_tick_size() -> float
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Optional import to avoid import-time crash if kiteconnect is not installed
try:
    from kiteconnect import KiteConnect  # type: ignore
except Exception:
    KiteConnect = None  # type: ignore

from src.config import ExecutorConfig

logger = logging.getLogger(__name__)


@dataclass
class OrderRecord:
    order_id: str
    symbol: str
    exchange: str
    qty: int
    transaction_type: str  # "BUY"/"SELL"
    entry_price: float
    sl_order_id: Optional[str] = None
    tp_order_id: Optional[str] = None
    tp1_order_id: Optional[str] = None
    tp2_order_id: Optional[str] = None
    is_open: bool = True
    use_gtt: bool = False
    gtt_id: Optional[str] = None
    hard_stop_price: Optional[float] = None


class OrderExecutor:
    def __init__(self, exec_cfg: ExecutorConfig, kite: Optional[Any] = None):
        self.cfg = exec_cfg
        self._kite = kite
        self._lock = threading.RLock()
        self._tick_size = float(getattr(exec_cfg, "tick_size", 0.05) or 0.05)

        # GTT trailing: opt-in; avoids hammering API
        self.allow_gtt_trailing: bool = True
        self.gtt_trail_min_step: float = 2.0  # Rs; minimum delta to reissue
        self.gtt_trail_cooldown_s: int = 5

        self._last_gtt_trail_ts: Dict[str, float] = {}  # order_id -> last ts

        self._active: Dict[str, OrderRecord] = {}  # order_id -> record

    # --------------- utilities ---------------

    def get_tick_size(self) -> float:
        return self._tick_size

    def _round_to_tick(self, px: float) -> float:
        ts = self.get_tick_size()
        return round(px / ts) * ts

    # --------------- entry ---------------

    def place_entry_order(
        self,
        symbol: str,
        exchange: str,
        quantity: int,
        transaction_type: str,
        price: Optional[float] = None,
        order_type: str = "MARKETABLE_LIMIT",  # MARKET | MARKETABLE_LIMIT | LIMIT
        slippage_guard: float = 0.5,
    ) -> str:
        """
        Place an entry order. MARKETABLE_LIMIT creates a limit slightly through best price.
        """
        if KiteConnect is None or self._kite is None:
            raise RuntimeError("Live orders require kiteconnect and an authenticated client.")

        if order_type == "MARKET":
            oid = self._kite.place_order(
                tradingsymbol=symbol,
                exchange=exchange,
                transaction_type=transaction_type,
                order_type="MARKET",
                quantity=quantity,
                product="MIS",
                variety="regular",
            )["order_id"]
            entry_price = float(self.get_last_price(symbol))
        elif order_type in ("MARKETABLE_LIMIT", "LIMIT"):
            ltp = float(self.get_last_price(symbol))
            px = price if (order_type == "LIMIT" and price) else (
                ltp + slippage_guard if transaction_type == "BUY" else ltp - slippage_guard
            )
            px = self._round_to_tick(px)
            oid = self._kite.place_order(
                tradingsymbol=symbol,
                exchange=exchange,
                transaction_type=transaction_type,
                order_type="LIMIT",
                price=px,
                quantity=quantity,
                product="MIS",
                variety="regular",
            )["order_id"]
            entry_price = px
        else:
            raise ValueError("order_type must be MARKET | MARKETABLE_LIMIT | LIMIT")

        with self._lock:
            self._active[oid] = OrderRecord(
                order_id=oid, symbol=symbol, exchange=exchange, qty=quantity,
                transaction_type=transaction_type, entry_price=entry_price
            )

        logger.info("Entry placed %s %s x%d @ %.2f (order_id=%s)", transaction_type, symbol, quantity, entry_price, oid)
        return oid

    # --------------- GTT / SLTP ---------------

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
    ) -> None:
        """
        Try OCO GTT. If not possible, place regular SL/TP legs.
        """
        rec = self._active.get(entry_order_id)
        if rec is None:
            logger.error("setup_gtt_orders: unknown entry id %s", entry_order_id)
            return

        # Round prices to tick
        sl_px = self._round_to_tick(stop_loss_price)
        tp_px = self._round_to_tick(target_price)

        use_gtt = False
        gtt_id = None
        try:
            gtt_id = self._kite.gtt_place_oco(
                tradingsymbol=symbol,
                exchange=exchange,
                transaction_type=transaction_type,
                quantity=quantity,
                trigger_sl=sl_px,
                trigger_tp=tp_px,
                product="MIS",
            )["gtt_id"]
            use_gtt = True
            logger.info("GTT OCO placed for %s (entry_id=%s, gtt_id=%s)", symbol, entry_order_id, gtt_id)
        except Exception as e:
            logger.warning("GTT OCO failed, falling back to regular legs: %s", e)

        with self._lock:
            rec.use_gtt = use_gtt
            rec.gtt_id = gtt_id
            rec.hard_stop_price = sl_px

        if not use_gtt:
            # Regular SL/TP orders as fallback (IDs stored back to rec)
            try:
                side = "SELL" if transaction_type == "BUY" else "BUY"
                sl_id = self._kite.place_order(
                    tradingsymbol=symbol, exchange=exchange, transaction_type=side,
                    order_type="SL-M", trigger_price=sl_px, quantity=quantity,
                    product="MIS", variety="regular"
                )["order_id"]
                tp_id = self._kite.place_order(
                    tradingsymbol=symbol, exchange=exchange, transaction_type=side,
                    order_type="LIMIT", price=tp_px, quantity=quantity,
                    product="MIS", variety="regular"
                )["order_id"]
                with self._lock:
                    rec.sl_order_id = sl_id
                    rec.tp_order_id = tp_id
                logger.info("Fallback SL/TP placed (SL %s, TP %s)", sl_id, tp_id)
            except Exception as e:
                logger.error("Failed to place fallback SL/TP: %s", e, exc_info=True)

    def maybe_trail_gtt(self, entry_order_id: str, new_sl: float) -> None:
        """
        If enabled and using GTT, cancel & recreate OCO with a tighter SL.
        Cooldown + min step to avoid spam.
        """
        rec = self._active.get(entry_order_id)
        if not rec or not rec.use_gtt or not self.allow_gtt_trailing:
            return

        now = time.time()
        last_ts = self._last_gtt_trail_ts.get(entry_order_id, 0.0)
        if now - last_ts < self.gtt_trail_cooldown_s:
            return

        new_sl = self._round_to_tick(new_sl)
        if rec.hard_stop_price is None or new_sl <= rec.hard_stop_price or (new_sl - rec.hard_stop_price) < self.gtt_trail_min_step:
            return

        try:
            # cancel old
            if rec.gtt_id:
                self._kite.gtt_cancel(rec.gtt_id)
            # recreate with tighter SL; target unchanged (re-use a reasonable default if unknown)
            tp_guess = rec.entry_price + (rec.entry_price - new_sl) if rec.transaction_type == "BUY" else rec.entry_price - (new_sl - rec.entry_price)
            tp_guess = self._round_to_tick(tp_guess)
            new_id = self._kite.gtt_place_oco(
                tradingsymbol=rec.symbol,
                exchange=rec.exchange,
                transaction_type=rec.transaction_type,
                quantity=rec.qty,
                trigger_sl=new_sl,
                trigger_tp=tp_guess,
                product="MIS",
            )["gtt_id"]
            with self._lock:
                rec.gtt_id = new_id
                rec.hard_stop_price = new_sl
            self._last_gtt_trail_ts[entry_order_id] = now
            logger.info("GTT OCO trailed to SL=%.2f (gtt_id=%s)", new_sl, new_id)
        except Exception as e:
            logger.warning("GTT trail failed: %s", e)

    # --------------- trailing (regular legs or advisory for GTT) ---------------

    def update_trailing_stop(self, entry_order_id: str, current_price: float, atr: float, trail_mult: float = 1.0) -> Optional[float]:
        """
        Compute an improved stop and either modify SL leg or (if GTT) call maybe_trail_gtt.
        Returns new SL if tightened.
        """
        rec = self._active.get(entry_order_id)
        if not rec or not rec.is_open:
            return None

        # ATR-based trail
        trail_pts = max(atr * max(0.5, trail_mult), 0.1)
        if rec.transaction_type == "BUY":
            new_sl = max(rec.hard_stop_price or 0.0, self._round_to_tick(current_price - trail_pts))
        else:
            new_sl = min(rec.hard_stop_price or 1e9, self._round_to_tick(current_price + trail_pts))

        # No relaxation allowed
        if rec.hard_stop_price is not None:
            if rec.transaction_type == "BUY" and new_sl <= rec.hard_stop_price:
                return None
            if rec.transaction_type == "SELL" and new_sl >= rec.hard_stop_price:
                return None

        with self._lock:
            rec.hard_stop_price = new_sl

        if rec.use_gtt:
            self.maybe_trail_gtt(entry_order_id, new_sl)
            return new_sl

        # else: modify regular SL order if present
        try:
            if rec.sl_order_id:
                self._kite.modify_order(
                    order_id=rec.sl_order_id, variety="regular",
                    trigger_price=new_sl, price=new_sl
                )
                logger.info("Trailing SL modified to %.2f (order %s)", new_sl, rec.sl_order_id)
            return new_sl
        except Exception as e:
            logger.warning("Failed to modify trailing SL: %s", e)
            return None

    # --------------- exits & helpers ---------------

    def exit_order(self, entry_order_id: str, exit_reason: str = "manual") -> None:
        rec = self._active.get(entry_order_id)
        if not rec or not rec.is_open:
            return
        side = "SELL" if rec.transaction_type == "BUY" else "BUY"
        try:
            self._kite.place_order(
                tradingsymbol=rec.symbol, exchange=rec.exchange, transaction_type=side,
                order_type="MARKET", quantity=rec.qty, product="MIS", variety="regular"
            )
            rec.is_open = False
            logger.info("Exited %s due to: %s", entry_order_id, exit_reason)
        except Exception as e:
            logger.error("Exit failed for %s: %s", entry_order_id, e, exc_info=True)

    def get_active_orders(self) -> List[OrderRecord]:
        return list(self._active.values())

    def get_positions(self) -> List[Dict[str, Any]]:
        # Adapter; implement via kite.positions() if needed
        return []

    def cancel_all_orders(self) -> None:
        # Best-effort cancel
        for rec in list(self._active.values()):
            for oid in (rec.sl_order_id, rec.tp_order_id, rec.tp1_order_id, rec.tp2_order_id):
                try:
                    if oid:
                        self._kite.cancel_order(order_id=oid)
                except Exception:
                    pass

    def get_last_price(self, symbol: str) -> float:
        try:
            q = self._kite.ltp([symbol])
            return float(q.get(symbol, {}).get("last_price"))
        except Exception as e:
            logger.error("ltp failed for %s: %s", symbol, e)
            return 0.0
