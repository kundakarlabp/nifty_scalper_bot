# src/execution/order_executor.py
"""
Order execution (live + light state) for Zerodha Kite.

Adds:
- TP1/TP2 partial exits driven by live LTP checks (no order webhooks required).
- Breakeven hop after TP1 (configurable tick offset).
- Works with both GTT-OCO and regular SL-M fallback:
  * On TP1 with GTT: cancel old OCO and recreate with remaining qty + new SL.
  * On TP1 with regular legs: exit partial via MARKET and re-create SL-M for remaining qty.
- Clean finalization on TP2 (flatten remaining, cancel legs/GTT).

Also includes:
- MARKET / MARKETABLE_LIMIT / LIMIT entries.
- ATR-based trailing; optional GTT reissue to mimic trailing.
- Robust LTP fetch for "NFO:SYMBOL"/"NSE:SYMBOL" or plain tradingsymbol+exchange.

NOTE: Runner must call executor.manage_partials() periodically (e.g., each loop).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Optional import to avoid import-time crash if kiteconnect is not installed
try:
    from kiteconnect import KiteConnect  # type: ignore
except Exception:
    KiteConnect = None  # type: ignore

from src.config import ExecutorConfig

logger = logging.getLogger(__name__)


# ------------ Data model for in-memory tracking ------------

@dataclass
class OrderRecord:
    order_id: str
    symbol: str
    exchange: str
    qty: int                             # CURRENT remaining qty
    transaction_type: str                # "BUY" / "SELL"
    entry_price: float
    # Initial risk/target used for R-based logic & breakeven
    initial_stop: Optional[float] = None
    initial_target: Optional[float] = None

    # Exit legs (fallback path)
    sl_order_id: Optional[str] = None    # SL-M trigger leg id
    tp_order_id: Optional[str] = None    # (unused when we self-manage TP)
    tp1_order_id: Optional[str] = None   # reserved; not used in this self-managed approach
    tp2_order_id: Optional[str] = None

    # GTT state
    use_gtt: bool = False
    gtt_id: Optional[str] = None

    # State
    is_open: bool = True
    hard_stop_price: Optional[float] = None  # last committed stop (no relaxation)

    # Partials bookkeeping
    tp1_price: Optional[float] = None
    tp2_price: Optional[float] = None
    tp_ratio: float = 0.5                # 50% by default
    breakeven_after_tp1: bool = True
    breakeven_offset_ticks: int = 1
    tp1_done: bool = False
    tp2_done: bool = False


# ------------ Executor ------------

class OrderExecutor:
    def __init__(self, exec_cfg: ExecutorConfig, kite: Optional[Any] = None):
        self.cfg = exec_cfg
        self._kite = kite
        self._lock = threading.RLock()
        self._tick_size = float(getattr(exec_cfg, "tick_size", 0.05) or 0.05)

        # GTT trailing controls
        self.allow_gtt_trailing: bool = True
        self.gtt_trail_min_step: float = 2.0     # Rs; minimum delta to re-issue GTT
        self.gtt_trail_cooldown_s: int = 5       # seconds between GTT reissues

        self._last_gtt_trail_ts: Dict[str, float] = {}  # entry_order_id -> last ts
        self._active: Dict[str, OrderRecord] = {}       # entry_order_id -> record

        # Partials global toggles (can be adjusted at runtime via Telegram /config if you wire it)
        self.partial_tp_enable: bool = True
        self.partial_tp_ratio: float = 0.50
        self.partial_tp2_r_mult: float = 2.0     # if no explicit target supplied, TP2 ~ 2R
        self.breakeven_after_tp1: bool = True
        self.breakeven_offset_ticks: int = 1

    # ---------- utilities ----------

    def get_tick_size(self) -> float:
        return self._tick_size

    def _round_to_tick(self, px: float) -> float:
        ts = self.get_tick_size()
        steps = round(px / ts)
        return round(steps * ts, 2)

    def _opp_side(self, side: str) -> str:
        return "SELL" if side.upper() == "BUY" else "BUY"

    # ---------- entry ----------

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

        side = transaction_type.upper()
        if side not in ("BUY", "SELL"):
            raise ValueError("transaction_type must be BUY or SELL")

        if order_type == "MARKET":
            resp = self._kite.place_order(
                tradingsymbol=symbol,
                exchange=exchange,
                transaction_type=side,
                order_type="MARKET",
                quantity=quantity,
                product="MIS",
                variety="regular",
            )
            oid = resp["order_id"]
            entry_price = float(self.get_last_price(symbol, exchange=exchange))
        elif order_type in ("MARKETABLE_LIMIT", "LIMIT"):
            ltp = float(self.get_last_price(symbol, exchange=exchange))
            px = price if (order_type == "LIMIT" and price is not None) else (
                ltp + slippage_guard if side == "BUY" else ltp - slippage_guard
            )
            px = self._round_to_tick(px)
            resp = self._kite.place_order(
                tradingsymbol=symbol,
                exchange=exchange,
                transaction_type=side,
                order_type="LIMIT",
                price=px,
                quantity=quantity,
                product="MIS",
                variety="regular",
            )
            oid = resp["order_id"]
            entry_price = px
        else:
            raise ValueError("order_type must be MARKET | MARKETABLE_LIMIT | LIMIT")

        with self._lock:
            self._active[oid] = OrderRecord(
                order_id=oid, symbol=symbol, exchange=exchange, qty=quantity,
                transaction_type=side, entry_price=entry_price
            )

        logger.info("Entry placed %s %s x%d @ %.2f (order_id=%s)", side, symbol, quantity, entry_price, oid)
        return oid

    # ---------- GTT / SLTP (initial) ----------

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
        Try GTT OCO for TP/SL. If unavailable, place regular SL-M + LIMIT legs.
        We still self-manage partials (TP1 via market) to avoid oversell risks.
        """
        rec = self._active.get(entry_order_id)
        if rec is None:
            logger.error("setup_gtt_orders: unknown entry id %s", entry_order_id)
            return

        sl_px = self._round_to_tick(stop_loss_price)
        tp_px = self._round_to_tick(target_price)

        # Store baseline risk & TP ladder on the record
        with self._lock:
            rec.initial_stop = sl_px
            rec.initial_target = tp_px
            # Compute TP1/TP2 thresholds
            side = transaction_type.upper()
            rec.tp_ratio = float(self.partial_tp_ratio)
            rec.breakeven_after_tp1 = bool(self.breakeven_after_tp1)
            rec.breakeven_offset_ticks = int(self.breakeven_offset_ticks)

            if self.partial_tp_enable:
                if side == "BUY":
                    rec.tp1_price = self._round_to_tick(entry_price + rec.tp_ratio * (tp_px - entry_price))
                    rec.tp2_price = tp_px
                else:
                    rec.tp1_price = self._round_to_tick(entry_price - rec.tp_ratio * (entry_price - tp_px))
                    rec.tp2_price = tp_px
            else:
                rec.tp1_price = None
                rec.tp2_price = tp_px

        # Prefer GTT OCO (lets us revise target/SL later on partials/trails)
        use_gtt = False
        gtt_id: Optional[str] = None
        try:
            gtt_id = self._kite.gtt_place_oco(  # type: ignore[attr-defined]
                tradingsymbol=symbol,
                exchange=exchange,
                transaction_type=transaction_type.upper(),
                quantity=quantity,
                trigger_sl=sl_px,
                trigger_tp=tp_px,
                product="MIS",
            )["gtt_id"]
            use_gtt = True
            logger.info("GTT OCO placed for %s (entry_id=%s, gtt_id=%s)", symbol, entry_order_id, gtt_id)
        except Exception as e:
            logger.warning("GTT OCO failed; falling back to regular SL/TP: %s", e)

        with self._lock:
            rec.use_gtt = use_gtt
            rec.gtt_id = gtt_id
            rec.hard_stop_price = sl_px

        if use_gtt:
            return

        # Fallback legs: place only SL-M; TP will be handled via market exits (TP1/TP2) by manage_partials()
        try:
            exit_side = self._opp_side(transaction_type)
            sl_id = self._kite.place_order(
                tradingsymbol=symbol, exchange=exchange, transaction_type=exit_side,
                order_type="SL-M", trigger_price=sl_px, quantity=quantity,
                product="MIS", variety="regular"
            )["order_id"]
            with self._lock:
                rec.sl_order_id = sl_id
                rec.tp_order_id = None  # we don't keep passive TP legs to avoid oversell
            logger.info("Fallback SL placed (SL %s), TP handled actively by executor.", sl_id)
        except Exception as e:
            logger.error("Failed to place fallback SL: %s", e, exc_info=True)

    # ---------- Partials management (call this in your main loop) ----------

    def manage_partials(self) -> None:
        """
        Iterate open records and execute partials if thresholds are crossed.
        Safe for both GTT-OCO and fallback SL management.
        """
        for entry_id, rec in list(self._active.items()):
            if not rec.is_open or not self.partial_tp_enable:
                continue

            try:
                ltp = float(self.get_last_price(rec.symbol, exchange=rec.exchange))
            except Exception:
                continue

            # TP1 trigger?
            if not rec.tp1_done and rec.tp1_price is not None:
                hit_tp1 = (ltp >= rec.tp1_price) if rec.transaction_type == "BUY" else (ltp <= rec.tp1_price)
                if hit_tp1:
                    self._execute_tp1(entry_id, rec, ltp)

            # TP2 trigger?
            if rec.is_open and not rec.tp2_done and rec.tp2_price is not None:
                hit_tp2 = (ltp >= rec.tp2_price) if rec.transaction_type == "BUY" else (ltp <= rec.tp2_price)
                if hit_tp2:
                    self._execute_tp2(entry_id, rec, ltp)

    def _execute_tp1(self, entry_id: str, rec: OrderRecord, ltp: float) -> None:
        """Exit partial qty at market; hop SL to breakeven; reissue GTT/SL for remaining qty."""
        try:
            qty1 = max(1, int(round(rec.qty * rec.tp_ratio)))
            if qty1 >= rec.qty:
                qty1 = rec.qty // 2 or 1

            exit_side = self._opp_side(rec.transaction_type)
            self._kite.place_order(
                tradingsymbol=rec.symbol, exchange=rec.exchange, transaction_type=exit_side,
                order_type="MARKET", quantity=qty1, product="MIS", variety="regular"
            )
            logger.info("TP1 market exit: %s qty=%d @ ~%.2f", rec.symbol, qty1, ltp)

            remaining = rec.qty - qty1
            if remaining <= 0:
                rec.qty = 0
                rec.is_open = False
                self._cleanup_after_close(rec)
                rec.tp1_done = True
                rec.tp2_done = True
                return

            # Update remaining qty
            with self._lock:
                rec.qty = remaining
                rec.tp1_done = True

            # Breakeven hop (after TP1)
            if rec.breakeven_after_tp1:
                be = self._breakeven_price(rec)
                if be is not None:
                    with self._lock:
                        rec.hard_stop_price = be

            # Re-create protection (GTT or SL-M) for remaining qty
            if rec.use_gtt:
                # Cancel old and reissue with remaining qty and same TP2 (if any)
                try:
                    if rec.gtt_id:
                        self._kite.gtt_cancel(rec.gtt_id)  # type: ignore[attr-defined]
                except Exception:
                    pass
                tp_for_remaining = rec.tp2_price if rec.tp2_price is not None else (rec.initial_target or ltp)
                new_id = self._kite.gtt_place_oco(  # type: ignore[attr-defined]
                    tradingsymbol=rec.symbol,
                    exchange=rec.exchange,
                    transaction_type=rec.transaction_type,
                    quantity=remaining,
                    trigger_sl=self._round_to_tick(rec.hard_stop_price or ltp),
                    trigger_tp=self._round_to_tick(tp_for_remaining),
                    product="MIS",
                )["gtt_id"]
                with self._lock:
                    rec.gtt_id = new_id
                logger.info("Reissued GTT after TP1 (qty=%d, gtt_id=%s)", remaining, new_id)
            else:
                # Fallback: cancel old SL (if any) and re-place for remaining qty
                try:
                    if rec.sl_order_id:
                        self._kite.cancel_order(variety="regular", order_id=rec.sl_order_id)
                except Exception:
                    pass
                new_sl_id = self._kite.place_order(
                    tradingsymbol=rec.symbol, exchange=rec.exchange, transaction_type=self._opp_side(rec.transaction_type),
                    order_type="SL-M", trigger_price=self._round_to_tick(rec.hard_stop_price or ltp),
                    quantity=remaining, product="MIS", variety="regular"
                )["order_id"]
                with self._lock:
                    rec.sl_order_id = new_sl_id
                logger.info("Replaced SL-M after TP1 (qty=%d, order_id=%s)", remaining, new_sl_id)

        except Exception as e:
            logger.error("TP1 handling failed: %s", e, exc_info=True)

    def _execute_tp2(self, entry_id: str, rec: OrderRecord, ltp: float) -> None:
        """Exit remaining at market; close record; cleanup SL/GTT."""
        try:
            if rec.qty <= 0:
                return
            exit_side = self._opp_side(rec.transaction_type)
            self._kite.place_order(
                tradingsymbol=rec.symbol, exchange=rec.exchange, transaction_type=exit_side,
                order_type="MARKET", quantity=rec.qty, product="MIS", variety="regular"
            )
            logger.info("TP2 market exit: %s qty=%d @ ~%.2f", rec.symbol, rec.qty, ltp)

            rec.qty = 0
            rec.is_open = False
            rec.tp2_done = True
            self._cleanup_after_close(rec)
        except Exception as e:
            logger.error("TP2 handling failed: %s", e, exc_info=True)

    def _breakeven_price(self, rec: OrderRecord) -> Optional[float]:
        """Compute breakeven price with tick offset after TP1."""
        try:
            offset = max(0, int(rec.breakeven_offset_ticks)) * self._tick_size
            if rec.transaction_type == "BUY":
                return self._round_to_tick((rec.entry_price + offset))
            else:
                return self._round_to_tick((rec.entry_price - offset))
        except Exception:
            return None

    def _cleanup_after_close(self, rec: OrderRecord) -> None:
        """Cancel any remaining protection orders after final close."""
        try:
            if rec.use_gtt and rec.gtt_id:
                try:
                    self._kite.gtt_cancel(rec.gtt_id)  # type: ignore[attr-defined]
                except Exception:
                    pass
            # Cancel any stray regular legs
            for oid in (rec.sl_order_id, rec.tp_order_id, rec.tp1_order_id, rec.tp2_order_id):
                try:
                    if oid:
                        self._kite.cancel_order(variety="regular", order_id=oid)
                except Exception:
                    pass
        except Exception:
            pass

    # ---------- trailing (regular legs or advisory to GTT) ----------

    def update_trailing_stop(self, entry_order_id: str, current_price: float, atr: float, trail_mult: float = 1.0) -> Optional[float]:
        """
        Compute an improved stop and either modify SL leg or (if GTT) call maybe_trail_gtt.
        Returns the new SL if tightened, else None.
        """
        rec = self._active.get(entry_order_id)
        if not rec or not rec.is_open:
            return None

        # ATR-based trail (no relaxation)
        trail_pts = max(atr * max(0.5, trail_mult), 0.1)
        if rec.transaction_type == "BUY":
            candidate = self._round_to_tick(current_price - trail_pts)
            new_sl = max(rec.hard_stop_price or 0.0, candidate)
            if rec.hard_stop_price is not None and new_sl <= rec.hard_stop_price:
                return None
        else:
            candidate = self._round_to_tick(current_price + trail_pts)
            new_sl = min(rec.hard_stop_price or 1e9, candidate)
            if rec.hard_stop_price is not None and new_sl >= rec.hard_stop_price:
                return None

        with self._lock:
            rec.hard_stop_price = new_sl

        if rec.use_gtt:
            self.maybe_trail_gtt(entry_order_id, new_sl)
            return new_sl

        # Modify regular SL-M leg (use only trigger_price for SL-M)
        try:
            if rec.sl_order_id:
                self._kite.modify_order(
                    variety="regular",
                    order_id=rec.sl_order_id,
                    trigger_price=new_sl,
                    # Do NOT send price for SL-M
                )
                logger.info("Trailing SL modified to %.2f (order %s)", new_sl, rec.sl_order_id)
            return new_sl
        except Exception as e:
            logger.warning("Failed to modify trailing SL: %s", e)
            return None

    def maybe_trail_gtt(self, entry_order_id: str, new_sl: float) -> None:
        """
        If using GTT and allowed, cancel & recreate OCO with tighter SL.
        Respects cooldown and minimum step to avoid API spam.
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
            # Cancel previous GTT
            if rec.gtt_id:
                try:
                    self._kite.gtt_cancel(rec.gtt_id)  # type: ignore[attr-defined]
                except Exception:
                    pass

            # Recreate with tighter SL; keep same target if set
            tp_guess = rec.tp2_price if rec.tp2_price is not None else (rec.initial_target or new_sl)
            tp_guess = self._round_to_tick(tp_guess)

            new_id = self._kite.gtt_place_oco(  # type: ignore[attr-defined]
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

    # ---------- exits & helpers ----------

    def exit_order(self, entry_order_id: str, exit_reason: str = "manual") -> None:
        rec = self._active.get(entry_order_id)
        if not rec or not rec.is_open:
            return
        side = self._opp_side(rec.transaction_type)
        try:
            self._kite.place_order(
                tradingsymbol=rec.symbol, exchange=rec.exchange, transaction_type=side,
                order_type="MARKET", quantity=rec.qty, product="MIS", variety="regular"
            )
            rec.is_open = False
            logger.info("Exited %s due to: %s", entry_order_id, exit_reason)
            self._cleanup_after_close(rec)
        except Exception as e:
            logger.error("Exit failed for %s: %s", entry_order_id, e, exc_info=True)

    def get_active_orders(self) -> List[OrderRecord]:
        return list(self._active.values())

    def get_positions(self) -> List[Dict[str, Any]]:
        # Adapter; implement via kite.positions() if you want live positions
        return []

    def cancel_all_orders(self) -> None:
        # Best-effort cancel of legs/GTT
        for rec in list(self._active.values()):
            try:
                if rec.use_gtt and rec.gtt_id:
                    try:
                        self._kite.gtt_cancel(rec.gtt_id)  # type: ignore[attr-defined]
                    except Exception:
                        pass
                for oid in (rec.sl_order_id, rec.tp_order_id, rec.tp1_order_id, rec.tp2_order_id):
                    try:
                        if oid:
                            self._kite.cancel_order(variety="regular", order_id=oid)
                    except Exception:
                        pass
            except Exception:
                pass

    def get_last_price(self, symbol: str, exchange: Optional[str] = None) -> float:
        """
        Robust LTP fetch:
        - Accepts "NFO:SYMBOL"/"NSE:SYMBOL" OR plain tradingsymbol with exchange provided.
        """
        try:
            if ":" in symbol:
                key = symbol
            else:
                ex = (exchange or "NFO").upper()
                key = f"{ex}:{symbol}"
            q = self._kite.ltp([key])
            return float(q.get(key, {}).get("last_price", 0.0))
        except Exception as e:
            logger.error("ltp failed for %s: %s", symbol, e)
            return 0.0
