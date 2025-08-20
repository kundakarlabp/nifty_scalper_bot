# src/execution/order_executor.py
"""
Order execution module (live + simulation) with bracket-like behavior,
lot-aware partial profit-taking, breakeven hop after TP1, trailing-SL support,
hard-stop failsafe, and exchange freeze-quantity chunking.

Public API:
- place_entry_order(token, symbol, side, quantity, price) -> record_id | None
- setup_gtt_orders(record_id, sl_price, tp_price) -> None  (REGULAR fallback when partial TP enabled)
- update_trailing_stop(record_id, current_price, atr, atr_multiplier) -> bool
- exit_order(record_id, exit_reason="manual") -> bool
- get_active_orders() -> list[_OrderRecord]
- get_positions() -> dict
- get_positions_kite() -> dict
- cancel_all_orders() -> None
- get_last_price(symbol) -> Optional[float]
- get_tick_size(token=None) -> float
- sync_and_enforce_oco() -> list[tuple[str, float]]   # (record_id, fill_price)
"""

from __future__ import annotations

import logging
import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List, Tuple

from src.config import settings
from src.utils.retry import retry

try:
    # optional import for typing and side values
    from kiteconnect import KiteConnect  # type: ignore
except Exception:  # pragma: no cover
    KiteConnect = object  # type: ignore

logger = logging.getLogger(__name__)


# ---------------------------- rounding helpers -------------------------- #

def _round_to_tick(x: float, tick: float) -> float:
    """Round price to nearest tick size."""
    try:
        if tick <= 0:
            return float(x)
        return round(float(x) / tick) * tick
    except Exception:
        return float(x)


def _round_to_qty(x: int, step: int) -> int:
    """Round quantity DOWN to nearest multiple of step."""
    try:
        x = int(x)
        step = max(1, int(step))
        return (x // step) * step
    except Exception:
        return int(x)


# ---------------------------- data model -------------------------------- #

@dataclass
class _OrderRecord:
    """Internal state for a managed order set (entry + exits)."""
    instrument_token: int
    symbol: str
    side: str  # "BUY" | "SELL"
    quantity: int  # total units (multiple of lot_size)
    entry_price: float
    order_variety: str
    order_product: str
    exchange: str
    tick_size: float
    lot_size: int
    freeze_qty: int
    record_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    # entry orders can be chunked; store all child order_ids
    entry_order_ids: List[str] = field(default_factory=list)
    is_open: bool = True
    entry_filled_qty: int = 0
    avg_entry_price: float = 0.0

    # Exit orders (REGULAR)
    sl_order_id: Optional[str] = None
    tp_order_id: Optional[str] = None
    # Partial profit targets (REGULAR)
    partial_enabled: bool = False
    tp1_order_id: Optional[str] = None
    tp2_order_id: Optional[str] = None
    tp1_qty: int = 0
    tp2_qty: int = 0
    breakeven_done: bool = False

    # GTT (placed as separate singles)
    gtt_sl_id: Optional[int] = None
    gtt_tp_id: Optional[int] = None

    # Remember desired exit prices (for (re)arm logic)
    desired_sl: Optional[float] = None
    desired_tp: Optional[float] = None


# ---------------------------- executor ---------------------------------- #

class OrderExecutor:
    """
    Manages live order execution with state-tracking and OCO/GTT logic.

    Uses settings.executor for defaults; safe when unset.
    """

    def __init__(self, config: Optional[object] = None, kite: Any = None, data_source: Any = None) -> None:
        self.config = config or getattr(settings, "executor", object())
        self.kite = kite
        self.data_source = data_source

        # lock + in-memory state
        self._lock = threading.Lock()
        self._active: Dict[str, _OrderRecord] = {}
        self._positions: Dict[str, Any] = {}
        self._tick_cache: Dict[int, float] = {}

        # derived config with safe defaults
        self.exchange: str = str(getattr(self.config, "exchange", "NFO"))
        self.order_variety: str = str(getattr(self.config, "order_variety", "regular"))
        self.order_product: str = str(getattr(self.config, "order_product", "NRML"))
        self.entry_order_type: str = str(getattr(self.config, "entry_order_type", "LIMIT"))  # LIMIT / MARKET
        self.use_slm_exit: bool = bool(getattr(self.config, "use_slm_exit", True))
        self.preferred_exit_mode: str = str(getattr(self.config, "preferred_exit_mode", "REGULAR")).upper()
        self.enable_trailing: bool = bool(getattr(self.config, "enable_trailing", True))
        self.trailing_atr_multiplier: float = float(getattr(self.config, "trailing_atr_multiplier", 1.5))
        self.partial_tp_enable: bool = bool(getattr(self.config, "partial_tp_enable", False))
        self.tp1_qty_ratio: float = float(getattr(self.config, "tp1_qty_ratio", 0.5))  # 50% at TP1
        self.breakeven_ticks: int = int(getattr(self.config, "breakeven_ticks", 2))   # hop to BE±ticks after TP1
        self.default_tick_size: float = float(getattr(self.config, "tick_size", 0.05))
        self.freeze_qty_default: int = int(getattr(self.config, "exchange_freeze_qty", 1800))
        self.lot_size: int = int(getattr(getattr(settings, "instruments", object()), "nifty_lot_size", 75))

    # --------------- market utils ---------------- #

    def get_tick_size(self, token: Optional[int] = None) -> float:
        """Fetch instrument tick size, cached per token."""
        if token is None:
            return self.default_tick_size
        if token in self._tick_cache:
            return self._tick_cache[token]

        tick = self.default_tick_size
        try:
            if not self.kite:
                raise RuntimeError("kite not available")
            # Try NFO first; fallback to NSE
            for seg in ("NFO", "NSE"):
                instruments = self.kite.instruments(seg)
                for inst in instruments:
                    if int(inst.get("instrument_token", -1)) == int(token):
                        tick = float(inst.get("tick_size", tick))
                        freeze = int(inst.get("freeze_quantity", self.freeze_qty_default)) if "freeze_quantity" in inst else self.freeze_qty_default
                        self._tick_cache[token] = tick
                        # keep a freeze cache by token
                        return tick
        except Exception as e:
            logger.warning("Tick size fetch failed for token %s, defaulting to %.2f: %s", token, tick, e)

        self._tick_cache[token] = tick
        return tick

    def get_last_price(self, symbol: str) -> Optional[float]:
        """Fetch LTP from attached data source."""
        try:
            if not self.data_source:
                return None
            return self.data_source.get_last_price(symbol)
        except Exception:
            return None

    def get_active_orders(self) -> List[Any]:
        with self._lock:
            return list(self._active.values())

    def get_positions(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._positions)

    def get_positions_kite(self) -> Dict[str, Any]:
        try:
            if not self.kite:
                return {}
            pos = self.kite.positions().get("day", [])
            return {p["tradingsymbol"]: p for p in pos}
        except Exception as e:
            logger.error("Failed to fetch live positions from Kite: %s", e)
            return {}

    # ---------------- kite wrappers (retry) ---------------- #

    @retry(exceptions=(Exception,))
    def _kite_place_order(self, **kwargs) -> str:
        """Wrapper for kite.place_order with retry & logging."""
        return self.kite.place_order(**kwargs)

    @retry(exceptions=(Exception,))
    def _kite_modify_order(self, **kwargs) -> str:
        """Wrapper for kite.modify_order with retry & logging."""
        return self.kite.modify_order(**kwargs)

    @retry(exceptions=(Exception,))
    def _kite_cancel(self, **kwargs) -> bool:
        """Wrapper for kite.cancel_order with retry & logging."""
        try:
            self.kite.cancel_order(**kwargs)
            return True
        except Exception as e:
            logger.warning("Failed to cancel order (%s): %s", kwargs, e)
            return False

    # ---------------- entry placement (with chunking) ---------------- #

    def _chunk_quantity(self, units: int, step: int, freeze: int) -> List[int]:
        """Split a quantity (units) into chunks obeying lot step and freeze limit."""
        units = _round_to_qty(units, step)
        if units <= 0:
            return []
        freeze_adj = _round_to_qty(min(freeze, units), step)
        if freeze_adj <= 0:
            freeze_adj = step
        chunks = []
        remaining = units
        while remaining > 0:
            take = min(remaining, freeze_adj)
            take = _round_to_qty(take, step)
            if take <= 0:
                break
            chunks.append(take)
            remaining -= take
        return chunks

    def place_entry_order(
        self,
        token: int,
        symbol: str,
        side: str,
        quantity: int,   # UNITS (must be multiple of lot_size)
        price: float,
    ) -> Optional[str]:
        """
        Place entry orders (chunked by freeze-limit).
        Returns record_id or None.
        """
        if not self.kite:
            logger.error("Kite not available; cannot place live orders.")
            return None

        if quantity <= 0:
            logger.warning("Attempted to place order with quantity <= 0. Skipping.")
            return None

        tick = self.get_tick_size(token)
        step = self.lot_size
        freeze = int(getattr(self.config, "exchange_freeze_qty", self.freeze_qty_default))

        with self._lock:
            # Prevent duplicate open record for same symbol
            for rec in self._active.values():
                if rec.symbol == symbol and rec.is_open:
                    logger.warning("Open order/position already exists for %s; skipping new entry.", symbol)
                    return None

        chunks = self._chunk_quantity(quantity, step, freeze)
        if not chunks:
            logger.warning("Quantity %d could not be chunked; check lot_size/freeze limits.", quantity)
            return None

        entry_ids: List[str] = []
        for q in chunks:
            try:
                params = {
                    "variety": self.order_variety,
                    "exchange": self.exchange,
                    "tradingsymbol": symbol,
                    "transaction_type": side,
                    "quantity": q,
                    "product": self.order_product,
                    "order_type": self.entry_order_type,
                }
                if self.entry_order_type.upper() == "LIMIT":
                    params["price"] = _round_to_tick(price, tick)
                oid = self._kite_place_order(**params)
                entry_ids.append(oid)
                logger.info("🟢 Entry chunk placed %s: %s %d@%.2f -> %s", symbol, side, q, price, oid)
            except Exception as e:
                logger.error("Failed to place entry chunk %s %d: %s", symbol, q, e)

        if not entry_ids:
            return None

        rec = _OrderRecord(
            instrument_token=token,
            symbol=symbol,
            side=side,
            quantity=sum(chunks),
            entry_price=price,
            order_variety=self.order_variety,
            order_product=self.order_product,
            exchange=self.exchange,
            tick_size=tick,
            lot_size=step,
            freeze_qty=freeze,
            entry_order_ids=entry_ids,
        )
        with self._lock:
            self._active[rec.record_id] = rec
        return rec.record_id

    # ---------------- exits (REGULAR & GTT) ---------------- #

    def _place_sl_regular(self, rec: _OrderRecord, sl_price: float, qty: int) -> Optional[str]:
        """Place/replace a regular SL(-M) order for given qty."""
        order_type = "SL-M" if self.use_slm_exit else "SL"
        params = {
            "variety": rec.order_variety,
            "exchange": rec.exchange,
            "tradingsymbol": rec.symbol,
            "transaction_type": "SELL" if rec.side == "BUY" else "BUY",
            "quantity": qty,
            "product": rec.order_product,
            "order_type": order_type,
            "trigger_price": _round_to_tick(sl_price, rec.tick_size),
        }
        if order_type == "SL":  # protective offset for limit SL
            params["price"] = _round_to_tick(sl_price, rec.tick_size)
        try:
            oid = self._kite_place_order(**params)
            return oid
        except Exception as e:
            logger.error("SL order placement failed for %s: %s", rec.symbol, e)
            return None

    def _place_tp_regular(self, rec: _OrderRecord, tp_price: float, qty: int) -> Optional[str]:
        """Place/replace a regular TP limit order for given qty."""
        params = {
            "variety": rec.order_variety,
            "exchange": rec.exchange,
            "tradingsymbol": rec.symbol,
            "transaction_type": "SELL" if rec.side == "BUY" else "BUY",
            "quantity": qty,
            "product": rec.order_product,
            "order_type": "LIMIT",
            "price": _round_to_tick(tp_price, rec.tick_size),
        }
        try:
            oid = self._kite_place_order(**params)
            return oid
        except Exception as e:
            logger.error("TP order placement failed for %s: %s", rec.symbol, e)
            return None

    def setup_gtt_orders(self, record_id: str, sl_price: float, tp_price: float) -> None:
        """
        Arms exits for the record. Behavior:
          - If partial TP is enabled => use REGULAR orders (two TPs if configured).
          - Else:
              * If preferred_exit_mode is GTT/AUTO => attempt two single GTTs (SL & TP),
                else REGULAR SL/TP.
        """
        with self._lock:
            rec = self._active.get(record_id)
        if not rec or not rec.is_open:
            logger.warning("Record %s not found or not open; cannot arm exits.", record_id)
            return

        rec.desired_sl = sl_price
        rec.desired_tp = tp_price

        # Partial TP => REGULAR orders always
        if self.partial_tp_enable:
            with self._lock:
                rec.partial_enabled = True

            # qty split: tp1_ratio for TP1, remaining for TP2
            tp1_qty = _round_to_qty(int(rec.quantity * self.tp1_qty_ratio), rec.lot_size)
            tp1_qty = max(rec.lot_size, tp1_qty) if tp1_qty > 0 else rec.lot_size
            tp2_qty = _round_to_qty(rec.quantity - tp1_qty, rec.lot_size)

            # Place SL for full qty first
            sl_id = self._place_sl_regular(rec, sl_price, rec.quantity)
            # Place TP1 at provided target
            tp1_id = self._place_tp_regular(rec, tp_price, tp1_qty)

            # Optional TP2 further out (2x distance)
            dist = abs(tp_price - rec.entry_price)
            tp2_price = rec.entry_price + (dist * 2.0) * (1 if rec.side == "BUY" else -1)
            tp2_id = None
            if tp2_qty > 0:
                tp2_id = self._place_tp_regular(rec, tp2_price, tp2_qty)

            with self._lock:
                rec.sl_order_id = sl_id
                rec.tp1_order_id = tp1_id
                rec.tp2_order_id = tp2_id
                rec.tp1_qty = tp1_qty
                rec.tp2_qty = tp2_qty
            logger.info("Exits armed (REGULAR partial): SL=%s, TP1=%s, TP2=%s", sl_id, tp1_id, tp2_id)
            return

        # Non-partial: try GTT if preferred, otherwise REGULAR
        if self.preferred_exit_mode in ("GTT", "AUTO"):
            try:
                # Two single GTTs (SL & TP). If this fails, fallback to regular.
                # NOTE: Kite's GTT signature may vary; this is wrapped in try.
                sl_gtt = self.kite.place_gtt(
                    trigger_type="single",
                    tradingsymbol=rec.symbol,
                    exchange=rec.exchange,
                    trigger_values=[_round_to_tick(sl_price, rec.tick_size)],
                    last_price=self.get_last_price(rec.symbol) or rec.entry_price,
                    orders=[{
                        "transaction_type": "SELL" if rec.side == "BUY" else "BUY",
                        "quantity": rec.quantity,
                        "order_type": "SL-M" if self.use_slm_exit else "SL",
                        "product": rec.order_product,
                        "price": _round_to_tick(sl_price, rec.tick_size),
                    }],
                )
                tp_gtt = self.kite.place_gtt(
                    trigger_type="single",
                    tradingsymbol=rec.symbol,
                    exchange=rec.exchange,
                    trigger_values=[_round_to_tick(tp_price, rec.tick_size)],
                    last_price=self.get_last_price(rec.symbol) or rec.entry_price,
                    orders=[{
                        "transaction_type": "SELL" if rec.side == "BUY" else "BUY",
                        "quantity": rec.quantity,
                        "order_type": "LIMIT",
                        "product": rec.order_product,
                        "price": _round_to_tick(tp_price, rec.tick_size),
                    }],
                )
                with self._lock:
                    rec.gtt_sl_id = int(sl_gtt)
                    rec.gtt_tp_id = int(tp_gtt)
                logger.info("Exits armed (GTT): SL GTT=%s, TP GTT=%s", sl_gtt, tp_gtt)
                return
            except Exception as e:
                logger.warning("GTT arming failed (%s); falling back to REGULAR exits.", e)

        # Fallback REGULAR exits
        sl_id = self._place_sl_regular(rec, sl_price, rec.quantity)
        tp_id = self._place_tp_regular(rec, tp_price, rec.quantity)
        with self._lock:
            rec.sl_order_id = sl_id
            rec.tp_order_id = tp_id
        logger.info("Exits armed (REGULAR): SL=%s, TP=%s", sl_id, tp_id)

    # ---------------- trailing stop ---------------- #

    def update_trailing_stop(self, record_id: str, current_price: float, atr: float, atr_multiplier: Optional[float] = None) -> bool:
        """
        Tighten SL according to ATR trail (never loosen).
        For REGULAR SL: modify order; for GTT SL: cancel & recreate.
        """
        with self._lock:
            rec = self._active.get(record_id)
        if not rec or not rec.is_open or not self.enable_trailing:
            return False
        if atr is None or atr <= 0:
            return False

        mult = float(atr_multiplier if atr_multiplier is not None else self.trailing_atr_multiplier)
        if mult <= 0:
            return False

        # Compute proposed new SL
        if rec.side == "BUY":
            proposed = current_price - atr * mult
            new_sl = max(float(rec.desired_sl or proposed), proposed)
            # never below original desired SL; allow only tighten (increase)
            if rec.desired_sl is not None:
                new_sl = max(rec.desired_sl, new_sl)
        else:
            proposed = current_price + atr * mult
            new_sl = min(float(rec.desired_sl or proposed), proposed)
            if rec.desired_sl is not None:
                new_sl = min(rec.desired_sl, new_sl)

        new_sl = _round_to_tick(new_sl, rec.tick_size)

        # If SL is unchanged or worse, skip
        old_sl = rec.desired_sl or new_sl
        if rec.side == "BUY" and new_sl <= old_sl:
            return False
        if rec.side == "SELL" and new_sl >= old_sl:
            return False

        # Apply
        try:
            if rec.sl_order_id:  # REGULAR SL
                params = {
                    "variety": rec.order_variety,
                    "order_id": rec.sl_order_id,
                    "quantity": rec.tp1_qty + rec.tp2_qty if rec.partial_enabled else rec.quantity,
                    "trigger_price": new_sl,
                }
                if not self.use_slm_exit:
                    params["price"] = new_sl
                    params["order_type"] = "SL"
                self._kite_modify_order(**params)
            elif rec.gtt_sl_id:  # GTT SL: cancel & recreate
                try:
                    self.kite.cancel_gtt(rec.gtt_sl_id)
                except Exception:
                    pass
                sl_gtt = self.kite.place_gtt(
                    trigger_type="single",
                    tradingsymbol=rec.symbol,
                    exchange=rec.exchange,
                    trigger_values=[new_sl],
                    last_price=self.get_last_price(rec.symbol) or rec.entry_price,
                    orders=[{
                        "transaction_type": "SELL" if rec.side == "BUY" else "BUY",
                        "quantity": rec.quantity,
                        "order_type": "SL-M" if self.use_slm_exit else "SL",
                        "product": rec.order_product,
                        "price": new_sl,
                    }],
                )
                with self._lock:
                    rec.gtt_sl_id = int(sl_gtt)
            else:
                return False

            with self._lock:
                rec.desired_sl = new_sl
            logger.info("Trailing SL updated for %s -> %.2f", rec.symbol, new_sl)
            return True
        except Exception as e:
            logger.error("Failed to update trailing SL for %s: %s", rec.symbol, e)
            return False

    # ---------------- exits / cancel / hard stop ---------------- #

    def exit_order(self, record_id: str, exit_reason: str = "manual") -> bool:
        """Hard exit via MARKET and cancel all linked exits."""
        with self._lock:
            rec = self._active.get(record_id)
        if not rec or not rec.is_open:
            logger.warning("Record %s not found or already closed.", record_id)
            return False

        try:
            # Cancel GTTs
            if rec.gtt_sl_id:
                try: self.kite.cancel_gtt(rec.gtt_sl_id)
                except Exception: pass
            if rec.gtt_tp_id:
                try: self.kite.cancel_gtt(rec.gtt_tp_id)
                except Exception: pass

            # Cancel regular exits
            if rec.sl_order_id:
                self._kite_cancel(variety=rec.order_variety, order_id=rec.sl_order_id)
            if rec.tp_order_id:
                self._kite_cancel(variety=rec.order_variety, order_id=rec.tp_order_id)
            if rec.tp1_order_id:
                self._kite_cancel(variety=rec.order_variety, order_id=rec.tp1_order_id)
            if rec.tp2_order_id:
                self._kite_cancel(variety=rec.order_variety, order_id=rec.tp2_order_id)

            # Market close for remaining qty (use side inverse)
            remaining_qty = max(0, rec.quantity - rec.entry_filled_qty) or rec.quantity
            self._kite_place_order(
                variety=rec.order_variety,
                exchange=rec.exchange,
                tradingsymbol=rec.symbol,
                transaction_type="SELL" if rec.side == "BUY" else "BUY",
                quantity=remaining_qty,
                product=rec.order_product,
                order_type="MARKET",
            )
            with self._lock:
                rec.is_open = False
            logger.info("🟥 Exited %s due to: %s", rec.symbol, exit_reason)
            return True
        except Exception as e:
            logger.error("Failed to exit %s: %s", rec.symbol, e)
            return False

    def cancel_all_orders(self) -> None:
        with self._lock:
            records = list(self._active.values())
        for rec in records:
            try:
                if rec.sl_order_id:
                    self._kite_cancel(variety=rec.order_variety, order_id=rec.sl_order_id)
                if rec.tp_order_id:
                    self._kite_cancel(variety=rec.order_variety, order_id=rec.tp_order_id)
                if rec.tp1_order_id:
                    self._kite_cancel(variety=rec.order_variety, order_id=rec.tp1_order_id)
                if rec.tp2_order_id:
                    self._kite_cancel(variety=rec.order_variety, order_id=rec.tp2_order_id)
                if rec.gtt_sl_id:
                    try: self.kite.cancel_gtt(rec.gtt_sl_id)
                    except Exception: pass
                if rec.gtt_tp_id:
                    try: self.kite.cancel_gtt(rec.gtt_tp_id)
                    except Exception: pass
                # Cancel entries
                for oid in rec.entry_order_ids:
                    self._kite_cancel(variety=rec.order_variety, order_id=oid)
                with self._lock:
                    rec.is_open = False
            except Exception as e:
                logger.warning("Failed to cancel some orders for %s: %s", rec.symbol, e)
        logger.info("All open orders cancelled.")

    # ---------------- sync + OCO enforcement ---------------- #

    def _refresh_entry_fills(self, rec: _OrderRecord, live_orders_by_id: Dict[str, Any]) -> None:
        """Update filled qty/avg price for entry orders."""
        total_filled = 0
        sum_price_qty = 0.0
        for oid in rec.entry_order_ids:
            o = live_orders_by_id.get(oid)
            if not o:
                continue
            filled = int(o.get("filled_quantity", 0))
            price = float(o.get("average_price", 0.0) or 0.0)
            total_filled += filled
            sum_price_qty += price * filled
        if total_filled > 0:
            rec.entry_filled_qty = total_filled
            rec.avg_entry_price = sum_price_qty / max(1, total_filled)

    def _reduce_sl_after_tp1(self, rec: _OrderRecord) -> None:
        """After TP1 fill: reduce SL qty to remaining & hop to breakeven ± ticks."""
        if not rec.partial_enabled or not rec.tp1_order_id or not rec.sl_order_id:
            return
        remaining = max(0, rec.quantity - rec.tp1_qty)
        if remaining <= 0:
            return
        # Modify SL qty
        try:
            self._kite_modify_order(
                variety=rec.order_variety,
                order_id=rec.sl_order_id,
                quantity=_round_to_qty(remaining, rec.lot_size),
            )
        except Exception as e:
            logger.warning("Failed to reduce SL qty after TP1 for %s: %s", rec.symbol, e)

        # Breakeven hop (tighten-only)
        be = rec.avg_entry_price if rec.avg_entry_price > 0 else rec.entry_price
        hop = rec.breakeven_done
        be_ticks = self.breakeven_ticks * rec.tick_size
        if rec.side == "BUY":
            new_sl = max(rec.desired_sl or be, be + be_ticks)
            if rec.desired_sl is None or new_sl > rec.desired_sl:
                try:
                    self._kite_modify_order(
                        variety=rec.order_variety,
                        order_id=rec.sl_order_id,
                        trigger_price=_round_to_tick(new_sl, rec.tick_size),
                        price=_round_to_tick(new_sl, rec.tick_size) if not self.use_slm_exit else None,
                    )
                    rec.desired_sl = new_sl
                    rec.breakeven_done = True
                except Exception:
                    pass
        else:
            new_sl = min(rec.desired_sl or be, be - be_ticks)
            if rec.desired_sl is None or new_sl < rec.desired_sl:
                try:
                    self._kite_modify_order(
                        variety=rec.order_variety,
                        order_id=rec.sl_order_id,
                        trigger_price=_round_to_tick(new_sl, rec.tick_size),
                        price=_round_to_tick(new_sl, rec.tick_size) if not self.use_slm_exit else None,
                    )
                    rec.desired_sl = new_sl
                    rec.breakeven_done = True
                except Exception:
                    pass

    def sync_and_enforce_oco(self) -> List[Tuple[str, float]]:
        """
        Poll broker, synchronize records, and enforce OCO (cancel opposite exit on fill).
        Return fills as (record_id, fill_price).
        """
        if not self.kite:
            return []

        try:
            live = self.kite.orders()
            live_by_id = {str(o.get("order_id")): o for o in live}
        except Exception as e:
            logger.error("Failed to fetch live orders: %s", e)
            return []

        # GTT status (best-effort)
        try:
            gtts = self.kite.gtts()
            gtt_by_id = {int(g.get("id", g.get("gtt_id"))): g for g in gtts}
        except Exception:
            gtt_by_id = {}

        fills: List[Tuple[str, float]] = []

        with self._lock:
            recs = list(self._active.values())

        for rec in recs:
            if not rec.is_open:
                continue

            # update entry fill status
            self._refresh_entry_fills(rec, live_by_id)

            # If entry completely filled and no exits armed (REGULAR), arm them if we have desired SL/TP
            if rec.entry_filled_qty >= rec.quantity and not any([rec.sl_order_id, rec.tp_order_id, rec.tp1_order_id, rec.gtt_sl_id, rec.gtt_tp_id]) and rec.desired_sl and rec.desired_tp:
                logger.info("Entry filled for %s; auto-arming exits.", rec.symbol)
                self.setup_gtt_orders(rec.record_id, rec.desired_sl, rec.desired_tp)

            # Check REGULAR exits for fills
            def _is_complete(oid: Optional[str]) -> bool:
                if not oid:
                    return False
                s = live_by_id.get(oid, {})
                return str(s.get("status", "")).upper() == "COMPLETE"

            sl_hit = _is_complete(rec.sl_order_id)
            tp_hit = _is_complete(rec.tp_order_id) or _is_complete(rec.tp1_order_id) or _is_complete(rec.tp2_order_id)

            # Handle TP1 fill (partial)
            if rec.partial_enabled and rec.tp1_order_id and _is_complete(rec.tp1_order_id):
                logger.info("TP1 filled for %s; reducing SL & breakeven hop.", rec.symbol)
                self._reduce_sl_after_tp1(rec)

            # OCO: if one exit filled, cancel the other(s) and close record
            if sl_hit or tp_hit:
                # capture price
                fill_price = 0.0
                for oid in [rec.sl_order_id, rec.tp_order_id, rec.tp1_order_id, rec.tp2_order_id]:
                    if oid and _is_complete(oid):
                        fill_price = float(live_by_id[oid].get("average_price", 0.0) or 0.0)
                        break

                # cancel the rest
                for oid in [rec.sl_order_id, rec.tp_order_id, rec.tp1_order_id, rec.tp2_order_id]:
                    if oid and not _is_complete(oid):
                        try:
                            self._kite_cancel(variety=rec.order_variety, order_id=oid)
                        except Exception:
                            pass

                with self._lock:
                    rec.is_open = False
                fills.append((rec.record_id, fill_price))
                logger.info("OCO closed %s @ %.2f (sl_hit=%s, tp_hit=%s)", rec.symbol, fill_price, sl_hit, tp_hit)
                continue

            # GTT paths (best-effort): if either GTT fired, mark closed
            if rec.gtt_sl_id and isinstance(gtt_by_id.get(rec.gtt_sl_id), dict):
                gs = gtt_by_id[rec.gtt_sl_id]
                if str(gs.get("status", "")).lower() in ("triggered", "cancelled", "deleted", "rejected"):
                    with self._lock:
                        rec.is_open = False
                    fills.append((rec.record_id, 0.0))
                    continue
            if rec.gtt_tp_id and isinstance(gtt_by_id.get(rec.gtt_tp_id), dict):
                gs = gtt_by_id[rec.gtt_tp_id]
                if str(gs.get("status", "")).lower() in ("triggered", "cancelled", "deleted", "rejected"):
                    with self._lock:
                        rec.is_open = False
                    fills.append((rec.record_id, 0.0))
                    continue

        return fills
