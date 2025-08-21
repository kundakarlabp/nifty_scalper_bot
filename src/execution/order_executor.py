# src/execution/order_executor.py
from __future__ import annotations
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    from kiteconnect import KiteConnect  # type: ignore
    from kiteconnect.exceptions import NetworkException, TokenException, InputException  # type: ignore
except Exception:  # pragma: no cover
    KiteConnect = None  # type: ignore
    NetworkException = TokenException = InputException = Exception  # type: ignore

try:
    from src.data.source import DataSource  # type: ignore
except Exception:  # pragma: no cover
    DataSource = object  # type: ignore

try:
    from src.config import settings  # type: ignore
except Exception:  # pragma: no cover
    settings = None  # type: ignore

log = logging.getLogger(__name__)


def _retry_call(fn, *args, tries: int = 3, base_delay: float = 0.25, **kwargs):
    """Lightweight retry wrapper with exponential backoff for transient Kite errors."""
    delay = base_delay
    for i in range(tries):
        try:
            return fn(*args, **kwargs)
        except (NetworkException, TokenException, InputException) as e:
            if i == tries - 1:
                raise
            log.warning(
                "Transient broker error in %s (try %d/%d): %s",
                getattr(fn, "__name__", "call"),
                i + 1,
                tries,
                e,
            )
            time.sleep(delay)
            delay *= 2.0


def _round_to_tick(x: float, tick: float) -> float:
    try:
        if tick <= 0:
            return float(x)
        return round(x / tick) * tick
    except Exception:
        return float(x)


def _round_to_step(qty: int, step: int) -> int:
    try:
        if step <= 0:
            return int(qty)
        return int(math.floor(qty / step) * step)
    except Exception:
        return int(qty)


def _chunks(total: int, chunk: int) -> List[int]:
    if chunk <= 0:
        return [total]
    out, left = [], total
    while left > 0:
        take = min(chunk, left)
        out.append(take)
        left -= take
    return out


@dataclass
class _OrderRecord:
    order_id: str
    instrument_token: int
    symbol: str
    side: str  # BUY/SELL
    quantity: int
    entry_price: float
    tick_size: float = 0.05

    # state
    is_open: bool = True
    filled_qty: int = 0

    # exits
    sl_gtt_id: Optional[int] = None
    sl_price: Optional[float] = None

    tp1_order_id: Optional[str] = None
    tp2_order_id: Optional[str] = None
    tp_price: Optional[float] = None
    tp1_done: bool = False

    # config/behavior
    exchange: str = "NFO"
    product: str = "NRML"
    variety: str = "regular"
    entry_order_type: str = "LIMIT"
    freeze_qty: int = 900
    use_slm_exit: bool = True  # kept for compatibility; GTT leg uses MARKET

    # partial/tp/breakeven/trailing controls
    partial_enabled: bool = False
    tp1_ratio: float = 0.5           # 0..1
    breakeven_ticks: int = 2
    trailing_enabled: bool = True
    trailing_mult: float = 1.5

    # entry children
    child_order_ids: List[str] = field(default_factory=list)

    @property
    def record_id(self) -> str:
        return self.order_id

    @property
    def remaining_qty(self) -> int:
        return max(0, self.quantity - self.filled_qty)

    def side_sign(self) -> int:
        return 1 if self.side.upper() == "BUY" else -1


class OrderExecutor:
    """
    Entry -> REGULAR
    Exits -> TP1/TP2 REGULAR LIMITs + one SL GTT (MARKET) for remaining qty.
    Our code enforces OCO: when one exit fills, we cancel the others.
    """

    def __init__(self, config: Any, kite: Optional[KiteConnect], data_source: Optional[DataSource]) -> None:
        self.kite = kite
        self.data_source = data_source
        self._lock = threading.Lock()
        self._active: Dict[str, _OrderRecord] = {}

        # executor config sourced from settings.executor safely
        self.exchange = getattr(config, "exchange", "NFO")
        self.product = getattr(config, "order_product", "NRML")
        self.variety = getattr(config, "order_variety", "regular")
        self.entry_order_type = getattr(config, "entry_order_type", "LIMIT")
        self.tick_size = float(getattr(config, "tick_size", 0.05))
        self.freeze_qty = int(getattr(config, "exchange_freeze_qty", 900))
        self.lot_size = int(getattr(getattr(settings, "instruments", object()), "nifty_lot_size", 75))

        self.partial_enable = bool(getattr(config, "partial_tp_enable", False))
        self.tp1_ratio = float(getattr(config, "tp1_qty_ratio", 0.5))
        self.breakeven_ticks = int(getattr(config, "breakeven_ticks", 2))
        self.enable_trailing = bool(getattr(config, "enable_trailing", True))
        self.trailing_mult = float(getattr(config, "trailing_atr_multiplier", 1.5))
        self.use_slm_exit = bool(getattr(config, "use_slm_exit", True))

    # --------- read helpers ---------
    def get_active_orders(self) -> List[_OrderRecord]:
        with self._lock:
            return list(self._active.values())

    def get_positions_kite(self) -> Dict[str, Any]:
        try:
            if not self.kite:
                return {}
            data = _retry_call(self.kite.positions, tries=2) or {}
            day = data.get("day", [])
            return {p.get("tradingsymbol"): p for p in day}
        except Exception as e:
            log.error("positions() failed: %s", e)
            return {}

    # --------- entry placement ---------
    def place_entry_order(self, token: int, symbol: str, side: str, quantity: int, price: float) -> Optional[str]:
        if not self.kite:
            log.warning("place_entry_order: kite is None (shadow).")
            return None
        if quantity <= 0:
            log.warning("place_entry_order: quantity <= 0")
            return None

        qty = _round_to_step(quantity, self.lot_size)
        if qty <= 0:
            return None

        with self._lock:
            for rec in self._active.values():
                if rec.symbol == symbol and rec.is_open:
                    log.warning("open record exists on %s; skip new entry.", symbol)
                    return None

        parts = _chunks(qty, self.freeze_qty if self.exchange.upper().startswith("NFO") else qty)
        child_ids: List[str] = []
        record_id: Optional[str] = None

        for q in parts:
            params = dict(
                variety=self.variety, exchange=self.exchange,
                tradingsymbol=symbol, transaction_type=side.upper(),
                quantity=int(q), product=self.product,
                order_type=self.entry_order_type.upper(),
            )
            if self.entry_order_type.upper() == "LIMIT":
                params["price"] = _round_to_tick(float(price), self.tick_size)

            try:
                oid = _retry_call(self.kite.place_order, tries=2, **params)  # <-- tries handled by wrapper
                child_ids.append(oid)
                record_id = record_id or oid
                log.info(
                    "Entry child placed: %s %s qty=%d price=%s -> %s",
                    symbol, side, q, params.get("price"), oid
                )
            except Exception as e:
                log.error("place_order failed chunk %d: %s", q, e)

        if not record_id:
            return None

        rec = _OrderRecord(
            order_id=record_id, instrument_token=int(token), symbol=symbol, side=side.upper(),
            quantity=qty, entry_price=float(price), tick_size=self.tick_size,
            exchange=self.exchange, product=self.product, variety=self.variety,
            entry_order_type=self.entry_order_type, freeze_qty=self.freeze_qty,
            use_slm_exit=self.use_slm_exit, partial_enabled=self.partial_enable,
            tp1_ratio=self.tp1_ratio, breakeven_ticks=self.breakeven_ticks,
            trailing_enabled=self.enable_trailing, trailing_mult=self.trailing_mult,
            child_order_ids=child_ids,
        )
        with self._lock:
            self._active[rec.record_id] = rec
        return rec.record_id

    # --------- exits setup (TP1/TP2 + SL GTT) ---------
    def setup_gtt_orders(self, record_id: str, sl_price: float, tp_price: float) -> None:
        rec = None
        with self._lock:
            rec = self._active.get(record_id)
        if not rec or not rec.is_open or not self.kite:
            return

        qty = rec.remaining_qty or rec.quantity
        if qty <= 0:
            return

        exit_side = "SELL" if rec.side == "BUY" else "BUY"
        tp_price = _round_to_tick(float(tp_price), rec.tick_size)
        sl_price = _round_to_tick(float(sl_price), rec.tick_size)

        # Cancel previous targets if any (idempotent)
        for old_tid in (rec.tp1_order_id, rec.tp2_order_id):
            if old_tid:
                try:
                    _retry_call(self.kite.cancel_order, tries=2, variety=rec.variety, order_id=old_tid)
                except Exception:
                    pass
        rec.tp1_order_id = rec.tp2_order_id = None

        # Partial qty split
        if rec.partial_enabled:
            q_tp1 = _round_to_step(int(round(qty * max(0.0, min(1.0, rec.tp1_ratio)))), self.lot_size)
            q_tp1 = min(max(q_tp1, self.lot_size), qty)  # at least one lot, not above total
            q_tp2 = qty - q_tp1
        else:
            q_tp1, q_tp2 = 0, qty  # no partials -> all in TP2

        # place TP1
        if q_tp1 > 0:
            try:
                rec.tp1_order_id = _retry_call(
                    self.kite.place_order, tries=2,
                    variety=rec.variety, exchange=rec.exchange, tradingsymbol=rec.symbol,
                    transaction_type=exit_side, quantity=int(q_tp1), product=rec.product,
                    order_type="LIMIT", price=tp_price
                )
                log.info("TP1 placed %s qty=%d @ %.2f -> %s", rec.symbol, q_tp1, tp_price, rec.tp1_order_id)
            except Exception as e:
                log.error("TP1 place failed: %s", e)

        # place TP2
        if q_tp2 > 0:
            try:
                rec.tp2_order_id = _retry_call(
                    self.kite.place_order, tries=2,
                    variety=rec.variety, exchange=rec.exchange, tradingsymbol=rec.symbol,
                    transaction_type=exit_side, quantity=int(q_tp2), product=rec.product,
                    order_type="LIMIT", price=tp_price
                )
                log.info("TP2 placed %s qty=%d @ %.2f -> %s", rec.symbol, q_tp2, tp_price, rec.tp2_order_id)
            except Exception as e:
                log.error("TP2 place failed: %s", e)

        rec.tp_price = tp_price

        # --- SL as single GTT MARKET for total remaining qty ---
        self._refresh_sl_gtt(rec, sl_price=sl_price, qty=qty)

    def _refresh_sl_gtt(self, rec: _OrderRecord, *, sl_price: float, qty: int) -> None:
        if not self.kite:
            return
        # cancel previous SL GTT
        if rec.sl_gtt_id is not None:
            try:
                _retry_call(self.kite.cancel_gtt, tries=2, gtt_id=rec.sl_gtt_id)
            except Exception:
                pass
            rec.sl_gtt_id = None

        # Zerodha GTT API needs: trigger_values=[...], last_price (current LTP), and order payloads (LIMIT/MARKET).
        # We emulate SL-M by using order_type="MARKET" with a single trigger.
        exit_side = "SELL" if rec.side == "BUY" else "BUY"

        # try to get a reasonable last_price (required by GTT)
        last_price = None
        try:
            if self.data_source and hasattr(self.data_source, "get_last_price"):
                last_price = self.data_source.get_last_price(rec.symbol)  # type: ignore[attr-defined]
        except Exception:
            last_price = None
        if last_price is None:
            # fallback if LTP not available
            last_price = float(sl_price)

        sl_leg = {
            "exchange": rec.exchange,
            "tradingsymbol": rec.symbol,
            "transaction_type": exit_side,
            "quantity": int(qty),
            "product": rec.product,
            "order_type": "MARKET",   # GTT supports MARKET/LIMIT; we use MARKET for stop
            "price": 0.0,             # ignored for MARKET
        }

        try:
            gid = _retry_call(
                self.kite.place_gtt,
                tries=2,
                trigger_type="single",
                tradingsymbol=rec.symbol,
                exchange=rec.exchange,
                trigger_values=[sl_price],   # <-- FIX: correct param name & type
                last_price=float(last_price),
                orders=[sl_leg],
            )
            # Kite returns dict or id; normalize to int if possible
            rec.sl_gtt_id = int(gid if isinstance(gid, int) else gid.get("id") or gid.get("gtt_id"))
            rec.sl_price = sl_price
            log.info("SL GTT set for %s qty=%d @ %.2f -> %s", rec.symbol, qty, sl_price, rec.sl_gtt_id)
        except Exception as e:
            log.error("SL GTT placement failed: %s", e)

    # --------- trailing stop (tighten-only) ---------
    def update_trailing_stop(self, record_id: str, *, current_price: float, atr: float, atr_multiplier: Optional[float] = None) -> None:
        rec = None
        with self._lock:
            rec = self._active.get(record_id)
        if not rec or not rec.is_open or not self.kite:
            return
        if not rec.trailing_enabled or atr <= 0 or current_price <= 0:
            return

        m = float(atr_multiplier if atr_multiplier is not None else rec.trailing_mult)
        if rec.side == "BUY":
            proposed = _round_to_tick(current_price - m * atr, rec.tick_size)
            if rec.tp1_done and rec.breakeven_ticks > 0:
                proposed = max(
                    proposed,
                    _round_to_tick(rec.entry_price + rec.breakeven_ticks * rec.tick_size, rec.tick_size),
                )
            if rec.sl_price is not None and proposed <= rec.sl_price:
                return
        else:
            proposed = _round_to_tick(current_price + m * atr, rec.tick_size)
            if rec.tp1_done and rec.breakeven_ticks > 0:
                proposed = min(
                    proposed,
                    _round_to_tick(rec.entry_price - rec.breakeven_ticks * rec.tick_size, rec.tick_size),
                )
            if rec.sl_price is not None and proposed >= rec.sl_price:
                return

        qty = rec.remaining_qty or rec.quantity
        if qty <= 0:
            return
        self._refresh_sl_gtt(rec, sl_price=proposed, qty=qty)

    # --------- manual exit ---------
    def exit_order(self, record_id: str, exit_reason: str = "manual") -> bool:
        rec = None
        with self._lock:
            rec = self._active.get(record_id)
        if not rec or not rec.is_open or not self.kite:
            return False

        # Cancel SL GTT and TPs
        if rec.sl_gtt_id is not None:
            try:
                _retry_call(self.kite.cancel_gtt, tries=2, gtt_id=rec.sl_gtt_id)
            except Exception:
                pass
            rec.sl_gtt_id = None
        for tid in (rec.tp1_order_id, rec.tp2_order_id):
            if tid:
                try:
                    _retry_call(self.kite.cancel_order, tries=2, variety=rec.variety, order_id=tid)
                except Exception:
                    pass

        qty = rec.remaining_qty or rec.quantity
        if qty <= 0:
            rec.is_open = False
            return True

        try:
            oid = _retry_call(
                self.kite.place_order,
                tries=2,
                variety=rec.variety, exchange=rec.exchange, tradingsymbol=rec.symbol,
                transaction_type=("SELL" if rec.side == "BUY" else "BUY"),
                quantity=int(qty), product=rec.product, order_type="MARKET"
            )
            log.info("Exit %s qty=%d -> %s (%s)", rec.symbol, qty, oid, exit_reason)
            rec.is_open = False
            return True
        except Exception as e:
            log.error("exit_order failed: %s", e)
            return False

    def cancel_all_orders(self) -> None:
        if not self.kite:
            with self._lock:
                self._active.clear()
            return
        with self._lock:
            recs = list(self._active.values())
        for rec in recs:
            # cancel all children and exits
            for oid in rec.child_order_ids:
                try:
                    _retry_call(self.kite.cancel_order, tries=2, variety=rec.variety, order_id=oid)
                except Exception:
                    pass
            for tid in (rec.tp1_order_id, rec.tp2_order_id):
                if tid:
                    try:
                        _retry_call(self.kite.cancel_order, tries=2, variety=rec.variety, order_id=tid)
                    except Exception:
                        pass
            if rec.sl_gtt_id is not None:
                try:
                    _retry_call(self.kite.cancel_gtt, tries=2, gtt_id=rec.sl_gtt_id)
                except Exception:
                    pass
            rec.is_open = False
        with self._lock:
            self._active = {k: v for k, v in self._active.items() if v.is_open}

    # --------- sync (detect TP1/TP2 fills, adjust SL, report) ---------
    def sync_and_enforce_oco(self) -> List[Tuple[str, float]]:
        if not self.kite:
            return []
        fills: List[Tuple[str, float]] = []

        # 1) orders snapshot
        try:
            live_orders = _retry_call(self.kite.orders, tries=2) or []
        except Exception as e:
            log.error("orders() failed: %s", e)
            live_orders = []
        omap: Dict[str, Dict[str, Any]] = {o.get("order_id"): o for o in live_orders if isinstance(o, dict)}

        # 2) gtts snapshot (for SL status)
        try:
            live_gtts = _retry_call(self.kite.gtts, tries=2) or []
        except Exception as e:
            log.error("gtts() failed: %s", e)
            live_gtts = []
        gmap: Dict[int, Dict[str, Any]] = {}
        for g in live_gtts:
            gid = g.get("id") or g.get("gtt_id")
            if gid is not None:
                gmap[int(gid)] = g

        with self._lock:
            recs = list(self._active.values())

        for rec in recs:
            if not rec.is_open:
                continue

            # update entry filled qty
            total_filled = 0
            for oid in rec.child_order_ids:
                o = omap.get(oid, {})
                if (o.get("status") or "").upper() == "COMPLETE":
                    total_filled += int(o.get("filled_quantity", o.get("quantity", 0)) or 0)
            if total_filled > rec.filled_qty:
                rec.filled_qty = total_filled

            # TP1/TP2 status (OCO enforcement on success)
            tp1_complete = False
            tp2_complete = False

            if rec.tp1_order_id:
                o1 = omap.get(rec.tp1_order_id, {})
                tp1_complete = (o1.get("status", "") or "").upper() == "COMPLETE"
            else:
                tp1_complete = True  # no TP1 means treat as done for closure logic

            if rec.tp2_order_id:
                o2 = omap.get(rec.tp2_order_id, {})
                tp2_complete = (o2.get("status", "") or "").upper() == "COMPLETE"

            # If TP1 just filled: hop SL to BE and shrink SL qty to remaining
            if rec.partial_enabled and not rec.tp1_done and rec.tp1_order_id and tp1_complete:
                rec.tp1_done = True
                rem = rec.remaining_qty or rec.quantity
                if rem > 0:
                    be = _round_to_tick(
                        rec.entry_price + rec.side_sign() * rec.breakeven_ticks * rec.tick_size,
                        rec.tick_size
                    )
                    self._refresh_sl_gtt(rec, sl_price=be, qty=rem)
                # cancel TP1 idempotently (already complete; no-op if cancel fails)

            # If TP2 filled OR both TPs complete => position closed, cancel SL GTT
            if tp2_complete and tp1_complete:
                if rec.sl_gtt_id is not None:
                    try:
                        _retry_call(self.kite.cancel_gtt, tries=2, gtt_id=rec.sl_gtt_id)
                    except Exception:
                        pass
                    rec.sl_gtt_id = None
                rec.is_open = False
                fills.append((rec.record_id, float(rec.tp_price or 0.0)))
                continue

            # SL triggered? (GTT state)
            if rec.sl_gtt_id is not None:
                g = gmap.get(rec.sl_gtt_id, {})
                g_state = (g.get("status") or g.get("state") or "").lower()
                if g_state in ("triggered", "cancelled", "deleted", "expired"):
                    # Cancel any pending TP legs to enforce OCO
                    for tid in (rec.tp1_order_id, rec.tp2_order_id):
                        if tid:
                            try:
                                _retry_call(self.kite.cancel_order, tries=2, variety=rec.variety, order_id=tid)
                            except Exception:
                                pass
                    rec.is_open = False
                    fills.append((rec.record_id, float(rec.sl_price or 0.0)))
                    continue

        # purge closed
        if fills:
            with self._lock:
                self._active = {k: v for k, v in self._active.items() if v.is_open}
        return fills