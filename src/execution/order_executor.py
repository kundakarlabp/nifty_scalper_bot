# Path: src/execution/order_executor.py
from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# --- Optional broker SDK (graceful fallback in paper mode) ---
try:
    from kiteconnect import KiteConnect  # type: ignore
    from kiteconnect.exceptions import (  # type: ignore
        NetworkException,
        TokenException,
        InputException,
    )
except Exception:
    KiteConnect = None  # type: ignore
    NetworkException = TokenException = InputException = Exception  # type: ignore

# --- Local imports (settings is optional for tests) ---
try:
    from src.config import settings
except Exception:  # pragma: no cover
    settings = None  # type: ignore

log = logging.getLogger(__name__)


# ------------------- small helpers -------------------
def _retry_call(fn, *args, tries: int = 3, base_delay: float = 0.25, **kwargs):
    """Retry transient Kite errors with exponential backoff."""
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
        return round(float(x) / tick) * tick if tick > 0 else float(x)
    except Exception:
        return float(x)


def _round_to_step(qty: int, step: int) -> int:
    try:
        return int(math.floor(qty / step) * step) if step > 0 else int(qty)
    except Exception:
        return int(qty)


def _chunks(total: int, chunk: int) -> List[int]:
    """Split a quantity into exchange-compliant child orders."""
    if total <= 0:
        return []
    if chunk <= 0 or chunk >= total:
        return [total]
    out, left = [], total
    while left > 0:
        take = min(chunk, left)
        out.append(take)
        left -= take
    return out


# ------------------- in-memory order record -------------------
@dataclass
class _OrderRecord:
    order_id: str
    instrument_token: int
    symbol: str
    side: str  # BUY/SELL
    quantity: int
    entry_price: float
    tick_size: float = 0.05

    is_open: bool = True
    filled_qty: int = 0

    # SL GTT
    sl_gtt_id: Optional[int] = None
    sl_price: Optional[float] = None

    # TPs (optional split)
    tp1_order_id: Optional[str] = None
    tp2_order_id: Optional[str] = None
    tp_price: Optional[float] = None
    tp1_done: bool = False

    # broker params
    exchange: str = "NFO"
    product: str = "NRML"
    variety: str = "regular"
    entry_order_type: str = "LIMIT"
    freeze_qty: int = 900
    use_slm_exit: bool = True

    # trailing / partials
    partial_enabled: bool = False
    tp1_ratio: float = 0.5  # 0..1
    breakeven_ticks: int = 2
    trailing_enabled: bool = True
    trailing_mult: float = 1.5

    # children
    child_order_ids: List[str] = field(default_factory=list)

    @property
    def record_id(self) -> str:
        return self.order_id

    @property
    def remaining_qty(self) -> int:
        return max(0, self.quantity - self.filled_qty)

    def side_sign(self) -> int:
        return 1 if self.side.upper() == "BUY" else -1


# ------------------- executor -------------------
class OrderExecutor:
    """
    Thin, robust wrapper around Kite order placement/management.
    Matches the runner/telegram expectations:
      - place_order(payload) -> bool/str (record_id/"PAPER"/None)
      - setup_gtt_orders(record_id, sl_price, tp_price)
      - update_trailing_stop(...)
      - sync_and_enforce_oco() -> fills list
      - get_active_orders(), get_positions_kite(), cancel_all_orders()
      - health_check(), shutdown()

    Extra:
      - set_live_broker(kite) / set_kite(kite) to flip live/paper mode safely
      - togglers for trailing/partial/breakeven/tp1_ratio
      - last_error for diag
    """

    def __init__(self, kite: Optional[KiteConnect], telegram_controller: Any = None) -> None:
        self._lock = threading.Lock()
        self.kite = kite
        self._live = kite is not None
        self.telegram = telegram_controller
        self._active: Dict[str, _OrderRecord] = {}
        self.last_error: Optional[str] = None

        # execution config (default-safe if settings missing)
        ex = getattr(settings, "executor", object()) if settings else object()
        ins = getattr(settings, "instruments", object()) if settings else object()

        self.exchange = getattr(ex, "exchange", "NFO")
        self.product = getattr(ex, "order_product", "NRML")
        self.variety = getattr(ex, "order_variety", "regular")
        self.entry_order_type = getattr(ex, "entry_order_type", "LIMIT")
        self.tick_size = float(getattr(ex, "tick_size", 0.05))
        self.freeze_qty = int(getattr(ex, "exchange_freeze_qty", 900))
        self.lot_size = int(getattr(ins, "nifty_lot_size", 75))  # NIFTY lot

        # exits / risk
        self.partial_enable = bool(getattr(ex, "partial_tp_enable", False))
        self.tp1_ratio = float(getattr(ex, "tp1_qty_ratio", 0.5))
        self.breakeven_ticks = int(getattr(ex, "breakeven_ticks", 2))
        self.enable_trailing = bool(getattr(ex, "enable_trailing", True))
        self.trailing_mult = float(getattr(ex, "trailing_atr_multiplier", 1.5))
        self.use_slm_exit = bool(getattr(ex, "use_slm_exit", True))

        # track last notification to throttle duplicates
        self._last_notification: Tuple[str, float] = ("", 0.0)

    # ----------- live/paper control ----------
    def set_live_broker(self, kite: Optional[KiteConnect]) -> None:
        """Hot-swap Kite session (None => paper)."""
        with self._lock:
            self.kite = kite
            self._live = kite is not None

    # Back-compat alias used by some runners
    def set_kite(self, kite: Optional[KiteConnect]) -> None:
        self.set_live_broker(kite)

    # ----------- public info ----------
    def get_active_orders(self) -> List[_OrderRecord]:
        """Return a snapshot of open records; touch broker for fresh state."""
        self.last_error = None
        with self._lock:
            recs = list(self._active.values())
        if not self.kite:
            return recs
        try:
            _retry_call(self.kite.orders, tries=2)
        except Exception as e:
            self.last_error = f"orders: {e}"
            log.error("orders() failed: %s", e)
        return recs

    @property
    def open_count(self) -> int:
        with self._lock:
            return sum(1 for r in self._active.values() if r.is_open)

    def get_positions_kite(self) -> Dict[str, Any]:
        self.last_error = None
        if not self.kite:
            return {}
        try:
            data = _retry_call(self.kite.positions, tries=2) or {}
            return {p.get("tradingsymbol"): p for p in data.get("day", [])}
        except Exception as e:
            self.last_error = f"positions: {e}"
            log.error("positions() failed: %s", e)
            return {}

    # ----------- entry API (runner calls place_order with payload) ----------
    def place_order(self, payload: Dict[str, Any]) -> Optional[str]:
        """
        payload = {
          action: 'BUY'|'SELL',
          quantity: int (in units, not lots),
          entry_price: float,
          stop_loss: float,
          take_profit: float,
          option_type: 'CE'|'PE',
          strike: float,
          (symbol|instrument_token) optional if you resolve elsewhere
        }
        """
        self.last_error = None

        action = str(payload.get("action", "")).upper()
        qty = int(payload.get("quantity", 0))
        price = float(payload.get("entry_price", 0))
        symbol = payload.get("symbol") or self._infer_symbol(payload)
        token = int(payload.get("instrument_token") or 0)

        if action not in ("BUY", "SELL") or qty <= 0 or price <= 0 or not symbol:
            self.last_error = "invalid_payload"
            log.warning("Invalid order payload: %s", payload)
            return None

        # round down to lot step
        qty = _round_to_step(qty, self.lot_size)
        if qty <= 0:
            self.last_error = "qty_rounded_to_zero"
            return None

        # prevent duplicate symbol entries (case-insensitive)
        norm_symbol = str(symbol).upper()
        with self._lock:
            if any(rec.symbol.upper() == norm_symbol and rec.is_open for rec in self._active.values()):
                self.last_error = "duplicate_symbol_open"
                log.warning("Open record exists on %s; skip new entry.", norm_symbol)
                return None

        if not self.kite:
            # PAPER MODE: create a synthetic record so the rest of the flow works
            rid = f"PAPER-{int(time.time() * 1000)}"
            rec = _OrderRecord(
                order_id=rid,
                instrument_token=token,
                symbol=symbol,
                side=action,
                quantity=qty,
                entry_price=float(price),
                tick_size=self.tick_size,
                exchange=self.exchange,
                product=self.product,
                variety=self.variety,
                entry_order_type=self.entry_order_type,
                freeze_qty=self.freeze_qty,
                use_slm_exit=self.use_slm_exit,
                partial_enabled=self.partial_enable,
                tp1_ratio=self.tp1_ratio,
                breakeven_ticks=self.breakeven_ticks,
                trailing_enabled=self.enable_trailing,
                trailing_mult=self.trailing_mult,
                child_order_ids=[],
            )
            with self._lock:
                self._active[rec.record_id] = rec
            # Notify (best-effort)
            self._notify(f"🧪 PAPER entry: {symbol} {action} qty={qty} @ {price}")
            log.info("Paper entry recorded: %s %s qty=%d @%s", symbol, action, qty, price)
            return rec.record_id

        # LIVE MODE
        parts = _chunks(qty, self.freeze_qty if self.exchange.upper() == "NFO" else qty)
        child_ids: List[str] = []
        record_id: Optional[str] = None

        for q in parts:
            params = dict(
                variety=self.variety,
                exchange=self.exchange,
                tradingsymbol=symbol,
                transaction_type=action,
                quantity=int(q),
                product=self.product,
                order_type=self.entry_order_type.upper(),
            )
            if self.entry_order_type.upper() == "LIMIT":
                params["price"] = _round_to_tick(price, self.tick_size)

            try:
                oid = _retry_call(self.kite.place_order, **params)
                if not oid:
                    continue
                child_ids.append(oid)
                record_id = record_id or oid
                log.info(
                    "Entry child placed: %s %s qty=%d price=%s -> %s",
                    symbol, action, q, params.get("price"), oid,
                )
            except Exception as e:
                self.last_error = f"place_order: {e}"
                log.error("place_order failed: %s", e)

        if not record_id:
            return None

        rec = _OrderRecord(
            order_id=record_id,
            instrument_token=token,
            symbol=symbol,
            side=action,
            quantity=qty,
            entry_price=float(price),
            tick_size=self.tick_size,
            exchange=self.exchange,
            product=self.product,
            variety=self.variety,
            entry_order_type=self.entry_order_type,
            freeze_qty=self.freeze_qty,
            use_slm_exit=self.use_slm_exit,
            partial_enabled=self.partial_enable,
            tp1_ratio=self.tp1_ratio,
            breakeven_ticks=self.breakeven_ticks,
            trailing_enabled=self.enable_trailing,
            trailing_mult=self.trailing_mult,
            child_order_ids=child_ids,
        )
        with self._lock:
            self._active[rec.record_id] = rec

        self._notify(f"✅ LIVE entry: {symbol} {action} qty={qty} @ {price} (rid={rec.record_id})")
        return rec.record_id

    # ----------- SL/TP orchestration ----------
    def setup_gtt_orders(self, record_id: str, sl_price: float, tp_price: float) -> None:
        """Place/refresh SL-GTT and TP limit orders for an active record."""
        with self._lock:
            rec = self._active.get(record_id)
        if not rec or not rec.is_open or not self.kite:
            return

        qty = rec.remaining_qty or rec.quantity
        if qty <= 0:
            return

        exit_side = "SELL" if rec.side == "BUY" else "BUY"
        tp_price = _round_to_tick(tp_price, rec.tick_size)
        sl_price = _round_to_tick(sl_price, rec.tick_size)

        # cancel old TPs if any
        for old in (rec.tp1_order_id, rec.tp2_order_id):
            if old:
                try:
                    _retry_call(self.kite.cancel_order, variety=rec.variety, order_id=old, tries=2)
                except Exception:
                    log.debug("Failed to cancel TP order %s", old, exc_info=True)
        rec.tp1_order_id = rec.tp2_order_id = None

        # partial split
        if rec.partial_enabled:
            q_tp1 = _round_to_step(int(round(qty * max(0, min(1, rec.tp1_ratio)))), self.lot_size)
            q_tp1 = min(max(q_tp1, self.lot_size), qty)
            q_tp2 = qty - q_tp1
        else:
            q_tp1, q_tp2 = 0, qty

        if q_tp1 > 0:
            try:
                rec.tp1_order_id = _retry_call(
                    self.kite.place_order,
                    variety=rec.variety,
                    exchange=rec.exchange,
                    tradingsymbol=rec.symbol,
                    transaction_type=exit_side,
                    quantity=q_tp1,
                    product=rec.product,
                    order_type="LIMIT",
                    price=tp_price,
                    tries=2,
                )
            except Exception as e:
                self.last_error = f"tp1: {e}"
                log.error("TP1 failed: %s", e)
        if q_tp2 > 0:
            try:
                rec.tp2_order_id = _retry_call(
                    self.kite.place_order,
                    variety=rec.variety,
                    exchange=rec.exchange,
                    tradingsymbol=rec.symbol,
                    transaction_type=exit_side,
                    quantity=q_tp2,
                    product=rec.product,
                    order_type="LIMIT",
                    price=tp_price,
                    tries=2,
                )
            except Exception as e:
                self.last_error = f"tp2: {e}"
                log.error("TP2 failed: %s", e)
        rec.tp_price = tp_price

        self._refresh_sl_gtt(rec, sl_price=sl_price, qty=qty)

    def _refresh_sl_gtt(self, rec: _OrderRecord, *, sl_price: float, qty: int) -> None:
        """(Re)create SL GTT as single-leg order."""
        if not self.kite or qty <= 0:
            return
        # cancel existing GTT
        if rec.sl_gtt_id:
            try:
                _retry_call(self.kite.cancel_gtt, rec.sl_gtt_id, tries=2)
            except Exception:
                log.debug("Failed to cancel existing SL GTT %s", rec.sl_gtt_id, exc_info=True)
            rec.sl_gtt_id = None

        exit_side = "SELL" if rec.side == "BUY" else "BUY"
        sl_leg = {
            "exchange": rec.exchange,
            "tradingsymbol": rec.symbol,
            "transaction_type": exit_side,
            "quantity": int(qty),
            "order_type": "SL-M" if rec.use_slm_exit else "SL",
            "product": rec.product,
            "price": None if rec.use_slm_exit else sl_price,
            "trigger_price": sl_price,
        }
        try:
            # Note: Real Kite API expects trigger_values list etc.
            # We keep current signature as per your existing integration.
            res = _retry_call(
                self.kite.place_gtt,
                trigger_type="single",
                tradingsymbol=rec.symbol,
                exchange=rec.exchange,
                trigger_price=sl_price,
                last_price=None,
                orders=[sl_leg],
                tries=2,
            )
            gid = res["id"] if isinstance(res, dict) else int(res)
            rec.sl_gtt_id, rec.sl_price = gid, sl_price
        except Exception as e:
            self.last_error = f"sl_gtt: {e}"
            log.error("SL GTT failed: %s", e)

    # ----------- trailing stop maintenance ----------
    def update_trailing_stop(
        self,
        record_id: str,
        *,
        current_price: float,
        atr: float,
        atr_multiplier: Optional[float] = None,
    ) -> None:
        with self._lock:
            rec = self._active.get(record_id)
        if not rec or not rec.is_open or not self.kite:
            return
        if not rec.trailing_enabled or atr <= 0 or current_price <= 0:
            return

        m = float(atr_multiplier or rec.trailing_mult)
        if rec.side == "BUY":
            proposed = _round_to_tick(current_price - m * atr, rec.tick_size)
            if rec.tp1_done:
                proposed = max(
                    proposed,
                    _round_to_tick(rec.entry_price + rec.breakeven_ticks * rec.tick_size, rec.tick_size),
                )
            if rec.sl_price and proposed <= rec.sl_price:
                return
        else:
            proposed = _round_to_tick(current_price + m * atr, rec.tick_size)
            if rec.tp1_done:
                proposed = min(
                    proposed,
                    _round_to_tick(rec.entry_price - rec.breakeven_ticks * rec.tick_size, rec.tick_size),
                )
            if rec.sl_price and proposed >= rec.sl_price:
                return

        self._refresh_sl_gtt(rec, sl_price=proposed, qty=rec.remaining_qty or rec.quantity)

    # ----------- polling / OCO enforcement ----------
    def sync_and_enforce_oco(self) -> List[Tuple[str, float]]:
        """Refresh local records from broker; return fills as [(record_id, px), ...]."""
        if not self.kite:
            return []
        fills: List[Tuple[str, float]] = []
        try:
            omap = {o.get("order_id"): o for o in _retry_call(self.kite.orders, tries=2) or []}
        except Exception as e:
            self.last_error = f"orders: {e}"
            log.error("orders() failed: %s", e)
            omap = {}
        try:
            gmap = {}
            for g in _retry_call(self.kite.gtts, tries=2) or []:
                gid = g.get("id") or g.get("gtt_id")
                if gid:
                    gmap[int(gid)] = g
        except Exception as e:
            self.last_error = f"gtts: {e}"
            log.error("gtts() failed: %s", e)
            gmap = {}

        with self._lock:
            recs = list(self._active.values())
        for rec in recs:
            if not rec.is_open:
                continue

            # entry fills (aggregate child orders)
            total_filled = sum(
                int(omap.get(oid, {}).get("filled_quantity", 0))
                for oid in rec.child_order_ids
                if (omap.get(oid, {}).get("status", "").upper() == "COMPLETE")
            )
            if total_filled > rec.filled_qty:
                rec.filled_qty = total_filled

            # TP fills / breakeven bump after TP1
            for tid in ("tp1_order_id", "tp2_order_id"):
                oid = getattr(rec, tid)
                if not oid:
                    continue
                o = omap.get(oid, {})
                if o.get("status", "").upper() == "COMPLETE":
                    if tid == "tp1_order_id" and rec.partial_enabled and not rec.tp1_done:
                        rec.tp1_done = True
                        self._refresh_sl_gtt(
                            rec,
                            sl_price=_round_to_tick(
                                rec.entry_price + rec.side_sign() * rec.breakeven_ticks * rec.tick_size,
                                rec.tick_size,
                            ),
                            qty=rec.remaining_qty or rec.quantity,
                        )

            # SL GTT state
            if rec.sl_gtt_id:
                g = (gmap.get(rec.sl_gtt_id) or {})
                s = (g.get("status") or g.get("state") or "").lower()
                if s in ("triggered", "cancelled", "deleted", "expired", "completed"):
                    rec.is_open = False
                    fills.append((rec.record_id, float(rec.sl_price or 0.0)))
                    continue

            # All TPs done?
            tp1_done = (not rec.tp1_order_id) or (omap.get(rec.tp1_order_id, {}).get("status", "").upper() == "COMPLETE")
            tp2_done = (rec.tp2_order_id and omap.get(rec.tp2_order_id, {}).get("status", "").upper() == "COMPLETE")
            if tp1_done and tp2_done:
                rec.is_open = False
                fills.append((rec.record_id, float(rec.tp_price or 0.0)))
                if rec.sl_gtt_id:
                    try:
                        _retry_call(self.kite.cancel_gtt, rec.sl_gtt_id, tries=2)
                    except Exception:
                        log.debug("Failed to cancel SL GTT %s during sweep", rec.sl_gtt_id, exc_info=True)

        if fills:
            with self._lock:
                self._active = {k: v for k, v in self._active.items() if v.is_open}
            # Notify (best-effort)
            try:
                if self.telegram and hasattr(self.telegram, "notify_fills"):
                    self.telegram.notify_fills(fills)
                else:
                    self._notify("📬 Fills: " + ", ".join([f"{rid}@{px}" for rid, px in fills]))
            except Exception:
                log.debug("Failed to send fills notification", exc_info=True)

        return fills

    # ----------- admin ----------
    def cancel_all_orders(self) -> None:
        """Best-effort cancel of all live child orders + SL GTTs for our records."""
        if not self.kite:
            with self._lock:
                self._active.clear()
            log.info("Paper mode: cleared in-memory orders.")
            return

        with self._lock:
            recs = list(self._active.values())

        for rec in recs:
            for oid in (rec.tp1_order_id, rec.tp2_order_id, *rec.child_order_ids):
                if not oid:
                    continue
                try:
                    _retry_call(self.kite.cancel_order, variety=rec.variety, order_id=oid, tries=2)
                except Exception:
                    log.debug("Failed to cancel order %s", oid, exc_info=True)
            if rec.sl_gtt_id:
                try:
                    _retry_call(self.kite.cancel_gtt, rec.sl_gtt_id, tries=2)
                except Exception:
                    log.debug("Failed to cancel SL GTT %s", rec.sl_gtt_id, exc_info=True)
            rec.is_open = False

        with self._lock:
            self._active.clear()
        log.info("All open executor orders cancelled.")

    def health_check(self) -> None:
        """No-op placeholder (runner calls this periodically)."""
        try:
            if not self.kite:
                return
            # optionally touch lightweight endpoints
            # _retry_call(self.kite.profile, tries=1)
        except Exception as e:
            log.warning("OrderExecutor health warning: %s", e)

    def shutdown(self) -> None:
        """Graceful teardown."""
        try:
            self.cancel_all_orders()
        except Exception:
            log.debug("Error during executor shutdown", exc_info=True)

    # ----------- telegram-wired mutators ----------
    def set_trailing_enabled(self, on: bool) -> None:
        self.enable_trailing = bool(on)
        with self._lock:
            for r in self._active.values():
                r.trailing_enabled = self.enable_trailing

    def set_trailing_mult(self, x: float) -> None:
        if x <= 0:
            return
        self.trailing_mult = float(x)
        with self._lock:
            for r in self._active.values():
                r.trailing_mult = self.trailing_mult

    def set_partial_enabled(self, on: bool) -> None:
        self.partial_enable = bool(on)
        with self._lock:
            for r in self._active.values():
                r.partial_enabled = self.partial_enable

    def set_tp1_ratio(self, pct: float) -> None:
        # accepts 0..100 or 0..1
        v = float(pct)
        v = v / 100.0 if v > 1.0 else v
        v = max(0.0, min(1.0, v))
        self.tp1_ratio = v
        with self._lock:
            for r in self._active.values():
                r.tp1_ratio = self.tp1_ratio

    def set_breakeven_ticks(self, ticks: int) -> None:
        t = int(max(0, ticks))
        self.breakeven_ticks = t
        with self._lock:
            for r in self._active.values():
                r.breakeven_ticks = self.breakeven_ticks

    # ----------- util ----------
    @staticmethod
    def _infer_symbol(payload: Dict[str, Any]) -> Optional[str]:
        """
        If not provided, compose a reasonable NFO symbol from strike & option_type.
        Example placeholder: NIFTY24500CE  (expiry not encoded here).
        """
        try:
            base = getattr(settings.instruments, "trade_symbol", "NIFTY") if settings else "NIFTY"
        except Exception:
            base = "NIFTY"
        strike = payload.get("strike")
        opt = str(payload.get("option_type", "")).upper()  # CE/PE
        if strike is None or opt not in ("CE", "PE"):
            return None
        try:
            s = int(float(strike))
        except Exception:
            return None
        return f"{base}{s}{opt}"

    # ----------- internal notify helper ----------
    def _notify(self, text: str) -> None:
        now = time.time()
        last_msg, last_ts = self._last_notification
        if text == last_msg and (now - last_ts) < 300:
            return
        self._last_notification = (text, now)
        try:
            if self.telegram:
                if hasattr(self.telegram, "send_message"):
                    self.telegram.send_message(text)
                elif hasattr(self.telegram, "notify"):
                    self.telegram.notify(text)  # legacy fallback
        except Exception:
            log.debug("Failed to send notification", exc_info=True)
