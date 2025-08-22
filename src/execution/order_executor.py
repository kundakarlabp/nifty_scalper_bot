# src/execution/orderexecutor.py
from __future__ import annotations
import logging, math, threading, time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    from kiteconnect import KiteConnect  # type: ignore
    from kiteconnect.exceptions import NetworkException, TokenException, InputException  # type: ignore
except Exception:
    KiteConnect = None  # type: ignore
    NetworkException = TokenException = InputException = Exception  # type: ignore

try:
    from src.data.source import DataSource
except Exception:
    DataSource = object  # type: ignore

try:
    from src.config import settings
except Exception:
    settings = None  # type: ignore

log = logging.getLogger(__name__)

# ------------------- helpers -------------------

def _retry_call(fn, *args, tries: int = 3, base_delay: float = 0.25, **kwargs):
    delay = base_delay
    for i in range(tries):
        try:
            return fn(*args, **kwargs)
        except (NetworkException, TokenException, InputException) as e:
            if i == tries - 1:
                raise
            log.warning("Transient broker error in %s (try %d/%d): %s",
                        getattr(fn, "__name__", "call"), i + 1, tries, e)
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
    if chunk <= 0: return [total]
    out, left = [], total
    while left > 0:
        take = min(chunk, left)
        out.append(take)
        left -= take
    return out

# ------------------- record -------------------

@dataclass
class _OrderRecord:
    order_id: str
    instrument_token: int
    symbol: str
    side: str
    quantity: int
    entry_price: float
    tick_size: float = 0.05

    is_open: bool = True
    filled_qty: int = 0

    sl_gtt_id: Optional[int] = None
    sl_price: Optional[float] = None

    tp1_order_id: Optional[str] = None
    tp2_order_id: Optional[str] = None
    tp_price: Optional[float] = None
    tp1_done: bool = False

    exchange: str = "NFO"
    product: str = "NRML"
    variety: str = "regular"
    entry_order_type: str = "LIMIT"
    freeze_qty: int = 900
    use_slm_exit: bool = True

    partial_enabled: bool = False
    tp1_ratio: float = 0.5
    breakeven_ticks: int = 2
    trailing_enabled: bool = True
    trailing_mult: float = 1.5

    child_order_ids: List[str] = field(default_factory=list)

    @property
    def record_id(self) -> str: return self.order_id
    @property
    def remaining_qty(self) -> int: return max(0, self.quantity - self.filled_qty)
    def side_sign(self) -> int: return 1 if self.side.upper() == "BUY" else -1

# ------------------- executor -------------------

class OrderExecutor:
    def __init__(self, config: Any, kite: Optional[KiteConnect], data_source: Optional[DataSource]) -> None:
        self.kite, self.data_source = kite, data_source
        self._lock = threading.Lock()
        self._active: Dict[str, _OrderRecord] = {}

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

    # --- helpers ---
    def get_active_orders(self) -> List[_OrderRecord]:
        with self._lock: return list(self._active.values())

    def get_positions_kite(self) -> Dict[str, Any]:
        if not self.kite: return {}
        try:
            data = _retry_call(self.kite.positions, tries=2) or {}
            return {p.get("tradingsymbol"): p for p in data.get("day", [])}
        except Exception as e:
            log.error("positions() failed: %s", e); return {}

    # --- entry placement ---
    def place_entry_order(self, token: int, symbol: str, side: str, quantity: int, price: float) -> Optional[str]:
        if not self.kite or quantity <= 0: return None
        qty = _round_to_step(quantity, self.lot_size)
        if qty <= 0: return None

        with self._lock:
            if any(rec.symbol == symbol and rec.is_open for rec in self._active.values()):
                log.warning("open record exists on %s; skip new entry.", symbol); return None

        parts = _chunks(qty, self.freeze_qty if self.exchange.upper()=="NFO" else qty)
        child_ids, record_id = [], None
        for q in parts:
            params = dict(variety=self.variety, exchange=self.exchange,
                          tradingsymbol=symbol, transaction_type=side.upper(),
                          quantity=int(q), product=self.product,
                          order_type=self.entry_order_type.upper())
            if self.entry_order_type.upper()=="LIMIT":
                params["price"] = _round_to_tick(float(price), self.tick_size)
            try:
                oid = _retry_call(self.kite.place_order, **params)
                child_ids.append(oid); record_id = record_id or oid
                log.info("Entry child placed: %s %s qty=%d price=%s -> %s",
                         symbol, side, q, params.get("price"), oid)
            except Exception as e: log.error("place_order failed: %s", e)

        if not record_id: return None
        rec = _OrderRecord(order_id=record_id, instrument_token=token, symbol=symbol,
                           side=side.upper(), quantity=qty, entry_price=float(price),
                           tick_size=self.tick_size, exchange=self.exchange, product=self.product,
                           variety=self.variety, entry_order_type=self.entry_order_type,
                           freeze_qty=self.freeze_qty, use_slm_exit=self.use_slm_exit,
                           partial_enabled=self.partial_enable, tp1_ratio=self.tp1_ratio,
                           breakeven_ticks=self.breakeven_ticks, trailing_enabled=self.enable_trailing,
                           trailing_mult=self.trailing_mult, child_order_ids=child_ids)
        with self._lock: self._active[rec.record_id] = rec
        return rec.record_id

    # --- exits (SL GTT + TPs) ---
    def setup_gtt_orders(self, record_id: str, sl_price: float, tp_price: float) -> None:
        with self._lock: rec = self._active.get(record_id)
        if not rec or not rec.is_open or not self.kite: return

        qty, exit_side = rec.remaining_qty or rec.quantity, ("SELL" if rec.side=="BUY" else "BUY")
        tp_price, sl_price = _round_to_tick(tp_price, rec.tick_size), _round_to_tick(sl_price, rec.tick_size)

        # cancel old TPs
        for old in (rec.tp1_order_id, rec.tp2_order_id):
            if old:
                try: _retry_call(self.kite.cancel_order, variety=rec.variety, order_id=old, tries=2)
                except Exception: pass
        rec.tp1_order_id = rec.tp2_order_id = None

        # partial split
        if rec.partial_enabled:
            q_tp1 = _round_to_step(int(round(qty * max(0,min(1,rec.tp1_ratio)))), self.lot_size)
            q_tp1 = min(max(q_tp1, self.lot_size), qty); q_tp2 = qty - q_tp1
        else: q_tp1, q_tp2 = 0, qty

        if q_tp1>0:
            try:
                rec.tp1_order_id = _retry_call(self.kite.place_order,
                    variety=rec.variety, exchange=rec.exchange, tradingsymbol=rec.symbol,
                    transaction_type=exit_side, quantity=q_tp1, product=rec.product,
                    order_type="LIMIT", price=tp_price, tries=2)
            except Exception as e: log.error("TP1 failed: %s", e)
        if q_tp2>0:
            try:
                rec.tp2_order_id = _retry_call(self.kite.place_order,
                    variety=rec.variety, exchange=rec.exchange, tradingsymbol=rec.symbol,
                    transaction_type=exit_side, quantity=q_tp2, product=rec.product,
                    order_type="LIMIT", price=tp_price, tries=2)
            except Exception as e: log.error("TP2 failed: %s", e)
        rec.tp_price = tp_price

        self._refresh_sl_gtt(rec, sl_price=sl_price, qty=qty)

    def _refresh_sl_gtt(self, rec: _OrderRecord, *, sl_price: float, qty: int) -> None:
        if not self.kite: return
        if rec.sl_gtt_id:
            try: _retry_call(self.kite.cancel_gtt, rec.sl_gtt_id, tries=2)
            except Exception: pass
            rec.sl_gtt_id = None

        exit_side = "SELL" if rec.side=="BUY" else "BUY"
        sl_leg = {"exchange": rec.exchange,"tradingsymbol": rec.symbol,"transaction_type": exit_side,
                  "quantity": int(qty),"order_type": "SL-M" if rec.use_slm_exit else "SL",
                  "product": rec.product,"price": None if rec.use_slm_exit else sl_price,
                  "trigger_price": sl_price}
        try:
            res = _retry_call(self.kite.place_gtt, trigger_type="single",
                              tradingsymbol=rec.symbol, exchange=rec.exchange,
                              trigger_price=sl_price, last_price=None,
                              orders=[sl_leg], tries=2)
            gid = res["id"] if isinstance(res, dict) else int(res)
            rec.sl_gtt_id, rec.sl_price = gid, sl_price
        except Exception as e: log.error("SL GTT failed: %s", e)

    # --- trailing ---
    def update_trailing_stop(self, record_id: str, *, current_price: float, atr: float, atr_multiplier: Optional[float]=None) -> None:
        with self._lock: rec = self._active.get(record_id)
        if not rec or not rec.is_open or not self.kite: return
        if not rec.trailing_enabled or atr<=0 or current_price<=0: return
        m = float(atr_multiplier or rec.trailing_mult)
        if rec.side=="BUY":
            proposed = _round_to_tick(current_price - m*atr, rec.tick_size)
            if rec.tp1_done: proposed = max(proposed, _round_to_tick(rec.entry_price+rec.breakeven_ticks*rec.tick_size, rec.tick_size))
            if rec.sl_price and proposed <= rec.sl_price: return
        else:
            proposed = _round_to_tick(current_price + m*atr, rec.tick_size)
            if rec.tp1_done: proposed = min(proposed, _round_to_tick(rec.entry_price-rec.breakeven_ticks*rec.tick_size, rec.tick_size))
            if rec.sl_price and proposed >= rec.sl_price: return
        self._refresh_sl_gtt(rec, sl_price=proposed, qty=rec.remaining_qty or rec.quantity)

    # --- manual exit & cancel_all remain same ---

    def sync_and_enforce_oco(self) -> List[Tuple[str,float]]:
        if not self.kite: return []
        fills: List[Tuple[str,float]] = []
        try: omap = {o.get("order_id"):o for o in _retry_call(self.kite.orders, tries=2) or []}
        except Exception as e: log.error("orders() failed: %s", e); omap={}
        try:
            gmap = {}
            for g in _retry_call(self.kite.gtts, tries=2) or []:
                gid = g.get("id") or g.get("gtt_id"); 
                if gid: gmap[int(gid)] = g
        except Exception as e: log.error("gtts() failed: %s", e); gmap={}

        with self._lock:
            recs = list(self._active.values())
        for rec in recs:
            if not rec.is_open: continue
            total_filled = sum(int(omap.get(oid,{}).get("filled_quantity",0)) for oid in rec.child_order_ids if (omap.get(oid,{}).get("status","").upper()=="COMPLETE"))
            if total_filled > rec.filled_qty: rec.filled_qty = total_filled
            for tid in ("tp1_order_id","tp2_order_id"):
                oid = getattr(rec,tid); o=omap.get(oid,{})
                if o.get("status","").upper()=="COMPLETE":
                    if tid=="tp1_order_id" and rec.partial_enabled and not rec.tp1_done:
                        rec.tp1_done=True
                        self._refresh_sl_gtt(rec, sl_price=_round_to_tick(rec.entry_price+rec.side_sign()*rec.breakeven_ticks*rec.tick_size,rec.tick_size),
                                             qty=rec.remaining_qty or rec.quantity)
            if rec.sl_gtt_id:
                g=(gmap.get(rec.sl_gtt_id) or {}); s=(g.get("status") or g.get("state") or "").lower()
                if s in ("triggered","cancelled","deleted","expired"):
                    rec.is_open=False; fills.append((rec.record_id,float(rec.sl_price or 0.0))); continue
            tp1_done=(not rec.tp1_order_id) or (omap.get(rec.tp1_order_id,{}).get("status","").upper()=="COMPLETE")
            tp2_done=(rec.tp2_order_id and omap.get(rec.tp2_order_id,{}).get("status","").upper()=="COMPLETE")
            if tp1_done and tp2_done:
                rec.is_open=False; fills.append((rec.record_id,float(rec.tp_price or 0.0)))
                if rec.sl_gtt_id: 
                    try:_retry_call(self.kite.cancel_gtt, rec.sl_gtt_id, tries=2)
                    except Exception: pass
        if fills:
            with self._lock: self._active={k:v for k,v in self._active.items() if v.is_open}
        return fills