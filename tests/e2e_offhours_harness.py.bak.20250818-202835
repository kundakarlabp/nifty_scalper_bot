# tests/e2e_offhours_harness.py
import time
import math
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

import pandas as pd

from src.data_streaming.realtime_trader import RealTimeTrader

# ---------- Dummy Kite client ----------
class DummyKite:
    """
    Zerodha-like stub:
    - instruments(): provides 1 CE + 1 PE for nearest expiry
    - ltp()/quote(): synthetic prices + small spread
    - historical_data(): simple series for indicators
    - place_order()/modify_order()/cancel_order(): record orders locally
    - orders(): returns broker-style order states so the bot can detect fills
    """

    VARIETY_REGULAR = "regular"
    ORDER_TYPE_LIMIT = "LIMIT"
    ORDER_TYPE_SL = "SL"
    ORDER_TYPE_MARKET = "MARKET"
    GTT_TYPE_OCO = "two-leg"

    def __init__(self):
        self._expiry = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
        self._nfo = [
            {
                "tradingsymbol": "NIFTY" + datetime.now().strftime("%y%b").upper() + "22500CE",
                "instrument_token": 111111,
                "name": "NIFTY",
                "instrument_type": "CE",
                "strike": 22500,
                "expiry": self._expiry,
            },
            {
                "tradingsymbol": "NIFTY" + datetime.now().strftime("%y%b").upper() + "22500PE",
                "instrument_token": 222222,
                "name": "NIFTY",
                "instrument_type": "PE",
                "strike": 22500,
                "expiry": self._expiry,
            },
        ]
        self._nse = [{
            "tradingsymbol": "NIFTY 50",
            "instrument_token": 256265,
            "segment": "INDICES",
        }]

        # dynamic last prices
        self._last_price: Dict[str, float] = {
            "NSE:NIFTY 50": 22500.0,
            f"NFO:{self._nfo[0]['tradingsymbol']}": 120.0,  # CE
            f"NFO:{self._nfo[1]['tradingsymbol']}": 110.0,  # PE
        }

        # internal order book
        self._orders: List[Dict[str, Any]] = []
        self._next_id = 1000

    # -------- instruments ----------
    def instruments(self, ex):
        return self._nfo if ex == "NFO" else self._nse

    # -------- pricing ----------
    def ltp(self, symbols):
        out = {}
        for s in symbols:
            lp = self._last_price.get(s, 100.0)
            out[s] = {"last_price": round(lp, 2)}
        return out

    def quote(self, symbols):
        out = {}
        for s in symbols:
            lp = self._last_price.get(s, 100.0)
            out[s] = {
                "last_price": round(lp, 2),
                "depth": {
                    "buy": [{"price": round(lp - 0.2, 2)}],
                    "sell": [{"price": round(lp + 0.2, 2)}],
                }
            }
        return out

    # -------- historical ----------
    def historical_data(self, token, start, end, tf, oi=False):
        n = int(max(30, (end - start).total_seconds() // 60))
        base = 22500 if token == 256265 else (120 if token == 111111 else 110)
        rows = []
        t = start
        for i in range(n):
            px = base + 0.6 * math.sin(i / 5)
            rows.append({
                "date": t,
                "open": px,
                "high": px + 1,
                "low": px - 1,
                "close": px + 0.2 * math.cos(i / 3),
                "volume": 1000 + i
            })
            t += timedelta(minutes=1)
        return rows

    # -------- orders ----------
    def _gen_id(self) -> str:
        self._next_id += 1
        return str(self._next_id)

    def place_order(self, **kwargs):
        oid = self._gen_id()
        rec = {
            "order_id": oid,
            "tradingsymbol": kwargs.get("tradingsymbol"),
            "exchange": kwargs.get("exchange"),
            "transaction_type": kwargs.get("transaction_type"),
            "quantity": int(kwargs.get("quantity", 0)),
            "order_type": kwargs.get("order_type"),
            "price": float(kwargs.get("price") or 0.0),
            "trigger_price": float(kwargs.get("trigger_price") or 0.0),
            "status": "OPEN",
            "average_price": None,
            "variety": kwargs.get("variety", self.VARIETY_REGULAR),
        }
        self._orders.append(rec)
        return {"order_id": oid}

    def modify_order(self, **kwargs):
        oid = str(kwargs.get("order_id"))
        for o in self._orders:
            if o["order_id"] == oid:
                if "price" in kwargs and kwargs["price"] is not None:
                    o["price"] = float(kwargs["price"])
                if "trigger_price" in kwargs and kwargs["trigger_price"] is not None:
                    o["trigger_price"] = float(kwargs["trigger_price"])
                return True
        return False

    def cancel_order(self, **kwargs):
        oid = str(kwargs.get("order_id"))
        for o in self._orders:
            if o["order_id"] == oid and o["status"] == "OPEN":
                o["status"] = "CANCELLED"
                return True
        return False

    def place_gtt(self, **kwargs):
        # not used in harness; we run REGULAR exits
        return {"trigger_id": int(time.time())}

    def delete_gtt(self, *args, **kwargs):
        return True

    # broker order monitor (bot polls this)
    def orders(self):
        return list(self._orders)

    # harness helpers
    def set_price(self, symbol: str, px: float):
        self._last_price[symbol] = float(px)

    def mark_fill_if_hit(self, symbol: str):
        """Mark TP/SL fills when price crosses the order levels."""
        lp = self._last_price.get(symbol)
        if lp is None:
            return
        for o in self._orders:
            if o["tradingsymbol"] != symbol or o["status"] != "OPEN":
                continue
            if o["order_type"] == self.ORDER_TYPE_LIMIT:
                # fill TP if price trades through limit in favor
                if o["transaction_type"] == "SELL" and lp >= o["price"]:
                    o["status"] = "COMPLETE"
                    o["average_price"] = o["price"]
                if o["transaction_type"] == "BUY" and lp <= o["price"]:
                    o["status"] = "COMPLETE"
                    o["average_price"] = o["price"]
            elif o["order_type"] == self.ORDER_TYPE_SL:
                # stop order: fill if price crosses trigger
                trig = o.get("trigger_price") or o.get("price")
                if o["transaction_type"] == "SELL" and lp <= trig:
                    o["status"] = "COMPLETE"
                    o["average_price"] = trig
                if o["transaction_type"] == "BUY" and lp >= trig:
                    o["status"] = "COMPLETE"
                    o["average_price"] = trig


# ---------- Force a guaranteed signal ----------
class AlwaysBuyStrategy:
    """Replaces the real strategy so entries are guaranteed for testing."""
    def __init__(self, atr=4.0):
        self.atr = atr
        self._token = 0
    def generate_signal(self, df: pd.DataFrame, current_price: float) -> Optional[Dict[str, Any]]:
        self._token += 1
        return {
            "signal": "BUY",
            "score": 10,
            "confidence": 9.5,
            "entry_price": round(float(current_price), 2),
            "stop_loss": round(float(current_price) - 6.0, 2),
            "target": round(float(current_price) + 10.0, 2),
            "market_volatility": self.atr,
            "reasons": ["forced-test"],
        }

# ---------- Harness runner ----------
def run_scenario_trend_up_then_trail(trader: RealTimeTrader, kite: DummyKite) -> Dict[str, Any]:
    """Price goes up to TP1 → breakeven hop → trails up → pullback hits trailed SL."""
    # ensure a fresh cycle opens 1 trade
    trader._refresh_instruments_cache(force=True)
    trader.is_trading = True

    # pull a fetch+process to open entry + exits
    trader.fetch_and_process_data()

    # grab the active order symbol
    act = trader.order_executor.get_active_orders()
    assert act, "No active order opened"
    entry_id, rec = next(iter(act.items()))
    symbol = rec.symbol

    base = kite._last_price.get(symbol, 120.0)

    # Step 1: rise towards TP1
    for step in range(8):
        kite.set_price(symbol, base + step * 1.2)
        kite.mark_fill_if_hit(symbol)
        trader._oco_and_housekeeping_tick()
        trader._trailing_tick()
        time.sleep(0.1)

    # Step 2: small extra rise to trigger trailing to tighten SL further
    for step in range(4):
        kite.set_price(symbol, base + 8*1.2 + step*0.8)
        kite.mark_fill_if_hit(symbol)
        trader._trailing_tick()
        time.sleep(0.1)

    # Step 3: pull back to hit trailed SL on remaining qty
    pullback = base + 8*1.2 - 3.0
    for _ in range(6):
        pullback -= 0.8
        kite.set_price(symbol, pullback)
        kite.mark_fill_if_hit(symbol)
        trader._oco_and_housekeeping_tick()
        time.sleep(0.1)

    closed = symbol not in [r.symbol for r in trader.order_executor.get_active_orders().values()]
    return {
        "scenario": "trend_up_then_trail",
        "closed": closed,
        "trades_today": len(trader.trades),
    }

def run_scenario_gap_down_hard_stop(trader: RealTimeTrader, kite: DummyKite) -> Dict[str, Any]:
    """Open new trade then gap down fast to test hard-stop handling."""
    # reset for a new trade
    trader.is_trading = True
    trader.fetch_and_process_data()
    act = trader.order_executor.get_active_orders()
    assert act, "No active order opened (hard stop scenario)"
    entry_id, rec = next(iter(act.items()))
    symbol = rec.symbol

    # sudden gap below SL
    lp = kite._last_price.get(symbol, 120.0)
    gap_px = lp - 15.0
    for _ in range(3):
        kite.set_price(symbol, gap_px)
        kite.mark_fill_if_hit(symbol)
        trader._oco_and_housekeeping_tick()
        time.sleep(0.1)

    closed = symbol not in [r.symbol for r in trader.order_executor.get_active_orders().values()]
    return {
        "scenario": "gap_down_hard_stop",
        "closed": closed,
        "trades_today": len(trader.trades),
    }

def main():
    # Build trader with dummy kite + guaranteed strategy
    t = RealTimeTrader()
    dk = DummyKite()
    t.order_executor.kite = dk
    t.strategy =
