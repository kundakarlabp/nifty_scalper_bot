from __future__ import annotations

"""Simulated order connector for backtests."""

from dataclasses import dataclass
from typing import Dict, Tuple
from datetime import datetime
from zoneinfo import ZoneInfo

from src.config import settings
from src.risk.greeks import estimate_greeks_from_mid


def synth_book_from_mid(mid: float, settings) -> tuple[float, float]:
    est = getattr(settings.executor, "default_spread_pct_est", 0.25)
    spr = mid * (est / 100.0)
    return mid - spr / 2.0, mid + spr / 2.0


@dataclass
class CostModel:
    """Transaction cost parameters."""

    brokerage_per_order: float = 20.0
    exchange_fees_pct: float = 0.0005
    stt_sell_pct: float = 0.000625
    gst_pct: float = 0.18
    stamp_buy_pct: float = 0.00003


@dataclass
class MicroModel:
    """Simple microstructure assumptions."""

    base_spread_pct: float = getattr(settings.executor, "default_spread_pct_est", 0.25)
    open_spread_pct: float = 0.35
    close_spread_pct: float = 0.45
    depth_per_lot: int = 5

    def spread_pct_for_time(self, t: tuple[int, int]) -> float:
        """Return the spread percentage for the given ``(hour, minute)``."""

        if t <= (10, 0):
            return self.open_spread_pct
        if t >= (15, 10):
            return self.close_spread_pct
        return self.base_spread_pct


class SimConnector:
    """Synthetic option book and fill simulator."""

    def __init__(
        self,
        lot_size: int = 50,
        costs: CostModel | None = None,
        micro: MicroModel | None = None,
    ) -> None:
        self.lot_size = lot_size
        self.costs = costs or CostModel()
        self.micro = micro or MicroModel()

    def synth_option_book(self, spot: float, strike: float, opt_type: str, now: datetime, atr_pct: float) -> Dict[str, float]:
        """Return a synthetic top-of-book quote for an option."""

        now_tz = now if now.tzinfo else now.replace(tzinfo=ZoneInfo("Asia/Kolkata"))
        estimate_greeks_from_mid(
            S=spot, K=strike, mid=max(1.0, 0.5), opt=opt_type, now=now_tz, atr_pct=atr_pct
        )
        intrinsic = max(0.0, spot - strike) if opt_type == "CE" else max(0.0, strike - spot)
        tv = max(0.5, spot * (atr_pct / 100.0) * 0.25)
        mid = max(1.0, intrinsic + tv)
        spr_pct = self.micro.spread_pct_for_time((now.hour, now.minute))
        spr = mid * spr_pct / 100.0
        bid = mid - spr / 2.0
        ask = mid + spr / 2.0
        if bid <= 0 or ask <= 0:
            bid, ask = synth_book_from_mid(mid, settings)
        depth_units = self.micro.depth_per_lot * self.lot_size
        return {"bid": bid, "ask": ask, "bid5": depth_units * 10, "ask5": depth_units * 10, "mid": mid, "spread_pct": spr_pct}

    def ladder_prices(self, mid: float, spread: float) -> Tuple[float, float]:
        """Return two ladder prices inside the spread."""

        return mid + 0.25 * spread, mid + 0.40 * spread

    def fill_limit_buy(self, price: float, bid: float, ask: float) -> Tuple[bool, float]:
        """Simulate a limit buy fill."""

        if price >= ask:
            return True, ask
        if price >= bid + 0.6 * (ask - bid):
            return True, price
        return False, 0.0

    def fill_limit_sell(self, price: float, bid: float, ask: float) -> Tuple[bool, float]:
        """Simulate a limit sell fill."""

        if price <= bid:
            return True, bid
        if price <= ask - 0.6 * (ask - bid):
            return True, price
        return False, 0.0

    def apply_costs(self, side: str, fill_price: float, qty: int) -> float:
        """Return total transaction costs for the trade."""

        notional = fill_price * qty
        brk = self.costs.brokerage_per_order
        exch = notional * self.costs.exchange_fees_pct
        stt = notional * self.costs.stt_sell_pct if side == "SELL" else 0.0
        stamp = notional * self.costs.stamp_buy_pct if side == "BUY" else 0.0
        gst = (brk + exch) * self.costs.gst_pct
        return brk + exch + stt + stamp + gst
