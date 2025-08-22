"""
src/risk/position_sizing.py

Position sizing for Nifty options:
- Uses account equity and risk-per-trade % to cap rupee risk
- Converts risk into lots using stop distance (points) × lot_size
- Obeys min/max lots, exchange freeze quantity, and tick size rounding
- Optional daily risk guards: max trades/day, max drawdown %, loss streak
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List
import math
import logging

log = logging.getLogger(__name__)


# ---------- Data structures ----------

@dataclass
class DailyRiskState:
    """Snapshot of the current session used for guard-rails."""
    trades_placed: int = 0
    realized_pnl: float = 0.0
    peak_equity: Optional[float] = None
    loss_streak: int = 0


@dataclass
class SizingInputs:
    """Inputs required to size the trade."""
    equity: float                     # total equity (rupees)
    risk_per_trade: float             # 0.01 = 1% of equity
    entry_price: float                # option entry price (rupees)
    stop_price: float                 # option stop price (rupees)
    lot_size: int                     # exchange lot size (e.g., 75 for NIFTY options)
    tick_size: float = 0.05
    freeze_qty: int = 1800            # NFO freeze qty per order (Railway config)
    min_lots: int = 1
    max_lots: int = 15

    # Optional daily limits (if provided, they will be enforced)
    max_trades_per_day: Optional[int] = None
    max_daily_drawdown_pct: Optional[float] = None  # 0.05 = 5%
    max_consecutive_losses: Optional[int] = None


@dataclass
class SizingDecision:
    ok: bool
    lots: int = 0
    qty: int = 0
    per_order_qty: int = 0              # capped by freeze_qty
    num_orders: int = 0                 # if >1, the executor should split
    rupee_risk_cap: float = 0.0
    rupee_risk_at_size: float = 0.0
    stop_points: float = 0.0
    reasons: List[str] = field(default_factory=list)


# ---------- Helpers ----------

def _round_to_tick(price: float, tick: float) -> float:
    if tick <= 0:
        return price
    return round(round(price / tick) * tick, 2)


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


# ---------- Core sizer ----------

class PositionSizer:
    """
    Converts risk% and stop distance to lots/quantity.
    """

    def __init__(self) -> None:
        pass

    def check_daily_guards(
        self,
        inputs: SizingInputs,
        daily: Optional[DailyRiskState] = None,
    ) -> Optional[str]:
        """
        Returns a string reason if any daily guard is violated, else None.
        """
        if daily is None:
            return None

        if inputs.max_trades_per_day is not None:
            if daily.trades_placed >= inputs.max_trades_per_day:
                return f"max_trades_per_day reached ({daily.trades_placed}/{inputs.max_trades_per_day})"

        if inputs.max_consecutive_losses is not None:
            if daily.loss_streak >= inputs.max_consecutive_losses:
                return f"loss streak guard ({daily.loss_streak} >= {inputs.max_consecutive_losses})"

        if inputs.max_daily_drawdown_pct is not None and inputs.max_daily_drawdown_pct > 0:
            # Estimate drawdown from peak_equity if provided
            if daily.peak_equity is not None and daily.peak_equity > 0:
                dd = (daily.peak_equity - (daily.peak_equity + daily.realized_pnl)) / daily.peak_equity
                # realized_pnl is negative when losing; above formula simplifies to -realized_pnl/peak_equity
                dd = max(0.0, -daily.realized_pnl / daily.peak_equity)
                if dd >= inputs.max_daily_drawdown_pct:
                    pct = round(dd * 100.0, 2)
                    return f"daily drawdown {pct}% exceeds limit {inputs.max_daily_drawdown_pct * 100:.2f}%"

        return None

    def size(
        self,
        inputs: SizingInputs,
        daily: Optional[DailyRiskState] = None,
    ) -> SizingDecision:
        """
        Compute a sizing decision.

        - If stop distance is invalid or equity small, returns ok=False with reasons.
        - qty returned is total quantity (lots * lot_size).
        - per_order_qty and num_orders tell the executor how to split to respect freeze qty.
        """
        reasons: List[str] = []

        # Daily guards
        guard = self.check_daily_guards(inputs, daily)
        if guard:
            return SizingDecision(ok=False, reasons=[guard])

        # Basic validation
        if inputs.equity <= 0:
            return SizingDecision(ok=False, reasons=["equity <= 0"])
        if inputs.risk_per_trade <= 0:
            return SizingDecision(ok=False, reasons=["risk_per_trade <= 0"])
        if inputs.lot_size <= 0:
            return SizingDecision(ok=False, reasons=["lot_size <= 0"])

        # Compute stop distance (points)
        stop_points = abs(inputs.entry_price - inputs.stop_price)
        stop_points = _round_to_tick(stop_points, inputs.tick_size)

        if stop_points <= 0:
            return SizingDecision(
                ok=False,
                reasons=[f"invalid stop distance (entry={inputs.entry_price}, stop={inputs.stop_price})"],
            )

        # Rupee risk cap for this trade
        rupee_risk_cap = inputs.equity * inputs.risk_per_trade

        # For options, risk per *lot* ~ stop_points * lot_size
        rupee_risk_per_lot = stop_points * inputs.lot_size

        if rupee_risk_per_lot <= 0:
            return SizingDecision(ok=False, reasons=["rupee_risk_per_lot <= 0"])

        raw_lots = math.floor(rupee_risk_cap / rupee_risk_per_lot)

        # Respect min/max lots
        lots = _clamp(raw_lots, inputs.min_lots, inputs.max_lots)

        if lots <= 0:
            reasons.append(
                f"risk too small for 1 lot: cap={rupee_risk_cap:.2f} < per_lot_risk={rupee_risk_per_lot:.2f}"
            )
            return SizingDecision(
                ok=False,
                rupee_risk_cap=rupee_risk_cap,
                stop_points=stop_points,
                reasons=reasons,
            )

        qty = lots * inputs.lot_size

        # Exchange freeze quantity constraint (per order)
        per_order_qty = inputs.freeze_qty if inputs.freeze_qty and inputs.freeze_qty > 0 else qty
        per_order_qty = max(inputs.lot_size, (per_order_qty // inputs.lot_size) * inputs.lot_size)
        num_orders = int(math.ceil(qty / per_order_qty)) if per_order_qty > 0 else 1

        # Effective risk at chosen size
        rupee_risk_at_size = rupee_risk_per_lot * lots

        reasons.append(
            f"sized {lots} lots (qty={qty}) using stop={stop_points:.2f} pts, "
            f"risk_cap={rupee_risk_cap:.2f}, risk_at_size={rupee_risk_at_size:.2f}"
        )

        return SizingDecision(
            ok=True,
            lots=lots,
            qty=qty,
            per_order_qty=per_order_qty,
            num_orders=num_orders,
            rupee_risk_cap=rupee_risk_cap,
            rupee_risk_at_size=rupee_risk_at_size,
            stop_points=stop_points,
            reasons=reasons,
        )