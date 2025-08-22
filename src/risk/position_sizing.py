from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DailyRiskState:
    trades_placed: int = 0
    realized_pnl: float = 0.0
    peak_equity: Optional[float] = None
    loss_streak: int = 0


@dataclass
class SizingInputs:
    equity: float
    risk_per_trade: float  # 0.01 = 1%
    entry_price: float
    stop_price: float
    lot_size: int
    tick_size: float
    freeze_qty: int
    min_lots: int
    max_lots: int
    max_trades_per_day: int
    max_daily_drawdown_pct: float
    max_consecutive_losses: int


@dataclass
class SizingDecision:
    ok: bool
    qty: int = 0
    lots: int = 0
    per_order_qty: int = 0
    num_orders: int = 0
    stop_points: float = 0.0
    reasons: List[str] = None  # type: ignore


class PositionSizer:
    def size(self, inp: SizingInputs, daily: Optional[DailyRiskState] = None) -> SizingDecision:
        rs: List[str] = []

        if daily:
            if daily.trades_placed >= inp.max_trades_per_day:
                return SizingDecision(False, reasons=["max trades reached"])
            if daily.loss_streak >= inp.max_consecutive_losses:
                return SizingDecision(False, reasons=["loss streak gate"])
            if inp.max_daily_drawdown_pct > 0 and daily.peak_equity is not None:
                if daily.realized_pnl < -inp.equity * inp.max_daily_drawdown_pct:
                    return SizingDecision(False, reasons=["daily drawdown limit"])

        stop_points = abs(inp.entry_price - inp.stop_price)
        if stop_points <= 0:
            return SizingDecision(False, reasons=["invalid stop"])

        money_risk = inp.equity * max(0.0, inp.risk_per_trade)
        qty_float = money_risk / (stop_points)
        qty = max(0, int(qty_float))

        # Convert to lots & obey bounds
        lots = max(inp.min_lots, min(inp.max_lots, qty // inp.lot_size))
        qty = lots * inp.lot_size

        # Freeze qty split (per order limit)
        per_order_qty = min(inp.freeze_qty, qty) if inp.freeze_qty > 0 else qty
        num_orders = (qty + per_order_qty - 1) // per_order_qty if per_order_qty else 0

        if lots <= 0 or qty <= 0:
            return SizingDecision(False, reasons=["lot gate"])

        rs.append(f"risk_amount={money_risk:.2f}")
        rs.append(f"stop_points={stop_points:.2f}")
        rs.append(f"lots={lots} qty={qty} orders={num_orders} size/order={per_order_qty}")

        return SizingDecision(
            ok=True,
            qty=qty,
            lots=lots,
            per_order_qty=per_order_qty,
            num_orders=num_orders,
            stop_points=stop_points,
            reasons=rs,
        )