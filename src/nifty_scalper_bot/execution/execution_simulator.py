"""Microstructure-aware execution simulation backed by canonical live costs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal

from nifty_scalper_bot.risk.cost_model import estimate_order_cost


@dataclass(slots=True)
class FillEvent:
    """Single fill for a simulated order."""

    qty: int
    price: float
    ts_ns: int


@dataclass(slots=True)
class CommissionModel:
    """Adapter to the canonical live NSE option cost model."""

    def calculate(self, turnover: float, *, side: Literal["BUY", "SELL"]) -> float:
        """Return live-cost-model fees for one executed option order."""

        return estimate_order_cost(turnover=turnover, side=side)


@dataclass(slots=True)
class ExecutionResult:
    """Outcome of an execution attempt in the simulator."""

    fills: List[FillEvent]
    remaining_qty: int
    fees: float

    @property
    def average_price(self) -> float:
        """Return the volume-weighted average price of completed fills."""

        if not self.fills:
            return 0.0
        turnover = sum(fill.price * fill.qty for fill in self.fills)
        quantity = sum(fill.qty for fill in self.fills)
        return turnover / quantity


@dataclass(slots=True)
class ExecutionSimulator:
    """Simulate depth-limited, spread-aware market order execution."""

    aggressiveness: float = 1.0
    commission_model: CommissionModel = field(default_factory=CommissionModel)
    tick_size: float = 0.05

    def simulate_market_order(
        self,
        *,
        side: Literal["BUY", "SELL"],
        qty: int,
        bid: float,
        ask: float,
        size_at_best: int,
        ts_ns: int,
        minimum_slippage_bps: float = 0.0,
        minimum_impact: float = 0.0,
    ) -> ExecutionResult:
        """Return fills, remainder, and canonical fees for one market order."""

        if qty <= 0:
            raise ValueError("Quantity must be positive")
        if ask <= bid:
            raise ValueError("Invalid order book snapshot")

        spread = ask - bid
        mid = (ask + bid) / 2.0
        aggressiveness = max(0.0, min(self.aggressiveness, 1.0))
        depth_ratio = min(1.0, max(size_at_best, 0) / qty)
        multiplier = 0.4 + 0.6 * aggressiveness + 0.4 * (1.0 - depth_ratio)
        multiplier = min(max(multiplier, 0.4), 1.0)
        adjustment = max(
            multiplier * spread,
            max(0.0, minimum_slippage_bps) / 10_000.0 * mid,
            max(0.0, minimum_impact),
        )
        direction = 1.0 if side == "BUY" else -1.0
        price = round((mid + direction * adjustment) / self.tick_size) * self.tick_size
        filled_qty = min(qty, max(size_at_best, 0))
        fills = (
            [FillEvent(qty=filled_qty, price=price, ts_ns=ts_ns)]
            if filled_qty > 0
            else []
        )
        turnover = sum(fill.price * fill.qty for fill in fills)
        fees = self.commission_model.calculate(turnover, side=side) if turnover else 0.0
        return ExecutionResult(
            fills=fills,
            remaining_qty=qty - filled_qty,
            fees=fees,
        )


__all__ = [
    "CommissionModel",
    "ExecutionResult",
    "ExecutionSimulator",
    "FillEvent",
]
