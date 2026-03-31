"""Microstructure-aware execution simulator for backtests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal


@dataclass(slots=True)
class FillEvent:
    """Single fill for an order during backtest execution."""

    qty: int
    price: float
    ts_ns: int


@dataclass(slots=True)
class CommissionModel:
    """Parameterised commission model for NSE index options."""

    brokerage_per_order: float = 40.0
    exchange_fee_rate: float = 0.0005
    sebi_fee_rate: float = 0.000001
    stamp_duty_rate: float = 0.0003
    gst_rate: float = 0.18

    def calculate(self, turnover: float) -> float:
        """Return the total fees for the given traded turnover."""

        variable = turnover * (
            self.exchange_fee_rate + self.sebi_fee_rate + self.stamp_duty_rate
        )
        brokerage = self.brokerage_per_order
        gst_base = brokerage + turnover * (self.exchange_fee_rate + self.sebi_fee_rate)
        gst = gst_base * self.gst_rate
        return brokerage + variable + gst


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
    """Simulate microstructure-aware market order execution.

    Example:
        >>> sim = ExecutionSimulator(aggressiveness=0.6)
        >>> result = sim.simulate_market_order(
        ...     side="BUY", qty=50, bid=100.0, ask=101.0, size_at_best=40, ts_ns=1
        ... )
        >>> result.remaining_qty
        10
    """

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
    ) -> ExecutionResult:
        """Simulate a single market order fill attempt.

        Args:
            side: Order side (``"BUY"`` or ``"SELL"``).
            qty: Requested contracts.
            bid: Best bid price.
            ask: Best ask price.
            size_at_best: Contracts currently available at the best price level.
            ts_ns: Timestamp of the book snapshot.

        Returns:
            ExecutionResult describing fills, remainder, and fees.
        """

        if qty <= 0:
            raise ValueError("Quantity must be positive")
        if ask <= bid:
            raise ValueError("Invalid order book snapshot")

        spread = ask - bid
        mid = (ask + bid) / 2.0
        aggressiveness = max(0.0, min(self.aggressiveness, 1.0))
        depth_ratio = min(1.0, max(size_at_best, 0) / qty) if qty > 0 else 1.0
        k = 0.4 + 0.6 * aggressiveness + 0.4 * (1.0 - depth_ratio)
        k = min(max(k, 0.4), 1.0)
        direction = 1.0 if side == "BUY" else -1.0
        raw_price = mid + direction * k * spread
        price = round(raw_price / self.tick_size) * self.tick_size
        filled_qty = min(qty, max(size_at_best, 0))
        fills = []
        if filled_qty > 0:
            fills.append(FillEvent(qty=filled_qty, price=price, ts_ns=ts_ns))
        remaining = qty - filled_qty
        turnover = sum(fill.price * fill.qty for fill in fills)
        fees = self.commission_model.calculate(turnover) if turnover else 0.0
        return ExecutionResult(fills=fills, remaining_qty=remaining, fees=fees)


__all__ = [
    "CommissionModel",
    "ExecutionResult",
    "ExecutionSimulator",
    "FillEvent",
]
