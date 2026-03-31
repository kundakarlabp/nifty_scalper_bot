"""Session KPI aggregation utilities for discretionary reviews."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, List

Direction = str


@dataclass(slots=True)
class TradeSample:
    """Executed trade enriched with path-wise analytics."""

    symbol: str
    direction: Direction
    quantity: int
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    mae: float
    mfe: float
    spread_cost: float = 0.0
    slippage: float = 0.0
    decay: float = 0.0

    @property
    def pnl(self) -> float:
        """Return the net profit/loss of the trade in currency units."""

        sign = 1.0 if self.direction.upper().startswith("L") else -1.0
        return (self.exit_price - self.entry_price) * sign * float(self.quantity)

    @property
    def return_pct(self) -> float:
        """Return the percentage return relative to entry notional."""

        notional = self.entry_price * float(self.quantity)
        if notional == 0:
            return 0.0
        return self.pnl / notional


@dataclass(slots=True)
class KPIReport:
    """Aggregate analytics across a collection of trades."""

    total_trades: int
    hit_rate: float
    average_mae: float
    average_mfe: float
    total_spread_cost: float
    average_slippage: float
    average_decay: float
    net_pnl: float
    average_return_pct: float
    per_hour_returns: dict[int, float] = field(default_factory=dict)

    def format_text(self) -> str:
        """Render the KPI snapshot as multi-line human readable text."""

        if self.total_trades == 0:
            return "No trades recorded for the requested window."
        lines = [
            f"Trades: {self.total_trades} | Hit-rate: {self.hit_rate:.1%}",
            f"MAE: {self.average_mae:.2f} | MFE: {self.average_mfe:.2f}",
            (
                f"Spread cost: {self.total_spread_cost:.2f} | "
                f"Slippage: {self.average_slippage:.2f}"
            ),
            f"Decay impact: {self.average_decay:.2f} | Net PnL: {self.net_pnl:.2f}",
            f"Avg return: {self.average_return_pct:.2%}",
        ]
        if self.per_hour_returns:
            per_hour = ", ".join(
                f"{hour:02d}h: {ret:.2%}"
                for hour, ret in sorted(self.per_hour_returns.items())
            )
            lines.append(f"Per-hour returns: {per_hour}")
        return "\n".join(lines)


def summarize_trades(trades: Iterable[TradeSample]) -> KPIReport:
    """Compute :class:`KPIReport` for the provided trades."""

    samples: List[TradeSample] = list(trades)
    total = len(samples)
    if total == 0:
        return KPIReport(
            total_trades=0,
            hit_rate=0.0,
            average_mae=0.0,
            average_mfe=0.0,
            total_spread_cost=0.0,
            average_slippage=0.0,
            average_decay=0.0,
            net_pnl=0.0,
            average_return_pct=0.0,
            per_hour_returns={},
        )
    wins = sum(1 for trade in samples if trade.pnl > 0)
    mae = sum(trade.mae for trade in samples) / total
    mfe = sum(trade.mfe for trade in samples) / total
    spread_cost = sum(trade.spread_cost for trade in samples)
    avg_slippage = sum(trade.slippage for trade in samples) / total
    avg_decay = sum(trade.decay for trade in samples) / total
    net_pnl = sum(trade.pnl for trade in samples)
    avg_return = sum(trade.return_pct for trade in samples) / total
    per_hour: dict[int, List[float]] = {}
    for trade in samples:
        hour = int(trade.entry_time.hour)
        per_hour.setdefault(hour, []).append(trade.return_pct)
    per_hour_returns = {
        hour: sum(values) / len(values) for hour, values in per_hour.items() if values
    }
    return KPIReport(
        total_trades=total,
        hit_rate=wins / total,
        average_mae=mae,
        average_mfe=mfe,
        total_spread_cost=spread_cost,
        average_slippage=avg_slippage,
        average_decay=avg_decay,
        net_pnl=net_pnl,
        average_return_pct=avg_return,
        per_hour_returns=per_hour_returns,
    )


class KPITracker:
    """Collect trades in-memory and expose session KPI summaries."""

    def __init__(self) -> None:
        self._trades: list[TradeSample] = []

    def record(self, trade: TradeSample) -> None:
        """Append ``trade`` to the internal ledger."""

        self._trades.append(trade)

    def extend(self, trades: Iterable[TradeSample]) -> None:
        """Append a sequence of trades."""

        for trade in trades:
            self.record(trade)

    def reset(self) -> None:
        """Clear collected trades."""

        self._trades.clear()

    def summarize(self) -> KPIReport:
        """Return a :class:`KPIReport` snapshot for recorded trades."""

        return summarize_trades(self._trades)

    def format_report(self) -> str:
        """Return a formatted string for human consumption."""

        return self.summarize().format_text()


__all__ = ["KPITracker", "KPIReport", "TradeSample", "summarize_trades"]
