"""Backtest helpers for the premium decay short strangle strategy."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Mapping, MutableMapping, Sequence

from nifty_scalper_bot.execution.order_manager import OrderType
from nifty_scalper_bot.options.strike_selector import SelectedContract
from nifty_scalper_bot.strategies.premium_decay import PremiumDecayStrategy
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class BacktestBar:
    """Container representing a single backtest observation."""

    timestamp: datetime
    indicators: Mapping[str, Any]
    option_chain: Sequence[SelectedContract]
    option_prices: Mapping[str, float]
    iv: float
    underlying: str = "NIFTY"


class _BacktestOrderManager:
    """Minimal order manager that tracks theoretical fills and PnL."""

    def __init__(self) -> None:
        self._positions: MutableMapping[str, tuple[str, int, float]] = {}
        self.realized_pnl: float = 0.0
        self.orders: list[Mapping[str, Any]] = []

    def place_order(
        self,
        *,
        symbol: str,
        side: Literal["BUY", "SELL"],
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        price: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        product: str | None = None,
        tag: str | None = None,
        trailing_spec: Any | None = None,
        max_slippage_per_lot: float | None = None,
    ) -> str:
        """Store the order request and update running PnL for BUY legs."""

        del order_type, stop_loss, take_profit, product, tag, trailing_spec
        del max_slippage_per_lot
        price_value = 0.0
        if price is not None:
            try:
                price_value = float(price)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                price_value = 0.0
        payload = {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price_value,
        }
        self.orders.append(payload)
        position = self._positions.get(symbol)
        if side == "SELL":
            self._positions[symbol] = ("SHORT", quantity, price_value)
        elif side == "BUY" and position is not None:
            _, entry_qty, entry_price = position
            qty = min(quantity, entry_qty)
            self.realized_pnl += (entry_price - price_value) * qty
            self._positions.pop(symbol, None)
        return f"BT-{len(self.orders)}"


class _BacktestRiskManager:
    """Risk manager stub approving all orders during simulation."""

    def check_order(self, signal: Any, live_enabled: bool) -> tuple[bool, str]:
        """Return an approval for every signal received."""

        del signal, live_enabled
        return True, ""


class _BacktestOrchestrator:
    """Orchestrator stub that never blocks signals."""

    def filter_signal(
        self, signal: Any, indicators: Mapping[str, Any], position_manager: Any
    ) -> Any:
        """Return the incoming signal unchanged."""

        del indicators, position_manager
        return signal

    def notify_submission(self, signal: Any, underlying: str) -> None:
        """Track submissions without side effects."""

        del signal, underlying

    def notify_exit(self, underlying: str) -> None:
        """Track exits without side effects."""

        del underlying


class _BacktestPositionManager:
    """Position manager stub satisfying orchestrator protocol."""

    def get_all_positions(self) -> Sequence[Any]:
        """Return an empty collection representing no open positions."""

        return []


def run_premium_decay_backtest(bars: Sequence[BacktestBar]) -> Mapping[str, Any]:
    """Simulate premium decay strategy on the provided observations.

    Args:
        bars: Chronologically ordered market snapshots including option chains and
            indicator context.

    Returns:
        Mapping summarising aggregate PnL, trade count, and exit count.

    Raises:
        ValueError: If *bars* is empty.
    """

    LOGGER.debug(
        "Entered run_premium_decay_backtest",
        extra={"event": "premium_decay_backtest_enter", "bars": len(bars)},
    )
    if not bars:
        raise ValueError("bars must not be empty")
    order_manager = _BacktestOrderManager()
    strategy = PremiumDecayStrategy(
        order_manager=order_manager,
        risk_manager=_BacktestRiskManager(),
        orchestrator=_BacktestOrchestrator(),
        position_manager=_BacktestPositionManager(),
        lots=1,
    )
    trades = 0
    exits = 0
    for bar in bars:
        if strategy.evaluate_entry(
            underlying=bar.underlying,
            indicators=bar.indicators,
            option_chain=bar.option_chain,
            iv=bar.iv,
        ):
            trades += 1
        had_position = strategy._active is not None  # noqa: SLF001 - deterministic test
        strategy.evaluate_exit(
            option_prices=bar.option_prices,
            current_iv=bar.iv,
            now=bar.timestamp,
        )
        if had_position and strategy._active is None:
            exits += 1
    if strategy._active is not None:
        LOGGER.info(
            "Condition met: premium_decay_backtest_forced_exit",
            extra={"event": "premium_decay_backtest_forced_exit"},
        )
        strategy.evaluate_exit(
            option_prices=bars[-1].option_prices,
            current_iv=bars[-1].iv,
            now=datetime.now(timezone.utc),
        )
        exits += 1
    result = {
        "trades": trades,
        "exits": exits,
        "pnl": round(order_manager.realized_pnl, 2),
    }
    LOGGER.info(
        "Condition met: premium_decay_backtest_complete",
        extra={"event": "premium_decay_backtest_complete", **result},
    )
    return result


__all__ = ["BacktestBar", "run_premium_decay_backtest"]
