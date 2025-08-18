# src/risk/session.py
"""
Manages the state of a single trading session, including P&L,
active trades, and risk limit enforcement.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List

from src.config import RiskConfig

logger = logging.getLogger(__name__)


class Trade:
    """Represents a single trade from entry to exit."""
    def __init__(self, symbol: str, direction: str, entry_price: float, quantity: int, order_id: str, atr: float):
        self.symbol = symbol
        self.direction = direction  # "BUY" or "SELL"
        self.entry_price = entry_price
        self.exit_price: float | None = None
        self.quantity = quantity
        self.order_id = order_id
        self.entry_time = datetime.now()
        self.exit_time: datetime | None = None
        self.pnl: float = 0.0
        self.net_pnl: float = 0.0
        self.fees: float = 0.0
        self.status: str = "OPEN"  # OPEN, CLOSED
        self.atr_at_entry = atr

    def close(self, exit_price: float, fees: float = 0.0) -> None:
        """Mark the trade as closed and calculate P&L."""
        if self.status == "CLOSED":
            logger.warning(f"Trade {self.order_id} is already closed.")
            return

        self.exit_price = exit_price
        self.exit_time = datetime.now()

        if self.direction == "BUY":
            self.pnl = (self.exit_price - self.entry_price) * self.quantity
        else:  # SELL
            self.pnl = (self.entry_price - self.exit_price) * self.quantity

        self.fees = fees
        self.net_pnl = self.pnl - self.fees
        self.status = "CLOSED"
        logger.info(f"Closed trade {self.order_id} for {self.symbol}. Net P&L: {self.net_pnl:.2f}")


class TradingSession:
    """
    Manages all state for the current trading day, including equity, P&L,
    positions, and risk limits.
    """

    def __init__(self, risk_config: RiskConfig, executor_config: "ExecutorConfig", starting_equity: float):
        if not isinstance(risk_config, RiskConfig):
            raise TypeError("A valid RiskConfig instance is required.")

        self.risk_config = risk_config
        self.executor_config = executor_config
        self.start_equity = starting_equity
        self.current_equity = starting_equity

        self.daily_pnl: float = 0.0
        self.consecutive_losses: int = 0
        self.trades_today: int = 0

        self.active_trades: Dict[str, Trade] = {}  # order_id -> Trade
        self.trade_history: List[Trade] = []

    def add_trade(self, trade: Trade) -> None:
        """Adds a new active trade to the session."""
        if trade.order_id in self.active_trades:
            logger.warning(f"Attempted to add duplicate trade with order_id: {trade.order_id}")
            return

        self.active_trades[trade.order_id] = trade
        self.trades_today += 1
        logger.info(f"New trade added: {trade.direction} {trade.quantity} {trade.symbol} @ {trade.entry_price}")

    def finalize_trade(self, order_id: str, exit_price: float) -> Trade | None:
        """Moves a trade from active to history and updates session P&L."""
        trade = self.active_trades.pop(order_id, None)
        if not trade:
            logger.warning(f"Could not find active trade with order_id: {order_id} to finalize.")
            return None

        fees = (trade.quantity / self.executor_config.nifty_lot_size) * 20 # Assuming 20 per lot fees
        fees = (trade.quantity / self.executor_config.nifty_lot_size) * 20 # Assuming 20 per lot fees for backtesting
        trade.close(exit_price=exit_price, fees=fees)
        self.trade_history.append(trade)

        # Update session state using net_pnl
        self.daily_pnl += trade.net_pnl
        self.current_equity += trade.net_pnl

        if trade.net_pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0  # Reset on a winning trade

        return trade

    @property
    def drawdown_pct(self) -> float:
        """Calculates the current drawdown as a percentage of starting equity."""
        if self.start_equity == 0:
            return 0.0
        # Drawdown is positive
        return -min(0, self.daily_pnl) / self.start_equity

    def check_risk_limits(self) -> str | None:
        """
        Checks if any session-level risk limits have been breached.
        Returns a string reason if a limit is breached, otherwise None.
        """
        if self.trades_today >= self.risk_config.max_trades_per_day:
            return f"Max trades per day ({self.risk_config.max_trades_per_day}) reached."

        if self.consecutive_losses >= self.risk_config.consecutive_loss_limit:
            return f"Consecutive loss limit ({self.risk_config.consecutive_loss_limit}) reached."

        if self.drawdown_pct >= self.risk_config.max_daily_drawdown_pct:
            return (
                f"Max daily drawdown ({self.risk_config.max_daily_drawdown_pct:.2%}) breached. "
                f"Current DD: {self.drawdown_pct:.2%}"
            )

        return None
