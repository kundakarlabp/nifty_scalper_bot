"""
Adaptive position sizing module.

This module provides a ``PositionSizing`` class that determines how many
contract lots to trade based on the account size, risk settings and
market conditions.  It protects the account by enforcing per‑trade risk
limits, a daily drawdown cap and a maximum number of consecutive losses.

The calculation assumes that each point move in the underlying contract
is worth ``Config.NIFTY_LOT_SIZE`` rupees.  For example, if the stop
loss is 10 points and the lot size is 50 (Nifty index options), then
one lot risks 10 × 50 = ₹500.  If the risk budget per trade is 1 % of a
₹100 000 account (₹1 000), two lots may be traded.

The size is further scaled by the signal confidence (on a 0–10 scale) so
that lower conviction signals risk less capital.  Daily loss and
drawdown limits are enforced to stop trading once losses exceed
predefined thresholds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

# Use a relative import.  The ``risk`` package is a subpackage of ``src``.
from ..config import Config

logger = logging.getLogger(__name__)


@dataclass
class PositionSizing:
    """Risk manager that calculates position sizes and tracks drawdown."""

    account_size: float = Config.ACCOUNT_SIZE
    risk_per_trade: float = Config.RISK_PER_TRADE  # fraction (e.g. 0.01 for 1 %)
    daily_risk: float = Config.MAX_DRAWDOWN        # fraction (e.g. 0.05 for 5 %)
    max_drawdown: float = Config.MAX_DRAWDOWN      # fraction (same as daily_risk by default)
    lot_size: int = Config.NIFTY_LOT_SIZE          # number of units per contract
    min_lots: int = Config.MIN_LOTS
    max_lots: int = Config.MAX_LOTS
    consecutive_loss_limit: int = Config.CONSECUTIVE_LOSS_LIMIT

    # Internal state
    daily_loss: float = 0.0
    equity_peak: float = field(init=False)
    equity: float = field(init=False)
    consecutive_losses: int = 0

    def __post_init__(self) -> None:
        # Initialise equity to current account size
        self.equity = self.account_size
        self.equity_peak = self.account_size
        logger.debug(
            "PositionSizing initialised: account_size=%s, risk_per_trade=%s, "
            "daily_risk=%s, max_drawdown=%s",
            self.account_size,
            self.risk_per_trade,
            self.daily_risk,
            self.max_drawdown,
        )

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        signal_confidence: float,
        market_volatility: float = 0.0,
    ) -> Optional[Dict[str, int]]:
        """Calculate the quantity of lots to trade for a given setup.

        Args:
            entry_price: The expected entry price of the trade.
            stop_loss: The absolute stop loss price level.  The difference
                ``abs(entry_price - stop_loss)`` determines the points risk.
            signal_confidence: Trade confidence on a 0–10 scale.
            market_volatility: Additional risk factor (0–1).  High values
                reduce the position size to account for elevated volatility.

        Returns:
            A dictionary with a single key ``quantity`` specifying the number
            of lots, or ``None`` if trading should be skipped.
        """
        try:
            # Prevent trading if maximum consecutive losses has been hit
            if self.consecutive_losses >= self.consecutive_loss_limit:
                logger.warning("Consecutive loss limit reached. No new trades allowed.")
                return None

            # Calculate absolute risk in points
            sl_points = abs(entry_price - stop_loss)
            if sl_points <= 0:
                logger.warning("Stop loss and entry price are equal. Cannot calculate risk.")
                return None

            # Risk per lot in rupees (points * lot_size)
            risk_per_lot = sl_points * self.lot_size
            # Total risk budget per trade in rupees
            trade_risk_budget = self.account_size * self.risk_per_trade
            # Starting quantity before scaling by confidence
            qty = int(trade_risk_budget // risk_per_lot)
            if qty <= 0:
                logger.info(
                    "Calculated quantity is zero or negative. Risk per lot: %.2f, trade risk budget: %.2f",
                    risk_per_lot,
                    trade_risk_budget,
                )
                return None

            # Scale quantity by confidence (0–10).  Cap at 1.0 when confidence >= 10.
            confidence_factor = max(0.1, min(signal_confidence / 10.0, 1.0))
            qty = max(self.min_lots, int(qty * confidence_factor))

            # Adjust quantity for market volatility: halve size if volatility > 0.5
            if market_volatility > 0.5:
                qty = max(self.min_lots, qty // 2)

            # Ensure quantity does not exceed configured limits
            qty = max(self.min_lots, min(qty, self.max_lots))

            # Calculate potential maximum loss for this trade
            potential_loss = qty * risk_per_lot
            daily_risk_limit = self.account_size * self.daily_risk
            if self.daily_loss + potential_loss > daily_risk_limit:
                logger.warning(
                    "Daily risk limit exceeded. Potential loss %.2f + accumulated losses %.2f > %.2f",
                    potential_loss,
                    self.daily_loss,
                    daily_risk_limit,
                )
                return None

            return {"quantity": qty}
        except Exception as exc:
            logger.error(f"Error calculating position size: {exc}")
            return None

    def update_after_trade(self, realised_pnl: float) -> None:
        """Update internal state after a trade has closed.

        Args:
            realised_pnl: Profit or loss realised from the trade.  Positive
                values increase equity, negative values reduce it.
        """
        self.equity += realised_pnl
        # Update peak equity
        self.equity_peak = max(self.equity, self.equity_peak)
        # Update daily loss (losses accumulate as positive values)
        if realised_pnl < 0:
            self.daily_loss += abs(realised_pnl)
            self.consecutive_losses += 1
        else:
            # Reset consecutive loss counter on profitable trade
            self.consecutive_losses = 0

        # Check drawdown
        current_drawdown = (self.equity_peak - self.equity) / self.equity_peak
        if current_drawdown >= self.max_drawdown:
            logger.warning(
                "Max drawdown threshold reached (%.2f%%). Trading should be halted.",
                current_drawdown * 100,
            )

    def reset_daily_limits(self) -> None:
        """Reset daily loss and consecutive loss counters.  Should be called at the start of a new trading day."""
        logger.info("Resetting daily loss and consecutive loss counters.")
        self.daily_loss = 0.0
        self.consecutive_losses = 0

    def get_risk_status(self) -> Dict[str, float]:
        """Return a snapshot of current risk metrics."""
        return {
            "equity": self.equity,
            "equity_peak": self.equity_peak,
            "daily_loss": self.daily_loss,
            "consecutive_losses": self.consecutive_losses,
        }

    def update_position_status(self, is_open: bool) -> None:
        """Placeholder for compatibility with RealTimeTrader.

        In more advanced implementations this method could track the number
        of open positions and adjust risk accordingly.  It is included
        here for interface consistency and future expansion.
        """
        logger.debug("update_position_status called with is_open=%s", is_open)