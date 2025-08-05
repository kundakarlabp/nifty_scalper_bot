# src/risk/position_sizing.py
"""
Adaptive position sizing module.

This module provides a `PositionSizing` class that determines how many
contract lots to trade based on the live account balance from Zerodha,
risk settings, and market conditions. It protects the account by enforcing
per-trade risk limits, a daily drawdown cap, and a maximum number of
consecutive losses.

The calculation assumes that each point move in the underlying contract
is worth `Config.NIFTY_LOT_SIZE` rupees.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

# ✅ Corrected import path based on typical project structure
from src.config import Config
# ✅ Corrected import path
from src.auth.zerodha_auth import get_kite_client  # fetch live capital

logger = logging.getLogger(__name__)


def get_live_account_balance() -> float:
    """Fetches the live cash balance from Zerodha."""
    try:
        kite = get_kite_client()
        margins = kite.margins(segment='equity')
        cash = margins['available']['cash']
        balance = float(cash)
        logger.info(f"💰 Live account balance fetched: ₹{balance:.2f}")
        return balance
    except Exception as e:
        fallback = 30000.0
        logger.warning(f"⚠️ Failed to fetch live account balance, using fallback: {e}")
        return fallback


@dataclass
class PositionSizing:
    """Risk manager that calculates position sizes and tracks drawdown."""

    account_size: float = field(default_factory=get_live_account_balance)
    risk_per_trade: float = Config.RISK_PER_TRADE
    daily_risk: float = Config.MAX_DRAWDOWN # Often the same as max_drawdown
    max_drawdown: float = Config.MAX_DRAWDOWN
    lot_size: int = Config.NIFTY_LOT_SIZE
    min_lots: int = Config.MIN_LOTS
    max_lots: int = Config.MAX_LOTS
    consecutive_loss_limit: int = Config.CONSECUTIVE_LOSS_LIMIT

    # Internal state
    daily_loss: float = 0.0
    equity_peak: float = field(init=False)
    equity: float = field(init=False)
    consecutive_losses: int = 0

    def __post_init__(self) -> None:
        """Initialize equity tracking."""
        self.equity = self.account_size
        self.equity_peak = self.account_size
        logger.info(f"💰 Live account size initialized: ₹{self.account_size:.2f}")

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        signal_confidence: float, # Assuming 0-10 scale based on logic
        market_volatility: float = 0.0, # Assuming 0.0 - 1.0 scale
    ) -> Optional[Dict[str, int]]:
        """
        Calculates the number of lots to trade based on risk parameters.

        Args:
            entry_price: The price at which the trade is entered.
            stop_loss: The stop loss price for the trade.
            signal_confidence: A score (e.g., 0-10) indicating the strength of the signal.
                               Used to scale position size.
            market_volatility: A measure of market volatility (e.g., 0.0 - 1.0).
                               High volatility can lead to position size reduction.

        Returns:
            A dictionary like {"quantity": <int>} if calculation is successful and
            within limits, otherwise None.
        """
        try:
            # Basic input validation
            if entry_price <= 0 or stop_loss <= 0:
                 logger.warning("⚠️ Invalid entry price or stop loss (<= 0).")
                 return None

            if self.consecutive_losses >= self.consecutive_loss_limit:
                logger.warning("❌ Consecutive loss limit reached. No new trades allowed.")
                return None

            sl_points = abs(entry_price - stop_loss)
            if sl_points <= 0:
                logger.warning("⚠️ Stop loss and entry price are equal or invalid.")
                return None

            risk_per_lot = sl_points * self.lot_size
            if risk_per_lot <= 0:
                 logger.warning("⚠️ Calculated risk per lot is zero or negative.")
                 return None

            trade_risk_budget = self.account_size * self.risk_per_trade

            # Initial quantity calculation
            qty_raw = trade_risk_budget / risk_per_lot # Use float division first
            if qty_raw <= 0:
                logger.info(
                    f"❌ Risk per lot ₹{risk_per_lot:.2f} exceeds trade budget ₹{trade_risk_budget:.2f}."
                )
                return None

            qty = int(qty_raw) # Truncate to integer lots

            # --- Apply Adjustments ---

            # 1. Confidence Factor (assuming signal_confidence is on a 0-10 scale)
            #    If it's 0-1, use signal_confidence directly or adjust the formula.
            confidence_factor = max(0.1, min(signal_confidence / 10.0, 1.0))
            qty = int(qty * confidence_factor)

            # 2. Volatility Adjustment
            if market_volatility > 0.5: # Threshold can be configurable
                qty = max(self.min_lots, qty // 2) # Reduce by half, but not below min

            # 3. Enforce Min/Max Lot Limits
            qty = max(self.min_lots, min(qty, self.max_lots))

            # --- Final Checks ---

            # Check if final quantity is valid
            if qty <= 0:
                 logger.info("❌ Calculated quantity is zero or negative after adjustments.")
                 return None

            # Check against daily risk limit
            potential_loss = qty * risk_per_lot
            daily_risk_limit = self.account_size * self.daily_risk
            if self.daily_loss + potential_loss > daily_risk_limit:
                logger.warning(
                    f"❌ Daily risk limit exceeded. Trade risk ₹{potential_loss:.2f} + "
                    f"accumulated ₹{self.daily_loss:.2f} > ₹{daily_risk_limit:.2f}"
                )
                return None

            logger.debug(f"✅ Calculated position size: {qty} lots (Entry: {entry_price}, SL: {stop_loss}, Conf: {signal_confidence}, Vol: {market_volatility})")
            return {"quantity": qty}

        except Exception as exc:
            logger.error(f"💥 Error calculating position size: {exc}", exc_info=True) # Include traceback
            return None

    def update_after_trade(self, realised_pnl: float) -> bool:
        """
        Updates internal state after a trade is closed.

        Args:
            realised_pnl: The profit or loss from the closed trade.

        Returns:
            True if trading should continue, False if limits (like max drawdown)
            are breached.
        """
        self.equity += realised_pnl
        self.equity_peak = max(self.equity, self.equity_peak)

        if realised_pnl < 0:
            self.daily_loss += abs(realised_pnl)
            self.consecutive_losses += 1
            logger.info(f"📉 Trade closed with loss: ₹{realised_pnl:.2f}. Consecutive losses: {self.consecutive_losses}")
        else:
            self.consecutive_losses = 0
            logger.info(f"📈 Trade closed with profit: ₹{realised_pnl:.2f}. Consecutive losses reset.")

        current_drawdown = (self.equity_peak - self.equity) / self.equity_peak if self.equity_peak > 0 else 0
        logger.debug(f"📊 Equity: ₹{self.equity:.2f}, Peak: ₹{self.equity_peak:.2f}, Drawdown: {current_drawdown*100:.2f}%, Daily Loss: ₹{self.daily_loss:.2f}")

        if current_drawdown >= self.max_drawdown:
            logger.critical(f"❗ Max drawdown of {self.max_drawdown*100:.2f}% reached: {current_drawdown*100:.2f}%. Trading should halt.")
            return False # Signal to stop trading

        # Optional: Check if daily loss limit is hit (might be redundant with per-trade risk check)
        # daily_risk_limit = self.account_size * self.daily_risk
        # if self.daily_loss >= daily_risk_limit:
        #     logger.warning(f"❗ Daily loss limit of ₹{daily_risk_limit:.2f} reached.")
        #     return False

        return True # Continue trading

    def reset_daily_limits(self) -> None:
        """Resets daily loss and consecutive loss counters, typically at the start of a new trading day."""
        logger.info("🔄 Resetting daily loss and consecutive loss counters.")
        self.daily_loss = 0.0
        self.consecutive_losses = 0

    def get_risk_status(self) -> Dict[str, float]:
        """Returns current risk metrics."""
        current_drawdown = (self.equity_peak - self.equity) / self.equity_peak if self.equity_peak > 0 else 0
        return {
            "equity": self.equity,
            "equity_peak": self.equity_peak,
            "current_drawdown": current_drawdown,
            "daily_loss": self.daily_loss,
            "consecutive_losses": float(self.consecutive_losses), # Ensure JSON serializable if needed
            "risk_level": current_drawdown # Simple representation, could be "LOW", "MEDIUM", "HIGH"
        }

    # ✅ Method exists but logic is minimal. Clarify purpose or remove if unused.
    def update_position_status(self, is_open: bool) -> None:
        """
        Updates the status of open positions.
        (Currently a placeholder - logic needs to be implemented based on requirements)
        """
        logger.debug(f"🧾 Position status updated: is_open={is_open}")
        # TODO: Implement logic to track open positions if needed for risk calcs
        # e.g., self._is_position_open = is_open
        # Or track multiple positions if the strategy allows.
