# src/risk/position_sizing.py
"""
A stateless utility for calculating position size based on risk parameters.
"""

from __future__ import annotations
import logging
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config import RiskConfig
    from src.risk.session import TradingSession

logger = logging.getLogger(__name__)


class PositionSizer:
    """
    A stateless calculator for determining position size in contracts.
    It takes risk parameters and session state to determine a safe trade size.
    """

    def __init__(self, risk_config: "RiskConfig"):
        if not hasattr(risk_config, "risk_per_trade_pct"):
             raise TypeError("A valid RiskConfig instance is required.")
        self.config = risk_config

    def calculate_quantity(
        self,
        session: "TradingSession",
        entry_price: float,
        stop_loss_price: float,
        lot_size: int,
    ) -> int:
        """
        Calculates the number of contracts (quantity) for a trade.

        Args:
            session: The current TradingSession, containing equity information.
            entry_price: The proposed entry price for the trade.
            stop_loss_price: The absolute stop-loss price for the trade.
            lot_size: The number of contracts in one lot for the instrument.

        Returns:
            The calculated quantity of contracts, or 0 if the trade is not viable.
        """
        try:
            if entry_price <= 0 or stop_loss_price <= 0:
                logger.warning("Invalid entry or stop-loss price for sizing.")
                return 0

            sl_points = abs(entry_price - stop_loss_price)
            if sl_points == 0:
                logger.warning("Stop-loss distance cannot be zero.")
                return 0

            # Determine the total capital available for risk calculation
            account_equity = session.current_equity
            if account_equity <= 0:
                logger.warning("Account equity is zero or negative. Cannot size position.")
                return 0

            # Calculate the monetary risk for this trade
            risk_per_contract = sl_points
            risk_budget = account_equity * self.config.risk_per_trade_pct

            if risk_per_contract == 0:
                logger.warning("Risk per contract is zero. Cannot size position.")
                return 0

            # Calculate the number of contracts based on the risk budget
            num_contracts = math.floor(risk_budget / risk_per_contract)

            # Adjust to be a multiple of the lot size
            if lot_size <= 0:
                logger.warning("Lot size must be positive.")
                return 0

            num_lots = math.floor(num_contracts / lot_size)

            # Clamp the number of lots to the configured min/max
            num_lots = max(self.config.min_lots, num_lots)
            num_lots = min(self.config.max_lots, num_lots)

            final_quantity = num_lots * lot_size

            if final_quantity <= 0:
                logger.info("Calculated position size is zero after adjustments.")
                return 0

            logger.info(
                f"Calculated position size: {final_quantity} contracts ({num_lots} lots). "
                f"Risk budget: {risk_budget:.2f}, SL points: {sl_points:.2f}"
            )
            return final_quantity

        except Exception as e:
            logger.error(f"Error calculating position size: {e}", exc_info=True)
            return 0