# src/risk/position_sizing.py
"""
Stateless position sizing based on risk budget and stop distance.

- Returns quantity in CONTRACTS (not lots).
- Never exceeds the risk budget. If min_lots cannot be afforded -> returns 0.
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
    A stateless calculator for determining position size (contracts).
    """

    def __init__(self, risk_config: "RiskConfig"):
        # basic duck-typing guard
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
        Args:
            session: TradingSession with `current_equity` (float).
            entry_price: proposed entry price.
            stop_loss_price: absolute stop-loss price.
            lot_size: contracts per lot (e.g., 75 for NIFTY).

        Returns:
            Quantity in contracts (multiple of lot_size), or 0 if not viable.
        """
        try:
            # Basic validation
            entry = float(entry_price)
            sl_px = float(stop_loss_price)
            if entry <= 0 or sl_px <= 0:
                logger.warning("Invalid entry or stop-loss price for sizing.")
                return 0

            sl_points = abs(entry - sl_px)
            if sl_points <= 0:
                logger.warning("Stop-loss distance cannot be zero.")
                return 0

            eq = float(getattr(session, "current_equity", 0.0) or 0.0)
            if eq <= 0:
                logger.warning("Account equity is zero/negative. Cannot size position.")
                return 0

            ls = int(lot_size or 0)
            if ls <= 0:
                logger.warning("Lot size must be positive.")
                return 0

            # Risk budget for this trade (money)
            risk_budget = eq * float(self.config.risk_per_trade_pct)

            # Money risk per 1 lot = sl_points * lot_size
            # Affordable lots strictly within budget:
            affordable_lots = math.floor(risk_budget / (sl_points * ls))

            # Respect configured caps:
            max_lots = int(self.config.max_lots)
            min_lots = int(self.config.min_lots)

            # If we cannot afford the configured minimum, do NOT upsize to it.
            if affordable_lots < min_lots:
                logger.info(
                    "Affordable lots=%d < min_lots=%d: skipping trade. "
                    "risk_budget=%.2f sl_points=%.2f lot_size=%d",
                    affordable_lots, min_lots, risk_budget, sl_points, ls,
                )
                return 0

            final_lots = min(affordable_lots, max_lots)
            final_qty = final_lots * ls

            if final_qty <= 0:
                logger.info("Calculated position size is zero after constraints.")
                return 0

            logger.info(
                "Position size: %d contracts (%d lots). "
                "Risk budget: %.2f | SL points: %.2f | affordable_lots: %d | clamp: [%d..%d]",
                final_qty, final_lots, risk_budget, sl_points, affordable_lots, min_lots, max_lots,
            )
            return final_qty

        except Exception as e:
            logger.error("Error calculating position size: %s", e, exc_info=True)
            return 0


# Backward-compat alias for older imports
PositionSizing = PositionSizer
