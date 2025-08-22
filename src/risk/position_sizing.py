# src/risk/position_sizing.py
"""
Position sizing for options: returns LOTS (contracts) based on risk budget and SL distance.

Formula:
    lots = floor( (equity * risk_per_trade) / (sl_points * lot_size) )

- `lot_size` = units per lot (e.g., 75 for NIFTY)
- We clamp to [min_lots, max_lots] from settings.instruments
- If computed lots < min_lots -> return 0 to skip unsafe trades
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from src.config import settings

log = logging.getLogger(__name__)


class PositionSizing:
    """Stateless helper converting account risk to option LOTS."""

    @staticmethod
    def lots_from_equity(
        *,
        equity: float,
        sl_points: float,
        lot_size: Optional[int] = None,
        risk_per_trade: Optional[float] = None,
    ) -> int:
        """
        Args:
            equity: current account equity (₹)
            sl_points: stop-loss distance in points (>0)
            lot_size: override lot size (default: settings.instruments.nifty_lot_size)
            risk_per_trade: override risk fraction (default: settings.risk.risk_per_trade)
        Returns:
            int: number of lots (0 means "do not trade")
        """
        # Read config safely
        try:
            _lot_size = int(lot_size or getattr(settings.instruments, "nifty_lot_size", 75))
            _risk = float(risk_per_trade if risk_per_trade is not None else getattr(settings.risk, "risk_per_trade", 0.01))
            min_lots = int(getattr(settings.instruments, "min_lots", 1))
            max_lots = int(getattr(settings.instruments, "max_lots", 10))
        except Exception as e:
            log.error("PositionSizing: settings read failed: %s", e)
            return 0

        # Validate inputs
        if equity <= 0 or sl_points <= 0 or _lot_size <= 0 or _risk <= 0:
            log.warning(
                "PositionSizing: invalid inputs (equity=%.2f, sl_points=%.2f, lot_size=%s, risk=%.4f)",
                equity, sl_points, _lot_size, _risk
            )
            return 0

        # Budget and per-lot risk
        rupees_at_risk = equity * _risk
        rupees_per_lot = sl_points * _lot_size

        try:
            lots = math.floor(rupees_at_risk / rupees_per_lot)
        except Exception as e:
            log.error("PositionSizing: math error: %s", e)
            return 0

        # Threshold and clamps
        if lots < min_lots:
            log.debug("PositionSizing: %d < min_lots(%d) -> 0 (skip).", lots, min_lots)
            return 0
        if lots > max_lots:
            lots = max_lots

        log.info(
            "Sizing: equity=₹%.2f risk=%.2f%% sl=%.2f pts lot_size=%d -> %d lots",
            equity, _risk * 100.0, sl_points, _lot_size, lots
        )
        return int(lots)


# Quick self-checks (optional, not executed in production)
if __name__ == "__main__":
    print("Lots ex1:",
          PositionSizing.lots_from_equity(equity=100_000, sl_points=20, lot_size=75, risk_per_trade=0.01))
    print("Lots ex2:",
          PositionSizing.lots_from_equity(equity=15_000, sl_points=50, lot_size=75, risk_per_trade=0.01))
    print("Lots ex3:",
          PositionSizing.lots_from_equity(equity=500_000, sl_points=10, lot_size=75, risk_per_trade=0.02))