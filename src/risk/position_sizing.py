# src/risk/position_sizing.py
"""
Position sizing for options: returns LOTS (contracts) based on risk budget and SL distance.

lots = floor( (account_equity * risk_per_trade) / (sl_points * lot_size) )

Class name MUST be `PositionSizing`.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from src.config import settings

log = logging.getLogger(__name__)


class PositionSizing:
    """
    Computes position size in terms of *lots* for a given trade.
    """

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
            equity: current total account equity (₹).
            sl_points: stop-loss distance in points (option points).
            lot_size: instrument lot size (defaults to settings.instruments.nifty_lot_size).
            risk_per_trade: fraction of equity to risk per trade (defaults to settings.risk.risk_per_trade).

        Returns:
            Lots (integer). 0 if below min_lots or invalid inputs.
        """
        try:
            _lot = int(lot_size or getattr(getattr(settings, "instruments", object()), "nifty_lot_size", 50))
            _risk = float(risk_per_trade if risk_per_trade is not None else getattr(getattr(settings, "risk", object()), "risk_per_trade", 0.01))
            min_lots = int(getattr(getattr(settings, "instruments", object()), "min_lots", 1))
            max_lots = int(getattr(getattr(settings, "instruments", object()), "max_lots", 100))
        except Exception as e:
            log.error("PositionSizing: settings load failed: %s", e)
            return 0

        if equity <= 0 or sl_points <= 0 or _lot <= 0 or _risk <= 0:
            log.warning("PositionSizing: invalid inputs (equity, sl_points, lot, risk must be > 0).")
            return 0

        try:
            rupees_at_risk = equity * _risk
            rupees_per_lot = sl_points * _lot  # points * qty
            lots = int(math.floor(rupees_at_risk / max(rupees_per_lot, 1e-9)))
        except Exception as e:
            log.exception("PositionSizing: math failed: %s", e)
            return 0

        if lots < min_lots:
            log.debug("PositionSizing: %d lots < min_lots (%d) -> 0", lots, min_lots)
            return 0

        clamped = min(lots, max_lots)
        log.info(
            "Sizing: equity=₹%.2f risk=%.2f%% sl_pts=%.2f lot=%d -> %d lots",
            equity, _risk * 100.0, sl_points, _lot, clamped
        )
        return clamped