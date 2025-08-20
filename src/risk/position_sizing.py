"""
Position sizing for options: returns LOTS (contracts) based on risk budget and SL distance.

Class name MUST be `PositionSizing` (not PositionSizer).
"""

from __future__ import annotations

import logging
import math

from src.config import settings

log = logging.getLogger(__name__)


class PositionSizing:
    """
    Compute lots = floor( (equity * risk_per_trade) / (sl_points * lot_size) )

    If result < min_lots => 0 (skip). Cap by MAX_LOTS.
    """

    @staticmethod
    def lots_from_equity(
        *,
        equity: float,
        sl_points: float,
        lot_size: int | None = None,
        risk_per_trade: float | None = None,
    ) -> int:
        lot_size = int(lot_size or getattr(settings.instruments, "nifty_lot_size", getattr(settings, "NIFTY_LOT_SIZE", 50)))
        risk = float(risk_per_trade if risk_per_trade is not None else getattr(settings.risk, "risk_per_trade", 0.01))

        try:
            equity = float(equity)
            sl_points = float(sl_points)
            if equity <= 0 or sl_points <= 0:
                return 0

            rupees_at_risk = equity * risk
            qty_contracts = math.floor(rupees_at_risk / max(0.01, sl_points * lot_size))
            if qty_contracts <= 0:
                return 0

            # clamp lot boundaries
            mn = int(getattr(settings.instruments, "min_lots", getattr(settings, "MIN_LOTS", 1)))
            mx = int(getattr(settings.instruments, "max_lots", getattr(settings, "MAX_LOTS", 10)))
            return max(mn, min(mx, qty_contracts))
        except Exception as e:
            log.warning("Position sizing error: %s", e)
            return 0