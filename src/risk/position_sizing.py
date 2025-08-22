# src/risk/position_sizing.py
"""
Position sizing for options: returns LOTS (contracts) based on risk budget and SL distance.

lots = floor( (account_equity * risk_per_trade) / (sl_points * lot_size) )
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

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
            Lots (integer). 0 if invalid inputs.
        """
        try:
            inst = getattr(settings, "instruments", object())
            risk = getattr(settings, "risk", object())

            _lot = int(lot_size or getattr(inst, "nifty_lot_size", 50))
            _risk = float(
                risk_per_trade if risk_per_trade is not None else getattr(risk, "risk_per_trade", 0.01)
            )
            min_lots = int(getattr(inst, "min_lots", 1))
            max_lots = int(getattr(inst, "max_lots", 100))
        except Exception as e:
            log.error("PositionSizing: settings load failed: %s", e)
            return 0

        # --- explicit validation (names each bad field) ---
        if equity is None or not math.isfinite(float(equity)) or float(equity) <= 0:
            log.warning("PositionSizing: invalid equity: %.4f", float(equity or 0.0))
            return 0
        if sl_points is None or not math.isfinite(float(sl_points)) or float(sl_points) <= 0:
            log.warning("PositionSizing: invalid sl_points: %.4f", float(sl_points or 0.0))
            return 0
        if _lot is None or int(_lot) <= 0:
            log.warning("PositionSizing: invalid lot_size: %s", _lot)
            return 0
        if _risk is None or not math.isfinite(float(_risk)) or float(_risk) <= 0:
            log.warning("PositionSizing: invalid risk_per_trade: %.6f", float(_risk or 0.0))
            return 0

        try:
            rupees_at_risk = float(equity) * float(_risk)
            rupees_per_lot = float(sl_points) * int(_lot)
            raw_lots = rupees_at_risk / max(rupees_per_lot, 1e-9)
            lots = int(math.floor(raw_lots))
        except Exception as e:
            log.exception("PositionSizing: math failed: %s", e)
            return 0

        # --- affordable-min logic ---
        # If we can afford at least 1 lot but lots < min_lots, take min_lots; else skip.
        if lots < min_lots:
            if raw_lots >= 1.0:
                log.debug(
                    "PositionSizing: raw_lots=%.2f < min_lots=%d, upgrading to min_lots.",
                    raw_lots, min_lots,
                )
                lots = min_lots
            else:
                log.debug(
                    "PositionSizing: raw_lots=%.2f < 1.0 → 0 lots (skip trade).",
                    raw_lots,
                )
                return 0

        clamped = min(lots, max_lots)

        log.debug(
            "Sizing: equity=₹%.2f risk=%.2f%% sl_pts=%.2f lot=%d → raw=%.2f lots, final=%d (min=%d, max=%d)",
            float(equity),
            float(_risk) * 100.0,
            float(sl_points),
            int(_lot),
            float(raw_lots),
            int(clamped),
            int(min_lots),
            int(max_lots),
        )
        return int(clamped)

    # ---------------- helpers ----------------

    @staticmethod
    def from_settings(sl_points: float, equity: Optional[float] = None) -> int:
        """
        Shortcut using defaults from settings.
        """
        risk_cfg = getattr(settings, "risk", object())
        inst_cfg = getattr(settings, "instruments", object())

        eq_default = float(getattr(risk_cfg, "default_equity", 0.0))
        eq = float(equity if equity is not None else eq_default)

        return PositionSizing.lots_from_equity(
            equity=eq,
            sl_points=float(sl_points),
            lot_size=int(getattr(inst_cfg, "nifty_lot_size", 50)),
            risk_per_trade=float(getattr(risk_cfg, "risk_per_trade", 0.01)),
        )

    @staticmethod
    def diagnostic(sl_points: float, equity: Optional[float] = None) -> Tuple[int, float]:
        """
        Returns (lots, rupees_at_risk) for diagnostics (/diag).
        """
        risk_cfg = getattr(settings, "risk", object())
        inst_cfg = getattr(settings, "instruments", object())

        eq_default = float(getattr(risk_cfg, "default_equity", 0.0))
        eq = float(equity if equity is not None else eq_default)

        lots = PositionSizing.from_settings(sl_points, equity=eq)
        lot_size = int(getattr(inst_cfg, "nifty_lot_size", 50))
        rupees_at_risk = float(lots) * float(sl_points) * lot_size
        return lots, rupees_at_risk