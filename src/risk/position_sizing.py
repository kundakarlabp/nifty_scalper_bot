# src/risk/position_sizing.py

from __future__ import annotations
import logging
import math
from typing import Optional, Tuple
from src.config import settings

# ✅ import Kite client wrapper (adjust path if your kite client is elsewhere)
try:
    from src.brokers.zerodha_client import kite
except ImportError:
    kite = None

log = logging.getLogger(__name__)


class PositionSizing:
    """Compute position size in lots."""

    @staticmethod
    def get_live_equity() -> float:
        """
        Get current equity from broker.
        Falls back to default_equity from .env if unavailable.
        """
        if kite is not None:
            try:
                margins = kite.margins("equity")
                net = float(margins.get("net", 0.0))
                if net > 0:
                    return net
            except Exception as e:
                log.warning("PositionSizing: live equity fetch failed: %s", e)

        # fallback
        risk_cfg = getattr(settings, "risk", object())
        return float(getattr(risk_cfg, "default_equity", 0.0))

    @staticmethod
    def lots_from_equity(
        *,
        equity: float,
        sl_points: float,
        lot_size: Optional[int] = None,
        risk_per_trade: Optional[float] = None,
    ) -> int:
        ...
        # 🔑 keep your existing validation/math unchanged
        ...

    @staticmethod
    def from_settings(sl_points: float, equity: Optional[float] = None) -> int:
        """
        Uses live broker equity if not supplied.
        """
        inst_cfg = getattr(settings, "instruments", object())
        risk_cfg = getattr(settings, "risk", object())

        eq = float(equity if equity is not None else PositionSizing.get_live_equity())

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
        inst_cfg = getattr(settings, "instruments", object())
        eq = float(equity if equity is not None else PositionSizing.get_live_equity())

        lots = PositionSizing.from_settings(sl_points, equity=eq)
        lot_size = int(getattr(inst_cfg, "nifty_lot_size", 50))
        rupees_at_risk = float(lots) * float(sl_points) * lot_size
        return lots, rupees_at_risk