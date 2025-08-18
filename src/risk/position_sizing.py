# src/risk/position_sizing.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from src.config import RiskConfig


@dataclass
class PositionSizer:
    """
    Risk-aware position sizing with lot guards and daily gates.

    - risk_per_trade_pct: fraction of equity risked per trade (e.g., 0.01 = 1%)
    - min_lots / max_lots: hard lot clamps
    - consecutive loss / daily DD / max trades checks are handled in TradingSession,
      but the sizer exposes a `check_risk_gates(session)` helper for convenience.
    """
    risk_cfg: RiskConfig
    account_size: float = 100_000.0

    def set_account_size(self, equity: float) -> None:
        self.account_size = max(100.0, float(equity))

    # ---------- sizing for options (primary path) ----------

    def calculate_lot_quantity(
        self,
        lot_size: int,
        sl_points: Optional[float] = None,
    ) -> int:
        """
        Return lot count (not units). If sl_points is provided, compute risk-based lot count.
        Otherwise, default to min_lots (safer in absence of SL info).
        """
        cfg = self.risk_cfg
        if sl_points is None or sl_points <= 0:
            return int(max(cfg.min_lots, 1))

        risk_rupees = self.account_size * float(cfg.risk_per_trade_pct)
        # Options risk approximation: SL points * lot_size * lots
        max_lots_risk_based = math.floor(risk_rupees / (float(sl_points) * int(lot_size) + 1e-9))
        lots = max(cfg.min_lots, min(cfg.max_lots, max_lots_risk_based))
        return int(max(lots, 0))

    def calculate_position_size(
        self,
        entry_price: float,
        sl_points: float,
        lot_size: Optional[int] = None,
    ) -> int:
        """
        Return UNITS (quantity), not lots. If lot_size given, converts lots → units.
        Primarily used by backtests (where lot_size may be absent).
        """
        if sl_points <= 0:
            return 0

        if lot_size:
            lots = self.calculate_lot_quantity(lot_size=lot_size, sl_points=sl_points)
            return int(lots * lot_size)

        # Generic fallback (non-lotted asset): risk = qty * sl_points
        risk_rupees = self.account_size * float(self.risk_cfg.risk_per_trade_pct)
        qty = math.floor(risk_rupees / float(sl_points))
        return int(max(qty, 0))

    # ---------- gates ----------

    def check_risk_gates(self, session: "TradingSession") -> Optional[str]:
        """
        Return a reason string if the trade should be denied, else None.
        """
        cfg = self.risk_cfg

        if session.trades_today >= cfg.max_trades_per_day:
            return "max trades per day reached"

        if session.consecutive_losses >= cfg.consecutive_loss_limit:
            return "consecutive loss limit reached"

        if session.daily_drawdown_pct <= -abs(cfg.max_daily_drawdown_pct):
            return "daily drawdown cap reached"

        return None
