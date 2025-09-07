from __future__ import annotations

"""Lightweight pre-trade risk guards."""

import os
import time
from dataclasses import dataclass
from typing import Optional

from src.utils.reliability import RateLimiter


@dataclass
class RiskConfig:
    """Configuration for :class:`RiskGuards`."""

    max_orders_per_min: int = int(os.getenv("MAX_ORDERS_PER_MIN", "30"))
    daily_loss_cap: float = float(os.getenv("DAILY_LOSS_CAP", "9999999"))
    trading_start_hm: str = os.getenv("TRADING_WINDOW_START", "09:20")
    trading_end_hm: str = os.getenv("TRADING_WINDOW_END", "15:25")
    kill_env: str = os.getenv("KILL_SWITCH_ENV", "ENABLE_TRADING")
    kill_file: str = os.getenv("KILL_SWITCH_FILE", "")


class RiskGuards:
    """Evaluate simple risk checks before placing trades."""

    def __init__(self, config: Optional[RiskConfig] = None) -> None:
        self.cfg = config or RiskConfig()
        self.rate = RateLimiter(self.cfg.max_orders_per_min)
        self._pnl_today = 0.0

    def set_pnl_today(self, value: float) -> None:
        """Update today's realised PnL."""
        self._pnl_today = float(value)

    # ------------------------------------------------------------------
    def _kill_switch(self) -> bool:
        env_block = os.getenv(self.cfg.kill_env, "true").lower() in {"false", "0", "no"}
        file_block = bool(self.cfg.kill_file and os.path.exists(self.cfg.kill_file))
        return env_block or file_block

    def _within_window(self) -> bool:
        hms = time.strftime("%H:%M", time.localtime())
        return self.cfg.trading_start_hm <= hms <= self.cfg.trading_end_hm

    def ok_to_trade(self, _decision: object | None = None) -> bool:
        """Return ``True`` if trading is permitted."""
        if self._kill_switch():
            return False
        if not self._within_window():
            return False
        if self._pnl_today <= -abs(self.cfg.daily_loss_cap):
            return False
        if not self.rate.allow():
            return False
        return True
