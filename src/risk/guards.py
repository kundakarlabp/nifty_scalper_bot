"""Runtime trading guardrails."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional

from src.config import settings
from src.utils.env import env_flag
from src.utils.reliability import RateLimiter


@dataclass
class RiskConfig:
    """Configuration for :class:`RiskGuards` sourced from ``settings.guards``."""

    max_orders_per_min: int = field(
        default_factory=lambda: settings.guards.max_orders_per_min
    )
    daily_loss_cap: float = field(default_factory=lambda: settings.guards.daily_loss_cap)
    trading_start_hm: str = field(
        default_factory=lambda: settings.guards.trading_start_hhmm
    )
    trading_end_hm: str = field(
        default_factory=lambda: settings.guards.trading_end_hhmm
    )
    kill_env: bool | str = field(default_factory=lambda: settings.guards.kill_env)
    kill_file: str = field(default_factory=lambda: settings.guards.kill_file)

    def __post_init__(self) -> None:
        self.max_orders_per_min = int(self.max_orders_per_min)
        self.daily_loss_cap = float(self.daily_loss_cap)
        self.trading_start_hm = str(self.trading_start_hm)
        self.trading_end_hm = str(self.trading_end_hm)
        self.kill_env = self._coerce_kill_flag(self.kill_env)
        self.kill_file = str(self.kill_file or "")

    @staticmethod
    def _coerce_kill_flag(value: bool | str) -> bool:
        if isinstance(value, bool):
            return value
        text = str(value or "").strip()
        if not text:
            return True
        lowered = text.lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        return env_flag(text, True)


class RiskGuards:
    """Lightweight pre-trade checks for live trading."""

    def __init__(self, config: Optional[RiskConfig] = None) -> None:
        self.cfg = config or RiskConfig()
        self.rate = RateLimiter(self.cfg.max_orders_per_min)
        self._pnl_today = 0.0

    # ------------------------------------------------------------------
    def set_pnl_today(self, value: float) -> None:
        """Record realised PnL for the current trading day."""

        self._pnl_today = float(value)

    # ------------------------------------------------------------------
    def _kill_switch(self) -> bool:
        """Return ``True`` if trading should be halted."""

        env_block = not self.cfg.kill_env
        file_block = bool(self.cfg.kill_file and os.path.exists(self.cfg.kill_file))
        return env_block or file_block

    def _within_window(self) -> bool:
        """Return ``True`` if current time is inside the trading window."""

        hms = time.strftime("%H:%M", time.localtime())
        return self.cfg.trading_start_hm <= hms <= self.cfg.trading_end_hm

    # ------------------------------------------------------------------
    def ok_to_trade(self, decision: object | None = None) -> bool:
        """Return ``True`` if a new order may be placed."""

        if self._kill_switch():
            return False
        if not self._within_window():
            return False
        if self._pnl_today <= -abs(self.cfg.daily_loss_cap):
            return False
        if not self.rate.allow():
            return False
        return True


__all__ = ["RiskConfig", "RiskGuards"]
