"""RiskManager: pre-trade gating and daily state tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional


@dataclass
class RiskState:
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    trade_count: int = 0
    consecutive_losses: int = 0
    is_paused: bool = False
    is_daily_limit_hit: bool = False
    is_cooldown_active: bool = False
    cooldown_until: Optional[datetime] = None
    open_positions_count: int = 0
    total_exposure: float = 0.0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class RiskManager:
    """Enforces all pre-trade risk rules. Single source of truth for trading permissions."""

    def __init__(
        self,
        capital: float,
        daily_loss_pct: float = 2.0,
        max_daily_trades: int = 30,
        max_concurrent: int = 3,
        consecutive_loss_cooldown: int = 3,
        cooldown_minutes: int = 30,
        vix_extreme_threshold: float = 30.0,
    ) -> None:
        self._capital = capital
        self._daily_loss_limit = capital * daily_loss_pct / 100.0
        self._max_daily_trades = max_daily_trades
        self._max_concurrent = max_concurrent
        self._consec_limit = consecutive_loss_cooldown
        self._cooldown_duration = timedelta(minutes=cooldown_minutes)
        self._vix_extreme = vix_extreme_threshold
        self._state = RiskState()
        self._current_vix: float = 0.0

    def can_trade(self) -> tuple[bool, str]:
        """Returns (allowed, reason). Call before every order."""
        s = self._state
        now = datetime.now(timezone.utc)

        if s.is_paused:
            return False, "MANUALLY_PAUSED"
        if s.is_daily_limit_hit:
            return False, "DAILY_LOSS_LIMIT"
        if s.trade_count >= self._max_daily_trades:
            return False, "MAX_DAILY_TRADES"
        if s.is_cooldown_active:
            if s.cooldown_until and now < s.cooldown_until:
                remaining = int((s.cooldown_until - now).total_seconds() / 60)
                return False, f"COOLDOWN_{remaining}MIN"
            else:
                self._state.is_cooldown_active = False
                self._state.cooldown_until = None
        if self._current_vix >= self._vix_extreme:
            return False, f"VIX_EXTREME_{self._current_vix:.1f}"

        return True, "OK"

    def record_trade(self, pnl: float) -> None:
        """Update state after a trade completes."""
        s = self._state
        s.daily_pnl += pnl
        s.daily_pnl_pct = (s.daily_pnl / self._capital) * 100.0
        s.trade_count += 1
        s.last_updated = datetime.now(timezone.utc)

        if pnl < 0:
            s.consecutive_losses += 1
            if s.consecutive_losses >= self._consec_limit:
                s.is_cooldown_active = True
                s.cooldown_until = datetime.now(timezone.utc) + self._cooldown_duration
        else:
            s.consecutive_losses = 0

        if abs(s.daily_pnl) >= self._daily_loss_limit and s.daily_pnl < 0:
            s.is_daily_limit_hit = True

    def update_vix(self, vix: float) -> None:
        self._current_vix = vix

    def update_positions(self, count: int, exposure: float) -> None:
        self._state.open_positions_count = count
        self._state.total_exposure = exposure

    def can_add_position(self) -> bool:
        return self._state.open_positions_count < self._max_concurrent

    def pause(self) -> None:
        self._state.is_paused = True

    def resume(self) -> None:
        self._state.is_paused = False

    def reset_daily(self) -> None:
        self._state = RiskState()

    def get_state(self) -> RiskState:
        return self._state
