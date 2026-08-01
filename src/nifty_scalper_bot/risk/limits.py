"""Risk switch helpers for absolute day loss and loss streak breakers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class SwitchSnapshot:
    """Immutable view of :class:`RiskSwitches` state."""

    day_loss: float
    max_day_loss: float
    consecutive_losses: int
    max_consecutive_losses: int
    cooldown_remaining: float


@dataclass(slots=True)
class RiskSwitches:
    """Track day loss, consecutive loss streaks and cooldowns."""

    max_day_loss: float
    max_consecutive_losses: int
    cooldown_minutes: float
    reset_hour_utc: int
    max_day_profit: float = 0.0
    clock: Callable[[], datetime] = _now_utc
    _day_anchor: datetime = field(init=False, repr=False)
    _day_pnl: float = field(init=False, repr=False, default=0.0)
    _consecutive_losses: int = field(init=False, repr=False, default=0)
    _cooldown_until: datetime | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        self.max_day_loss = max(0.0, float(self.max_day_loss))
        self.max_day_profit = max(0.0, float(self.max_day_profit))
        self.max_consecutive_losses = max(0, int(self.max_consecutive_losses))
        self.cooldown_minutes = max(0.0, float(self.cooldown_minutes))
        self.reset_hour_utc = int(self.reset_hour_utc)
        self._day_anchor = self._current_anchor()
        self._day_pnl = 0.0
        self._consecutive_losses = 0
        self._cooldown_until = None

    # Public API ---------------------------------------------------------
    def record_pnl(self, delta: float) -> None:
        """Record a realised PnL delta for the active trading day.

        Day P&L only. The loss streak is deliberately NOT advanced here: this
        is fed by incremental position-accounting deltas, so a single trade
        exiting in several partial slices produced one streak increment per
        slice and tripped the consecutive-loss breaker after one losing trade.
        Use :meth:`record_trade_result` for per-trade outcomes.
        """

        self._reset_if_needed()
        self._day_pnl += float(delta)

    def record_trade_result(self, net_pnl: float) -> None:
        """Advance the loss streak and cooldown from one completed trade."""

        self._reset_if_needed()
        value = float(net_pnl)
        if value < 0:
            self._consecutive_losses += 1
            if self.cooldown_minutes > 0:
                self._cooldown_until = self._now() + timedelta(
                    minutes=self.cooldown_minutes
                )
        elif value > 0:
            self._consecutive_losses = 0

    def restore_runtime(
        self,
        consecutive_losses: int = 0,
        cooldown_until_epoch: float | None = None,
    ) -> None:
        """Re-seed streak/cooldown state after an intraday process restart."""

        self._reset_if_needed()
        self._consecutive_losses = max(
            self._consecutive_losses, max(0, int(consecutive_losses))
        )
        if cooldown_until_epoch:
            until = datetime.fromtimestamp(float(cooldown_until_epoch), timezone.utc)
            if until > self._now() and (
                self._cooldown_until is None or until > self._cooldown_until
            ):
                self._cooldown_until = until

    def cooldown_until_epoch(self) -> float:
        """Return the active cooldown deadline as a UTC epoch (0.0 when idle)."""

        if self.cooldown_remaining() <= 0.0 or self._cooldown_until is None:
            return 0.0
        return self._cooldown_until.timestamp()

    def engage_cooldown(self, minutes: float | None = None) -> None:
        """Force a cooldown window, used for broker rejections."""

        self._reset_if_needed()
        duration = float(minutes) if minutes is not None else self.cooldown_minutes
        if duration <= 0:
            self._cooldown_until = None
            return
        self._cooldown_until = self._now() + timedelta(minutes=duration)

    def breach_reason(self) -> str | None:
        """Return reason string when a breaker has been tripped."""

        self._reset_if_needed()
        if self.max_day_loss > 0 and self.day_loss() >= self.max_day_loss:
            return "Max day loss reached"
        if self.max_day_profit > 0 and self.day_profit() >= self.max_day_profit:
            return "Daily profit target reached"
        if (
            self.max_consecutive_losses > 0
            and self._consecutive_losses >= self.max_consecutive_losses
        ):
            return "Max consecutive losses reached"
        return None

    def cooldown_remaining(self) -> float:
        self._reset_if_needed()
        if self._cooldown_until is None:
            return 0.0
        remaining = (self._cooldown_until - self._now()).total_seconds()
        if remaining <= 0:
            self._cooldown_until = None
            return 0.0
        return remaining

    def reset_day(self) -> None:
        """Reset counters for a new trading day."""

        self._day_anchor = self._current_anchor()
        self._day_pnl = 0.0
        self._consecutive_losses = 0
        self._cooldown_until = None

    def snapshot(self) -> SwitchSnapshot:
        return SwitchSnapshot(
            day_loss=self.day_loss(),
            max_day_loss=self.max_day_loss,
            consecutive_losses=self._consecutive_losses,
            max_consecutive_losses=self.max_consecutive_losses,
            cooldown_remaining=self.cooldown_remaining(),
        )

    # Accessors ----------------------------------------------------------
    def day_loss(self) -> float:
        self._reset_if_needed()
        pnl = self._day_pnl
        return max(-pnl, 0.0)

    def day_profit(self) -> float:
        self._reset_if_needed()
        return max(self._day_pnl, 0.0)

    def consecutive_losses(self) -> int:
        self._reset_if_needed()
        return self._consecutive_losses

    # Internal helpers ---------------------------------------------------
    def _now(self) -> datetime:
        value = self.clock()
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _current_anchor(self) -> datetime:
        now = self._now()
        anchor = now.replace(
            hour=self.reset_hour_utc,
            minute=0,
            second=0,
            microsecond=0,
        )
        if now < anchor:
            anchor -= timedelta(days=1)
        return anchor

    def _reset_if_needed(self) -> None:
        now = self._now()
        if now >= self._day_anchor + timedelta(days=1):
            self.reset_day()


__all__ = ["RiskSwitches", "SwitchSnapshot"]
