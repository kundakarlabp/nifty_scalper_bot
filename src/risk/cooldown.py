"""Adaptive cool-down utilities for loss streak management.

The bot previously halted trading for the rest of the session whenever the
configured consecutive-loss limit was breached.  That approach is safe but it
also means the strategy cannot recover after a brief rough patch, even if
market conditions improve later in the day.

This module implements :class:`LossCooldownManager`, a small state container
that keeps track of recent losses and intraday drawdown to dynamically size the
cool-down window.  It supports exponential backoff when losses keep piling up
and gradually relaxes the penalty once profitable trades return.  The logic is
completely self-contained so it can be unit-tested without booting the full
runner stack.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta

from src.config import RiskSettings


@dataclass
class CooldownState:
    """Public snapshot returned by :class:`LossCooldownManager`.

    Attributes
    ----------
    active_until:
        When the cool-down expires. ``None`` when no cool-down is in effect.
    severity:
        Scalar multiplier applied on top of the base cool-down duration.  A
        value of ``1.0`` means no backoff is currently applied.
    last_trigger:
        Timestamp of the last cool-down trigger, ``None`` if never triggered.
    """

    active_until: datetime | None
    severity: float
    last_trigger: datetime | None


class LossCooldownManager:
    """Maintain an adaptive cool-down after loss streaks.

    The manager keeps a "severity" multiplier that grows when fresh losses
    arrive and shrinks when profitable trades return.  The multiplier is bound
    between 1 and the ratio of ``loss_cooldown_max_minutes`` to
    ``loss_cooldown_minutes`` from :class:`RiskSettings`.
    """

    def __init__(self, cfg: RiskSettings) -> None:
        self._cfg = cfg
        self._level: float = 0.0
        self._active_until: datetime | None = None
        self._last_trigger: datetime | None = None

    # ------------------------------------------------------------------
    # state helpers
    # ------------------------------------------------------------------
    def reset_for_new_day(self) -> None:
        """Reset cool-down state at the start of a new session."""

        self._level = 0.0
        self._active_until = None
        self._last_trigger = None

    @property
    def severity(self) -> float:
        """Current severity multiplier used to scale cool-down windows."""

        base_minutes = self._base_minutes()
        if base_minutes <= 0.0:
            return 1.0
        factor = self._backoff_factor() ** self._level
        max_factor = self._max_factor(base_minutes)
        if max_factor > 0.0:
            factor = min(factor, max_factor)
        return max(1.0, factor)

    def snapshot(self) -> CooldownState:
        """Return a read-only view of the internal state."""

        return CooldownState(
            active_until=self._active_until,
            severity=self.severity,
            last_trigger=self._last_trigger,
        )

    # ------------------------------------------------------------------
    # configuration helpers
    # ------------------------------------------------------------------
    def _base_minutes(self) -> float:
        return float(max(0, int(self._cfg.loss_cooldown_minutes)))

    def _max_factor(self, base_minutes: float) -> float:
        if base_minutes <= 0.0:
            return 0.0
        max_minutes = float(max(base_minutes, int(self._cfg.loss_cooldown_max_minutes)))
        return max_minutes / base_minutes if max_minutes > 0 else 0.0

    def _drawdown_ratio(self, day_loss: float, max_daily_loss: float | None) -> float:
        if max_daily_loss is None or max_daily_loss <= 0:
            return 0.0
        if day_loss <= 0:
            return 0.0
        return max(0.0, min(1.0, float(day_loss) / float(max_daily_loss)))

    def _loss_streak_threshold(self) -> int:
        custom = getattr(self._cfg, "loss_cooldown_trigger_after_losses", None)
        if custom is None:
            return max(1, int(self._cfg.consecutive_loss_limit))
        return max(1, int(custom))

    def _should_trigger(self, streak: int, ratio: float) -> bool:
        if streak >= self._loss_streak_threshold():
            return True
        return ratio >= float(self._cfg.loss_cooldown_drawdown_pct)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def active_until(self, now: datetime) -> datetime | None:
        """Return the current cool-down expiry if it is still active."""

        if self._active_until is None:
            return None
        if now >= self._active_until:
            self._active_until = None
            return None
        return self._active_until

    def register_trade(
        self,
        *,
        now: datetime,
        pnl: float,
        streak: int,
        day_loss: float,
        max_daily_loss: float | None,
    ) -> datetime | None:
        """Update state after a trade and return the cool-down expiry.

        Parameters
        ----------
        now:
            Timestamp associated with the trade close.
        pnl:
            Realised P&L of the trade in rupees.
        streak:
            Current consecutive loss count *after* accounting for the latest
            trade.
        day_loss:
            Intraday realised loss after the trade.
        max_daily_loss:
            Configured daily loss cap used to compute drawdown ratios.
        """

        if pnl >= 0:
            self._on_winner()
            return self.active_until(now)

        ratio = self._drawdown_ratio(day_loss, max_daily_loss)
        if not self._should_trigger(streak, ratio):
            return self.active_until(now)

        base_minutes = self._base_minutes()
        if base_minutes <= 0:
            # Cool-down disabled; ensure we clear out any previous state.
            self._active_until = None
            self._last_trigger = now
            return None

        self._apply_backoff(ratio, base_minutes)
        minutes = min(base_minutes * self.severity, float(self._cfg.loss_cooldown_max_minutes))
        expiry = now + timedelta(minutes=minutes)
        if self._active_until is None or expiry > self._active_until:
            self._active_until = expiry
        self._last_trigger = now
        return self._active_until

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _apply_backoff(self, ratio: float, base_minutes: float) -> None:
        increment = 1.0
        threshold = float(self._cfg.loss_cooldown_drawdown_pct)
        if ratio >= threshold:
            overshoot = max(0.0, ratio - threshold)
            increment += overshoot * float(self._cfg.loss_cooldown_drawdown_scale)
        self._level += increment
        backoff = self._backoff_factor()
        if backoff > 1.0:
            max_factor = self._max_factor(base_minutes)
            if max_factor > 0.0:
                max_level = math.log(max_factor, backoff)
                self._level = min(self._level, max_level)

    def _on_winner(self) -> None:
        relax = float(self._cfg.loss_cooldown_relax_multiplier)
        if relax <= 0.0:
            self._level = 0.0
            return
        self._level = max(0.0, (self._level * relax) - (1.0 - relax))

    def _backoff_factor(self) -> float:
        return max(1.0, float(self._cfg.loss_cooldown_backoff))


__all__ = ["CooldownState", "LossCooldownManager"]
