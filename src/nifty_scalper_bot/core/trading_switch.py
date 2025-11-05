"""Centralized trading switch for operator pause and cooldown controls."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time

from nifty_scalper_bot.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class TradingSwitchState:
    """Snapshot of the trading switch state for inspection."""

    enabled: bool
    resume_at: float
    remaining_seconds: float
    can_trade: bool


class TradingSwitch:
    """Coordinate operator trading toggles across the application."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._enabled = True
        self._resume_at = 0.0

    def pause(self) -> None:
        """Pause trading until explicitly resumed.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered TradingSwitch.pause",
            extra={"event": "trading_switch_pause_enter"},
        )
        try:
            with self._lock:
                self._enabled = False
                self._resume_at = 0.0
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in TradingSwitch.pause: %s",
                exc,
                extra={"event": "trading_switch_pause_error"},
            )
            raise
        log.info(
            "Condition met: trading switch paused",
            extra={"event": "trading_switch_paused"},
        )

    def resume(self) -> None:
        """Resume trading immediately and clear cooldown timers.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered TradingSwitch.resume",
            extra={"event": "trading_switch_resume_enter"},
        )
        try:
            with self._lock:
                self._enabled = True
                self._resume_at = 0.0
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in TradingSwitch.resume: %s",
                exc,
                extra={"event": "trading_switch_resume_error"},
            )
            raise
        log.info(
            "Condition met: trading switch resumed",
            extra={"event": "trading_switch_resumed"},
        )

    def cooldown(self, seconds: float) -> None:
        """Pause trading for a finite number of seconds.

        Args:
            seconds: Cooldown duration in seconds.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered TradingSwitch.cooldown",
            extra={
                "event": "trading_switch_cooldown_enter",
                "seconds": float(seconds),
            },
        )
        duration = max(float(seconds), 0.0)
        try:
            now = time.time()
            with self._lock:
                self._enabled = True
                self._resume_at = now + duration if duration > 0.0 else 0.0
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in TradingSwitch.cooldown: %s",
                exc,
                extra={"event": "trading_switch_cooldown_error"},
            )
            raise
        if duration <= 0.0:
            log.info(
                "Condition met: trading cooldown skipped",
                extra={"event": "trading_switch_cooldown_skipped"},
            )
            return
        log.info(
            "Condition met: trading cooldown engaged",
            extra={
                "event": "trading_switch_cooldown_engaged",
                "seconds": duration,
            },
        )

    def can_trade(self) -> bool:
        """Return ``True`` when trading is allowed by the switch.

        Args:
            None.

        Returns:
            ``True`` when trading is permitted.

        Raises:
            None.
        """

        log.debug(
            "Entered TradingSwitch.can_trade",
            extra={"event": "trading_switch_can_trade_enter"},
        )
        try:
            with self._lock:
                enabled = self._enabled
                resume_at = self._resume_at
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in TradingSwitch.can_trade: %s",
                exc,
                extra={"event": "trading_switch_can_trade_error"},
            )
            raise
        now = time.time()
        allowed = bool(enabled and (resume_at == 0.0 or now >= resume_at))
        if not allowed:
            remaining = max(resume_at - now, 0.0)
            log.info(
                "Condition met: trading switch blocking",
                extra={
                    "event": "trading_switch_blocking",
                    "remaining": round(remaining, 3),
                    "enabled": enabled,
                    "resume_at": resume_at,
                },
            )
        return allowed

    def remaining(self) -> float:
        """Return the remaining cooldown seconds (0 if none).

        Args:
            None.

        Returns:
            Remaining seconds in cooldown window.

        Raises:
            None.
        """

        log.debug(
            "Entered TradingSwitch.remaining",
            extra={"event": "trading_switch_remaining_enter"},
        )
        try:
            with self._lock:
                resume_at = self._resume_at
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in TradingSwitch.remaining: %s",
                exc,
                extra={"event": "trading_switch_remaining_error"},
            )
            raise
        remaining = max(resume_at - time.time(), 0.0)
        return remaining

    def snapshot(self) -> TradingSwitchState:
        """Return a snapshot describing the current switch state.

        Args:
            None.

        Returns:
            TradingSwitchState describing the switch configuration.

        Raises:
            None.
        """

        log.debug(
            "Entered TradingSwitch.snapshot",
            extra={"event": "trading_switch_snapshot_enter"},
        )
        try:
            with self._lock:
                enabled = self._enabled
                resume_at = self._resume_at
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in TradingSwitch.snapshot: %s",
                exc,
                extra={"event": "trading_switch_snapshot_error"},
            )
            raise
        now = time.time()
        remaining = max(resume_at - now, 0.0)
        allowed = bool(enabled and (resume_at == 0.0 or remaining == 0.0))
        state = TradingSwitchState(
            enabled=enabled,
            resume_at=resume_at,
            remaining_seconds=remaining,
            can_trade=allowed,
        )
        log.info(
            "Condition met: trading switch snapshot",
            extra={
                "event": "trading_switch_snapshot",
                "enabled": enabled,
                "resume_at": resume_at,
                "remaining": round(remaining, 3),
                "can_trade": allowed,
            },
        )
        return state


_SWITCH = TradingSwitch()


def trading_switch() -> TradingSwitch:
    """Return the singleton trading switch instance.

    Args:
        None.

    Returns:
        TradingSwitch: Global operator trading switch.

    Raises:
        None.
    """

    log.debug(
        "Entered trading_switch",
        extra={"event": "trading_switch_singleton_enter"},
    )
    return _SWITCH


__all__ = ["TradingSwitch", "TradingSwitchState", "trading_switch"]
