"""Purpose: Coordinate operator entry controls and release-safe submission.

Responsibilities:
- TradingSwitch tracks pause/cooldown and entry-admission state.
- release_entry_hold_active reports the deployment's entry hold.
- live_entry_submission_slot synchronizes new broker entries with release handoff.

Boundaries: Entry control is not broker order placement or position truth.
Safety: Release holds block opening exposure, not protective exit orders.
"""

from __future__ import annotations

import fcntl
import os
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from nifty_scalper_bot.utils.logging import get_logger

log = get_logger(__name__)


def _release_control_path(name: str) -> Path:
    root = os.getenv("NIFTYBOT_CONFIG_DIR", "/home/ubuntu/.config/niftybot")
    return Path(root) / name


def release_entry_hold_active() -> bool:
    """Read the deployment entry hold; protective exits are never affected."""
    return _release_control_path("auto_release_entry_hold").is_file()


@contextmanager
def live_entry_submission_slot(payload: Mapping[str, object]) -> Iterator[None]:
    """Serialize LIVE entry submission with a deployment's broker-flat handoff.

    Every live opening order takes a shared advisory lock across its entire
    broker API call. The release runner exclusively holds the same lock from
    before the no-exposure snapshots through engine restart. Exit SELL orders
    do not take this lock and remain available for protective execution.
    """
    side = str(payload.get("side") or payload.get("transaction_type") or "").upper()
    live_mode = os.getenv("EXECUTION_MODE", "SHADOW").strip().upper() == "LIVE"
    if not live_mode or side == "SELL":
        yield
        return
    lock_path = _release_control_path("auto_release_entry_lock")
    descriptor = os.open(
        lock_path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH)
        if release_entry_hold_active():
            raise RuntimeError("release_entry_hold_blocks_new_entry")
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


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
        self._enabled = False
        self._resume_at = 0.0
        self._operator_paused = False

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
                self._operator_paused = True
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
                self._operator_paused = False
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

    def arm_for_runtime(self) -> bool:
        """Enable a pristine switch after canonical LIVE readiness passes.

        Runtime arming never overrides an explicit operator/emergency pause or
        an active cooldown. It synchronizes the readiness SSOT with the final
        OrderManager entry switch.

        Returns:
            ``True`` when the switch is ready for new entries.
        """

        now = time.time()
        changed = False
        try:
            with self._lock:
                if release_entry_hold_active():
                    return False
                if self._operator_paused:
                    return False
                if self._resume_at > now:
                    return False
                if self._resume_at > 0.0:
                    self._resume_at = 0.0
                if not self._enabled:
                    self._enabled = True
                    changed = True
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in TradingSwitch.arm_for_runtime: %s",
                exc,
                extra={"event": "trading_switch_runtime_arm_error"},
            )
            raise
        if changed:
            log.info(
                "Condition met: trading switch runtime armed",
                extra={"event": "trading_switch_runtime_armed"},
            )
        return True

    def disarm_for_runtime(self) -> None:
        """Disable runtime entries without creating an operator pause latch.

        Session/readiness transitions use this path so the next valid market
        session can re-arm normally. Existing operator-pause and cooldown state
        remains authoritative and is not cleared.
        """

        changed = False
        try:
            with self._lock:
                if self._enabled:
                    self._enabled = False
                    changed = True
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in TradingSwitch.disarm_for_runtime: %s",
                exc,
                extra={"event": "trading_switch_runtime_disarm_error"},
            )
            raise
        if changed:
            log.info(
                "Condition met: trading switch runtime disarmed",
                extra={"event": "trading_switch_runtime_disarmed"},
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
                operator_paused = self._operator_paused
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in TradingSwitch.can_trade: %s",
                exc,
                extra={"event": "trading_switch_can_trade_error"},
            )
            raise
        now = time.time()
        allowed = bool(
            enabled
            and not operator_paused
            and not release_entry_hold_active()
            and (resume_at == 0.0 or now >= resume_at)
        )
        if not allowed:
            remaining = max(resume_at - now, 0.0)
            log.info(
                "Condition met: trading switch blocking",
                extra={
                    "event": "trading_switch_blocking",
                    "remaining": round(remaining, 3),
                    "enabled": enabled,
                    "operator_paused": operator_paused,
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
                operator_paused = self._operator_paused
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in TradingSwitch.snapshot: %s",
                exc,
                extra={"event": "trading_switch_snapshot_error"},
            )
            raise
        now = time.time()
        remaining = max(resume_at - now, 0.0)
        allowed = bool(
            enabled
            and not operator_paused
            and not release_entry_hold_active()
            and (resume_at == 0.0 or remaining == 0.0)
        )
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
                "operator_paused": operator_paused,
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
