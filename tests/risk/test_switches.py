from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from nifty_scalper_bot.risk.limits import RiskSwitches


class Clock:
    def __init__(self) -> None:
        self.current = datetime(2024, 1, 1, tzinfo=timezone.utc)

    def advance(self, seconds: float) -> None:
        self.current += timedelta(seconds=seconds)

    def __call__(self) -> datetime:
        return self.current


def test_day_loss_breaker_triggers() -> None:
    clock = Clock()
    switches = RiskSwitches(
        max_day_loss=100.0,
        max_consecutive_losses=5,
        cooldown_minutes=0.0,
        reset_hour_utc=3,
        clock=clock,
    )

    switches.record_pnl(-60.0)
    switches.record_pnl(-50.0)

    assert switches.day_loss() == pytest.approx(110.0)
    assert switches.breach_reason() == "Max day loss reached"


def test_consecutive_losses_with_cooldown() -> None:
    clock = Clock()
    switches = RiskSwitches(
        max_day_loss=0.0,
        max_consecutive_losses=2,
        cooldown_minutes=1.0,
        reset_hour_utc=3,
        clock=clock,
    )

    switches.record_pnl(-10.0)
    switches.record_pnl(-5.0)

    assert switches.breach_reason() == "Max consecutive losses reached"
    assert switches.cooldown_remaining() > 0
    clock.advance(120)
    assert switches.cooldown_remaining() == pytest.approx(0.0)


def test_day_reset_reinitialises_state() -> None:
    clock = Clock()
    switches = RiskSwitches(
        max_day_loss=50.0,
        max_consecutive_losses=1,
        cooldown_minutes=0.0,
        reset_hour_utc=3,
        clock=clock,
    )

    switches.record_pnl(-40.0)
    clock.advance(24 * 3600 + 10)
    switches.record_pnl(0.0)

    assert switches.day_loss() == pytest.approx(0.0)
    assert switches.consecutive_losses() == 0
