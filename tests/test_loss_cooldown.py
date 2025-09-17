from __future__ import annotations

from datetime import datetime, timedelta

from src.config import RiskSettings
from src.risk.cooldown import LossCooldownManager


def _ts(minutes: int = 0) -> datetime:
    base = datetime(2024, 7, 1, 9, 30, 0)
    return base + timedelta(minutes=minutes)


def test_cooldown_triggers_at_threshold() -> None:
    cfg = RiskSettings(loss_cooldown_trigger_after_losses=2, loss_cooldown_minutes=30)
    manager = LossCooldownManager(cfg)

    # first loss should not trigger the cool-down yet
    first = manager.register_trade(
        now=_ts(),
        pnl=-150.0,
        streak=1,
        day_loss=150.0,
        max_daily_loss=10_000.0,
    )
    assert first is None

    # second consecutive loss reaches the configured trigger
    second = manager.register_trade(
        now=_ts(),
        pnl=-200.0,
        streak=2,
        day_loss=350.0,
        max_daily_loss=10_000.0,
    )
    assert second is not None
    assert (second - _ts()).total_seconds() / 60 >= cfg.loss_cooldown_minutes


def test_cooldown_backoff_and_relaxation() -> None:
    cfg = RiskSettings(
        loss_cooldown_trigger_after_losses=2,
        loss_cooldown_minutes=20,
        loss_cooldown_backoff=2.0,
        loss_cooldown_relax_multiplier=0.5,
    )
    manager = LossCooldownManager(cfg)

    first = manager.register_trade(
        now=_ts(),
        pnl=-100.0,
        streak=2,
        day_loss=200.0,
        max_daily_loss=10_000.0,
    )
    assert first is not None
    duration_first = (first - _ts()).total_seconds()

    later = _ts(45)
    second = manager.register_trade(
        now=later,
        pnl=-150.0,
        streak=3,
        day_loss=400.0,
        max_daily_loss=10_000.0,
    )
    assert second is not None
    duration_second = (second - later).total_seconds()
    assert duration_second > duration_first

    # a profitable trade should relax the multiplier back towards 1
    manager.register_trade(
        now=_ts(60),
        pnl=500.0,
        streak=0,
        day_loss=0.0,
        max_daily_loss=10_000.0,
    )
    third = manager.register_trade(
        now=_ts(75),
        pnl=-100.0,
        streak=2,
        day_loss=100.0,
        max_daily_loss=10_000.0,
    )
    assert third is not None
    duration_third = (third - _ts(75)).total_seconds()
    assert duration_third < duration_second


def test_drawdown_triggers_even_without_streak() -> None:
    cfg = RiskSettings(
        loss_cooldown_trigger_after_losses=5,
        loss_cooldown_drawdown_pct=0.25,
        loss_cooldown_minutes=25,
    )
    manager = LossCooldownManager(cfg)

    expiry = manager.register_trade(
        now=_ts(),
        pnl=-3_000.0,
        streak=1,
        day_loss=3_000.0,
        max_daily_loss=10_000.0,
    )
    assert expiry is not None
    assert manager.severity >= 1.0
