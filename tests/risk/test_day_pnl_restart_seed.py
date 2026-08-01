"""Intraday restart must not reset the daily-loss circuit (P1)."""

from __future__ import annotations

from types import MethodType, SimpleNamespace

from nifty_scalper_bot.risk.limits import RiskSwitches
from nifty_scalper_bot.risk.risk_manager import RiskManager


def _manager(realized: float, session_date: str | None, today: str = "2026-08-03"):
    switches = RiskSwitches(
        max_day_loss=1000.0,
        max_consecutive_losses=3,
        cooldown_minutes=0.0,
        reset_hour_utc=3,
    )
    tripped: list[str] = []
    stub = SimpleNamespace(
        _switches=switches,
        _logger=SimpleNamespace(warning=lambda *a, **k: None),
        settings=SimpleNamespace(max_consecutive_losses=3),
        _format_switch_reason=lambda reason: reason,
        _trip_breaker=lambda reason: tripped.append(reason),
        position_manager=SimpleNamespace(
            get_realized_pnl=lambda: realized,
            _pnl_trading_date=session_date,
            _trading_date_ist=lambda: today,
        ),
    )
    stub._seed_day_pnl_from_persisted_state = MethodType(
        RiskManager._seed_day_pnl_from_persisted_state, stub
    )
    return stub, switches, tripped


def test_same_day_loss_is_carried_across_restart() -> None:
    stub, switches, tripped = _manager(-400.0, "2026-08-03")
    stub._seed_day_pnl_from_persisted_state()

    assert switches.day_loss() == 400.0
    assert tripped == []


def test_seeded_loss_beyond_limit_trips_the_breaker() -> None:
    stub, switches, tripped = _manager(-1500.0, "2026-08-03")
    stub._seed_day_pnl_from_persisted_state()

    assert switches.day_loss() == 1500.0
    assert tripped == ["Max day loss reached"]


def test_previous_session_pnl_is_not_seeded() -> None:
    stub, switches, tripped = _manager(-400.0, "2026-08-01")
    stub._seed_day_pnl_from_persisted_state()

    assert switches.day_loss() == 0.0
    assert tripped == []


def test_unknown_session_date_is_not_seeded() -> None:
    stub, switches, _ = _manager(-400.0, None)
    stub._seed_day_pnl_from_persisted_state()

    assert switches.day_loss() == 0.0


def test_flat_pnl_is_a_no_op() -> None:
    stub, switches, _ = _manager(0.0, "2026-08-03")
    stub._seed_day_pnl_from_persisted_state()

    assert switches.day_loss() == 0.0
