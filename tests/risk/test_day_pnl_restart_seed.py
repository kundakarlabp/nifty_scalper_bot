"""Intraday restart must not reset the daily-loss circuit (P1)."""

from __future__ import annotations

from types import MethodType, SimpleNamespace

from nifty_scalper_bot.config.settings import RiskSettings
from nifty_scalper_bot.risk.limits import RiskSwitches
from nifty_scalper_bot.risk.risk_manager import RiskManager


def _manager(
    realized: float,
    session_date: str | None,
    today: str = "2026-08-03",
    circuit_date: str | None = None,
):
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
            get_risk_circuit_state=lambda: (
                {"trading_date": circuit_date} if circuit_date else {}
            ),
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


def test_same_day_risk_circuit_recovers_missing_pnl_session_date() -> None:
    stub, switches, tripped = _manager(
        -400.0,
        None,
        circuit_date="2026-08-03",
    )

    stub._seed_day_pnl_from_persisted_state()

    assert switches.day_loss() == 400.0
    assert tripped == []


def test_previous_risk_circuit_date_does_not_recover_missing_session_date() -> None:
    stub, switches, _ = _manager(-400.0, None, circuit_date="2026-08-01")

    stub._seed_day_pnl_from_persisted_state()

    assert switches.day_loss() == 0.0


def test_flat_pnl_is_a_no_op() -> None:
    stub, switches, _ = _manager(0.0, "2026-08-03")
    stub._seed_day_pnl_from_persisted_state()

    assert switches.day_loss() == 0.0


def _circuit_stub(state: dict):
    switches = RiskSwitches(
        max_day_loss=1000.0,
        max_consecutive_losses=3,
        cooldown_minutes=5.0,
        reset_hour_utc=3,
    )
    tripped: list[str] = []
    saved: list[dict] = []
    stub = SimpleNamespace(
        _switches=switches,
        _completed_trade_costs_today=0.0,
        _logger=SimpleNamespace(warning=lambda *a, **k: None),
        _format_switch_reason=lambda reason: reason,
        _trip_breaker=lambda reason: tripped.append(reason),
        position_manager=SimpleNamespace(
            get_risk_circuit_state=lambda: dict(state),
            persist_risk_circuit_state=lambda **kw: saved.append(kw),
        ),
    )
    for name in (
        "_restore_risk_circuit_from_persisted_state",
        "_persist_risk_circuit_state",
        "record_completed_trade",
    ):
        setattr(stub, name, MethodType(getattr(RiskManager, name), stub))
    return stub, switches, tripped, saved


def test_transaction_costs_survive_restart() -> None:
    stub, switches, tripped, _ = _circuit_stub(
        {"completed_trade_costs_today": 240.0, "consecutive_losses": 0}
    )
    stub._restore_risk_circuit_from_persisted_state()

    assert switches.day_loss() == 240.0
    assert stub._completed_trade_costs_today == 240.0
    assert tripped == []


def test_loss_streak_and_cooldown_survive_restart() -> None:
    import time

    stub, switches, tripped, _ = _circuit_stub(
        {
            "consecutive_losses": 3,
            "loss_cooldown_until_epoch": time.time() + 120.0,
        }
    )
    stub._restore_risk_circuit_from_persisted_state()

    assert switches.consecutive_losses() == 3
    assert switches.cooldown_remaining() > 0.0
    assert tripped == ["Max consecutive losses reached"]


def test_completed_trade_persists_the_circuit_state() -> None:
    stub, _switches_, _tripped, saved = _circuit_stub({})
    stub.record_completed_trade(-500.0, estimated_costs=60.0)

    assert saved and saved[-1]["completed_trade_costs_today"] == 60.0
    assert saved[-1]["consecutive_losses"] == 1
    assert saved[-1]["loss_cooldown_until_epoch"] > 0.0


def test_constructor_restores_breached_loss_streak_without_startup_failure() -> None:
    position_manager = SimpleNamespace(
        get_realized_pnl=lambda: 0.0,
        get_risk_circuit_state=lambda: {"consecutive_losses": 3},
    )

    risk = RiskManager(
        settings=RiskSettings(max_consecutive_losses=3),
        position_manager=position_manager,
        account_balance=50_000.0,
    )

    assert risk.is_circuit_breaker_tripped() == (
        True,
        "Consecutive loss limit reached (3/3)",
    )
