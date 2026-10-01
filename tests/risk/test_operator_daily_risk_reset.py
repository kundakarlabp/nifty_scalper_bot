from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest

from nifty_scalper_bot.risk.limits import RiskSwitches
from nifty_scalper_bot.risk.risk_manager import RiskManager


def test_rebase_day_pnl_preserves_streak_and_cooldown() -> None:
    switches = RiskSwitches(
        max_day_loss=500.0,
        max_consecutive_losses=3,
        cooldown_minutes=5.0,
        reset_hour_utc=3,
    )
    switches.record_pnl(-450.0)
    switches.record_trade_result(-100.0)

    switches.rebase_day_pnl()

    assert switches.day_loss() == 0.0
    assert switches.consecutive_losses() == 1
    assert switches.cooldown_remaining() > 0.0


def test_operator_reset_preserves_accounting_and_persists_baseline() -> None:
    persisted: dict[str, object] = {}
    manager = SimpleNamespace(
        get_realized_pnl=lambda: -400.0,
        persist_risk_circuit_state=lambda **values: persisted.update(values),
    )
    switches = RiskSwitches(
        max_day_loss=500.0,
        max_consecutive_losses=3,
        cooldown_minutes=0.0,
        reset_hour_utc=3,
    )
    switches.record_pnl(-464.0)
    switches.record_trade_result(-400.0)
    owner = SimpleNamespace(
        position_manager=manager,
        _switches=switches,
        _completed_trade_costs_today=64.0,
        _operator_reset_realized_baseline=0.0,
        _operator_reset_cost_baseline=0.0,
        _last_pnl_snapshot=-400.0,
        _breaker_tripped=True,
        _breaker_reason="Max day loss reached",
        _breaker_alerted=True,
        _shadow_forced=True,
        _last_rejection="DAY_LOSS_LIMIT",
        _logger=SimpleNamespace(warning=lambda *args, **kwargs: None),
        _format_switch_reason=lambda reason: reason,
    )
    owner._persist_risk_circuit_state = MethodType(
        RiskManager._persist_risk_circuit_state, owner
    )

    result = RiskManager.operator_reset_daily_loss_budget(owner)

    assert result["prior_day_loss"] == pytest.approx(464.0)
    assert result["new_day_loss"] == 0.0
    assert switches.consecutive_losses() == 1
    assert owner._breaker_tripped is False
    assert owner._operator_reset_realized_baseline == pytest.approx(-400.0)
    assert owner._operator_reset_cost_baseline == pytest.approx(64.0)
    assert persisted["operator_reset_realized_baseline"] == pytest.approx(-400.0)
    assert persisted["operator_reset_cost_baseline"] == pytest.approx(64.0)
    assert persisted["consecutive_losses"] == 1


def test_restart_seeds_only_loss_after_operator_baseline() -> None:
    today = "2026-10-01"
    manager = SimpleNamespace(
        get_realized_pnl=lambda: -500.0,
        _pnl_trading_date=today,
        _trading_date_ist=lambda: today,
    )
    switches = RiskSwitches(
        max_day_loss=500.0,
        max_consecutive_losses=3,
        cooldown_minutes=0.0,
        reset_hour_utc=3,
    )
    owner = SimpleNamespace(
        position_manager=manager,
        _switches=switches,
        _operator_reset_realized_baseline=-400.0,
        _last_pnl_snapshot=-500.0,
        _logger=SimpleNamespace(warning=lambda *args, **kwargs: None),
        _record_realized_pnl_metrics=lambda *args, **kwargs: None,
        _trip_on_switch_breach=lambda: None,
        _trip_breaker=lambda reason: None,
        _format_switch_reason=lambda reason: reason,
    )

    RiskManager._seed_day_pnl_from_persisted_state(owner)

    assert switches.day_loss() == pytest.approx(100.0)

def test_missing_operator_baseline_keeps_legacy_seed_behavior() -> None:
    today = "2026-10-01"
    manager = SimpleNamespace(
        get_realized_pnl=lambda: -400.0,
        _pnl_trading_date=today,
        _trading_date_ist=lambda: today,
    )
    switches = RiskSwitches(
        max_day_loss=500.0,
        max_consecutive_losses=3,
        cooldown_minutes=0.0,
        reset_hour_utc=3,
    )
    owner = SimpleNamespace(
        position_manager=manager,
        _switches=switches,
        _last_pnl_snapshot=-400.0,
        _logger=SimpleNamespace(warning=lambda *args, **kwargs: None),
        _record_realized_pnl_metrics=lambda *args, **kwargs: None,
        _trip_on_switch_breach=lambda: None,
        _trip_breaker=lambda reason: None,
        _format_switch_reason=lambda reason: reason,
    )

    RiskManager._seed_day_pnl_from_persisted_state(owner)

    assert switches.day_loss() == pytest.approx(400.0)
