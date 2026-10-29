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


def test_operator_risk_reset_is_at_most_once_per_trading_day() -> None:
    today = "2026-10-29"
    persisted: dict[str, object] = {}
    manager = SimpleNamespace(
        get_realized_pnl=lambda: -450.0,
        _trading_date_ist=lambda: today,
        get_risk_circuit_state=lambda: dict(persisted),
        persist_risk_circuit_state=lambda **updates: persisted.update(
            updates, trading_date=today
        ),
    )
    switches = RiskSwitches(
        max_day_loss=700.0,
        max_consecutive_losses=3,
        cooldown_minutes=1.0,
        reset_hour_utc=3,
    )
    switches.record_pnl(-450.0)
    switches.record_trade_result(-400.0)
    owner = SimpleNamespace(
        position_manager=manager,
        _switches=switches,
        _completed_trade_costs_today=35.0,
        _logger=SimpleNamespace(warning=lambda *args, **kwargs: None),
        _format_switch_reason=lambda reason: reason,
    )
    first = RiskManager.operator_reset_daily_loss_budget(owner)
    assert first["prior_day_loss"] == pytest.approx(450.0)
    assert switches.day_loss() == 0.0
    assert persisted["operator_reset_trading_date"] == today
    assert persisted["operator_reset_realized_baseline"] == -450.0
    assert switches.consecutive_losses() == 1
    assert switches.cooldown_remaining() > 0
    with pytest.raises(ValueError, match="daily_risk_reset_already_applied"):
        RiskManager.operator_reset_daily_loss_budget(owner)
    assert switches.day_loss() == 0.0


def test_failed_durable_write_does_not_clear_loss_allowance() -> None:
    switches = RiskSwitches(
        max_day_loss=700.0,
        max_consecutive_losses=3,
        cooldown_minutes=0.0,
        reset_hour_utc=3,
    )
    switches.record_pnl(-650.0)

    def fail_to_persist(**_updates):
        raise OSError("disk write unavailable")

    owner = SimpleNamespace(
        position_manager=SimpleNamespace(
            get_realized_pnl=lambda: -650.0,
            persist_risk_circuit_state=fail_to_persist,
        ),
        _switches=switches,
        _completed_trade_costs_today=0.0,
    )
    with pytest.raises(OSError, match="disk write unavailable"):
        RiskManager.operator_reset_daily_loss_budget(owner)
    assert switches.day_loss() == pytest.approx(650.0)


def test_restoring_stale_reset_baseline_cannot_erase_next_day_loss() -> None:
    today = "2026-10-29"
    owner = SimpleNamespace(
        position_manager=SimpleNamespace(
            _trading_date_ist=lambda: today,
            get_risk_circuit_state=lambda: {
                "trading_date": "2026-10-28",
                "operator_reset_trading_date": "2026-10-28",
                "operator_reset_realized_baseline": -800.0,
                "operator_reset_cost_baseline": 70.0,
            },
        ),
        _operator_reset_realized_baseline=0.0,
        _operator_reset_cost_baseline=0.0,
        _operator_reset_trading_date="",
    )
    RiskManager._restore_operator_reset_baseline(owner)
    assert owner._operator_reset_realized_baseline == 0.0
    assert owner._operator_reset_trading_date == ""


def test_operator_rebase_never_clears_an_unrelated_breaker() -> None:
    switches = RiskSwitches(
        max_day_loss=800.0,
        max_consecutive_losses=3,
        cooldown_minutes=0.0,
        reset_hour_utc=3,
    )
    switches.record_pnl(-100.0)
    owner = SimpleNamespace(
        position_manager=SimpleNamespace(
            get_realized_pnl=lambda: -100.0,
            persist_risk_circuit_state=lambda **_values: None,
        ),
        _switches=switches,
        _completed_trade_costs_today=0.0,
        _breaker_tripped=True,
        _breaker_reason="broker_position_reconciliation_failed",
        _shadow_forced=True,
        _last_rejection="broker_position_reconciliation_failed",
        _logger=SimpleNamespace(warning=lambda *args, **kwargs: None),
        _format_switch_reason=lambda reason: reason,
    )
    RiskManager.operator_reset_daily_loss_budget(owner)
    assert owner._breaker_tripped is True
    assert owner._breaker_reason == "broker_position_reconciliation_failed"
    assert owner._shadow_forced is True
    assert owner._last_rejection == "broker_position_reconciliation_failed"
