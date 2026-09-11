from __future__ import annotations

from types import SimpleNamespace

import pytest

from nifty_scalper_bot.risk.broker_pnl_risk_patch import (
    _patched_refresh_realized_pnl,
    _patched_seed_day_pnl_from_persisted_state,
    _resolve_broker_realized_pnl,
)
from nifty_scalper_bot.risk.limits import RiskSwitches


def _owner(broker_value: float):
    state = {"value": broker_value}
    position_manager = SimpleNamespace(
        refresh_broker_pnl_diagnostic=lambda force=False: {
            "account_realized": state["value"]
        },
        get_broker_account_realized_pnl=lambda force=False: state["value"],
    )
    switches = RiskSwitches(
        max_day_loss=1000.0,
        max_consecutive_losses=3,
        cooldown_minutes=0.0,
        reset_hour_utc=3,
    )
    tripped: list[str] = []
    logger = SimpleNamespace(
        info=lambda *_a, **_k: None,
        warning=lambda *_a, **_k: None,
        exception=lambda *_a, **_k: None,
    )
    owner = SimpleNamespace(
        position_manager=position_manager,
        _switches=switches,
        _last_pnl_snapshot=0.0,
        _logger=logger,
        _format_switch_reason=lambda reason: str(reason),
        _trip_breaker=lambda reason: tripped.append(str(reason)),
    )
    return owner, state, switches, tripped


def test_resolve_broker_realized_prefers_dedicated_account_source() -> None:
    manager = SimpleNamespace(
        refresh_broker_pnl_diagnostic=lambda force=False: None,
        get_broker_account_realized_pnl=lambda force=False: -321.5,
    )

    assert _resolve_broker_realized_pnl(manager, force=True) == pytest.approx(-321.5)


def test_daily_risk_seed_uses_broker_account_realized() -> None:
    owner, _state, switches, tripped = _owner(-400.0)

    _patched_seed_day_pnl_from_persisted_state(owner)

    assert owner._last_pnl_snapshot == pytest.approx(-400.0)
    assert switches.day_loss() == pytest.approx(400.0)
    assert tripped == []


def test_daily_risk_refresh_tracks_broker_delta_not_local_ledger() -> None:
    owner, state, switches, tripped = _owner(-400.0)
    _patched_seed_day_pnl_from_persisted_state(owner)
    state["value"] = -550.0

    _patched_refresh_realized_pnl(owner)

    assert owner._last_pnl_snapshot == pytest.approx(-550.0)
    assert switches.day_loss() == pytest.approx(550.0)
    assert tripped == []


def test_broker_daily_loss_still_trips_real_risk_circuit() -> None:
    owner, _state, switches, tripped = _owner(-1200.0)

    _patched_seed_day_pnl_from_persisted_state(owner)

    assert switches.day_loss() == pytest.approx(1200.0)
    assert tripped == ["Max day loss reached"]
