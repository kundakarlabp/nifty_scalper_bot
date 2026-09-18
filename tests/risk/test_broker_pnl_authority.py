from __future__ import annotations

from types import SimpleNamespace

import pytest

from nifty_scalper_bot.risk.limits import RiskSwitches
from nifty_scalper_bot.risk.risk_manager import (
    RiskManager,
    _resolve_broker_realized_pnl,
)


def _owner(monkeypatch: pytest.MonkeyPatch, broker_value: float):
    state = {"value": broker_value}
    position_manager = SimpleNamespace(
        refresh_broker_pnl_diagnostic=lambda force=False: {
            "account_realized": state["value"],
            "strategy_realized": state["value"],
            "difference": 0.0,
            "status": "matched",
            "diagnostic_only": True,
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
    owner = object.__new__(RiskManager)
    owner.position_manager = position_manager
    owner._switches = switches
    owner._last_pnl_snapshot = 0.0
    owner._logger = logger
    owner._soft_override = False
    monkeypatch.setattr(
        RiskManager,
        "_trip_breaker",
        lambda _self, reason: tripped.append(str(reason)),
    )
    return owner, state, switches, tripped


def test_resolve_broker_realized_prefers_dedicated_account_source() -> None:
    manager = SimpleNamespace(
        refresh_broker_pnl_diagnostic=lambda force=False: None,
        get_broker_account_realized_pnl=lambda force=False: -321.5,
    )

    assert _resolve_broker_realized_pnl(manager, force=True) == pytest.approx(-321.5)


def test_daily_risk_seed_uses_broker_account_realized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, _state, switches, tripped = _owner(monkeypatch, -400.0)

    RiskManager._seed_day_pnl_from_persisted_state(owner)

    assert owner._last_pnl_snapshot == pytest.approx(-400.0)
    assert switches.day_loss() == pytest.approx(400.0)
    assert tripped == []


def test_daily_risk_refresh_tracks_broker_delta_not_local_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, state, switches, tripped = _owner(monkeypatch, -400.0)
    RiskManager._seed_day_pnl_from_persisted_state(owner)
    state["value"] = -550.0

    RiskManager._refresh_realized_pnl(owner)

    assert owner._last_pnl_snapshot == pytest.approx(-550.0)
    assert switches.day_loss() == pytest.approx(550.0)
    assert tripped == []


def test_daily_risk_refresh_falls_back_to_local_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, _state, switches, tripped = _owner(monkeypatch, -400.0)
    owner.position_manager.get_broker_account_realized_pnl = lambda force=False: None
    owner.position_manager.get_realized_pnl = lambda: -450.0
    owner._last_pnl_snapshot = -400.0

    RiskManager._refresh_realized_pnl(owner)

    assert owner._last_pnl_snapshot == pytest.approx(-450.0)
    assert switches.day_loss() == pytest.approx(50.0)
    assert tripped == []


def test_broker_daily_loss_still_trips_real_risk_circuit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, _state, switches, tripped = _owner(monkeypatch, -1200.0)

    RiskManager._seed_day_pnl_from_persisted_state(owner)

    assert switches.day_loss() == pytest.approx(1200.0)
    assert tripped == ["Daily loss limit reached (1200.00/1000.00)"]


def test_broker_pnl_authority_is_native_not_import_time_patch() -> None:
    assert not hasattr(RiskManager, "_broker_pnl_risk_patch")


def test_resolve_broker_realized_rejects_explicit_diagnostic_mismatch() -> None:
    manager = SimpleNamespace(
        refresh_broker_pnl_diagnostic=lambda force=False: {
            "account_realized": 0.0,
            "strategy_realized": -507.0,
            "difference": 507.0,
            "status": "mismatch",
            "diagnostic_only": True,
        },
        get_broker_account_realized_pnl=lambda force=False: 0.0,
    )

    assert _resolve_broker_realized_pnl(manager, force=True) is None


def test_restart_mismatch_preserves_strategy_loss_plus_persisted_costs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    switches = RiskSwitches(
        max_day_loss=712.82,
        max_consecutive_losses=3,
        cooldown_minutes=0.0,
        reset_hour_utc=3,
    )
    tripped: list[str] = []
    position_manager = SimpleNamespace(
        refresh_broker_pnl_diagnostic=lambda force=False: {
            "account_realized": 0.0,
            "strategy_realized": -507.0,
            "difference": 507.0,
            "status": "mismatch",
            "diagnostic_only": True,
        },
        get_broker_account_realized_pnl=lambda force=False: 0.0,
        get_realized_pnl=lambda: -507.0,
        _pnl_trading_date="2026-09-18",
        _trading_date_ist=lambda: "2026-09-18",
        get_risk_circuit_state=lambda: {
            "trading_date": "2026-09-18",
            "completed_trade_costs_today": 444.75,
            "consecutive_losses": 2,
        },
    )
    owner = object.__new__(RiskManager)
    owner.position_manager = position_manager
    owner._switches = switches
    owner._last_pnl_snapshot = -507.0
    owner._completed_trade_costs_today = 0.0
    owner._logger = SimpleNamespace(
        info=lambda *_a, **_k: None,
        warning=lambda *_a, **_k: None,
        exception=lambda *_a, **_k: None,
    )
    owner._soft_override = False
    monkeypatch.setattr(
        RiskManager,
        "_trip_breaker",
        lambda _self, reason: tripped.append(str(reason)),
    )

    RiskManager._seed_day_pnl_from_persisted_state(owner)
    assert switches.day_loss() == pytest.approx(507.0)

    RiskManager._restore_risk_circuit_from_persisted_state(owner)

    assert switches.day_loss() == pytest.approx(951.75)
    assert owner._completed_trade_costs_today == pytest.approx(444.75)
    assert switches.consecutive_losses() == 2
    assert tripped
