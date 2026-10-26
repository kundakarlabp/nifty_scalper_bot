"""Regression tests for a daily loss envelope frozen to pre-entry equity."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nifty_scalper_bot.risk.limits import RiskSwitches
from nifty_scalper_bot.risk.risk_manager import RiskManager


class PositionEvidence:
    def __init__(self) -> None:
        self.entries = 0
        self.positions: list[object] = []
        self.state: dict[str, object] = {}

    def trades_today(self) -> int:
        return self.entries

    def get_open_positions(self) -> list[object]:
        return self.positions

    def _trading_date_ist(self) -> str:
        return "2026-10-26"

    def persist_risk_circuit_state(self, **values: object) -> None:
        self.state.update(values)
        self.state["trading_date"] = self._trading_date_ist()

    def get_risk_circuit_state(self) -> dict[str, object]:
        return dict(self.state)


def _owner(
    positions: PositionEvidence, *, starting_cap: float = 0.0
) -> RiskManager:
    manager = object.__new__(RiskManager)
    manager.position_manager = positions
    manager._switches = RiskSwitches(
        max_day_loss=starting_cap,
        max_consecutive_losses=3,
        cooldown_minutes=0.0,
        reset_hour_utc=3,
    )
    manager._completed_trade_costs_today = 0.0
    manager._last_pnl_snapshot = 0.0
    manager._breaker_tripped = False
    manager._logger = SimpleNamespace(
        info=lambda *_a, **_k: None,
        warning=lambda *_a, **_k: None,
    )
    # Test the broker-funds transition without initializing the full engine.
    manager._configured_daily_loss_cap = lambda funds: float(funds) * 0.05
    return manager


def test_premium_debit_and_release_cannot_rebase_traded_day() -> None:
    evidence = PositionEvidence()
    manager = _owner(evidence)
    RiskManager._sync_daily_cap_from_verified_balance(manager, 15_264.20)
    assert manager._switches.max_day_loss == pytest.approx(763.21)
    assert evidence.state["daily_loss_cap"] == pytest.approx(763.21)

    evidence.entries = 1
    evidence.positions = [object()]
    RiskManager._sync_daily_cap_from_verified_balance(manager, 5_731.95)
    assert manager._switches.max_day_loss == pytest.approx(763.21)

    evidence.positions = []
    RiskManager._sync_daily_cap_from_verified_balance(manager, 15_390.95)
    assert manager._switches.max_day_loss == pytest.approx(763.21)


def test_existing_broker_exposure_prevents_cap_rebase_before_local_trade_count() -> None:
    evidence = PositionEvidence()
    manager = _owner(evidence, starting_cap=763.21)
    evidence.positions = [object()]
    RiskManager._sync_daily_cap_from_verified_balance(manager, 5_731.95)
    assert manager._switches.max_day_loss == pytest.approx(763.21)


def test_same_day_restart_restores_original_cap_not_free_cash_cap() -> None:
    evidence = PositionEvidence()
    evidence.state = {
        "trading_date": evidence._trading_date_ist(),
        "daily_loss_cap": 763.21,
        "completed_trade_costs_today": 69.757154,
        "consecutive_losses": 0,
    }
    restarted = _owner(evidence, starting_cap=286.5975)
    RiskManager._restore_risk_circuit_from_persisted_state(restarted)
    assert restarted._switches.max_day_loss == pytest.approx(763.21)
    assert restarted._completed_trade_costs_today == pytest.approx(69.757154)


def test_previous_trading_day_cap_is_not_reused() -> None:
    evidence = PositionEvidence()
    evidence.state = {
        "trading_date": "2026-10-25",
        "daily_loss_cap": 763.21,
        "completed_trade_costs_today": 69.757154,
    }
    restarted = _owner(evidence, starting_cap=510.0)
    RiskManager._restore_risk_circuit_from_persisted_state(restarted)
    assert restarted._switches.max_day_loss == pytest.approx(510.0)
    assert restarted._completed_trade_costs_today == 0.0
