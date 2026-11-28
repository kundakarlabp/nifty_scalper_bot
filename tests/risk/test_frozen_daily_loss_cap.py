"""Regression tests for a daily loss envelope frozen to pre-entry equity."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nifty_scalper_bot.config.settings import RiskSettings
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


def _owner(positions: PositionEvidence, *, starting_cap: float = 0.0) -> RiskManager:
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
    manager.settings = RiskSettings(daily_loss_pct=5.0, per_trade_risk_pct=5.0)
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


def test_broker_exposure_freezes_cap_without_local_trade_count() -> None:
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


def test_explicit_rupee_ceiling_limits_five_percent_of_growing_balance() -> None:
    """An absolute operator ceiling wins over 5% of a larger cash balance."""
    evidence = PositionEvidence()
    manager = _owner(evidence)
    manager.settings.max_daily_loss_absolute = 773.0

    assert manager._configured_daily_loss_cap(15_699.0) == pytest.approx(773.0)
    RiskManager._sync_daily_cap_from_verified_balance(manager, 15_699.0)
    assert manager._switches.max_day_loss == pytest.approx(773.0)


def test_traded_day_accepts_stricter_absolute_ceiling_without_cash_rebase() -> None:
    """Premium-paid cash must not shrink a frozen cap below the INR ceiling."""
    evidence = PositionEvidence()
    evidence.entries = 1
    evidence.positions = [object()]
    manager = _owner(evidence, starting_cap=784.95)
    manager._completed_trade_costs_today = 71.506732
    manager.settings.max_daily_loss_absolute = 773.0

    RiskManager._sync_daily_cap_from_verified_balance(manager, 5_000.0)
    assert manager._switches.max_day_loss == pytest.approx(773.0)
    assert evidence.state["daily_loss_cap"] == pytest.approx(773.0)

    # Subsequent premium debit, cash release or balance increase cannot
    # rebase the already-frozen envelope.
    evidence.positions = []
    RiskManager._sync_daily_cap_from_verified_balance(manager, 2_000.0)
    RiskManager._sync_daily_cap_from_verified_balance(manager, 25_000.0)
    assert manager._switches.max_day_loss == pytest.approx(773.0)


def test_traded_day_never_increases_cap_when_absolute_override_is_higher() -> None:
    evidence = PositionEvidence()
    evidence.entries = 1
    manager = _owner(evidence, starting_cap=784.95)
    manager._completed_trade_costs_today = 70.0
    manager.settings.max_daily_loss_absolute = 900.0

    RiskManager._sync_daily_cap_from_verified_balance(manager, 25_000.0)
    assert manager._switches.max_day_loss == pytest.approx(784.95)
    assert evidence.state == {}


def test_traded_day_never_rebases_from_premium_without_absolute_ceiling() -> None:
    evidence = PositionEvidence()
    evidence.positions = [object()]
    manager = _owner(evidence, starting_cap=784.95)
    manager.settings.max_daily_loss_absolute = 0.0

    RiskManager._sync_daily_cap_from_verified_balance(manager, 4_000.0)
    assert manager._switches.max_day_loss == pytest.approx(784.95)
    assert evidence.state == {}


def test_already_tripped_circuit_can_only_be_tightened() -> None:
    evidence = PositionEvidence()
    manager = _owner(evidence, starting_cap=784.95)
    manager._breaker_tripped = True
    manager.settings.max_daily_loss_absolute = 773.0

    RiskManager._sync_daily_cap_from_verified_balance(manager, 15_699.0)
    assert manager._breaker_tripped is True
    assert manager._switches.max_day_loss == pytest.approx(773.0)
