from __future__ import annotations

import time
from dataclasses import dataclass
from types import MethodType, SimpleNamespace
from typing import Any

import pytest

from nifty_scalper_bot.execution.preflight_validator import (
    PreFlightValidator,
    PreFlightValidatorSettings,
)

FIXED_TIME = 1_700_000_000.0


@dataclass
class DummyPositionManager:
    positions: list[dict[str, Any]]

    def get_all_positions(self) -> list[dict[str, Any]]:  # noqa: D401 - test stub
        return list(self.positions)


@dataclass(slots=True)
class RiskSnapshotStub:
    daily_realized: float
    daily_loss_limit: float
    losses_in_row: int


class DummyRiskManager:
    def __init__(self) -> None:
        self.current_balance = 100_000.0
        self._daily_realized = 0.0
        self._daily_loss_limit = 20_000.0
        self._losses = 0
        self.position_manager = DummyPositionManager(positions=[])
        self.settings = SimpleNamespace(max_concurrent_positions=5)
        self._margin_available = 50_000.0

    def snapshot(self) -> RiskSnapshotStub:
        return RiskSnapshotStub(
            daily_realized=self._daily_realized,
            daily_loss_limit=self._daily_loss_limit,
            losses_in_row=self._losses,
        )

    def get_daily_pnl(self) -> float:
        return self._daily_realized

    def set_daily_pnl(self, value: float) -> None:
        self._daily_realized = value

    def set_daily_loss_limit(self, value: float) -> None:
        self._daily_loss_limit = value

    def get_consecutive_losses(self) -> int:
        return self._losses

    def set_consecutive_losses(self, value: int) -> None:
        self._losses = value

    def get_margin_available(self) -> float:
        return self._margin_available

    def set_margin_available(self, value: float) -> None:
        self._margin_available = value


class DummyRegimeManager:
    def __init__(self) -> None:
        self._regime = "CALM"
        self._confidence = 0.5

    def get_current_regime(self) -> str:
        return self._regime

    def set_regime(self, regime: str) -> None:
        self._regime = regime

    def get_regime_confidence(self) -> float:
        return self._confidence

    def set_confidence(self, value: float) -> None:
        self._confidence = value


class DummyDataHub:
    def __init__(self) -> None:
        self._tick: dict[str, Any] | None = {
            "timestamp": FIXED_TIME,
            "best_bid": 100.0,
            "best_ask": 100.5,
        }

    def set_tick(self, payload: dict[str, Any] | None) -> None:
        self._tick = payload

    def getlatesttick(self, symbol: str) -> dict[str, Any] | None:  # noqa: ARG002
        return None if self._tick is None else dict(self._tick)


class DummySessionGuard:
    def __init__(self) -> None:
        self._allowed = True
        self._reason = ""

    def set_market_open(self, is_open: bool) -> None:
        self._allowed = is_open
        self._reason = "Market closed" if not is_open else ""

    def cantradenow(self) -> tuple[bool, str]:
        return self._allowed, self._reason


@pytest.fixture(autouse=True)
def _freeze_time(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "time", lambda: FIXED_TIME)


@pytest.fixture()
def validator_components() -> dict[str, Any]:
    return {
        "risk": DummyRiskManager(),
        "regime": DummyRegimeManager(),
        "datahub": DummyDataHub(),
        "session": DummySessionGuard(),
    }


def _build_validator(components: dict[str, Any]) -> PreFlightValidator:
    settings = PreFlightValidatorSettings()
    return PreFlightValidator(
        risk_manager=components["risk"],
        regime_manager=components["regime"],
        datahub=components["datahub"],
        session_guard=components["session"],
        settings=settings,
    )


def test_preflight_allows_when_gates_pass(validator_components: dict[str, Any]) -> None:
    validator = _build_validator(validator_components)
    result = validator.validate(
        "NIFTY",
        context={
            "order_cost": 1_000.0,
            "margin_available": 5_000.0,
            "atr_percentile": 50.0,
        },
    )
    assert result.allowed is True
    assert result.reasons == []
    assert result.blocking_level == "NONE"


def test_preflight_uses_validation_cache(validator_components: dict[str, Any]) -> None:
    validator = _build_validator(validator_components)
    call_count = {"market_hours": 0}

    original_gate = validator._check_market_hours

    def wrapped_gate(self: PreFlightValidator, *args: Any, **kwargs: Any) -> Any:
        call_count["market_hours"] += 1
        return original_gate(*args, **kwargs)

    validator._check_market_hours = MethodType(wrapped_gate, validator)

    context = {
        "order_cost": 1_000.0,
        "margin_available": 5_000.0,
        "atr_percentile": 50.0,
    }

    first_result = validator.validate("NIFTY", context=context)
    second_result = validator.validate("NIFTY", context=context)

    assert call_count["market_hours"] == 1
    assert first_result.allowed is True
    assert second_result.allowed is True


def test_preflight_blocks_market_hours_and_stale_quote(
    validator_components: dict[str, Any],
) -> None:
    validator_components["session"].set_market_open(False)
    validator_components["datahub"].set_tick({"timestamp": FIXED_TIME - 10.0})
    validator = _build_validator(validator_components)
    result = validator.validate(
        "NIFTY", context={"order_cost": 100.0, "margin_available": 200.0}
    )
    gates = {reason["gate"] for reason in result.reasons}
    assert "market_hours" in gates
    assert "quote_staleness" in gates
    assert result.blocking_level == "HARD"
    assert result.allowed is False


def test_preflight_detects_drawdown_and_regime(
    validator_components: dict[str, Any],
) -> None:
    risk = validator_components["risk"]
    risk.set_daily_pnl(-20_000.0)
    validator_components["regime"].set_regime("VOLATILE")
    validator_components["regime"].set_confidence(0.95)
    validator = _build_validator(validator_components)
    result = validator.validate(
        "NIFTY",
        context={
            "account_balance": 50_000.0,
            "order_cost": 1_000.0,
            "margin_available": 5_000.0,
            "atr_percentile": 10.0,
        },
    )
    gates = {reason["gate"] for reason in result.reasons}
    assert "daily_loss_limit" in gates
    assert "regime_block" in gates
    assert "atr_range" in gates
    assert result.blocking_level == "HARD"
    assert result.allowed is False


def test_preflight_rate_limit_blocks_after_threshold(
    validator_components: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    validator = _build_validator(validator_components)
    # allow first batch at fixed timestamp
    for _ in range(validator._settings.order_rate_limit_per_sec):
        outcome = validator.validate(
            "NIFTY", context={"order_cost": 1.0, "margin_available": 5.0}
        )
        assert outcome.allowed is True
    # advance slightly but within same window to trigger block
    monkeypatch.setattr(time, "time", lambda: FIXED_TIME + 0.1)
    outcome = validator.validate(
        "NIFTY", context={"order_cost": 1.0, "margin_available": 5.0}
    )
    assert outcome.allowed is False
    gates = {reason["gate"] for reason in outcome.reasons}
    assert "rate_limit" in gates
    assert outcome.blocking_level == "HARD"
