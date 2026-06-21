from types import SimpleNamespace

import pytest

from nifty_scalper_bot.core.app import (
    _resolve_startup_risk_initial_balance,
    apply_broker_auth_failure_to_context,
)
from nifty_scalper_bot.utils.errors import BrokerBalanceUnavailableError


def test_live_risk_initial_balance_uses_validated_broker_balance(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    settings = SimpleNamespace(execution_mode="LIVE")
    config = SimpleNamespace(initial_balance=1_000_000.0)

    balance = _resolve_startup_risk_initial_balance(
        settings=settings,
        config=config,
        startup_available_balance=16_436.10,
    )

    assert balance == pytest.approx(16_436.10)


def test_live_risk_initial_balance_requires_broker_balance(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    settings = SimpleNamespace(execution_mode="LIVE")
    config = SimpleNamespace(initial_balance=1_000_000.0)

    with pytest.raises(BrokerBalanceUnavailableError):
        _resolve_startup_risk_initial_balance(
            settings=settings,
            config=config,
            startup_available_balance=None,
        )


def test_paper_risk_initial_balance_uses_configured_capital(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "PAPER")
    settings = SimpleNamespace(execution_mode="PAPER")
    config = SimpleNamespace(initial_balance=1_000_000.0)

    balance = _resolve_startup_risk_initial_balance(
        settings=settings,
        config=config,
        startup_available_balance=None,
    )

    assert balance == pytest.approx(1_000_000.0)


def test_auth_failure_callback_marks_context_fail_closed():
    ctx = SimpleNamespace(
        broker_auth_invalid=False,
        broker_auth_error=None,
        broker_auth_invalid_at=None,
        broker_ready=True,
        broker_balance_valid=True,
        broker_balance_error=None,
        live_orders_armed=True,
        execution_armed=True,
        trading_ready=True,
        live_block_reason=None,
        execution_block_reason=None,
    )

    apply_broker_auth_failure_to_context(
        ctx,
        {"reason": "Incorrect api_key or access_token", "generation": 7},
    )

    assert ctx.broker_auth_invalid is True
    assert ctx.broker_auth_error == "Incorrect api_key or access_token"
    assert ctx.broker_ready is False
    assert ctx.broker_balance_valid is False
    assert ctx.live_orders_armed is False
    assert ctx.execution_armed is False
    assert ctx.trading_ready is False
    assert ctx.live_block_reason == "broker_auth_invalid"
    assert ctx.execution_block_reason == "broker_auth_invalid"

@pytest.mark.parametrize("bad_balance", [float("nan"), float("inf"), -1.0])
def test_live_risk_initial_balance_rejects_invalid_numbers(monkeypatch, bad_balance):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    settings = SimpleNamespace(execution_mode="LIVE")
    config = SimpleNamespace(initial_balance=1_000_000.0)

    with pytest.raises(BrokerBalanceUnavailableError):
        _resolve_startup_risk_initial_balance(
            settings=settings,
            config=config,
            startup_available_balance=bad_balance,
        )


def test_live_risk_initial_balance_accepts_zero(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    settings = SimpleNamespace(execution_mode="LIVE")
    config = SimpleNamespace(initial_balance=1_000_000.0)
    assert _resolve_startup_risk_initial_balance(
        settings=settings,
        config=config,
        startup_available_balance=0.0,
    ) == 0.0
