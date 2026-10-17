from types import SimpleNamespace

import pytest

from nifty_scalper_bot.core.app import (
    _resolve_startup_risk_initial_balance,
    apply_broker_auth_failure_to_context,
    apply_broker_auth_recovery_to_context,
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


def test_auth_recovery_clears_only_auth_blockers() -> None:
    ctx = SimpleNamespace(
        broker_auth_invalid=True,
        broker_auth_error="expired",
        broker_auth_invalid_at=object(),
        broker_session_invalid=True,
        broker_ready=False,
        broker_balance_valid=False,
        broker_balance_error="expired",
        live_orders_armed=False,
        execution_armed=False,
        trading_ready=False,
        live_block_reason="execution_not_armed:broker_auth_invalid",
        execution_block_reason="broker_auth_invalid",
        runtime_readiness_recomputed_mono=123.0,
    )

    apply_broker_auth_recovery_to_context(ctx, {"valid": True, "generation": 8})

    assert ctx.broker_auth_invalid is False
    assert ctx.broker_auth_error is None
    assert ctx.broker_session_invalid is False
    assert ctx.broker_ready is True
    assert ctx.broker_balance_valid is False
    assert ctx.live_orders_armed is False
    assert ctx.execution_armed is False
    assert ctx.trading_ready is False
    assert ctx.live_block_reason is None
    assert ctx.execution_block_reason is None
    assert ctx.runtime_readiness_recomputed_mono == 0.0
