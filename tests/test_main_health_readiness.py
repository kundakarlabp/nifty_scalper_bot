import json
from types import SimpleNamespace

from nifty_scalper_bot import main


def _json(response):
    return json.loads(response.body.decode())


def _ctx(**overrides):
    decision = SimpleNamespace(
        blocker_list=tuple(overrides.pop("blockers", ())),
        primary_blocker=overrides.pop("primary_blocker", None),
        live_orders_armed=overrides.pop("live_orders_armed", False),
        execution_ready=overrides.pop("execution_ready", False),
    )
    base = dict(
        readiness_decision=decision,
        broker_ready=False,
        broker_auth_invalid=False,
        broker_balance_valid=False,
        last_valid_broker_balance=None,
        broker_balance_error=None,
        position_reconciliation_started=False,
        position_reconciliation_completed=False,
        position_reconciliation_failed=False,
        position_reconciliation_error=None,
        unprotected_broker_positions=set(),
        live_orders_armed=False,
        market_state="closed",
        live_block_reason=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def setup_function():
    main.app.state.bot = None
    main.app.state.bot_error = None
    main.app.state.bot_started = False
    main.app.state.bot_starting = False


def test_readyz_returns_503_until_startup_completed():
    response = main.readyz()
    body = _json(response)

    assert response.status_code == 503
    assert body["ready"] is False
    assert body["primary_blocker"] == "startup_incomplete"


def test_readyz_uses_context_safety_blockers():
    ctx = _ctx(
        blockers=("broker_auth_invalid",),
        primary_blocker="broker_auth_invalid",
        broker_auth_invalid=True,
        broker_balance_valid=False,
        position_reconciliation_completed=True,
    )
    main.app.state.bot_started = True
    main.app.state.bot = SimpleNamespace(_ctx=ctx)

    response = main.readyz()
    body = _json(response)

    assert response.status_code == 503
    assert body["ready"] is False
    assert body["primary_blocker"] == "broker_auth_invalid"
    assert "broker_auth_invalid" in body["blockers"]


def test_health_trading_exposes_canonical_context():
    ctx = _ctx(
        blockers=("position_reconciliation_failed",),
        primary_blocker="position_reconciliation_failed",
        broker_ready=True,
        broker_balance_valid=True,
        last_valid_broker_balance=16_436.10,
        position_reconciliation_started=True,
        position_reconciliation_failed=True,
        position_reconciliation_error="fetch failed",
        unprotected_broker_positions={"NFO:NIFTY26MAY23750CE"},
    )
    main.app.state.bot = SimpleNamespace(_ctx=ctx)

    response = main.health_trading()
    body = _json(response)

    assert response.status_code == 200
    assert body["status"] == "blocked"
    assert body["primary_blocker"] == "position_reconciliation_failed"
    assert body["broker"]["balance_valid"] is True
    assert body["broker"]["balance"] == 16_436.10
    assert body["reconciliation"]["failed"] is True
    assert body["reconciliation"]["unprotected_positions"] == ["NFO:NIFTY26MAY23750CE"]
