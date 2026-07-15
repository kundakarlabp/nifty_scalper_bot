import asyncio
import json
from types import SimpleNamespace

import pytest

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


class _DummyBot:
    def __init__(self, *, event=None, fail: Exception | None = None) -> None:
        self.event = event
        self.fail = fail
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True
        if self.fail is not None:
            raise self.fail
        if self.event is not None:
            await self.event.wait()

    async def stop(self) -> None:
        self.stopped = True


def _app_state() -> SimpleNamespace:
    return SimpleNamespace(
        state=SimpleNamespace(
            bot=None,
            bot_error=None,
            bot_started=False,
            bot_starting=False,
        )
    )


async def test_duplicate_startup_guard_blocks_simultaneous_start(monkeypatch):
    main._release_bot_start_guard()
    event = asyncio.Event()
    created: list[_DummyBot] = []

    def factory() -> _DummyBot:
        bot = _DummyBot(event=event)
        created.append(bot)
        return bot

    app1 = _app_state()
    app2 = _app_state()
    first = asyncio.create_task(
        main._run_bot_background(app1, startup_delay=0, app_factory=factory)
    )
    await asyncio.sleep(0)
    second = asyncio.create_task(
        main._run_bot_background(app2, startup_delay=0, app_factory=factory)
    )
    await asyncio.sleep(0.01)

    assert len(created) == 1
    assert app2.state.bot is None

    event.set()
    await first
    await second
    assert main._BOT_START_GUARD is False


async def test_startup_guard_allows_later_start_after_exit():
    main._release_bot_start_guard()
    created = 0

    def factory() -> _DummyBot:
        nonlocal created
        created += 1
        return _DummyBot()

    await main._run_bot_background(_app_state(), startup_delay=0, app_factory=factory)
    await main._run_bot_background(_app_state(), startup_delay=0, app_factory=factory)

    assert created == 2
    assert main._BOT_START_GUARD is False


async def test_startup_failure_resets_guard():
    main._release_bot_start_guard()

    def fail_factory() -> _DummyBot:
        return _DummyBot(fail=RuntimeError("boom"))

    app = _app_state()
    await main._run_bot_background(app, startup_delay=0, app_factory=fail_factory)

    assert app.state.bot_error == "boom"
    assert main._BOT_START_GUARD is False

    later = _app_state()
    await main._run_bot_background(
        later, startup_delay=0, app_factory=lambda: _DummyBot()
    )
    assert later.state.bot_started is True


async def test_startup_cancellation_resets_guard():
    main._release_bot_start_guard()
    event = asyncio.Event()
    task = asyncio.create_task(
        main._run_bot_background(
            _app_state(), startup_delay=0, app_factory=lambda: _DummyBot(event=event)
        )
    )
    await asyncio.sleep(0.01)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert main._BOT_START_GUARD is False


async def test_lifespan_restart_not_permanently_blocked_after_shutdown(monkeypatch):
    main._release_bot_start_guard()
    started = 0

    async def fake_runner(app):
        nonlocal started
        if not main._try_acquire_bot_start_guard():
            return
        try:
            started += 1
        finally:
            main._release_bot_start_guard()

    monkeypatch.setattr(main, "_run_bot_background", fake_runner)

    async with main.lifespan(_app_state()):
        await asyncio.sleep(0)
    async with main.lifespan(_app_state()):
        await asyncio.sleep(0)

    assert started == 2
    assert main._BOT_START_GUARD is False


def test_livez_returns_200_while_readyz_can_be_503() -> None:
    setup_function()

    live = main.livez()
    ready = main.readyz()

    assert live["status"] == "alive"
    assert ready.status_code == 503


def test_health_trading_structured_status_and_unknown_auth():
    class MDM:
        def get_tick_pressure_stats(self):
            return {"pending_ticks": 2, "active_drains": 0}

        def get_ohlc_bars(self, symbol):
            return [{}] * 30

    class Runner:
        def runner_history_count(self, symbol):
            return 30

        def indicator_history_count(self, symbol):
            return 30

    ctx = _ctx(
        blockers=(),
        primary_blocker=None,
        execution_ready=True,
        live_orders_armed=False,
        broker_ready=True,
        broker_balance_valid=True,
        position_reconciliation_completed=True,
        selected_ce="NFO:CE",
        selected_pe="NFO:PE",
        atm_strike=24000,
        market_data_manager=MDM(),
        strategy_runner=Runner(),
    )
    main.app.state.bot = SimpleNamespace(_ctx=ctx)
    response = main.health_trading()
    body = _json(response)
    assert body["primary_blocker"] == "startup_pipeline_incomplete"
    assert body["selected"] == {"atm": 24000, "ce": "NFO:CE", "pe": "NFO:PE"}
    assert body["history"]["ce"] == {"mdm": 30, "runner": 30, "indicator": 30}
    assert body["broker_authentication"] == "unknown"
    assert body["broker"]["authentication"] == "unknown"
    assert body["broker"]["authenticated"] is False
    assert body["tick_pressure"]["pending_ticks"] == 2


def test_readyz_live_requires_authenticated_broker(monkeypatch):
    monkeypatch.setenv("ENABLE_LIVE", "true")
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    ctx = _ctx(
        order_endpoint_verified=True,
        broker_balance_valid=True,
        position_reconciliation_completed=True,
    )
    main.app.state.bot_started = True
    main.app.state.bot = SimpleNamespace(_ctx=ctx)

    response = main.readyz()
    body = _json(response)

    assert response.status_code == 200
    assert body["ready"] is True


def test_readyz_live_blocks_unknown_broker_authentication(monkeypatch):
    monkeypatch.setenv("ENABLE_LIVE", "true")
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    ctx = _ctx(
        broker_balance_valid=True,
        position_reconciliation_completed=True,
    )
    main.app.state.bot_started = True
    main.app.state.bot = SimpleNamespace(_ctx=ctx)

    response = main.readyz()
    body = _json(response)

    assert response.status_code == 503
    assert body["primary_blocker"] == "broker_authentication_unknown"
    assert "broker_authentication_unknown" in body["blockers"]


def test_readyz_live_blocks_invalid_broker_authentication(monkeypatch):
    monkeypatch.setenv("ENABLE_LIVE", "true")
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    ctx = _ctx(
        broker_auth_invalid=True,
        broker_balance_valid=True,
        position_reconciliation_completed=True,
    )
    main.app.state.bot_started = True
    main.app.state.bot = SimpleNamespace(_ctx=ctx)

    response = main.readyz()
    body = _json(response)

    assert response.status_code == 503
    assert body["primary_blocker"] == "broker_authentication_invalid"
    assert "broker_authentication_invalid" in body["blockers"]


def test_health_trading_reconciliation_requires_authenticated_broker():
    ctx = _ctx(
        position_reconciliation_started=True, position_reconciliation_completed=True
    )
    main.app.state.bot = SimpleNamespace(_ctx=ctx)

    body = _json(main.health_trading())

    assert body["broker"]["authentication"] == "unknown"
    assert body["reconciliation"]["completed"] is False


def test_health_trading_reconciliation_completed_when_order_endpoint_verified():
    ctx = _ctx(
        order_endpoint_verified=True,
        position_reconciliation_started=True,
        position_reconciliation_completed=True,
    )
    main.app.state.bot = SimpleNamespace(_ctx=ctx)

    body = _json(main.health_trading())

    assert body["broker"]["authentication"] == "authenticated"
    assert body["broker"]["order_endpoint_verified"] is True
    assert body["reconciliation"]["completed"] is True


def test_health_trading_invalid_broker_forces_reconciliation_incomplete():
    ctx = _ctx(
        broker_auth_invalid=True,
        position_reconciliation_started=True,
        position_reconciliation_completed=True,
    )
    main.app.state.bot = SimpleNamespace(_ctx=ctx)

    body = _json(main.health_trading())

    assert body["broker"]["authentication"] == "invalid"
    assert body["reconciliation"]["completed"] is False


def test_readyz_live_unknown_broker_still_blocks_once(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ENABLE_LIVE", "true")
    ctx = _ctx(broker_balance_valid=True, position_reconciliation_completed=True)
    main.app.state.bot_started = True
    main.app.state.bot = SimpleNamespace(_ctx=ctx)

    body = _json(main.readyz())

    assert body["ready"] is False
    assert body["blockers"].count("broker_authentication_unknown") == 1


def test_health_trading_balance_success_does_not_mark_order_endpoint_authenticated():
    ctx = _ctx(broker_balance_valid=True, position_reconciliation_completed=True)
    main.app.state.bot = SimpleNamespace(_ctx=ctx)

    body = _json(main.health_trading())

    assert body["broker"]["funds_endpoint_verified"] is True
    assert body["broker"]["order_endpoint_verified"] is False
    assert body["broker"]["broker_session_state"] == "funds_verified"
    assert body["live_order_readiness"]["ready"] is False


def test_health_trading_order_readiness_requires_reconciliation_completion():
    ctx = _ctx(
        order_endpoint_verified=True,
        broker_balance_valid=True,
        evaluation_ready=True,
        position_reconciliation_started=True,
        position_reconciliation_completed=False,
    )
    main.app.state.bot = SimpleNamespace(_ctx=ctx)

    body = _json(main.health_trading())

    assert body["broker"]["order_endpoint_verified"] is True
    assert body["live_order_readiness"]["ready"] is False
    assert (
        "position_reconciliation_incomplete" in body["live_order_readiness"]["missing"]
    )


def test_health_trading_order_verified_and_reconciled_satisfies_readiness_portion():
    ctx = _ctx(
        order_endpoint_verified=True,
        broker_balance_valid=True,
        evaluation_ready=True,
        position_reconciliation_started=True,
        position_reconciliation_completed=True,
    )
    main.app.state.bot = SimpleNamespace(_ctx=ctx)

    body = _json(main.health_trading())

    assert body["broker"]["broker_session_state"] == "order_verified"
    assert body["reconciliation"]["reconciliation_completed"] is True
    assert "order_endpoint_unverified" not in body["live_order_readiness"]["missing"]
    assert (
        "position_reconciliation_incomplete"
        not in body["live_order_readiness"]["missing"]
    )


def test_generic_broker_auth_flags_do_not_verify_order_endpoint():
    ctx = _ctx(
        broker_authenticated=True,
        broker_auth_verified=True,
        broker_balance_valid=True,
        evaluation_ready=True,
        position_reconciliation_started=True,
        position_reconciliation_completed=True,
    )
    main.app.state.bot = SimpleNamespace(_ctx=ctx)

    body = _json(main.health_trading())

    assert body["broker"]["market_data_authenticated"] is True
    assert body["broker"]["funds_endpoint_verified"] is True
    assert body["broker"]["order_endpoint_verified"] is False
    assert body["broker"]["broker_session_state"] == "funds_verified"
    assert "order_endpoint_unverified" in body["live_order_readiness"]["missing"]
