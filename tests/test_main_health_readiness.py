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


def test_candle_identity_supports_datetimes_strings_and_bar_objects():
    from datetime import datetime, timezone

    stamp = datetime(2026, 10, 16, 7, 31, tzinfo=timezone.utc)
    expected = "2026-10-16T07:31Z"
    assert main._bar_minute_identity(stamp) == expected
    assert main._bar_minute_identity("2026-10-16T13:01:59+05:30") == expected
    assert main._bar_minute_identity({"timestamp": stamp}) == expected
    assert main._bar_minute_identity(SimpleNamespace(timestamp=stamp)) == expected
    assert main._bar_minute_identity(None) is None


def test_recent_candle_timestamp_identity_explains_runner_count_delta():
    from datetime import datetime, timedelta, timezone

    base = datetime(2026, 10, 16, 7, 20, tzinfo=timezone.utc)
    timestamps = [base + timedelta(minutes=minute) for minute in range(12)]
    symbol = "NFO:NIFTY26O1923150CE"

    class MDM:
        def get_ohlc_bars(self, _symbol, *, limit=None):
            rows = [{"timestamp": ts} for ts in timestamps]
            return rows[-limit:] if limit else rows

    class Runner:
        def history_tail_timestamps(self, _symbol, *, limit=12):
            return {
                "runner": timestamps[-limit:],
                "indicator": timestamps[-limit:],
            }

    ctx = _ctx(market_data_manager=MDM(), strategy_runner=Runner())
    result = main._history_tail_diagnostics(ctx, symbol)
    assert result["diagnostic_only"] is True
    assert result["classification"] == "recent_tails_aligned"
    assert result["sample_size"] == 12
    assert result["latest_minute_by_source"]["runner"] == "2026-10-16T07:31Z"
    assert result["latest_minute_by_source"]["mdm"] == "2026-10-16T07:31Z"


def test_recent_candle_timestamp_divergence_does_not_change_readiness():
    symbol = "NFO:NIFTY26O1923150PE"

    class MDM:
        def get_ohlc_bars(self, _symbol, *, limit=None):
            return [
                {"timestamp": "2026-10-16T07:25:00+00:00"},
                {"timestamp": "2026-10-16T07:26:00+00:00"},
            ]

    class Runner:
        def history_tail_timestamps(self, _symbol, *, limit=12):
            return {
                "runner": ["2026-10-16T07:25:00Z", "2026-10-16T07:27:00Z"],
                "indicator": ["2026-10-16T07:25:00Z", "2026-10-16T07:26:00Z"],
            }

    result = main._history_tail_diagnostics(
        _ctx(market_data_manager=MDM(), strategy_runner=Runner()),
        symbol,
    )
    assert result["classification"] == "recent_tail_divergence"
    assert result["latest_minute_by_source"]["runner"] == "2026-10-16T07:27Z"
    assert result["diagnostic_only"] is True



def test_recent_candle_diagnostics_locate_duplicates_and_missing_minutes():
    minutes = [
        f"2026-10-16T07:{minute:02d}:00Z" for minute in range(20, 24)
    ]

    class MDM:
        def get_ohlc_bars(self, _symbol, *, limit=12):
            return [{"timestamp": minute} for minute in minutes]

    class Runner:
        def history_tail_timestamps(self, _symbol, *, limit=12):
            return {
                "runner": [minutes[0], minutes[1], minutes[1], minutes[3]],
                "indicator": list(minutes),
            }

    result = main._history_tail_diagnostics(
        _ctx(market_data_manager=MDM(), strategy_runner=Runner()),
        "NFO:NIFTY26O1923100PE",
    )
    assert result["classification"] == "recent_tail_divergence"
    assert result["latest_minute_agreement"] is True
    assert result["matching_tail_suffix_bars"] == 1
    assert result["duplicate_minute_ids_by_source"]["runner"] == [
        "2026-10-16T07:21Z"
    ]
    assert result["missing_vs_mdm_by_source"]["runner"] == [
        "2026-10-16T07:22Z"
    ]
    assert result["extra_vs_mdm_by_source"]["runner"] == []
    assert result["missing_vs_mdm_by_source"]["indicator"] == []
    assert result["diagnostic_only"] is True


def test_recent_candle_diagnostics_distinguish_window_boundary_from_live_gap():
    canonical = [
        f"2026-10-16T07:{minute:02d}:00Z" for minute in range(20, 24)
    ]
    runner_rows = [
        "2026-10-16T07:19:00Z", *canonical[1:]
    ]

    class MDM:
        def get_ohlc_bars(self, _symbol, *, limit=12):
            return [{"timestamp": minute} for minute in canonical]

    class Runner:
        def history_tail_timestamps(self, _symbol, *, limit=12):
            return {"runner": runner_rows, "indicator": canonical}

    diagnostic = main._history_tail_diagnostics(
        _ctx(market_data_manager=MDM(), strategy_runner=Runner()),
        "NFO:NIFTY26O1923100CE",
    )
    assert diagnostic["latest_minute_agreement"] is True
    assert diagnostic["matching_tail_suffix_bars"] == 3
    assert diagnostic["sample_window_boundary_difference"] is True
    assert diagnostic["missing_vs_mdm_by_source"]["runner"] == ["2026-10-16T07:20Z"]
    assert diagnostic["extra_vs_mdm_by_source"]["runner"] == []
    assert diagnostic["classification"] == "recent_tail_divergence"


def test_recent_candle_diagnostics_missing_provider_has_no_false_gap():
    result = main._history_tail_diagnostics(_ctx(), "NFO:NIFTY26O1923100PE")
    assert result["classification"] == "identities_unavailable"
    assert result["latest_minute_agreement"] is None
    assert result["missing_vs_mdm_by_source"] == {}
    assert result["extra_vs_mdm_by_source"] == {}
    assert result["diagnostic_only"] is True


def test_history_identity_diagnostic_is_only_sampled_when_counts_differ():
    class Runner:
        def __init__(self):
            self.reads = 0

        def history_tail_timestamps(self, _symbol, *, limit=12):
            self.reads += 1
            return {"runner": [], "indicator": []}

    runner = Runner()
    history = {
        "required_execution_bars": 30,
        "ce": {"mdm": 50, "runner": 50, "indicator": 50},
        "pe": {"mdm": 50, "runner": 51, "indicator": 50},
    }
    result = main._history_count_diagnostics(
        history,
        ctx=_ctx(strategy_runner=runner),
        selected={"ce": "NFO:NIFTY26O1923100CE", "pe": "NFO:NIFTY26O1923100PE"},
    )
    assert runner.reads == 1
    assert "recent_timestamp_identity" not in result["per_side"]["ce"]
    assert result["execution_history_ready"] is True


def test_recent_candle_identity_handles_missing_providers_without_crashing():
    result = main._history_tail_diagnostics(_ctx(), "NFO:NIFTY26O1923150PE")
    assert result["classification"] == "identities_unavailable"
    assert result["diagnostic_only"] is True


def test_history_diagnostics_expose_count_delta_without_readiness_veto():
    history = {
        "required_execution_bars": 30,
        "ce": {"mdm": 70, "indicator": 70, "runner": 81},
        "pe": {"mdm": 70, "indicator": 70, "runner": 84},
    }

    diagnostic = main._history_count_diagnostics(history)
    assert diagnostic["count_mismatch"] is True
    assert diagnostic["execution_history_ready"] is True
    assert diagnostic["per_side"]["ce"]["runner_minus_indicator"] == 11
    assert diagnostic["per_side"]["pe"]["runner_minus_indicator"] == 14
    assert diagnostic["per_side"]["ce"]["mdm_minus_indicator"] == 0
    assert diagnostic["diagnostic_only"] is True


def test_history_diagnostics_fail_closed_on_missing_canonical_history():
    history = {
        "required_execution_bars": 30,
        "ce": {"mdm": 35, "indicator": 29, "runner": 50},
        "pe": {"mdm": 0, "indicator": 0, "runner": 50},
    }
    diagnostic = main._history_count_diagnostics(history)
    assert diagnostic["execution_history_ready"] is False
    assert diagnostic["per_side"]["ce"]["minimum_ready"] is False
    assert diagnostic["per_side"]["pe"]["minimum_ready"] is False
    assert diagnostic["diagnostic_only"] is True


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
    assert body["broker_authentication"] == "authenticated"
    assert body["broker"]["authentication"] == "authenticated"
    assert body["broker"]["authenticated"] is True
    assert body["broker"]["order_endpoint_verified"] is False
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

    assert response.status_code == 200
    assert body["ready"] is True
    assert "broker_authentication_unknown" not in body["blockers"]


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
    assert body["reconciliation"]["completed"] is True


def test_health_reads_order_endpoint_proof_from_runtime_manager():
    ctx = _ctx(
        broker_balance_valid=True,
        position_reconciliation_started=True,
        position_reconciliation_completed=True,
        order_manager=SimpleNamespace(order_endpoint_verified=True),
    )
    main.app.state.bot = SimpleNamespace(_ctx=ctx)

    body = _json(main.health_trading())

    assert body["broker"]["order_endpoint_verified"] is True
    assert body["broker"]["broker_session_state"] == "order_verified"


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
    assert body["reconciliation"]["completed"] is True


def test_readyz_live_unknown_broker_still_blocks_once(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ENABLE_LIVE", "true")
    ctx = _ctx(broker_balance_valid=True, position_reconciliation_completed=True)
    main.app.state.bot_started = True
    main.app.state.bot = SimpleNamespace(_ctx=ctx)

    body = _json(main.readyz())

    assert body["ready"] is True
    assert body["blockers"].count("broker_authentication_unknown") == 0


def test_health_trading_balance_success_does_not_mark_order_endpoint_authenticated():
    ctx = _ctx(
        broker_balance_valid=True,
        evaluation_ready=True,
        live_orders_armed=True,
        position_reconciliation_completed=True,
    )
    main.app.state.bot = SimpleNamespace(_ctx=ctx)

    body = _json(main.health_trading())

    assert body["broker"]["funds_endpoint_verified"] is True
    assert body["broker"]["order_endpoint_verified"] is False
    assert body["broker"]["order_endpoint_verification_scope"] == "process_local"
    assert (
        body["broker"]["order_endpoint_verification_state"]
        == "not_yet_proven_by_live_order"
    )
    assert body["broker"]["order_endpoint_verification_required_for_readiness"] is False
    assert body["broker"]["broker_session_state"] == "funds_verified"
    assert body["broker"]["authentication"] == "authenticated"
    assert body["live_order_readiness"]["ready"] is True
    assert "order_endpoint_unverified" not in body["live_order_readiness"]["missing"]


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
    assert body["broker"]["authentication"] == "authenticated"
    assert "order_endpoint_unverified" not in body["live_order_readiness"]["missing"]


def test_fresh_current_generation_feed_proves_market_data_authentication():
    class MDM:
        def trading_feed_health(self):
            return {
                "required_symbol_readiness": {
                    "NSE:NIFTY": {
                        "ready": True,
                        "subscription_confirmed": True,
                        "current_generation_tick_received": True,
                    }
                }
            }

    ctx = _ctx(
        market_data_manager=MDM(),
        broker_balance_valid=True,
        position_reconciliation_completed=True,
    )
    main.app.state.bot = SimpleNamespace(_ctx=ctx)

    body = _json(main.health_trading())

    assert body["broker"]["market_data_authenticated"] is True
    assert body["broker"]["funds_endpoint_verified"] is True
    assert body["broker"]["order_endpoint_verified"] is False
    assert body["broker"]["broker_session_state"] == "funds_verified"


async def test_lifespan_keeps_replication_alive_when_bot_startup_fails(
    monkeypatch,
    tmp_path,
):
    main._release_bot_start_guard()
    replication_started: list[object] = []
    replication_cancelled = asyncio.Event()

    async def replication_loop():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            replication_cancelled.set()
            raise

    def fake_replication_start(db_path):
        replication_started.append(db_path)
        return asyncio.create_task(replication_loop())

    async def failed_runner(app):
        app.state.bot_error = "broker auth invalid"
        app.state.bot_started = False

    monkeypatch.setattr(main, "get_data_dir", lambda: tmp_path)
    monkeypatch.setattr(main, "start_trade_replication_task", fake_replication_start)
    monkeypatch.setattr(main, "_run_bot_background", failed_runner)

    app = _app_state()
    async with main.lifespan(app):
        await asyncio.sleep(0)
        assert app.state.bot_error == "broker auth invalid"
        assert replication_started == [tmp_path / "trades.db"]
        assert replication_cancelled.is_set() is False

    assert replication_cancelled.is_set() is True


def test_trading_status_distinguishes_live_configuration_from_current_arming(
    monkeypatch,
) -> None:
    monkeypatch.setenv("ENABLE_LIVE", "true")
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    ctx = _ctx(
        broker_balance_valid=True,
        position_reconciliation_completed=True,
        live_orders_armed=False,
    )
    main.app.state.bot = SimpleNamespace(_ctx=ctx)

    blocked = main.trading_status()

    assert blocked["configured_for_live"] is True
    assert blocked["can_place_live_order_now"] is False
    assert blocked["will_trade"] is False

    ctx.live_orders_armed = True
    ctx.readiness_decision.live_orders_armed = True
    armed = main.trading_status()

    assert armed["configured_for_live"] is True
    assert armed["can_place_live_order_now"] is True
    assert armed["will_trade"] is True


def test_health_trading_exposes_signal_observability_without_changing_readiness():
    class Runner:
        def get_status(self):
            return {
                "tick_count": 11,
                "phase9_evaluation_count": 4,
                "legacy_activity_count": 15,
                "last_tick_age_sec": 0.8,
                "last_eval_age_sec": 17.2,
                "runner_state": "RunnerState.EXECUTION_ENABLED",
                "candidate_generated_count": 1,
                "approved_candidate_count": 0,
                "completed_strategy_evaluations_window": {
                    "SMC": 4,
                    "VWAPPro": 4,
                    "ORBPro": 4,
                },
                "completed_strategy_evaluations_total": {
                    "SMC": 10,
                    "VWAPPro": 10,
                    "ORBPro": 10,
                },
                "no_vote_reason_counts_by_strategy_window": {
                    "SMC": {"underlying_no_liquidity_sweep": 3},
                    "VWAPPro": {"distance_outside_band": 2},
                    "ORBPro": {"non_breakout_regime": 4},
                },
                "no_vote_reason_counts_by_strategy_total": {},
                "last_trigger_qualified_age_sec": 42.5,
                "setup_lifecycle": {"transition_counts": {"TRIGGER_QUALIFIED": 2}},
            }

    ctx = _ctx(
        blockers=(),
        execution_ready=True,
        live_orders_armed=True,
        broker_balance_valid=True,
        position_reconciliation_completed=True,
        evaluation_ready=True,
        strategy_runner=Runner(),
    )
    main.app.state.bot = SimpleNamespace(_ctx=ctx)

    body = _json(main.health_trading())

    observability = body["signal_observability"]
    assert observability["runner_tick_count"] == 11
    assert observability["phase9_evaluation_count"] == 4
    assert observability["legacy_activity_count"] == 15
    assert observability["last_tick_age_sec"] == 0.8
    assert observability["last_eval_age_sec"] == 17.2
    assert observability["runner_state"] == "RunnerState.EXECUTION_ENABLED"
    assert observability["completed_strategy_evaluations_window"]["SMC"] == 4
    assert (
        observability["no_vote_reason_counts_by_strategy_window"]["ORBPro"][
            "non_breakout_regime"
        ]
        == 4
    )
    assert observability["last_trigger_qualified_age_sec"] == 42.5
    assert body["live_order_readiness"]["ready"] is True
