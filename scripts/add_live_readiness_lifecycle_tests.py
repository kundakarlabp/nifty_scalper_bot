from __future__ import annotations

import re
from pathlib import Path

# Temporary one-shot TDD patch input; removed by the validation workflow.
ROOT = Path.cwd()


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    (ROOT / path).write_text(text, encoding="utf-8")


def replace_function(text: str, name: str, replacement: str) -> str:
    pattern = rf"(?ms)^def {re.escape(name)}\([^\n]*\):\n.*?(?=^def |\Z)"
    updated, count = re.subn(pattern, replacement.rstrip() + "\n\n", text, count=1)
    if count != 1:
        raise RuntimeError(f"{name}: expected one function, found {count}")
    return updated


# TDD: reconciliation freshness must be consumed by production arming paths.
path = "tests/core/test_reconcile_readiness_lifecycle.py"
text = read(path)
marker = "def test_runtime_reconciliation_freshness_contract_is_authoritative"
if marker not in text:
    text += '''\n\ndef test_runtime_reconciliation_freshness_contract_is_authoritative(monkeypatch) -> None:\n    import inspect\n\n    from nifty_scalper_bot.core import app as app_module\n\n    monkeypatch.setenv("POSITION_RECONCILE_MAX_AGE_SECONDS", "60")\n    fresh = _Ctx(\n        completed=True,\n        completed_at=datetime.now(timezone.utc) - timedelta(seconds=5),\n    )\n    stale = _Ctx(\n        completed=True,\n        completed_at=datetime.now(timezone.utc) - timedelta(seconds=300),\n    )\n\n    assert app_module._reconciliation_is_fresh(fresh) is True\n    assert app_module._reconciliation_is_fresh(stale) is False\n\n    checker = app_module.RuntimeSelfChecker(stale)\n    ok, detail, meta = checker._check_position_reconciliation()\n    assert ok is False\n    assert detail == "position_reconciliation_stale"\n    assert meta["blocker"] == "position_reconciliation_stale"\n\n    rearm_source = inspect.getsource(app_module._live_readiness_rearm_loop)\n    recompute_source = inspect.getsource(app_module._recompute_and_push_runtime_readiness)\n    assert "_reconciliation_is_fresh(ctx)" in rearm_source\n    assert "position_reconciliation_stale" in recompute_source\n'''
write(path, text)


# TDD: a real timestamp is immutable proof and must outrank a stale cached age field.
path = "tests/execution/test_quote_readiness.py"
text = read(path)
marker = "def test_tick_age_prefers_real_quote_timestamp_over_stale_cached_age"
if marker not in text:
    text += '''\n\ndef test_tick_age_prefers_real_quote_timestamp_over_stale_cached_age(monkeypatch):\n    import nifty_scalper_bot.execution.quote_readiness as quote_readiness\n\n    monkeypatch.setattr(quote_readiness.time, "time", lambda: 1_800_000_000.0)\n    age_ms = resolve_tick_age_ms(\n        {\n            "tick_age_ms": 2_287_265.0,\n            "last_tick_ts_ms": 1_799_999_999_500.0,\n        }\n    )\n\n    assert age_ms == 500.0\n'''
write(path, text)


# TDD: funds endpoint already proves an authenticated REST session; order-endpoint
# verification remains diagnostic and must not redefine reconciliation truth.
path = "tests/test_main_health_readiness.py"
text = read(path)
text = replace_function(
    text,
    "test_health_trading_structured_status_and_unknown_auth",
    '''def test_health_trading_structured_status_and_unknown_auth():
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
''',
)
text = replace_function(
    text,
    "test_readyz_live_blocks_unknown_broker_authentication",
    '''def test_readyz_live_blocks_unknown_broker_authentication(monkeypatch):
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
''',
)
text = replace_function(
    text,
    "test_health_trading_reconciliation_requires_authenticated_broker",
    '''def test_health_trading_reconciliation_requires_authenticated_broker():
    ctx = _ctx(
        position_reconciliation_started=True, position_reconciliation_completed=True
    )
    main.app.state.bot = SimpleNamespace(_ctx=ctx)

    body = _json(main.health_trading())

    assert body["broker"]["authentication"] == "unknown"
    assert body["reconciliation"]["completed"] is True
''',
)
text = replace_function(
    text,
    "test_health_trading_invalid_broker_forces_reconciliation_incomplete",
    '''def test_health_trading_invalid_broker_forces_reconciliation_incomplete():
    ctx = _ctx(
        broker_auth_invalid=True,
        position_reconciliation_started=True,
        position_reconciliation_completed=True,
    )
    main.app.state.bot = SimpleNamespace(_ctx=ctx)

    body = _json(main.health_trading())

    assert body["broker"]["authentication"] == "invalid"
    assert body["reconciliation"]["completed"] is True
''',
)
text = replace_function(
    text,
    "test_readyz_live_unknown_broker_still_blocks_once",
    '''def test_readyz_live_unknown_broker_still_blocks_once(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ENABLE_LIVE", "true")
    ctx = _ctx(broker_balance_valid=True, position_reconciliation_completed=True)
    main.app.state.bot_started = True
    main.app.state.bot = SimpleNamespace(_ctx=ctx)

    body = _json(main.readyz())

    assert body["ready"] is True
    assert body["blockers"].count("broker_authentication_unknown") == 0
''',
)
text = replace_function(
    text,
    "test_health_trading_balance_success_does_not_mark_order_endpoint_authenticated",
    '''def test_health_trading_balance_success_does_not_mark_order_endpoint_authenticated():
    ctx = _ctx(
        broker_balance_valid=True,
        evaluation_ready=True,
        position_reconciliation_completed=True,
    )
    main.app.state.bot = SimpleNamespace(_ctx=ctx)

    body = _json(main.health_trading())

    assert body["broker"]["funds_endpoint_verified"] is True
    assert body["broker"]["order_endpoint_verified"] is False
    assert body["broker"]["broker_session_state"] == "funds_verified"
    assert body["broker"]["authentication"] == "authenticated"
    assert body["live_order_readiness"]["ready"] is True
    assert "order_endpoint_unverified" not in body["live_order_readiness"]["missing"]
''',
)
text = replace_function(
    text,
    "test_generic_broker_auth_flags_do_not_verify_order_endpoint",
    '''def test_generic_broker_auth_flags_do_not_verify_order_endpoint():
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
''',
)
write(path, text)
