from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from nifty_scalper_bot.core import app


@pytest.mark.asyncio
async def test_market_open_rearm_loop_arms_when_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    async def _stop_sleep(_secs: float) -> None:
        if calls:
            raise asyncio.CancelledError
        calls.append('tick')

    async def _ensure(_ctx: object, *, reason: str) -> None:
        calls.append(reason)

    monkeypatch.setattr(app.asyncio, 'sleep', _stop_sleep)
    monkeypatch.setattr(app, 'get_market_state', lambda: app.MarketState.OPEN)
    monkeypatch.setattr(app, '_ensure_strategy_runner_started', _ensure)
    monkeypatch.setattr(app, '_runner_is_running', lambda _r: True)
    monkeypatch.setattr(app, '_resolve_quote_capability', lambda _ctx: {'available': True, 'error': None})

    mdm = SimpleNamespace(
        readiness_snapshot=lambda: {'spot_ready': True, 'missing_hard': []},
        has_ws_tradable_quote=lambda: True,
    )
    ctx = SimpleNamespace(
        settings=SimpleNamespace(execution_mode='LIVE'),
        market_data_manager=mdm,
        strategy_runner=object(),
        live_orders_armed=False,
        trading_ready=False,
        effective_mode='DATA_WARMUP',
    )

    with pytest.raises(asyncio.CancelledError):
        await app._live_readiness_rearm_loop(ctx)

    assert ctx.live_orders_armed is False
    assert ctx.trading_ready is False
    assert ctx.effective_mode == 'DATA_WARMUP'


@pytest.mark.asyncio
async def test_market_open_rearm_loop_waits_when_runner_not_running(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    async def _stop_sleep(_secs: float) -> None:
        nonlocal called
        if called:
            raise asyncio.CancelledError
        called = True

    monkeypatch.setattr(app.asyncio, 'sleep', _stop_sleep)
    monkeypatch.setattr(app, 'get_market_state', lambda: app.MarketState.OPEN)
    monkeypatch.setattr(app, '_runner_is_running', lambda _r: False)
    monkeypatch.setattr(app, '_resolve_quote_capability', lambda _ctx: {'available': True, 'error': None})

    async def _ensure(_ctx: object, *, reason: str) -> None:
        return None

    monkeypatch.setattr(app, '_ensure_strategy_runner_started', _ensure)

    mdm = SimpleNamespace(
        readiness_snapshot=lambda: {'spot_ready': True, 'missing_hard': []},
        has_ws_tradable_quote=lambda: True,
    )
    ctx = SimpleNamespace(
        settings=SimpleNamespace(execution_mode='LIVE'),
        market_data_manager=mdm,
        strategy_runner=object(),
        live_orders_armed=False,
        trading_ready=False,
        effective_mode='DATA_WARMUP',
    )

    with pytest.raises(asyncio.CancelledError):
        await app._live_readiness_rearm_loop(ctx)

    assert ctx.live_orders_armed is False
    assert ctx.trading_ready is False


@pytest.mark.asyncio
async def test_live_rearm_loop_uses_readiness_state_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {'snapshot': 0}

    called_sleep = False

    async def _stop_sleep(_secs: float) -> None:
        nonlocal called_sleep
        if called_sleep:
            raise asyncio.CancelledError
        called_sleep = True

    monkeypatch.setattr(app.asyncio, 'sleep', _stop_sleep)
    monkeypatch.setattr(app, 'get_market_state', lambda: app.MarketState.OPEN)
    monkeypatch.setattr(app, '_runner_is_running', lambda _r: True)
    monkeypatch.setattr(app, '_resolve_quote_capability', lambda _ctx: {'available': True, 'error': None})

    mdm = SimpleNamespace(
        wait_until_ready=lambda timeout=0.75: True,
        readiness_state_snapshot=lambda: called.__setitem__('snapshot', called['snapshot'] + 1) or {'spot_ready': True, 'missing_hard': []},
        has_ws_tradable_quote=lambda: True,
    )
    ctx = SimpleNamespace(settings=SimpleNamespace(execution_mode='LIVE'), market_data_manager=mdm, strategy_runner=object())
    with pytest.raises(asyncio.CancelledError):
        await app._live_readiness_rearm_loop(ctx)
    assert called['snapshot'] == 0


@pytest.mark.asyncio
async def test_live_rearm_skips_unchanged_healthy_full_recompute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleep_calls = 0
    recomputes: list[str] = []

    async def _stop_after_one_iteration(_secs: float) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls > 1:
            raise asyncio.CancelledError

    async def _recompute(_ctx: object, *, reason: str) -> None:
        recomputes.append(reason)

    monkeypatch.setattr(app.asyncio, "sleep", _stop_after_one_iteration)
    monkeypatch.setattr(app, "get_market_state", lambda: app.MarketState.OPEN)
    monkeypatch.setattr(app, "_runner_is_running", lambda _r: True)
    monkeypatch.setattr(app, "_recompute_and_push_runtime_readiness", _recompute)

    symbols = ["NSE:NIFTY", "NFO:NIFTY26AUGFUT", "NFO:CE", "NFO:PE"]
    mdm = SimpleNamespace(
        pipeline_overloaded=False,
        has_fresh_ws_ltp=lambda requested, max_age_seconds=60.0: bool(
            requested and requested[0] in symbols
        ),
    )
    ctx = SimpleNamespace(
        settings=SimpleNamespace(execution_mode="LIVE"),
        market_data_manager=mdm,
        strategy_runner=object(),
        live_orders_armed=True,
        trading_ready=True,
        broker_balance_valid=True,
        position_reconciliation_completed=True,
        position_reconciliation_completed_at=app.datetime.now(app.timezone.utc),
        position_reconciliation_failed=False,
        active_contract_basket={
            "spot_symbol": symbols[0],
            "futures_symbol": symbols[1],
            "selected_ce": symbols[2],
            "selected_pe": symbols[3],
        },
        runtime_readiness_fingerprint=tuple(symbols),
        runtime_readiness_recomputed_mono=time.monotonic(),
    )

    with pytest.raises(asyncio.CancelledError):
        await app._live_readiness_rearm_loop(ctx)

    assert recomputes == []


@pytest.mark.asyncio
async def test_rearm_loop_state_updates(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    async def _stop_sleep(_secs: float) -> None:
        nonlocal calls
        calls += 1
        if calls > 2:
            raise asyncio.CancelledError

    monkeypatch.setattr(app.asyncio, 'sleep', _stop_sleep)
    monkeypatch.setattr(app, 'get_market_state', lambda: app.MarketState.OPEN)
    monkeypatch.setattr(app, '_resolve_quote_capability', lambda _ctx: {'available': True, 'error': None})
    async def _ensure(_ctx: object, *, reason: str) -> None:
        return None
    monkeypatch.setattr(app, '_ensure_strategy_runner_started', _ensure)
    monkeypatch.setattr(app, '_runner_is_running', lambda _r: calls == 1)
    mdm = SimpleNamespace(
        readiness_state_snapshot=lambda: {'spot_ready': calls == 1, 'missing_hard': [] if calls == 1 else ['futures']},
        has_ws_tradable_quote=lambda: calls == 1,
        has_fresh_ws_ltp=lambda: calls == 1,
    )
    ctx = SimpleNamespace(settings=SimpleNamespace(execution_mode='LIVE'), market_data_manager=mdm, strategy_runner=object())
    with pytest.raises(asyncio.CancelledError):
        await app._live_readiness_rearm_loop(ctx)
    assert ctx.live_orders_armed is False
    assert ctx.trading_ready is False
    assert ctx.readiness_mode == 'DATA_WARMUP'
