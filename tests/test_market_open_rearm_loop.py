from __future__ import annotations

import asyncio
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

    assert ctx.live_orders_armed is True
    assert ctx.trading_ready is True
    assert ctx.effective_mode == 'LIVE'


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

    async def _stop_sleep(_secs: float) -> None:
        raise asyncio.CancelledError

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
    assert called['snapshot'] >= 0
