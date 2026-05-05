from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from nifty_scalper_bot.core import app


class _MdmStub:
    def get_fresh_spot_tick(self, symbol: str, require_ws: bool = False) -> dict[str, Any] | None:
        return {'symbol': symbol, 'last_price': 25000.0}


@pytest.mark.asyncio
async def test_refresh_readiness_starts_runner_after_first_tick(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    started_reasons: list[str] = []

    async def _fake_ensure_strategy_runner_started(ctx: Any, *, reason: str) -> None:
        started_reasons.append(reason)
        ctx.strategy_runner.started = True

    monkeypatch.setattr(app, '_ensure_strategy_runner_started', _fake_ensure_strategy_runner_started)

    ctx = SimpleNamespace(
        market_data_manager=_MdmStub(),
        strategy_runner=SimpleNamespace(_started=False, started=False, _active_symbols={'NSE:NIFTY'}),
        message_bus=SimpleNamespace(running=True),
        strategy_runner_task=None,
        data_observation_ready=False,
        live_orders_armed=False,
    )

    caplog.set_level('INFO', logger='nifty_scalper_bot.core.app')
    await app._refresh_readiness_after_first_tick(ctx, reason='first_spot_tick_listener')

    assert started_reasons == ['first_spot_tick_listener:symbols_ready_after_spot']
    assert 'STRATEGY_RUNNER_START_REQUESTED_AFTER_TICK' in caplog.text


@pytest.mark.asyncio
async def test_refresh_readiness_does_not_start_runner_without_active_symbols(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    started_reasons: list[str] = []

    async def _fake_ensure_strategy_runner_started(ctx: Any, *, reason: str) -> None:
        started_reasons.append(reason)
        ctx.strategy_runner.started = True

    monkeypatch.setattr(app, '_ensure_strategy_runner_started', _fake_ensure_strategy_runner_started)
    ctx = SimpleNamespace(
        market_data_manager=_MdmStub(),
        strategy_runner=SimpleNamespace(_started=False, started=False, _active_symbols=set()),
        message_bus=SimpleNamespace(running=True),
        strategy_runner_task=None,
        data_observation_ready=False,
    )
    caplog.set_level('INFO', logger='nifty_scalper_bot.core.app')
    await app._refresh_readiness_after_first_tick(ctx, reason='first_spot_tick_listener')
    assert ctx.data_observation_ready is True
    assert started_reasons == []
    assert 'LIVE_BASKET_BUILD_FAILED' in caplog.text


@pytest.mark.asyncio
async def test_first_spot_builds_basket_when_runner_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    started_reasons: list[str] = []

    async def _fake_ensure_strategy_runner_started(_ctx: Any, *, reason: str) -> None:
        started_reasons.append(reason)

    async def _fake_build(_ctx: Any, *, spot_ltp: float, configured_mode: str, hydrate: bool) -> dict[str, Any]:
        assert spot_ltp > 0
        assert configured_mode == 'LIVE'
        assert hydrate is True
        _ctx.strategy_runner._active_symbols = {'NSE:NIFTY', 'NFO:NIFTY24MAYFUT'}
        return {}

    monkeypatch.setattr(app, '_ensure_strategy_runner_started', _fake_ensure_strategy_runner_started)
    monkeypatch.setattr(app, '_build_and_hydrate_live_basket_from_spot', _fake_build)
    ctx = SimpleNamespace(
        market_data_manager=_MdmStub(),
        strategy_runner=SimpleNamespace(_started=False, started=False, _active_symbols=set()),
        message_bus=SimpleNamespace(running=True),
        strategy_runner_task=None,
        data_observation_ready=False,
    )
    await app._refresh_readiness_after_first_tick(ctx, reason='first_spot_tick_listener')
    assert started_reasons == ['first_spot_tick_listener:symbols_ready_after_spot']


@pytest.mark.asyncio
async def test_ensure_runner_started_no_false_positive(caplog: pytest.LogCaptureFixture) -> None:
    class _Runner:
        _running = False

        def start(self) -> None:
            return None

    ctx = SimpleNamespace(strategy_runner=_Runner(), runner_task=None, strategy_runner_task=None)
    caplog.set_level('INFO', logger='nifty_scalper_bot.core.app')
    await app._ensure_strategy_runner_started(ctx, reason='unit_test')
    assert 'STRATEGY_RUNNER_STARTED reason=unit_test' not in caplog.text
    assert 'STRATEGY_RUNNER_START_RETURNED_NOT_RUNNING' in caplog.text


@pytest.mark.asyncio
async def test_refresh_readiness_after_first_tick_uses_runner_is_running(monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = SimpleNamespace(
        market_data_manager=_MdmStub(),
        strategy_runner=SimpleNamespace(_running=True),
        message_bus=SimpleNamespace(running=True),
        strategy_runner_task=None,
        data_observation_ready=False,
    )
    monkeypatch.setattr(app, '_ensure_strategy_runner_started', lambda *_a, **_k: None)
    await app._refresh_readiness_after_first_tick(ctx, reason='unit')
    assert ctx.data_observation_ready is True
