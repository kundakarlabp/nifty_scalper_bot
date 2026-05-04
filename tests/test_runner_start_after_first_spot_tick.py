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
        strategy_runner=SimpleNamespace(_started=False, started=False),
        message_bus=SimpleNamespace(running=True),
        strategy_runner_task=None,
        data_observation_ready=False,
        live_orders_armed=False,
    )

    caplog.set_level('INFO', logger='nifty_scalper_bot.core.app')
    await app._refresh_readiness_after_first_tick(ctx, reason='first_spot_tick_listener')

    assert started_reasons == ['first_spot_tick_listener:data_pipeline_ready_after_tick']
    assert 'STRATEGY_RUNNER_START_REQUESTED_AFTER_TICK' in caplog.text


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
