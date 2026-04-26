from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from nifty_scalper_bot.core import app as app_module


class _Runner:
    def get_status(self):
        return {'running': False}


def test_health_check_uses_shared_log_throttled(monkeypatch) -> None:
    ctx = SimpleNamespace(
        strategy_runner=_Runner(),
        settings=SimpleNamespace(enable_live=False),
        shadow_mode_enabled=False,
    )
    monkeypatch.setattr(app_module, 'get_market_state', lambda: app_module.MarketState.CLOSED)
    app_module._health_check(ctx)


@pytest.mark.asyncio
async def test_stop_handles_failed_health_task() -> None:
    app = object.__new__(app_module.NiftyScalperApp)
    app._running = True
    app._shutdown_event = asyncio.Event()
    app._self_test_task = None
    app._ctx = SimpleNamespace(telegram_application=None, telegram_bot=None)
    app._telegram_application_started = False
    app._telegram_task = None

    async def _fail():
        raise RuntimeError('health-fail')

    app._health_task = asyncio.create_task(_fail())
    await asyncio.sleep(0)

    async def _shutdown(_ctx):
        return None

    original = app_module.shutdown_sequence
    app_module.shutdown_sequence = _shutdown
    try:
        await app.stop()
    finally:
        app_module.shutdown_sequence = original

    assert app._health_task is None
