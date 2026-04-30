from __future__ import annotations

import asyncio
from pathlib import Path
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


def test_bot_context_declares_effective_mode_and_started_mono() -> None:
    source = Path('src/nifty_scalper_bot/core/app.py').read_text(encoding='utf-8')
    assert 'effective_mode: str = "SHADOW"' in source
    assert 'started_mono: float = field(default_factory=time_module.monotonic)' in source


def test_health_check_uses_time_module_monotonic_not_datetime_time(monkeypatch, caplog) -> None:
    calls: list[str] = []

    def _mono() -> float:
        calls.append('mono')
        return 1000.0

    ctx = SimpleNamespace(
        strategy_runner=_Runner(),
        settings=SimpleNamespace(enable_live=True),
        shadow_mode_enabled=False,
        readiness_mode='DATA_WARMUP',
        effective_mode='DATA_WARMUP',
        trading_ready=False,
        live_orders_armed=False,
        started_mono=999.0,
    )
    monkeypatch.setattr(app_module, 'get_market_state', lambda: app_module.MarketState.OPEN)
    monkeypatch.setattr(app_module.time_module, 'monotonic', _mono)

    caplog.set_level('INFO')
    app_module._health_check(ctx)

    assert calls == ['mono']
    assert any('data_warmup' in rec.message for rec in caplog.records)


def test_health_check_market_closed_reason(monkeypatch, caplog) -> None:
    ctx = SimpleNamespace(
        strategy_runner=_Runner(),
        settings=SimpleNamespace(enable_live=True),
        shadow_mode_enabled=False,
        readiness_mode='LIVE',
        effective_mode='LIVE',
        trading_ready=True,
        live_orders_armed=True,
        started_mono=10.0,
    )
    monkeypatch.setattr(app_module, 'get_market_state', lambda: app_module.MarketState.CLOSED)
    monkeypatch.setattr(app_module.time_module, 'monotonic', lambda: 200.0)

    caplog.set_level('INFO')
    app_module._health_check(ctx)

    assert any('market_closed' in rec.message for rec in caplog.records)
