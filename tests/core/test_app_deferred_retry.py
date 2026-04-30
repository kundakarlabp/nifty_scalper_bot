from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from nifty_scalper_bot.core import app


@pytest.mark.asyncio
async def test_deferred_basket_retry_scheduled_on_spot_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _no_op_retry(
        _ctx: object, *, configured_mode: str, max_attempts: int = 24, delay_seconds: float = 5.0
    ) -> None:
        del _ctx, configured_mode, max_attempts, delay_seconds
        await asyncio.sleep(0)

    monkeypatch.setattr(app, "_deferred_basket_hydration_retry", _no_op_retry)
    ctx = SimpleNamespace(
        live_orders_armed=True,
        trading_ready=True,
        readiness_mode="LIVE",
        live_block_reason=None,
    )

    app._schedule_deferred_basket_retry(ctx, configured_mode="LIVE")

    assert ctx.live_orders_armed is False
    assert ctx.trading_ready is False
    assert ctx.readiness_mode == "DATA_WARMUP"
    assert ctx.live_block_reason == "fresh_ws_spot_unavailable"
    assert ctx.deferred_basket_retry_started is True  # noqa: SLF001
    assert isinstance(ctx.deferred_basket_retry_task, asyncio.Task)  # noqa: SLF001
    await ctx.deferred_basket_retry_task  # noqa: SLF001


@pytest.mark.asyncio
async def test_strategy_runner_start_idempotent() -> None:
    started = 0

    class _Runner:
        is_running = False

        async def start(self) -> None:
            nonlocal started
            started += 1

    existing = asyncio.create_task(asyncio.sleep(0.2))
    ctx = SimpleNamespace(strategy_runner=_Runner(), runner_task=existing)
    await app._ensure_strategy_runner_started(ctx, reason="test_existing")
    assert started == 0

    existing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await existing

    ctx.runner_task = None
    await app._ensure_strategy_runner_started(ctx, reason="test_start")
    assert started == 1
