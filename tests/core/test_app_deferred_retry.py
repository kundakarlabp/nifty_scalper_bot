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


@pytest.mark.asyncio
async def test_deferred_retry_no_duplicate_failed_log_for_unchanged_data_not_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """2026-07-20 incident: every 15s retry logged BOTH
    DEFERRED_BASKET_RETRY_WAITING and DEFERRED_BASKET_RETRY_FAILED for the
    identical data_not_ready state (95 times), implying repeated failures
    when nothing had actually changed. With an unchanged progress
    fingerprint (MDM bar counts for the committed basket's symbols), only
    the WAITING event fires and no FAILED/PROGRESS event is emitted; once
    the fingerprint changes, a PROGRESS event fires instead of FAILED."""
    from nifty_scalper_bot.utils.market_hours import MarketState

    events: list[str] = []

    class _Logger:
        def info(self, msg, *a, extra=None, **k):
            events.append((extra or {}).get("event", msg))

        def warning(self, *a, **k):
            pass

        def error(self, *a, **k):
            pass

    monkeypatch.setattr(app, "LOGGER", _Logger())
    monkeypatch.setattr(app, "get_market_state", lambda: MarketState.OPEN)
    monkeypatch.setattr(
        app, "_wait_for_live_spot_or_raise", lambda *a, **k: _coro(24000.0)
    )
    monkeypatch.setattr(
        app, "_build_and_hydrate_live_basket_from_spot", lambda *a, **k: _coro({})
    )
    monkeypatch.setattr(
        app, "_ensure_strategy_runner_started", lambda *a, **k: _coro(None)
    )
    monkeypatch.setattr(
        app, "_recompute_and_push_runtime_readiness", lambda *a, **k: _coro(None)
    )
    monkeypatch.setattr(app.asyncio, "sleep", lambda *_a, **_k: _coro(None))

    bar_counts = {"CALL": 41}

    class _MDM:
        def get_ohlc_bars(self, symbol):
            return list(range(bar_counts.get(symbol, 0)))

    ctx = SimpleNamespace(
        trading_ready=False,
        live_orders_armed=False,
        data_ready=False,
        evaluation_ready=False,
        selected_ce=None,
        market_data_manager=_MDM(),
        active_contract_basket={"symbols": ["CALL"]},
    )

    await app._deferred_basket_hydration_retry(
        ctx, configured_mode="LIVE", max_attempts=2, delay_seconds=0.0
    )

    waiting = events.count("DEFERRED_BASKET_RETRY_WAITING")
    failed = events.count("DEFERRED_BASKET_RETRY_FAILED")
    progress = events.count("DEFERRED_BASKET_RETRY_PROGRESS")
    assert failed == 0, "the duplicate FAILED event for an unchanged state must be gone"
    assert waiting >= 1
    # Fingerprint unchanged across both attempts (bar_counts static) -> no
    # PROGRESS event either; only the WAITING evidence is logged.
    assert progress == 0


def _coro(value):
    async def _inner(*_a, **_k):
        return value

    return _inner()
