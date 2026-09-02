"""Regression coverage for exact market-open readiness rearming."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from nifty_scalper_bot.core import session_boundary_rearm
from nifty_scalper_bot.core.session_boundary_rearm import apply_app_patch
from nifty_scalper_bot.utils.market_hours import MarketState


@pytest.mark.asyncio
async def test_periodic_rearm_gets_an_exact_market_open_boundary_wake() -> None:
    completed = asyncio.Event()
    calls: list[tuple[str, str]] = []
    state_calls = 0

    async def original_loop(_ctx) -> None:
        await completed.wait()

    def market_state() -> MarketState:
        nonlocal state_calls
        state_calls += 1
        return MarketState.CLOSED if state_calls == 1 else MarketState.OPEN

    async def ensure_started(_ctx, *, reason: str) -> None:
        calls.append(("runner", reason))

    async def recompute(_ctx, *, reason: str) -> None:
        calls.append(("readiness", reason))
        completed.set()

    app_module = SimpleNamespace(
        _live_readiness_rearm_loop=original_loop,
        get_market_state=market_state,
        _next_nse_open_after=lambda now: now,
        _ensure_strategy_runner_started=ensure_started,
        _recompute_and_push_runtime_readiness=recompute,
    )
    apply_app_patch(app_module)

    await asyncio.wait_for(app_module._live_readiness_rearm_loop(object()), timeout=1.0)

    assert calls == [
        ("runner", "market_open_boundary"),
        ("readiness", "market_open_boundary"),
    ]


@pytest.mark.asyncio
async def test_market_close_boundary_disarms_readiness_and_runtime_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OPEN -> CLOSED must clear both readiness SSOT and final entry switch."""

    state_calls = 0
    sleep_calls = 0
    calls: list[tuple[str, str]] = []
    switch_calls: list[str] = []

    def market_state() -> MarketState:
        nonlocal state_calls
        state_calls += 1
        return MarketState.OPEN if state_calls == 1 else MarketState.CLOSED

    async def fake_sleep(_seconds: float) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls >= 2:
            raise asyncio.CancelledError

    async def ensure_started(_ctx, *, reason: str) -> None:
        calls.append(("runner", reason))

    async def recompute(ctx, *, reason: str) -> None:
        calls.append(("readiness", reason))
        ctx.live_orders_armed = reason == "market_open_boundary"
        ctx.execution_armed = ctx.live_orders_armed
        ctx.live_block_reason = None if ctx.live_orders_armed else "market_closed"
        ctx.execution_block_reason = ctx.live_block_reason

    switch = SimpleNamespace(
        arm_for_runtime=lambda: switch_calls.append("arm") or True,
        disarm_for_runtime=lambda: switch_calls.append("disarm"),
    )
    monkeypatch.setattr(session_boundary_rearm, "trading_switch", lambda: switch)
    monkeypatch.setattr(session_boundary_rearm.asyncio, "sleep", fake_sleep)

    async def original_loop(_ctx) -> None:
        return None

    app_module = SimpleNamespace(
        _live_readiness_rearm_loop=original_loop,
        get_market_state=market_state,
        _next_nse_open_after=lambda now: now,
        _ensure_strategy_runner_started=ensure_started,
        _recompute_and_push_runtime_readiness=recompute,
    )
    apply_app_patch(app_module)
    ctx = SimpleNamespace(live_orders_armed=False, execution_armed=False)

    with pytest.raises(asyncio.CancelledError):
        await session_boundary_rearm._market_open_boundary_worker(app_module, ctx)

    assert calls == [
        ("runner", "market_open_boundary"),
        ("readiness", "market_open_boundary"),
        ("readiness", "market_close_boundary"),
    ]
    assert switch_calls == ["arm", "disarm"]
    assert ctx.live_orders_armed is False
    assert ctx.execution_armed is False
    assert ctx.live_block_reason == "market_closed"


@pytest.mark.asyncio
async def test_boundary_patch_preserves_original_loop_failure() -> None:
    async def original_loop(_ctx) -> None:
        raise RuntimeError("original failure")

    app_module = SimpleNamespace(
        _live_readiness_rearm_loop=original_loop,
        get_market_state=lambda: MarketState.CLOSED,
        _next_nse_open_after=lambda now: now,
        _ensure_strategy_runner_started=lambda *_args, **_kwargs: None,
        _recompute_and_push_runtime_readiness=lambda *_args, **_kwargs: None,
    )
    apply_app_patch(app_module)

    with pytest.raises(RuntimeError, match="original failure"):
        await app_module._live_readiness_rearm_loop(object())


@pytest.mark.asyncio
async def test_runtime_readiness_push_arms_pristine_trading_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    switch = SimpleNamespace(
        arm_for_runtime=lambda: calls.append("armed") or True,
    )
    monkeypatch.setattr(session_boundary_rearm, "trading_switch", lambda: switch)

    async def original_loop(_ctx) -> None:
        return None

    async def recompute(ctx, *, reason: str) -> None:
        ctx.live_orders_armed = True
        ctx.execution_armed = True
        ctx.live_block_reason = None
        ctx.execution_block_reason = None

    app_module = SimpleNamespace(
        _live_readiness_rearm_loop=original_loop,
        get_market_state=lambda: MarketState.CLOSED,
        _next_nse_open_after=lambda now: now,
        _ensure_strategy_runner_started=lambda *_args, **_kwargs: None,
        _recompute_and_push_runtime_readiness=recompute,
    )
    apply_app_patch(app_module)
    ctx = SimpleNamespace(live_orders_armed=False, execution_armed=False)

    await app_module._recompute_and_push_runtime_readiness(ctx, reason="test")

    assert calls == ["armed"]
    assert ctx.live_orders_armed is True
    assert ctx.execution_armed is True


@pytest.mark.asyncio
async def test_runtime_readiness_push_preserves_operator_pause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    switch = SimpleNamespace(arm_for_runtime=lambda: False)
    monkeypatch.setattr(session_boundary_rearm, "trading_switch", lambda: switch)

    async def original_loop(_ctx) -> None:
        return None

    async def recompute(ctx, *, reason: str) -> None:
        ctx.live_orders_armed = True
        ctx.execution_armed = True
        ctx.live_block_reason = None
        ctx.execution_block_reason = None

    app_module = SimpleNamespace(
        _live_readiness_rearm_loop=original_loop,
        get_market_state=lambda: MarketState.CLOSED,
        _next_nse_open_after=lambda now: now,
        _ensure_strategy_runner_started=lambda *_args, **_kwargs: None,
        _recompute_and_push_runtime_readiness=recompute,
    )
    apply_app_patch(app_module)
    ctx = SimpleNamespace(live_orders_armed=False, execution_armed=False)

    await app_module._recompute_and_push_runtime_readiness(ctx, reason="test")

    assert ctx.live_orders_armed is False
    assert ctx.execution_armed is False
    assert ctx.live_block_reason == "execution_not_armed:trading_switch_off"
    assert ctx.execution_block_reason == "execution_not_armed:trading_switch_off"
