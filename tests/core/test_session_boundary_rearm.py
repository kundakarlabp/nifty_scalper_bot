"""Regression coverage for exact market-open readiness rearming."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

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
