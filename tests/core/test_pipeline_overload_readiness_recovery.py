from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from nifty_scalper_bot.core import app


@pytest.mark.asyncio
async def test_dynamic_readiness_recomputes_as_soon_as_pipeline_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mdm = SimpleNamespace(pipeline_overloaded=True)
    ctx = SimpleNamespace(market_data_manager=mdm)
    recomputes: list[str] = []

    async def recompute(_ctx: object, *, reason: str) -> None:
        recomputes.append(reason)

    monkeypatch.setattr(app, "_recompute_and_push_runtime_readiness", recompute)

    async def recover() -> None:
        await asyncio.sleep(0)
        mdm.pipeline_overloaded = False

    recovery = asyncio.create_task(recover())
    recovered = await app._recompute_readiness_after_pipeline_recovery(
        ctx,
        reason="dynamic_basket_pipeline_recovered",
        timeout_seconds=0.1,
        poll_seconds=0,
    )
    await recovery

    assert recovered is True
    assert recomputes == ["dynamic_basket_pipeline_recovered"]


@pytest.mark.asyncio
async def test_dynamic_readiness_stays_fail_closed_while_pipeline_is_overloaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = SimpleNamespace(market_data_manager=SimpleNamespace(pipeline_overloaded=True))
    recomputes: list[str] = []

    async def recompute(_ctx: object, *, reason: str) -> None:
        recomputes.append(reason)

    monkeypatch.setattr(app, "_recompute_and_push_runtime_readiness", recompute)

    recovered = await app._recompute_readiness_after_pipeline_recovery(
        ctx,
        reason="dynamic_basket_pipeline_recovered",
        timeout_seconds=0,
        poll_seconds=0,
    )

    assert recovered is False
    assert recomputes == []
