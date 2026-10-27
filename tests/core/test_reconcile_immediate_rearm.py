"""Successful broker reconciliation should promptly refresh live arming."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from nifty_scalper_bot.core import app


@pytest.mark.asyncio
async def test_fresh_reconciliation_rechecks_blocked_readiness_immediately(monkeypatch):
    calls = []

    async def recompute(ctx, *, reason):
        calls.append(reason)
        ctx.live_orders_armed = True

    monkeypatch.setattr(app, "_recompute_and_push_runtime_readiness", recompute)
    ctx = SimpleNamespace(
        position_reconciliation_completed=True,
        position_reconciliation_failed=False,
        position_reconciliation_completed_at=datetime.now(timezone.utc),
        live_orders_armed=False,
    )
    await app._refresh_live_readiness_after_reconcile(ctx)
    assert calls == ["post_position_reconcile"]
    assert ctx.live_orders_armed is True


@pytest.mark.asyncio
async def test_failed_or_stale_reconciliation_never_rearms(monkeypatch):
    calls = []

    async def recompute(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(app, "_recompute_and_push_runtime_readiness", recompute)
    for completed, failed, age in (
        (False, False, 0),
        (True, True, 0),
        (True, False, 500),
    ):
        ctx = SimpleNamespace(
            position_reconciliation_completed=completed,
            position_reconciliation_failed=failed,
            position_reconciliation_completed_at=(
                datetime.now(timezone.utc) - timedelta(seconds=age)
            ),
            live_orders_armed=False,
        )
        await app._refresh_live_readiness_after_reconcile(ctx)
    assert calls == []


@pytest.mark.asyncio
async def test_already_armed_runtime_does_not_recompute_for_refresh(monkeypatch):
    calls = []

    async def recompute(*args, **kwargs):
        calls.append(1)

    monkeypatch.setattr(app, "_recompute_and_push_runtime_readiness", recompute)
    ctx = SimpleNamespace(
        position_reconciliation_completed=True,
        position_reconciliation_failed=False,
        position_reconciliation_completed_at=datetime.now(timezone.utc),
        live_orders_armed=True,
    )
    await app._refresh_live_readiness_after_reconcile(ctx)
    assert calls == []
