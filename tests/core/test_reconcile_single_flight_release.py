"""Regression coverage for the reconciliation single-flight slot.

``_reconcile_state`` coalesces concurrent runs on
``position_reconciliation_active_run_ids``.  The slot is only ever released on
the success and ``except Exception`` paths inside ``if ctx.position_manager:``,
so any exit that misses both leaks the run id permanently.  Every later call
then short-circuits as ``POSITION_RECONCILE_COALESCED``, freezing
``position_reconciliation_completed_at`` (its only write site) and eventually
raising ``position_reconciliation_stale`` forever.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from nifty_scalper_bot.core import app


def _ctx(position_manager: object | None) -> SimpleNamespace:
    return SimpleNamespace(
        position_manager=position_manager,
        order_manager=None,
        broker_client=None,
        bracket_manager=None,
        position_reconciliation_active_run_ids=set(),
    )


@pytest.mark.asyncio
async def test_missing_position_manager_releases_single_flight_slot() -> None:
    """A run that skips the body must not hold the slot for the process life."""
    ctx = _ctx(None)

    await app._reconcile_state(ctx, source="unit_test")

    assert ctx.position_reconciliation_active_run_ids == set()


@pytest.mark.asyncio
async def test_second_run_is_not_coalesced_after_body_is_skipped() -> None:
    """Reconciliation must still run again after a body-skipping run."""
    ctx = _ctx(None)

    await app._reconcile_state(ctx, source="first")
    await app._reconcile_state(ctx, source="second")

    assert ctx.position_reconciliation_active_run_ids == set()


@pytest.mark.asyncio
async def test_cancellation_releases_single_flight_slot(monkeypatch) -> None:
    """CancelledError is a BaseException, so it bypasses ``except Exception``."""

    async def _cancelled(*_args: object, **_kwargs: object) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(app.asyncio, "to_thread", _cancelled)
    ctx = _ctx(SimpleNamespace(get_open_positions=lambda: []))

    with pytest.raises(asyncio.CancelledError):
        await app._reconcile_state(ctx, source="unit_test")

    assert ctx.position_reconciliation_active_run_ids == set()
