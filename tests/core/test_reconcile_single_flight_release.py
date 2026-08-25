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


def _terminal_events(caplog: pytest.LogCaptureFixture) -> set[str]:
    return {
        str(getattr(record, "event", ""))
        for record in caplog.records
        if str(getattr(record, "event", "")).startswith("POSITION_RECONCILE_")
    }


@pytest.mark.asyncio
async def test_cancellation_clears_in_progress_and_records_terminal_state(
    monkeypatch, caplog
) -> None:
    """Cancellation must not leave in_progress=True with no terminal event."""

    async def _cancelled(*_args: object, **_kwargs: object) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(app.asyncio, "to_thread", _cancelled)
    ctx = _ctx(SimpleNamespace(get_open_positions=lambda: []))

    with caplog.at_level("INFO"):
        with pytest.raises(asyncio.CancelledError):
            await app._reconcile_state(ctx, source="unit_test")

    assert ctx.position_reconciliation_in_progress is False
    assert "POSITION_RECONCILE_CANCELLED" in _terminal_events(caplog)


@pytest.mark.asyncio
async def test_reconcile_loop_exit_is_reported(caplog) -> None:
    """An unexpected end of the periodic loop must not be silent."""

    async def _boom() -> None:
        raise RuntimeError("loop died")

    task = asyncio.ensure_future(_boom())
    with pytest.raises(RuntimeError):
        await task
    with caplog.at_level("ERROR"):
        app._log_reconciliation_task_exit(task)

    events = {str(getattr(r, "event", "")) for r in caplog.records}
    assert "POSITION_RECONCILE_LOOP_EXITED" in events


@pytest.mark.asyncio
async def test_reconcile_loop_cancellation_is_not_an_error(caplog) -> None:
    """Shutdown cancellation is expected and must stay out of the error channel."""

    async def _sleep_forever() -> None:
        await asyncio.sleep(3600)

    task = asyncio.ensure_future(_sleep_forever())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with caplog.at_level("INFO"):
        app._log_reconciliation_task_exit(task)

    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert not errors


@pytest.mark.asyncio
async def test_missing_position_manager_records_terminal_state(caplog) -> None:
    """A skipped run must terminate explicitly rather than silently."""
    ctx = _ctx(None)

    with caplog.at_level("INFO"):
        await app._reconcile_state(ctx, source="unit_test")

    assert ctx.position_reconciliation_in_progress is False
    events = _terminal_events(caplog)
    assert "POSITION_RECONCILE_SKIPPED" in events
    # A skipped run must not advance the success clock that gates arming.
    assert getattr(ctx, "position_reconciliation_completed_at", None) is None
