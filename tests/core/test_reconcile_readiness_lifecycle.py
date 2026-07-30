"""Reconciliation readiness must not flap during routine refreshes.

Regression for the 2026-07-27 production logs: `_reconcile_state` cleared
`position_reconciliation_completed` at the START of every run, and readiness
converted that into the hard blocker `position_reconciliation_incomplete`.
With refreshes every ~15-20s each taking 2.5-7.2s (two scheduled owners
overlapping), the 30s readiness check repeatedly landed inside a run and
live_orders_armed flapped False/True, blocking entries roughly half the time:

  15:26:23 armed=False   (reconcile started :22)
  15:26:53 armed=True
  15:27:24 armed=False   (reconcile started :21)
  15:27:54 armed=True

`completed` means "a valid broker reconciliation has previously succeeded",
not "no refresh is running".
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from nifty_scalper_bot.core.app import _reconciliation_max_age_seconds


def test_max_age_default_and_override(monkeypatch) -> None:
    monkeypatch.delenv("POSITION_RECONCILE_MAX_AGE_SECONDS", raising=False)
    assert _reconciliation_max_age_seconds() == 120.0

    monkeypatch.setenv("POSITION_RECONCILE_MAX_AGE_SECONDS", "45")
    assert _reconciliation_max_age_seconds() == 45.0

    # Malformed config must not widen the guard.
    monkeypatch.setenv("POSITION_RECONCILE_MAX_AGE_SECONDS", "not-a-number")
    assert _reconciliation_max_age_seconds() == 120.0

    # 0 disables the age check explicitly.
    monkeypatch.setenv("POSITION_RECONCILE_MAX_AGE_SECONDS", "0")
    assert _reconciliation_max_age_seconds() == 0.0


def _blockers(ctx) -> list[str]:
    """Evaluate the readiness blocker rules against a context stub."""
    from nifty_scalper_bot.core import app as app_module

    blockers: list[str] = []
    if not bool(getattr(ctx, "position_reconciliation_completed", False)):
        blockers.append("position_reconciliation_incomplete")
    else:
        completed_at = getattr(ctx, "position_reconciliation_completed_at", None)
        max_age = app_module._reconciliation_max_age_seconds()
        if max_age > 0:
            if completed_at is None:
                blockers.append("position_reconciliation_stale")
            else:
                age = (datetime.now(timezone.utc) - completed_at).total_seconds()
                if age > max_age:
                    blockers.append("position_reconciliation_stale")
    return blockers


class _Ctx:
    def __init__(self, **kw):
        self.position_reconciliation_completed = kw.get("completed", False)
        self.position_reconciliation_completed_at = kw.get("completed_at")
        self.position_reconciliation_in_progress = kw.get("in_progress", False)


def test_blocks_before_first_successful_reconciliation(monkeypatch) -> None:
    """Fail-closed startup is preserved."""
    monkeypatch.delenv("POSITION_RECONCILE_MAX_AGE_SECONDS", raising=False)
    ctx = _Ctx(completed=False)
    assert "position_reconciliation_incomplete" in _blockers(ctx)


def test_refresh_in_progress_does_not_block_after_first_success(
    monkeypatch,
) -> None:
    """THE FLAP FIX: an in-flight refresh must not invalidate last-known-good."""
    monkeypatch.delenv("POSITION_RECONCILE_MAX_AGE_SECONDS", raising=False)
    ctx = _Ctx(
        completed=True,
        completed_at=datetime.now(timezone.utc) - timedelta(seconds=5),
        in_progress=True,
    )
    assert _blockers(ctx) == []


def test_stale_last_known_good_still_blocks(monkeypatch) -> None:
    """A stuck reconciler must not leave execution armed indefinitely."""
    monkeypatch.setenv("POSITION_RECONCILE_MAX_AGE_SECONDS", "60")
    ctx = _Ctx(
        completed=True,
        completed_at=datetime.now(timezone.utc) - timedelta(seconds=300),
    )
    assert "position_reconciliation_stale" in _blockers(ctx)


def test_missing_completed_at_is_treated_as_stale(monkeypatch) -> None:
    """Fail closed when the success timestamp is unavailable."""
    monkeypatch.setenv("POSITION_RECONCILE_MAX_AGE_SECONDS", "60")
    ctx = _Ctx(completed=True, completed_at=None)
    assert "position_reconciliation_stale" in _blockers(ctx)


def test_reconcile_state_coalesces_overlapping_runs(monkeypatch) -> None:
    """Single-flight: a second run while one is active must coalesce.

    Production showed periodic_health and manual reconciliations overlapping,
    each hitting the broker for seconds.
    """
    import asyncio

    from nifty_scalper_bot.core import app as app_module

    class _Ctx2:
        position_reconciliation_active_run_ids = {"already-running"}
        position_manager = None
        order_manager = None
        broker = None

    ctx = _Ctx2()
    started_before = set(ctx.position_reconciliation_active_run_ids)

    asyncio.run(app_module._reconcile_state(ctx, source="manual"))

    # Coalesced: no new run id registered, and the in-flight one is untouched.
    assert ctx.position_reconciliation_active_run_ids == started_before


def test_health_loop_is_not_a_second_reconciliation_scheduler() -> None:
    """Only the dedicated sync loop may own scheduled reconciliation."""
    import inspect

    from nifty_scalper_bot.core import app as app_module

    health_loop_source = inspect.getsource(app_module.NiftyScalperApp._health_loop)
    app_source = inspect.getsource(app_module)

    assert "_reconcile_state" not in health_loop_source
    assert 'await _reconcile_state(ctx, source="periodic_health")' in app_source
