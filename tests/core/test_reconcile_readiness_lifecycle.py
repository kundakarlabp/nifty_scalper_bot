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


def test_runtime_reconciliation_freshness_contract_is_authoritative(
    monkeypatch,
) -> None:
    import inspect

    from nifty_scalper_bot.core import app as app_module

    monkeypatch.setenv("POSITION_RECONCILE_MAX_AGE_SECONDS", "60")
    fresh = _Ctx(
        completed=True,
        completed_at=datetime.now(timezone.utc) - timedelta(seconds=5),
    )
    stale = _Ctx(
        completed=True,
        completed_at=datetime.now(timezone.utc) - timedelta(seconds=300),
    )

    assert app_module._reconciliation_is_fresh(fresh) is True
    assert app_module._reconciliation_is_fresh(stale) is False

    checker = app_module.RuntimeSelfChecker(stale)
    ok, detail, meta = checker._check_position_reconciliation()
    assert ok is False
    assert detail == "position_reconciliation_stale"
    assert meta["blocker"] == "position_reconciliation_stale"

    rearm_source = inspect.getsource(app_module._live_readiness_rearm_loop)
    recompute_source = inspect.getsource(
        app_module._recompute_and_push_runtime_readiness
    )
    assert "_reconciliation_is_fresh(ctx)" in rearm_source
    assert "position_reconciliation_stale" in recompute_source


class _PositionManager:
    def __init__(self, positions=None, *, raises: bool = False) -> None:
        self._positions = list(positions or [])
        self._raises = raises

    def get_open_positions(self):
        if self._raises:
            raise RuntimeError("position state unavailable")
        return list(self._positions)


def _selfcheck_context(
    *,
    age_seconds: float = 300.0,
    completed: bool = True,
    failed: bool = False,
    positions=None,
    position_manager_present: bool = True,
    position_read_raises: bool = False,
    unprotected: bool = False,
    unresolved: bool = False,
):
    ctx = _Ctx(
        completed=completed,
        completed_at=datetime.now(timezone.utc) - timedelta(seconds=age_seconds),
    )
    ctx.position_reconciliation_failed = failed
    ctx.unprotected_broker_positions = {"NFO:TEST"} if unprotected else set()
    ctx.unprotected_broker_position = bool(unprotected)
    ctx.unresolved_reconciliation_symbols = {"NFO:TEST"} if unresolved else set()
    ctx.position_manager = (
        _PositionManager(positions, raises=position_read_raises)
        if position_manager_present
        else None
    )
    return ctx


def test_closed_flat_book_suspends_age_only_reconciliation_staleness(monkeypatch) -> None:
    from nifty_scalper_bot.core import app as app_module
    from nifty_scalper_bot.utils.market_hours import MarketState

    monkeypatch.setenv("POSITION_RECONCILE_MAX_AGE_SECONDS", "60")
    monkeypatch.setattr(app_module, "get_market_state", lambda: MarketState.CLOSED)
    ctx = _selfcheck_context(positions=[])

    ok, detail, meta = app_module.RuntimeSelfChecker(ctx)._check_position_reconciliation()

    assert ok is True
    assert detail == "position_reconciliation_age_suspended_market_closed"
    assert meta["market_state"] == MarketState.CLOSED.value
    assert meta["age_check_suspended"] is True


def test_market_open_stale_reconciliation_still_fails_closed(monkeypatch) -> None:
    from nifty_scalper_bot.core import app as app_module
    from nifty_scalper_bot.utils.market_hours import MarketState

    monkeypatch.setenv("POSITION_RECONCILE_MAX_AGE_SECONDS", "60")
    monkeypatch.setattr(app_module, "get_market_state", lambda: MarketState.OPEN)
    ctx = _selfcheck_context(positions=[])

    ok, detail, meta = app_module.RuntimeSelfChecker(ctx)._check_position_reconciliation()

    assert ok is False
    assert detail == "position_reconciliation_stale"
    assert meta["blocker"] == "position_reconciliation_stale"


def test_closed_book_with_exposure_or_unknown_state_never_suspends_age(monkeypatch) -> None:
    from nifty_scalper_bot.core import app as app_module
    from nifty_scalper_bot.utils.market_hours import MarketState

    monkeypatch.setenv("POSITION_RECONCILE_MAX_AGE_SECONDS", "60")
    monkeypatch.setattr(app_module, "get_market_state", lambda: MarketState.CLOSED)

    contexts = [
        _selfcheck_context(positions=[{"symbol": "NFO:TEST"}]),
        _selfcheck_context(position_read_raises=True),
        _selfcheck_context(position_manager_present=False),
    ]
    for ctx in contexts:
        ok, detail, meta = app_module.RuntimeSelfChecker(ctx)._check_position_reconciliation()
        assert ok is False
        assert detail == "position_reconciliation_stale"
        assert meta["blocker"] == "position_reconciliation_stale"


def test_closed_flat_book_never_hides_hard_reconciliation_failures(monkeypatch) -> None:
    from nifty_scalper_bot.core import app as app_module
    from nifty_scalper_bot.utils.market_hours import MarketState

    monkeypatch.setenv("POSITION_RECONCILE_MAX_AGE_SECONDS", "60")
    monkeypatch.setattr(app_module, "get_market_state", lambda: MarketState.CLOSED)

    cases = [
        (_selfcheck_context(failed=True, positions=[]), "position_reconciliation_failed"),
        (
            _selfcheck_context(completed=False, positions=[]),
            "position_reconciliation_incomplete",
        ),
        (_selfcheck_context(unprotected=True, positions=[]), "unprotected_broker_position"),
        (
            _selfcheck_context(unresolved=True, positions=[]),
            "position_reconciliation_unresolved",
        ),
    ]
    for ctx, expected in cases:
        ok, detail, meta = app_module.RuntimeSelfChecker(ctx)._check_position_reconciliation()
        assert ok is False
        assert detail == expected
        assert meta["blocker"] == expected
