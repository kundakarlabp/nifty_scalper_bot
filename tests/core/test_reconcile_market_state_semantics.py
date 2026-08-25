from __future__ import annotations

from datetime import datetime, timedelta, timezone

from nifty_scalper_bot.core import app as app_module
from nifty_scalper_bot.utils.market_hours import MarketState


class _PositionManager:
    def __init__(self, positions=None, *, raises: bool = False) -> None:
        self._positions = list(positions or [])
        self._raises = raises

    def get_open_positions(self):
        if self._raises:
            raise RuntimeError("position state unavailable")
        return list(self._positions)


class _Ctx:
    def __init__(
        self,
        *,
        completed: bool = True,
        failed: bool = False,
        age_seconds: float = 300.0,
        positions=None,
        position_manager_present: bool = True,
        position_read_raises: bool = False,
        unprotected: bool = False,
        unresolved: bool = False,
    ) -> None:
        self.position_reconciliation_completed = completed
        self.position_reconciliation_failed = failed
        self.position_reconciliation_completed_at = datetime.now(timezone.utc) - timedelta(
            seconds=age_seconds
        )
        self.unprotected_broker_positions = {"NFO:TEST"} if unprotected else set()
        self.unprotected_broker_position = bool(unprotected)
        self.unresolved_reconciliation_symbols = {"NFO:TEST"} if unresolved else set()
        self.position_manager = (
            _PositionManager(positions, raises=position_read_raises)
            if position_manager_present
            else None
        )


def _checker(ctx: _Ctx) -> app_module.RuntimeSelfChecker:
    return app_module.RuntimeSelfChecker(ctx)


def test_closed_flat_book_suspends_age_only_reconciliation_staleness(monkeypatch) -> None:
    """Scheduler and self-check must agree when closed+flat refresh is intentionally skipped."""
    monkeypatch.setenv("POSITION_RECONCILE_MAX_AGE_SECONDS", "60")
    monkeypatch.setattr(app_module, "get_market_state", lambda: MarketState.CLOSED)
    ctx = _Ctx(age_seconds=300, positions=[])

    ok, detail, meta = _checker(ctx)._check_position_reconciliation()

    assert ok is True
    assert detail == "position_reconciliation_age_suspended_market_closed"
    assert meta["market_state"] == MarketState.CLOSED.value


def test_market_open_stale_reconciliation_still_fails_closed(monkeypatch) -> None:
    monkeypatch.setenv("POSITION_RECONCILE_MAX_AGE_SECONDS", "60")
    monkeypatch.setattr(app_module, "get_market_state", lambda: MarketState.OPEN)
    ctx = _Ctx(age_seconds=300, positions=[])

    ok, detail, meta = _checker(ctx)._check_position_reconciliation()

    assert ok is False
    assert detail == "position_reconciliation_stale"
    assert meta["blocker"] == "position_reconciliation_stale"


def test_closed_book_with_open_position_does_not_suspend_staleness(monkeypatch) -> None:
    monkeypatch.setenv("POSITION_RECONCILE_MAX_AGE_SECONDS", "60")
    monkeypatch.setattr(app_module, "get_market_state", lambda: MarketState.CLOSED)
    ctx = _Ctx(age_seconds=300, positions=[object()])

    ok, detail, _ = _checker(ctx)._check_position_reconciliation()

    assert ok is False
    assert detail == "position_reconciliation_stale"


def test_closed_book_with_unknown_position_state_does_not_suspend_staleness(monkeypatch) -> None:
    monkeypatch.setenv("POSITION_RECONCILE_MAX_AGE_SECONDS", "60")
    monkeypatch.setattr(app_module, "get_market_state", lambda: MarketState.CLOSED)
    ctx = _Ctx(age_seconds=300, position_read_raises=True)

    ok, detail, _ = _checker(ctx)._check_position_reconciliation()

    assert ok is False
    assert detail == "position_reconciliation_stale"


def test_closed_flat_book_never_hides_explicit_reconciliation_failure(monkeypatch) -> None:
    monkeypatch.setattr(app_module, "get_market_state", lambda: MarketState.CLOSED)
    ctx = _Ctx(failed=True, positions=[])

    ok, detail, meta = _checker(ctx)._check_position_reconciliation()

    assert ok is False
    assert detail == "position_reconciliation_failed"
    assert meta["blocker"] == "position_reconciliation_failed"


def test_closed_flat_book_never_hides_unprotected_position(monkeypatch) -> None:
    monkeypatch.setattr(app_module, "get_market_state", lambda: MarketState.CLOSED)
    ctx = _Ctx(unprotected=True, positions=[])

    ok, detail, meta = _checker(ctx)._check_position_reconciliation()

    assert ok is False
    assert detail == "unprotected_broker_position"
    assert meta["blocker"] == "unprotected_broker_position"


def test_closed_flat_book_never_hides_unresolved_reconciliation(monkeypatch) -> None:
    monkeypatch.setattr(app_module, "get_market_state", lambda: MarketState.CLOSED)
    ctx = _Ctx(unresolved=True, positions=[])

    ok, detail, meta = _checker(ctx)._check_position_reconciliation()

    assert ok is False
    assert detail == "position_reconciliation_unresolved"
    assert meta["blocker"] == "position_reconciliation_unresolved"


def test_incomplete_reconciliation_is_not_excused_off_hours(monkeypatch) -> None:
    monkeypatch.setattr(app_module, "get_market_state", lambda: MarketState.CLOSED)
    ctx = _Ctx(completed=False, positions=[])

    ok, detail, meta = _checker(ctx)._check_position_reconciliation()

    assert ok is False
    assert detail == "position_reconciliation_incomplete"
    assert meta["blocker"] == "position_reconciliation_incomplete"
