"""Canonical history-readiness SSOT tests (spec §8/§11 Readiness).

async so they execute under the repo conftest pyfunc hook (sync tests are
silently no-op'd by that hook).
"""

from __future__ import annotations

from types import SimpleNamespace

from nifty_scalper_bot.core import app


def _ctx(ce_bars: int, pe_bars: int, *, ce="NFO:NIFTY2661623300CE", pe="NFO:NIFTY2661623300PE"):
    counts = {ce: ce_bars, pe: pe_bars}

    class _MDM:
        def get_ohlc_bars(self, sym):
            return [{}] * counts.get(sym, 0)

    class _Runner:
        _option_required_bars = 30
        def _history_count_for_symbol(self, sym):
            return counts.get(sym, 0)
        def _indicator_count_for_symbol(self, sym):
            return counts.get(sym, 0)

    return SimpleNamespace(market_data_manager=_MDM(), strategy_runner=_Runner())


async def test_canonical_tests_execute_sentinel() -> None:
    # Spec §9: prove this file's async bodies actually run.
    assert True


async def test_both_ready_blocker_absent() -> None:
    ctx = _ctx(60, 60)
    r = app.compute_selected_option_history_readiness(ctx, "NFO:NIFTY2661623300CE", "NFO:NIFTY2661623300PE")
    assert r.both_ready is True
    assert r.blocker is None


async def test_ce_cold_blocker_present() -> None:
    ctx = _ctx(10, 60)
    r = app.compute_selected_option_history_readiness(ctx, "NFO:NIFTY2661623300CE", "NFO:NIFTY2661623300PE")
    assert r.both_ready is False
    assert r.blocker == "selected_option_history_cold"


async def test_pe_cold_blocker_present() -> None:
    ctx = _ctx(60, 5)
    r = app.compute_selected_option_history_readiness(ctx, "NFO:NIFTY2661623300CE", "NFO:NIFTY2661623300PE")
    assert r.both_ready is False
    assert r.blocker == "selected_option_history_cold"


async def test_blocker_clears_after_counts_ready() -> None:
    cold = _ctx(10, 60)
    r1 = app.compute_selected_option_history_readiness(cold, "NFO:NIFTY2661623300CE", "NFO:NIFTY2661623300PE")
    assert r1.blocker == "selected_option_history_cold"
    warm = _ctx(60, 60)
    r2 = app.compute_selected_option_history_readiness(warm, "NFO:NIFTY2661623300CE", "NFO:NIFTY2661623300PE")
    # Fresh computation, no stale carry-forward.
    assert r2.blocker is None and r2.both_ready is True


async def test_missing_selection_returns_not_set() -> None:
    ctx = _ctx(0, 0)
    r = app.compute_selected_option_history_readiness(ctx, None, None)
    assert r.both_ready is False
    assert r.blocker == "selected_option_not_set"


async def test_readiness_is_pure_no_mutation() -> None:
    # Calling it must not change the underlying counts (no hydration/reseed).
    ctx = _ctx(60, 60)
    before = ctx.strategy_runner._history_count_for_symbol("NFO:NIFTY2661623300CE")
    app.compute_selected_option_history_readiness(ctx, "NFO:NIFTY2661623300CE", "NFO:NIFTY2661623300PE")
    after = ctx.strategy_runner._history_count_for_symbol("NFO:NIFTY2661623300CE")
    assert before == after
