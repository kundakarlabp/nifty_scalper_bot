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
        def runner_history_count(self, sym):
            return counts.get(sym, 0)
        def indicator_history_count(self, sym):
            return counts.get(sym, 0)
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
    assert r.blocker == "selected_ce_history_cold"


async def test_pe_cold_blocker_present() -> None:
    ctx = _ctx(60, 5)
    r = app.compute_selected_option_history_readiness(ctx, "NFO:NIFTY2661623300CE", "NFO:NIFTY2661623300PE")
    assert r.both_ready is False
    assert r.blocker == "selected_pe_history_cold"


async def test_blocker_clears_after_counts_ready() -> None:
    cold = _ctx(10, 60)
    r1 = app.compute_selected_option_history_readiness(cold, "NFO:NIFTY2661623300CE", "NFO:NIFTY2661623300PE")
    assert r1.blocker == "selected_ce_history_cold"
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
    before = ctx.strategy_runner.runner_history_count("NFO:NIFTY2661623300CE")
    app.compute_selected_option_history_readiness(ctx, "NFO:NIFTY2661623300CE", "NFO:NIFTY2661623300PE")
    after = ctx.strategy_runner.runner_history_count("NFO:NIFTY2661623300CE")
    assert before == after


async def test_runner_cold_indicator_warm_not_ready() -> None:
    # Spec §5/§12: distinct counts — runner history cold must block even if
    # indicator history is warm.
    from types import SimpleNamespace as NS
    class _MDM:
        def get_ohlc_bars(self, sym):
            return [{}] * 60
    class _Runner:
        _option_required_bars = 30
        def runner_history_count(self, sym):
            return 5  # cold
        def indicator_history_count(self, sym):
            return 60  # warm
    ctx = NS(market_data_manager=_MDM(), strategy_runner=_Runner())
    r = app.compute_selected_option_history_readiness(ctx, "NFO:NIFTY2661623300CE", "NFO:NIFTY2661623300PE")
    assert r.both_ready is False
    assert r.blocker == "selected_option_history_cold"


async def test_shared_role_resolver_spot_aliases_are_consistent() -> None:
    from nifty_scalper_bot.core.history_roles import resolve_symbol_history_role

    aliases = ["NIFTY", "NSE:NIFTY", "NIFTY50", "NSE:NIFTY50"]
    for configured in aliases:
        for symbol in aliases:
            assert resolve_symbol_history_role(symbol=symbol, spot_symbol=configured) == "spot_context"


async def test_shared_role_resolver_keeps_selected_option_exact() -> None:
    from nifty_scalper_bot.core.history_roles import resolve_symbol_history_role

    assert resolve_symbol_history_role(
        symbol="NFO:NIFTY26JUN24000CE",
        selected_ce="NFO:NIFTY26JUN24000CE",
        selected_pe="NFO:NIFTY26JUN24000PE",
        spot_symbol="NSE:NIFTY",
    ) == "selected_option"
    assert resolve_symbol_history_role(
        symbol="NFO:NIFTY26JUN24100CE",
        selected_ce="NFO:NIFTY26JUN24000CE",
        selected_pe="NFO:NIFTY26JUN24000PE",
        spot_symbol="NSE:NIFTY",
    ) == "option_context"


def test_startup_sequence_never_uses_max_of_runner_and_mdm_for_readiness() -> None:
    """Static architecture check (2026-07-20 hydration-deadlock incident):
    core/app.py's startup readiness loop previously computed
    `effective_bar_count = max(runner_history_count, mdm_ohlc_count)`, so a
    STALE runner/indicator count from before a restart (or from before MDM's
    cache shrank) could satisfy readiness on its own even though canonical
    MDM history was short (production: mdm=41, stale runner/indicator=77,
    required=50 -> incorrectly marked ready via max(77,41)=77>=50).

    `startup_sequence` is a large top-level orchestration function unsuitable
    for direct unit invocation here; this is a deliberate static check (the
    class of check called for in the incident's own architecture-review
    step), not a substitute for the full-cycle behavioral tests above.
    """
    import inspect

    from nifty_scalper_bot.core import app as app_module

    src = inspect.getsource(app_module.startup_sequence)
    assert "max(runner_history_count, mdm_ohlc_count)" not in src
    assert "effective_bar_count = mdm_ohlc_count" in src
    # The sync-from-MDM trigger must fire whenever the runner LAGS canonical
    # MDM history, not merely when it is exactly empty - otherwise a stale
    # nonzero runner count never gets a chance to be corrected.
    assert "mdm_ohlc_count > runner_history_count" in src
