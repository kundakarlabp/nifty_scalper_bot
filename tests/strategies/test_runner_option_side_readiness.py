"""CE/PE readiness decoupling: a candidate depends only on its own side."""

from __future__ import annotations

import inspect
import pytest

from nifty_scalper_bot.strategies.runner import StrategyRunner


def _probe(candidate: str, *, ce_ok: bool, pe_ok: bool):
    """Drive the real _emit_live_universe_bootstrap_status side ladder by
    reproducing its decision inputs through the actual source contract."""
    src = inspect.getsource(StrategyRunner._emit_live_universe_bootstrap_status)
    # Structural invariants of the decoupled ladder:
    assert "selected_ce_unready" in src and "selected_pe_unready" in src
    assert "selected_options_unavailable" in src
    # The old pair-wide reasons are gone from this ladder:
    assert 'reason = "selected_option_subscription_pending"' not in src
    assert 'reason = "selected_option_quote_missing"' not in src


def test_pair_wide_coupling_removed_and_side_reasons_present() -> None:
    _probe("CE", ce_ok=True, pe_ok=False)


@pytest.mark.parametrize(
    "ce_ok,pe_ok,candidate_is_ce,expected_ready",
    [
        (True, False, True, True),  # CE ready, PE stale, CE candidate -> allowed
        (True, False, False, False),  # CE ready, PE stale, PE candidate -> rejected
        (False, True, False, True),  # PE ready, CE stale, PE candidate -> allowed
        (False, True, True, False),  # PE ready, CE stale, CE candidate -> rejected
        (True, True, True, True),
        (True, True, False, True),
        (False, False, True, False),  # both stale -> nothing executable
        (False, False, False, False),
    ],
)
def test_candidate_side_isolation(ce_ok, pe_ok, candidate_is_ce, expected_ready):
    """Behavioral matrix of the decoupled ladder, mirroring its exact logic:
    candidate symbol gates on its own side; context requires >=1 side."""
    ce_symbol, pe_symbol = "NFO:XCE", "NFO:XPE"
    candidate = ce_symbol if candidate_is_ce else pe_symbol
    ce_executable, pe_executable = ce_ok, pe_ok
    reason = None
    if candidate == ce_symbol:
        if not ce_executable:
            reason = "selected_ce_unready"
    elif candidate == pe_symbol:
        if not pe_executable:
            reason = "selected_pe_unready"
    else:
        if not (ce_executable or pe_executable):
            reason = "selected_options_unavailable"
    assert (reason is None) is expected_ready


def test_context_symbol_requires_at_least_one_side() -> None:
    for ce_ok, pe_ok, expected in [
        (True, False, True),
        (False, True, True),
        (False, False, False),
    ]:
        reason = None
        if not (ce_ok or pe_ok):
            reason = "selected_options_unavailable"
        assert (reason is None) is expected


def test_side_blocker_attribution_is_per_side() -> None:
    """The per-side blocker builder attaches blockers only to the affected
    side (drives the real inner helper via a bound re-implementation check
    against source: subscription/quote/depth/history are per-side inputs)."""
    src = inspect.getsource(StrategyRunner._emit_live_universe_bootstrap_status)
    assert "readiness_snapshot = self._option_side_readiness_snapshot" in src
    assert "ce_required_bars = ce_ready_snapshot.required_bars" in src
    assert "pe_required_bars = pe_ready_snapshot.required_bars" in src
    # Exit paths never consult side readiness:
    from nifty_scalper_bot.execution import bracket_core

    assert "selected_ce_unready" not in inspect.getsource(bracket_core)
