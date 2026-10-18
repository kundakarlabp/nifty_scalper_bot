"""Offline-only ORB regime-transition ablation.

These tests must not alter live/paper ORB eligibility. All existing breakout,
completed-bar, direction, volume and execution-quality gates remain intact.
"""

from __future__ import annotations

import pytest

from .test_orb_pro_v2 import (
    CE,
    FUTURE,
    PE,
    _bar,
    _base_indicators,
    _opening_rows,
    _strategy,
)


def _breakout_rows(side: str, *, volume: float = 3_000.0) -> list[dict[str, object]]:
    rows = _opening_rows()
    if side == "CE":
        rows.append(
            _bar(
                15,
                open_=24_008.0,
                high=24_034.0,
                low=24_006.0,
                close=24_030.0,
                volume=volume,
            )
        )
    else:
        rows.append(
            _bar(
                15,
                open_=23_990.0,
                high=23_992.0,
                low=23_955.0,
                close=23_962.0,
                volume=volume,
            )
        )
    return rows


@pytest.mark.parametrize("side,symbol", [("CE", CE), ("PE", PE)])
def test_range_transition_disabled_by_default(
    monkeypatch, side: str, symbol: str
) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.delenv("ORB_RESEARCH_ALLOW_RANGE_TRANSITION", raising=False)
    rows = _breakout_rows(side)
    indicators = _base_indicators(side, rows[-1]["timestamp"])
    indicators["regime"] = "RANGE"
    strategy = _strategy({FUTURE: rows})

    assert strategy.generate_signal(symbol, indicators, 50.0) is None
    assert strategy.last_no_vote_reason == "non_breakout_regime"


@pytest.mark.parametrize("side,symbol", [("CE", CE), ("PE", PE)])
def test_shadow_range_transition_needs_actual_futures_breakout(
    monkeypatch, side: str, symbol: str
) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setenv("ORB_RESEARCH_ALLOW_RANGE_TRANSITION", "true")
    rows = _breakout_rows(side)
    indicators = _base_indicators(side, rows[-1]["timestamp"])
    indicators["regime"] = "RANGE"
    strategy = _strategy({FUTURE: rows})

    signal = strategy.generate_signal(symbol, indicators, 50.0)

    assert signal is not None
    assert signal.symbol == symbol
    assert signal.metadata["opening_range_source"] == "futures"
    assert signal.metadata["setup_pass"] is True


@pytest.mark.parametrize("mode", ["LIVE", "PAPER"])
def test_experimental_flag_never_changes_live_or_paper_eligibility(
    monkeypatch, mode: str
) -> None:
    monkeypatch.setenv("EXECUTION_MODE", mode)
    monkeypatch.setenv("ORB_RESEARCH_ALLOW_RANGE_TRANSITION", "true")
    rows = _breakout_rows("CE")
    indicators = _base_indicators("CE", rows[-1]["timestamp"])
    indicators["regime"] = "RANGE"
    strategy = _strategy({FUTURE: rows})

    assert strategy.generate_signal(CE, indicators, 50.0) is None
    assert strategy.last_no_vote_reason == "non_breakout_regime"


def test_low_activity_remains_hard_veto_even_during_shadow_ablation(
    monkeypatch,
) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setenv("ORB_RESEARCH_ALLOW_RANGE_TRANSITION", "true")
    rows = _breakout_rows("CE")
    indicators = _base_indicators("CE", rows[-1]["timestamp"])
    indicators["regime"] = "LOW_ACTIVITY"
    strategy = _strategy({FUTURE: rows})

    assert strategy.generate_signal(CE, indicators, 50.0) is None
    assert strategy.last_no_vote_reason == "non_breakout_regime"


def test_range_without_breakout_does_not_generate_signal(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setenv("ORB_RESEARCH_ALLOW_RANGE_TRANSITION", "true")
    rows = _opening_rows()
    rows.append(
        _bar(
            15,
            open_=24_000.0,
            high=24_015.0,
            low=23_995.0,
            close=24_010.0,
            volume=3_000.0,
        )
    )
    indicators = _base_indicators("CE", rows[-1]["timestamp"])
    indicators["regime"] = "RANGE"
    strategy = _strategy({FUTURE: rows})

    assert strategy.generate_signal(CE, indicators, 50.0) is None
    assert strategy.last_no_vote_reason == "no_fresh_underlying_breakout"


def test_low_volume_breakout_still_requires_retest_and_structure(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setenv("ORB_RESEARCH_ALLOW_RANGE_TRANSITION", "true")
    rows = _breakout_rows("CE", volume=900.0)
    indicators = _base_indicators("CE", rows[-1]["timestamp"])
    indicators["regime"] = "RANGE"
    strategy = _strategy({FUTURE: rows})

    assert strategy.generate_signal(CE, indicators, 50.0) is None
    assert strategy.last_no_vote_reason == "awaiting_orb_retest"


@pytest.mark.parametrize("side,symbol", [("CE", CE), ("PE", PE)])
def test_shadow_strict_retest_wick_invalidates_deep_reclaim(
    monkeypatch, side: str, symbol: str
) -> None:
    """Deep intrabar breach is an opt-in research rejection, not a live rule."""
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setenv("ORB_MOMENTUM_BRANCH_ENABLED", "false")
    monkeypatch.setenv("ORB_BALANCED_RANGE_MAX_ATR", "2.0")
    rows = _breakout_rows(side, volume=1_600.0)
    if side == "CE":
        retest = _bar(
            16,
            open_=24_025.0,
            high=24_028.0,
            low=23_995.0,
            close=24_022.0,
            volume=400.0,
        )
    else:
        retest = _bar(
            16,
            open_=23_970.0,
            high=24_005.0,
            low=23_968.0,
            close=23_975.0,
            volume=400.0,
        )
    rows.append(retest)
    indicators = _base_indicators(side, retest["timestamp"])

    monkeypatch.delenv("ORB_RESEARCH_STRICT_RETEST_WICK", raising=False)
    baseline = _strategy({FUTURE: rows}).generate_signal(symbol, indicators, 50.0)
    assert baseline is not None and baseline.metadata["entry_branch"] == "retest"
    assert baseline.metadata["underlying_retest_wick_excursion_atr"] > 0.15

    monkeypatch.setenv("ORB_RESEARCH_STRICT_RETEST_WICK", "true")
    strict = _strategy({FUTURE: rows})
    assert strict.generate_signal(symbol, indicators, 50.0) is None
    assert strict.last_no_vote_reason == "orb_breakout_invalidated"


@pytest.mark.parametrize("mode", ["LIVE", "PAPER"])
def test_strict_retest_wick_flag_never_changes_live_paper_eligibility(
    monkeypatch, mode: str
) -> None:
    monkeypatch.setenv("EXECUTION_MODE", mode)
    monkeypatch.setenv("ORB_MOMENTUM_BRANCH_ENABLED", "false")
    monkeypatch.setenv("ORB_BALANCED_RANGE_MAX_ATR", "2.0")
    monkeypatch.setenv("ORB_RESEARCH_STRICT_RETEST_WICK", "true")
    rows = _breakout_rows("CE", volume=1_600.0)
    rows.append(
        _bar(
            16,
            open_=24_025.0,
            high=24_028.0,
            low=23_995.0,
            close=24_022.0,
            volume=400.0,
        )
    )
    signal = _strategy({FUTURE: rows}).generate_signal(
        CE, _base_indicators("CE", rows[-1]["timestamp"]), 50.0
    )
    assert signal is not None
    assert signal.metadata["entry_branch"] == "retest"
    assert signal.metadata["retest_wick_research_guard"] is False


def test_shadow_strict_wick_preserves_ordinary_retest_and_restart_causality(
    monkeypatch,
) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setenv("ORB_MOMENTUM_BRANCH_ENABLED", "false")
    monkeypatch.setenv("ORB_BALANCED_RANGE_MAX_ATR", "2.0")
    monkeypatch.setenv("ORB_RESEARCH_STRICT_RETEST_WICK", "true")
    rows = _breakout_rows("CE", volume=1_600.0)
    normal = _bar(
        16, open_=24_024.0, high=24_026.0, low=24_019.0,
        close=24_021.0, volume=400.0,
    )
    current = _strategy({FUTURE: [*rows, normal]}).generate_signal(
        CE, _base_indicators("CE", normal["timestamp"]), 50.0
    )
    assert current is not None
    assert current.metadata["retest_wick_research_guard"] is True

    deep = _bar(
        16, open_=24_025.0, high=24_028.0, low=23_995.0,
        close=24_022.0, volume=400.0,
    )
    later = _bar(
        17, open_=24_022.0, high=24_027.0, low=24_018.0,
        close=24_023.0, volume=400.0,
    )
    restarted = _strategy({FUTURE: [*rows, deep, later]})
    assert (
        restarted.generate_signal(
            CE, _base_indicators("CE", later["timestamp"]), 50.0
        )
        is None
    )
    assert restarted.last_no_vote_reason == "no_fresh_underlying_breakout"
