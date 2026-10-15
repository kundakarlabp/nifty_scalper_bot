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
