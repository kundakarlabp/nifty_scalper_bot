"""Causal, SHADOW-only failed-breakout reversal research regression cases."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from nifty_scalper_bot.backtesting import directional_hypotheses as research


_START = datetime(2026, 10, 1, 10, 0, tzinfo=timezone(timedelta(hours=5, minutes=30)))
_FUTURE = "NFO:NIFTY26OCTFUT"
_CE = "NFO:NIFTY26O0624000CE"
_PE = "NFO:NIFTY26O0624000PE"


def _bar(index, close=24000.0, volume=1000.0):
    return {
        "timestamp": _START + timedelta(minutes=index),
        "open": close,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": volume,
        "is_complete": True,
        "is_provisional": False,
    }


def _reversal_bars(side):
    rows = [_bar(index) for index in range(45)]
    if side == "PE":
        rows[-2] = {
            **_bar(43, volume=1600),
            "high": 24005.0,
            "low": 23999.0,
        }
        rows[-1] = {
            **_bar(44, 23998.0, 1200),
            "open": 24000.0,
            "high": 24000.5,
            "low": 23997.0,
        }
    else:
        rows[-2] = {
            **_bar(43, 23999.8, 1600),
            "open": 24000.0,
            "low": 23995.5,
            "high": 24001.0,
        }
        rows[-1] = {
            **_bar(44, 24002.0, 1200),
            "open": 24000.0,
            "low": 23999.5,
            "high": 24003.0,
        }
    return rows


@pytest.mark.parametrize("side", ["CE", "PE"])
def test_failed_breakout_requires_prior_sweep_and_later_completed_confirmation(side):
    rows = _reversal_bars(side)
    observed = research.classify_completed_futures_bars(rows)
    assert observed["failed_breakout_reversal"] == side
    assert research.classify_completed_futures_bars(rows[:-1]) == {}

    rejected = [dict(row) for row in rows]
    rejected[-2]["is_complete"] = False
    assert research.classify_completed_futures_bars(rejected) == {}
    rejected = [dict(row) for row in rows]
    rejected[-1]["is_provisional"] = True
    assert research.classify_completed_futures_bars(rejected) == {}


@pytest.mark.parametrize("side", ["CE", "PE"])
def test_failed_breakout_does_not_fade_accepted_break_or_untraded_sweep(side):
    rows = _reversal_bars(side)
    clean = [dict(row) for row in rows]
    clean[-2]["volume"] = 0
    assert "failed_breakout_reversal" not in research.classify_completed_futures_bars(
        clean
    )

    accepted = [dict(row) for row in rows]
    if side == "PE":
        accepted[-2]["close"] = 24003.0
    else:
        accepted[-2]["close"] = 23997.0
    assert "failed_breakout_reversal" not in research.classify_completed_futures_bars(
        accepted
    )


class _Engine:
    def __init__(self, bars):
        self.bars = bars

    def get_history(self, symbol, count, *, field):
        assert field == "bars"
        return self.bars[-count:] if symbol == _FUTURE else []

    def get_atr(self, symbol):
        return 2.0 if symbol in {_CE, _PE} else None


def test_reversal_only_buys_premium_with_existing_research_geometry(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    rows = _reversal_bars("PE")
    candidate = research.ArchivedDirectionalHypothesis(
        "failed_breakout_reversal", _Engine(rows)
    )
    context = {"futures_symbol": _FUTURE, "bar_timestamp": rows[-1]["timestamp"]}
    assert candidate.generate_signal(_CE, context, 60.0) is None
    signal = candidate.generate_signal(_PE, context, 60.0)
    assert signal is not None
    assert signal.action == "BUY"
    assert signal.stop_loss == pytest.approx(57.6)
    assert signal.take_profit == pytest.approx(64.8)
    assert signal.metadata["signal_family"] == "research_only"
    assert signal.metadata["contract_side"] == "PE"

    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    with pytest.raises(ValueError, match="reversal_research_requires_shadow_process"):
        research.ArchivedDirectionalHypothesis("failed_breakout_reversal", _Engine(rows))


def test_reversal_is_opt_in_to_archived_replay_and_never_promoted(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setattr(research, "load_archive", lambda _: ({}, {}, "archive_sha"))
    monkeypatch.setattr(
        research, "get_settings", lambda: SimpleNamespace(elite=object())
    )
    observed = []

    def fake_scenario(*_args, **kwargs):
        observed.append(
            {
                "components": kwargs["components"],
                "factory_names": [
                    strategy.name
                    for strategy in kwargs["strategy_factory"](_Engine([]))
                ],
            }
        )
        return {"strategies": {}}

    monkeypatch.setattr(research, "_scenario", fake_scenario)
    monkeypatch.delenv("RESEARCH_FAILED_BREAKOUT_REVERSAL_ENABLED", raising=False)
    baseline = research.run_directional_hypothesis_research(Path("/unused"))
    assert baseline["failed_breakout_reversal_opt_in"] is False
    assert "FailedBreakoutResearch" not in observed[0]["components"]
    assert "FailedBreakoutResearch" not in observed[0]["factory_names"]

    observed.clear()
    monkeypatch.setenv("RESEARCH_FAILED_BREAKOUT_REVERSAL_ENABLED", "true")
    candidate = research.run_directional_hypothesis_research(Path("/unused"))
    assert candidate["failed_breakout_reversal_opt_in"] is True
    assert candidate["promotion_eligible"] is False
    assert candidate["selected_for_live"] is None
    assert len(observed) == 3
    assert all(
        "FailedBreakoutResearch" in row["components"]
        and "FailedBreakoutResearch" in row["factory_names"]
        for row in observed
    )
