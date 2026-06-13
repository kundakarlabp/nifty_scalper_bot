from __future__ import annotations

from nifty_scalper_bot.core import app


def test_compute_history_readiness_ready_and_cold() -> None:
    ready = app.compute_history_readiness(symbol="NFO:CE", role="selected_option", required_bars=30, mdm_bars=30, runner_bars=30, indicator_bars=30)
    cold = app.compute_history_readiness(symbol="NFO:PE", role="selected_option", required_bars=30, mdm_bars=29, runner_bars=30, indicator_bars=30)
    assert ready.minimum_ready is True
    assert cold.minimum_ready is False


def test_selected_history_blocker_absent_when_both_ready() -> None:
    ce = app.compute_history_readiness(symbol="NFO:CE", role="selected_option", required_bars=30, mdm_bars=30, runner_bars=30, indicator_bars=30)
    pe = app.compute_history_readiness(symbol="NFO:PE", role="selected_option", required_bars=30, mdm_bars=30, runner_bars=30, indicator_bars=30)
    aggregate = app.SelectedOptionHistoryReadiness("NFO:CE", "NFO:PE", ce, pe, ce.minimum_ready and pe.minimum_ready, None if ce.minimum_ready and pe.minimum_ready else "selected_option_history_cold")
    assert aggregate.both_ready
    assert aggregate.blocker is None


def test_selected_history_blocker_present_for_either_cold() -> None:
    ce = app.compute_history_readiness(symbol="NFO:CE", role="selected_option", required_bars=30, mdm_bars=10, runner_bars=30, indicator_bars=30)
    pe = app.compute_history_readiness(symbol="NFO:PE", role="selected_option", required_bars=30, mdm_bars=30, runner_bars=30, indicator_bars=30)
    aggregate = app.SelectedOptionHistoryReadiness("NFO:CE", "NFO:PE", ce, pe, ce.minimum_ready and pe.minimum_ready, None if ce.minimum_ready and pe.minimum_ready else "selected_option_history_cold")
    assert not aggregate.both_ready
    assert aggregate.blocker == "selected_option_history_cold"
