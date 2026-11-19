"""Use the existing protective bracket exit dispatcher for VWAP max-hold."""

from __future__ import annotations

import time

from nifty_scalper_bot.execution.bracket_core import BracketManager, BracketState


def _bracket(*, setup: str, held_seconds: float, progress: float) -> BracketState:
    now = time.time()
    bracket = BracketState(
        entry_order_id="range-test",
        symbol="NFO:NIFTY26NOV25250CE",
        side="BUY",
        quantity=65,
        entry_price=100.0,
        sl_trigger_price=90.0,
        tp_trigger_price=125.0,
        initial_sl_trigger_price=90.0,
        remaining_quantity=65,
        trade_provenance={
            "setup_name": setup,
            "range_option_hold_minutes": 15.0,
        },
    )
    bracket.entry_fill_ts = now - held_seconds
    bracket.highest_ltp = 100.0 + progress
    return bracket


def test_range_hard_time_stop_uses_existing_bracket_exit() -> None:
    manager = object.__new__(BracketManager)
    manager._time_stop_seconds = 12 * 60
    manager._time_stop_min_progress_r = 0.5
    # Strong early MFE normally exempts a trade from the stalled-time-stop.
    matured = _bracket(
        setup="session_vwap_reentry_live",
        held_seconds=16 * 60,
        progress=12.0,
    )
    exit_action = manager._time_stop_action(matured, 104.0)
    assert exit_action is not None
    assert exit_action["type"] == "TIME_STOP"
    assert "RANGE_VWAP_MAX_HOLD" in exit_action["reason"]
    assert exit_action["qty"] == 65
    assert manager._time_stop_action(
        _bracket(setup="session_vwap_reentry_live", held_seconds=13 * 60, progress=12),
        104.0,
    ) is None
    # Existing non-range strategies retain their original progress exemption.
    assert manager._time_stop_action(
        _bracket(setup="vwap_continuation", held_seconds=16 * 60, progress=12),
        104.0,
    ) is None
    # The shared stalled-position time-stop remains effective for range trades.
    old = _bracket(
        setup="session_vwap_reentry_live", held_seconds=13 * 60, progress=1.0
    )
    assert manager._time_stop_action(old, 99.0)["type"] == "TIME_STOP"
