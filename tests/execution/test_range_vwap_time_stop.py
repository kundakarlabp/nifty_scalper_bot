"""Use the existing protective bracket exit dispatcher for VWAP max-hold."""

from __future__ import annotations

import time
from types import SimpleNamespace

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
    assert (
        manager._time_stop_action(
            _bracket(
                setup="session_vwap_reentry_live", held_seconds=13 * 60, progress=12
            ),
            104.0,
        )
        is None
    )
    # Existing non-range strategies retain their original progress exemption.
    assert (
        manager._time_stop_action(
            _bracket(setup="vwap_continuation", held_seconds=16 * 60, progress=12),
            104.0,
        )
        is None
    )
    # The shared stalled-position time-stop remains effective for range trades.
    old = _bracket(
        setup="session_vwap_reentry_live", held_seconds=13 * 60, progress=1.0
    )
    assert manager._time_stop_action(old, 99.0)["type"] == "TIME_STOP"


def test_range_hard_max_hold_is_independent_of_global_time_stop() -> None:
    manager = object.__new__(BracketManager)
    manager._time_stop_min_progress_r = 0.5
    for global_seconds in (0, 30 * 60):
        manager._time_stop_seconds = global_seconds
        matured = _bracket(
            setup="session_vwap_reentry_live",
            held_seconds=16 * 60,
            progress=12.0,
        )
        action = manager._time_stop_action(matured, 104.0)
        assert action is not None
        assert action["type"] == "TIME_STOP"
        assert action["qty"] == 65
        assert "RANGE_VWAP_MAX_HOLD" in action["reason"]
        # The hard max cannot trigger before 15m, even with global timer off.
        early = _bracket(
            setup="session_vwap_reentry_live",
            held_seconds=13 * 60,
            progress=12.0,
        )
        assert manager._time_stop_action(early, 104.0) is None
    # Disabling the shared policy still disables it for other strategies.
    manager._time_stop_seconds = 0
    other = _bracket(setup="vwap_continuation", held_seconds=16 * 60, progress=0)
    assert manager._time_stop_action(other, 104.0) is None


def test_futures_vwap_arrival_and_invalidation_use_shared_exit_dispatcher() -> None:
    manager = object.__new__(BracketManager)
    manager._exit_quotes = {}
    manager._time_stop_seconds = 12 * 60
    manager._time_stop_min_progress_r = 0.5
    bracket = _bracket(setup="session_vwap_reentry_live", held_seconds=30, progress=0.0)
    bracket.trade_provenance.update(
        range_futures_contract="NFO:NIFTY26NOVFUT",
        range_futures_side="CE",
        range_futures_target=25250.0,
        range_futures_invalidation=25230.0,
    )
    current = [25251.0]
    active = ["NFO:NIFTY26NOVFUT"]
    tick_age = [0.1]

    def cached(symbol, *, max_age_seconds, require_ws):
        assert symbol == "NFO:NIFTY26NOVFUT"
        assert max_age_seconds == 2.0
        assert require_ws is False
        return current[0]

    manager._market_data = SimpleNamespace(
        get_active_nifty_future_symbol_cached=lambda: active[0],
        get_cached_ltp=cached,
        time_since_last_tick=lambda _symbol: tick_age[0],
    )
    target = manager._evaluate_exit_fast(bracket, 105.0)
    assert target is not None
    assert target["type"] == "FINAL_TP"
    assert target["reason"] == "RANGE_VWAP_FUTURES_TARGET"

    current[0] = 25229.0
    stop = manager._evaluate_exit_fast(bracket, 95.0)
    assert stop is not None
    assert stop["type"] == "SL"
    assert "HARD_SL_BREACH" in stop["reason"]

    # Stale or rotated underlying cannot authorize an exit. Protective
    # option-price SL/TP and time-stop decisions remain available.
    current[0] = None
    assert manager._evaluate_exit_fast(bracket, 105.0) is None
    current[0] = 25251.0
    active[0] = "NFO:NIFTY26DECFUT"
    assert manager._evaluate_exit_fast(bracket, 105.0) is None

    active[0] = "NFO:NIFTY26NOVFUT"
    tick_age[0] = 4.0
    assert manager._evaluate_exit_fast(bracket, 105.0) is None
    tick_age[0] = 0.1
    bracket.trade_provenance.update(
        range_futures_side="PE",
        range_futures_target=25230.0,
        range_futures_invalidation=25250.0,
    )
    current[0] = 25228.0
    assert manager._evaluate_exit_fast(bracket, 105.0)["type"] == "FINAL_TP"
    current[0] = 25251.0
    assert manager._evaluate_exit_fast(bracket, 105.0)["type"] == "SL"


def test_futures_exit_accepts_existing_datahub_facade_not_fake_mdm_methods() -> None:
    """Production DataHub exposes get_active_futures_symbol, not the MDM alias."""
    manager = object.__new__(BracketManager)
    bracket = _bracket(setup="session_vwap_reentry_live", held_seconds=30, progress=0.0)
    bracket.trade_provenance.update(
        range_futures_contract="NFO:NIFTY26NOVFUT",
        range_futures_side="CE",
        range_futures_target=25250.0,
        range_futures_invalidation=25230.0,
    )
    active = ["NFO:NIFTY26NOVFUT"]
    fresh = [True]
    current = [25251.0]
    observed = []

    def guarded_cached_ltp(symbol, *, max_age_seconds, require_ws):
        observed.append((symbol, max_age_seconds, require_ws))
        return current[0] if fresh[0] else None

    manager._market_data = SimpleNamespace(
        get_active_futures_symbol=lambda: active[0],
        get_cached_ltp=guarded_cached_ltp,
        # No time_since_last_tick; DataHub enforces max age in get_cached_ltp.
    )
    target = manager._range_vwap_underlying_exit_action(bracket, 105.0)
    assert target is not None and target["type"] == "FINAL_TP"
    assert observed[-1] == ("NFO:NIFTY26NOVFUT", 2.0, False)

    current[0] = 25229.0
    stop = manager._range_vwap_underlying_exit_action(bracket, 105.0)
    assert stop is not None and stop["type"] == "SL"

    fresh[0] = False
    assert manager._range_vwap_underlying_exit_action(bracket, 105.0) is None
    fresh[0] = True
    active[0] = "NFO:NIFTY26DECFUT"
    assert manager._range_vwap_underlying_exit_action(bracket, 105.0) is None
