from __future__ import annotations

import threading

import pytest

from nifty_scalper_bot.core.live_ws_tick_receipts import apply_patch
from nifty_scalper_bot.strategies.runner import StrategyRunner


def _runner(*, selected_eval_age_s: float, now: float) -> StrategyRunner:
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._eval_gate_lock = threading.Lock()
    runner._entry_eval_active = False
    runner._entry_eval_active_started_at = None
    runner._entry_eval_active_symbol = None
    runner._entry_eval_active_phase = None
    runner._entry_eval_drain_scheduled = False
    runner._pending_entry_eval_symbols = set()
    runner._entry_eval_drain_count = 0
    runner._entry_eval_shutdown = False
    runner._runtime_loop_attached = True
    runner._entry_eval_last_progress_ts = now - 3600.0
    runner._last_tick_seen_ts = now - 0.2
    runner._entry_eval_dispatch_stall_s = 120.0
    runner._active_selected_ce = "NFO:NIFTY2690124150CE"
    runner._active_selected_pe = "NFO:NIFTY2690124150PE"
    runner._last_selected_option_tick_ts = now - 0.2
    runner._last_selected_candidate_eval_completed_ts = 0.0
    # Production #1157 producer used the `_at` spelling while the watchdog
    # consumed `_ts`. Reproduce that exact state.
    runner._last_selected_candidate_eval_completed_at = now - selected_eval_age_s
    runner._runner_started_mono = now - 3600.0
    return runner


def test_recent_selected_eval_producer_timestamp_keeps_watchdog_alive() -> None:
    assert apply_patch() is True
    now = 10_000.0
    runner = _runner(selected_eval_age_s=1.0, now=now)

    state = runner._entry_eval_liveness_snapshot(now)

    assert runner._last_selected_candidate_eval_completed_ts == pytest.approx(now - 1.0)
    assert state["evaluation_alive"] is True
    assert state["dispatch_stalled"] is False
    assert state["worker_stalled"] is False


def test_stale_selected_eval_still_trips_fail_closed_watchdog() -> None:
    assert apply_patch() is True
    now = 10_000.0
    runner = _runner(selected_eval_age_s=300.0, now=now)

    state = runner._entry_eval_liveness_snapshot(now)

    assert runner._last_selected_candidate_eval_completed_ts == pytest.approx(now - 300.0)
    assert state["evaluation_alive"] is False
    assert state["dispatch_stalled"] is True
    assert state["worker_stalled"] is True
