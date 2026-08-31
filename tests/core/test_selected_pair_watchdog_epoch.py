from __future__ import annotations

from types import SimpleNamespace

from nifty_scalper_bot.core.strategy_runner_dynamic_universe_safety import (
    _apply_selected_pair_epoch_liveness,
    _refresh_selected_pair_epoch,
)


NEW_PAIR = ("NFO:NIFTY2690124100CE", "NFO:NIFTY2690124100PE")
OLD_PAIR = ("NFO:NIFTY2690124050CE", "NFO:NIFTY2690124050PE")


def _runner() -> SimpleNamespace:
    return SimpleNamespace(
        _active_selected_ce=NEW_PAIR[0],
        _active_selected_pe=NEW_PAIR[1],
        _entry_eval_dispatch_stall_s=120.0,
        _last_selected_candidate_eval_completed_ts=10.0,
        _last_selected_option_tick_ts=104.0,
    )


def _base_state(**overrides: object) -> dict[str, object]:
    state: dict[str, object] = {
        "tick_age_s": 1.0,
        "dispatch_stalled": True,
        "pending_entry_eval": [],
        "drain_scheduled": False,
        "drain_active": False,
        "drain_active_age_s": 0.0,
        "last_progress_age_s": 500.0,
        "selected_eval_age_s": 500.0,
        "evaluation_alive": False,
        "work_outstanding": False,
        "drain_stranded": False,
        "worker_stalled": True,
    }
    state.update(overrides)
    return state


def test_pair_switch_starts_new_watchdog_epoch_without_faking_completion() -> None:
    runner = _runner()
    runner._selected_entry_eval_epoch_pair = OLD_PAIR
    runner._selected_entry_eval_epoch_started_at = 1.0

    pair, epoch_started = _refresh_selected_pair_epoch(runner, now=100.0)
    state = _apply_selected_pair_epoch_liveness(runner, _base_state(), now=105.0)

    assert pair == NEW_PAIR
    assert epoch_started == 100.0
    assert runner._last_selected_candidate_eval_completed_ts == 10.0
    assert state["selected_pair_epoch_age_s"] == 5.0
    assert state["selected_eval_age_s"] is None
    assert state["dispatch_stalled"] is False
    assert state["worker_stalled"] is False
    assert state["evaluation_alive"] is True


def test_new_pair_without_completion_still_dispatch_stalls_after_timeout() -> None:
    runner = _runner()
    runner._selected_entry_eval_epoch_pair = NEW_PAIR
    runner._selected_entry_eval_epoch_started_at = 100.0
    runner._last_selected_option_tick_ts = 219.0

    state = _apply_selected_pair_epoch_liveness(runner, _base_state(), now=220.5)

    assert state["selected_eval_age_s"] is None
    assert state["tick_age_s"] == 1.5
    assert state["dispatch_stalled"] is True
    assert state["worker_stalled"] is True
    assert state["evaluation_alive"] is False


def test_pair_epoch_does_not_mask_a_genuinely_overdue_active_worker() -> None:
    runner = _runner()
    runner._selected_entry_eval_epoch_pair = NEW_PAIR
    runner._selected_entry_eval_epoch_started_at = 100.0

    state = _apply_selected_pair_epoch_liveness(
        runner,
        _base_state(
            dispatch_stalled=False,
            drain_active=True,
            drain_active_age_s=91.0,
            work_outstanding=True,
        ),
        now=105.0,
    )

    assert state["dispatch_stalled"] is False
    assert state["worker_stalled"] is True


def test_tick_from_previous_pair_cannot_trigger_new_pair_dispatch_stall() -> None:
    runner = _runner()
    runner._selected_entry_eval_epoch_pair = NEW_PAIR
    runner._selected_entry_eval_epoch_started_at = 100.0
    runner._last_selected_option_tick_ts = 99.0

    state = _apply_selected_pair_epoch_liveness(runner, _base_state(), now=250.0)

    assert state["tick_age_s"] is None
    assert state["dispatch_stalled"] is False
    assert state["worker_stalled"] is False


def test_current_epoch_completion_preserves_canonical_liveness_decision() -> None:
    runner = _runner()
    runner._selected_entry_eval_epoch_pair = NEW_PAIR
    runner._selected_entry_eval_epoch_started_at = 100.0
    runner._last_selected_candidate_eval_completed_ts = 104.0
    base = _base_state(
        dispatch_stalled=False,
        worker_stalled=False,
        selected_eval_age_s=1.0,
        last_progress_age_s=1.0,
        evaluation_alive=True,
    )

    state = _apply_selected_pair_epoch_liveness(runner, base, now=105.0)

    assert state["dispatch_stalled"] is False
    assert state["worker_stalled"] is False
    assert state["selected_eval_age_s"] == 1.0
    assert state["last_progress_age_s"] == 1.0
    assert state["evaluation_alive"] is True
    assert state["selected_pair_epoch_current"] is True
