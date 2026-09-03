from __future__ import annotations

import threading
from types import SimpleNamespace

from nifty_scalper_bot.core.strategy_runner_dynamic_universe_safety import (
    _apply_selected_pair_transition_liveness,
    _record_selected_pair_transition,
    _schedule_selected_pair_evaluation,
)

OLD_PAIR = ("NFO:NIFTY2690124050CE", "NFO:NIFTY2690124050PE")
NEW_PAIR = ("NFO:NIFTY2690124000CE", "NFO:NIFTY2690124000PE")


class _Logger:
    def info(self, *_args: object, **_kwargs: object) -> None:
        return None

    def warning(self, *_args: object, **_kwargs: object) -> None:
        return None


def _runner() -> SimpleNamespace:
    return SimpleNamespace(
        _active_selected_ce=NEW_PAIR[0],
        _active_selected_pe=NEW_PAIR[1],
        _entry_eval_dispatch_stall_s=120.0,
        _last_selected_candidate_eval_completed_ts=10.0,
        _last_selected_option_tick_ts=104.0,
    )


def _state(**overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
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
    result.update(overrides)
    return result


def test_initial_selection_does_not_reset_canonical_watchdog_baseline() -> None:
    runner = _runner()
    changed = _record_selected_pair_transition(
        runner, (None, None), NEW_PAIR, now=100.0
    )

    assert changed is False
    assert not hasattr(runner, "_selected_entry_eval_epoch_started_at")
    canonical = _state(dispatch_stalled=True, worker_stalled=True)
    assert _apply_selected_pair_transition_liveness(runner, canonical, now=105.0) == canonical


def test_real_pair_rotation_ignores_only_previous_pair_completion_age() -> None:
    runner = _runner()
    assert _record_selected_pair_transition(runner, OLD_PAIR, NEW_PAIR, now=100.0)

    adjusted = _apply_selected_pair_transition_liveness(runner, _state(), now=105.0)

    assert runner._last_selected_candidate_eval_completed_ts == 10.0
    assert adjusted["selected_pair_epoch_age_s"] == 5.0
    assert adjusted["selected_eval_age_s"] is None
    assert adjusted["dispatch_stalled"] is False
    assert adjusted["worker_stalled"] is False
    assert adjusted["evaluation_alive"] is True


def test_non_selected_progress_cannot_mask_selected_pair_stall_without_rotation() -> None:
    runner = _runner()
    runner._last_selected_candidate_eval_completed_ts = 10.0
    canonical = _state(
        dispatch_stalled=True,
        worker_stalled=True,
        last_progress_age_s=130.0,
        selected_eval_age_s=130.0,
        evaluation_alive=False,
    )

    adjusted = _apply_selected_pair_transition_liveness(runner, canonical, now=200.0)

    assert adjusted == canonical
    assert adjusted["dispatch_stalled"] is True
    assert adjusted["worker_stalled"] is True


def test_rotated_pair_without_completion_stalls_after_configured_timeout() -> None:
    runner = _runner()
    _record_selected_pair_transition(runner, OLD_PAIR, NEW_PAIR, now=100.0)
    runner._last_selected_option_tick_ts = 219.0

    adjusted = _apply_selected_pair_transition_liveness(runner, _state(), now=220.5)

    assert adjusted["tick_age_s"] == 1.5
    assert adjusted["dispatch_stalled"] is True
    assert adjusted["worker_stalled"] is True
    assert adjusted["evaluation_alive"] is False


def test_rotated_pair_fails_closed_on_short_selected_pair_deadline(monkeypatch) -> None:
    monkeypatch.setenv("SELECTED_PAIR_EVAL_STALL_SECONDS", "15")
    runner = _runner()
    _record_selected_pair_transition(runner, OLD_PAIR, NEW_PAIR, now=100.0)
    runner._last_selected_option_tick_ts = 115.5

    adjusted = _apply_selected_pair_transition_liveness(
        runner,
        _state(work_outstanding=True, drain_active=False),
        now=116.0,
    )

    assert adjusted["selected_pair_eval_stall_s"] == 15.0
    assert adjusted["worker_stalled"] is True
    assert adjusted["evaluation_alive"] is False


def test_rotated_pair_active_worker_over_90_seconds_still_fails_closed() -> None:
    runner = _runner()
    _record_selected_pair_transition(runner, OLD_PAIR, NEW_PAIR, now=100.0)

    adjusted = _apply_selected_pair_transition_liveness(
        runner,
        _state(
            dispatch_stalled=False,
            drain_active=True,
            drain_active_age_s=91.0,
            work_outstanding=True,
        ),
        now=105.0,
    )

    assert adjusted["dispatch_stalled"] is False
    assert adjusted["worker_stalled"] is True


def test_old_pair_tick_does_not_count_as_current_pair_dispatch_evidence() -> None:
    runner = _runner()
    _record_selected_pair_transition(runner, OLD_PAIR, NEW_PAIR, now=100.0)
    runner._last_selected_option_tick_ts = 99.0

    adjusted = _apply_selected_pair_transition_liveness(runner, _state(), now=250.0)

    assert adjusted["tick_age_s"] is None
    assert adjusted["dispatch_stalled"] is False


def test_current_pair_completion_restores_canonical_liveness_result() -> None:
    runner = _runner()
    _record_selected_pair_transition(runner, OLD_PAIR, NEW_PAIR, now=100.0)
    runner._last_selected_candidate_eval_completed_ts = 104.0
    canonical = _state(
        dispatch_stalled=False,
        worker_stalled=False,
        selected_eval_age_s=1.0,
        last_progress_age_s=1.0,
        evaluation_alive=True,
    )

    adjusted = _apply_selected_pair_transition_liveness(runner, canonical, now=105.0)

    assert adjusted["selected_eval_age_s"] == 1.0
    assert adjusted["last_progress_age_s"] == 1.0
    assert adjusted["worker_stalled"] is False
    assert adjusted["evaluation_alive"] is True


def test_pair_rotation_forces_both_selected_legs_and_reschedules_stale_latch() -> None:
    calls: list[bool] = []
    runner = SimpleNamespace(
        _eval_gate_lock=threading.Lock(),
        _pending_entry_eval_symbols=set(),
        _entry_eval_generation_by_symbol={},
        _entry_eval_shutdown=False,
        _entry_eval_active=False,
        _entry_eval_drain_scheduled=True,
        _schedule_entry_eval_drain=lambda: (calls.append(True), True)[1],
        _logger=_Logger(),
    )

    assert _schedule_selected_pair_evaluation(runner, NEW_PAIR, now=100.0) is True

    assert runner._pending_entry_eval_symbols == set(NEW_PAIR)
    assert runner._entry_eval_generation_by_symbol[NEW_PAIR[0]] == 1
    assert runner._entry_eval_generation_by_symbol[NEW_PAIR[1]] == 1
    assert calls == [True]
    assert runner._entry_eval_drain_scheduled is True


def test_pair_rotation_schedule_failure_disarms_immediately() -> None:
    runner = SimpleNamespace(
        _eval_gate_lock=threading.Lock(),
        _pending_entry_eval_symbols=set(),
        _entry_eval_generation_by_symbol={},
        _entry_eval_shutdown=False,
        _entry_eval_active=False,
        _entry_eval_drain_scheduled=True,
        _schedule_entry_eval_drain=lambda: False,
        _entry_eval_stall_disarmed=False,
        _runtime_live_orders_armed=True,
        _runtime_readiness_reason=None,
        _logger=_Logger(),
    )

    assert _schedule_selected_pair_evaluation(runner, NEW_PAIR, now=100.0) is False

    assert runner._entry_eval_drain_scheduled is False
    assert runner._entry_eval_stall_disarmed is True
    assert runner._runtime_live_orders_armed is False
    assert runner._runtime_readiness_reason == "strategy_evaluation_stalled"
