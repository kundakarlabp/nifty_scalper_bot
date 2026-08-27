"""A stall that happens *before* work is queued must still be detected.

``_entry_eval_liveness_snapshot`` derived ``worker_stalled`` from
``work_outstanding = pending or drain_active or drain_scheduled``.  When live
ticks stop producing entry-evaluation work at all, none of those flags is ever
set, so continuous market data with a silent evaluator was indistinguishable
from a legitimately idle one: no stall log, no disarm, no recovery, and
/health kept reporting armed.
"""

from __future__ import annotations

import inspect
import threading
import time

from nifty_scalper_bot.strategies.runner import StrategyRunner


def _runner(*, tick_age_s: float, progress_age_s: float, **overrides: object):
    runner = StrategyRunner.__new__(StrategyRunner)
    now = time.monotonic()
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
    runner._entry_eval_last_progress_ts = now - progress_age_s
    runner._last_tick_seen_ts = now - tick_age_s
    runner._entry_eval_dispatch_stall_s = 120.0
    for name, value in overrides.items():
        setattr(runner, name, value)
    return runner, now


def test_fresh_ticks_with_no_evaluation_progress_is_a_stall() -> None:
    """Live ticks plus a silent evaluator must report a stall, not idle."""
    runner, now = _runner(tick_age_s=0.5, progress_age_s=300.0)

    state = runner._entry_eval_liveness_snapshot(now)

    assert state["work_outstanding"] is False
    assert state["dispatch_stalled"] is True
    assert state["worker_stalled"] is True


def test_fresh_ticks_within_the_dispatch_window_are_not_a_stall() -> None:
    """Normal bar-close evaluation cadence must not trip the detector."""
    runner, now = _runner(tick_age_s=0.5, progress_age_s=61.0)

    state = runner._entry_eval_liveness_snapshot(now)

    assert state["dispatch_stalled"] is False
    assert state["worker_stalled"] is False


def test_no_ticks_is_not_a_dispatch_stall() -> None:
    """Without live ticks the evaluator is legitimately idle; feed gates own that."""
    runner, now = _runner(tick_age_s=600.0, progress_age_s=300.0)

    state = runner._entry_eval_liveness_snapshot(now)

    assert state["dispatch_stalled"] is False
    assert state["worker_stalled"] is False


def test_dispatch_stall_disarms_new_entries() -> None:
    """The stall must fail closed rather than keep advertising armed."""
    runner, now = _runner(tick_age_s=0.5, progress_age_s=300.0)
    runner._logger = _NullLogger()
    runner._entry_eval_stall_disarmed = False
    runner._runtime_live_orders_armed = True
    runner._runtime_readiness_reason = None

    state = runner._entry_eval_liveness_snapshot(now)
    assert runner._disarm_stalled_entry_worker_if_needed(state) is True

    assert runner._runtime_live_orders_armed is False
    assert runner._runtime_readiness_reason == "strategy_evaluation_stalled"


def test_dispatch_stall_clears_when_evaluation_resumes() -> None:
    """Derived state: recovery must not require a drain to clear the stall."""
    runner, now = _runner(tick_age_s=0.5, progress_age_s=1.0)

    state = runner._entry_eval_liveness_snapshot(now)

    assert state["dispatch_stalled"] is False


def test_watchdog_interval_reads_the_configured_environment_value() -> None:
    """parse_float_env takes the value, so passing the name ignored the config."""
    src = " ".join(inspect.getsource(StrategyRunner.__init__).split())

    for name in (
        "HEALTH_WATCHDOG_INTERVAL_SECONDS",
        "ENTRY_EVAL_DISPATCH_STALL_SECONDS",
    ):
        assert f'parse_float_env( os.getenv("{name}")' in src, (
            f"{name} must be read through os.getenv; passing the bare name "
            "makes parse_float_env log a warning and use the default"
        )


class _NullLogger:
    def critical(self, *_args: object, **_kwargs: object) -> None:
        return None

    def warning(self, *_args: object, **_kwargs: object) -> None:
        return None
