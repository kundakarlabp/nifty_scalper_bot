"""Evaluation-stall recovery must actually recover, and never block ticks.

Post-#943 audit, P0/P1. _recover_strategy_eval_stall_once() had two defects:

1. It logged action=recompute_readiness_and_restart_loop and called
   self.start(), but start() is a no-op while the runner is already running --
   which it always is when a stall is detected. The advertised restart never
   happened, and because the recovery was marked attempted, escalation was
   suppressed.

2. Its readiness-recompute fallback called asyncio.run(result) when no loop was
   running on the calling thread. This function runs inside _health_watchdog ->
   _on_tick_safe, i.e. on the market-data tick thread, so it blocked tick
   ingestion for the full recompute -- the same defect fixed for Telegram.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
import time
from unittest.mock import MagicMock

from nifty_scalper_bot.strategies.runner import StrategyRunner


def _source() -> str:
    return inspect.getsource(StrategyRunner._recover_strategy_eval_stall_once)


def test_recovery_does_not_block_the_tick_thread_with_asyncio_run() -> None:
    """THE BLOCKING FIX: no synchronous asyncio.run on the tick path."""
    src = _source()
    assert "asyncio.run(result)" not in src, (
        "recovery runs on the market-data tick thread; asyncio.run() there "
        "blocks tick ingestion for the whole recompute"
    )
    assert "run_coroutine_threadsafe" in src


def test_recovery_reports_when_start_was_a_noop() -> None:
    """Operators must see that the advertised restart did nothing."""
    src = _source()
    assert "start_was_noop" in src
    assert "STRATEGY_EVAL_STALL_WORKER_STATE" in src


def test_recovery_nudges_the_real_evaluation_mechanism() -> None:
    """Post-#918 the evaluation mechanism is the coalesced entry-eval drain."""
    src = _source()
    assert "_schedule_entry_eval_drain" in src


def test_recovery_captures_evaluation_worker_state() -> None:
    """Stall logs must explain WHY evaluation stalled, not only that it did."""
    src = _source()
    for field in (
        "pending_entry_eval",
        "drain_scheduled",
        "drain_active",
        "runtime_loop_attached",
        "executor_alive",
    ):
        assert field in src, f"missing worker-state field: {field}"


def test_recovery_reschedules_a_stranded_pending_drain() -> None:
    """A pending symbol with no drain scheduled must be re-nudged."""
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._logger = MagicMock()
    runner._eval_gate_lock = threading.Lock()
    runner._last_global_eval_ts = time.monotonic() - 120.0
    runner._running = True
    runner._entry_eval_shutdown = False
    runner._entry_eval_active = False
    runner._entry_eval_drain_scheduled = False
    runner._entry_eval_drain_count = 3
    runner._runtime_loop_attached = True
    runner._pending_entry_eval_symbols = {"NFO:NIFTY26JUN24000CE"}
    runner._entry_eval_executor = None
    runner._runtime_readiness_recompute_callback = None
    runner._last_eval_ts = {}
    runner._last_periodic_eval_at_by_symbol = {}
    runner.start = lambda: None

    scheduled: list[bool] = []
    runner._schedule_entry_eval_drain = lambda: (scheduled.append(True), True)[1]

    runner._recover_strategy_eval_stall_once(time.monotonic())

    assert scheduled == [True], "stranded pending drain was not rescheduled"
    payload = runner._logger.warning.call_args[1]["extra"]
    assert payload["event"] == "STRATEGY_EVAL_STALL_WORKER_STATE"
    assert payload["start_was_noop"] is True
    assert payload["drain_rescheduled"] is True
    assert payload["pending_entry_eval"] == ["NFO:NIFTY26JUN24000CE"]


def test_recovery_does_not_reschedule_when_a_drain_is_already_active() -> None:
    """No duplicate drain when one is already running."""
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._logger = MagicMock()
    runner._eval_gate_lock = threading.Lock()
    runner._last_global_eval_ts = time.monotonic() - 120.0
    runner._running = True
    runner._entry_eval_shutdown = False
    runner._entry_eval_active = True          # already draining
    runner._entry_eval_drain_scheduled = False
    runner._entry_eval_drain_count = 1
    runner._runtime_loop_attached = True
    runner._pending_entry_eval_symbols = {"NFO:NIFTY26JUN24000CE"}
    runner._entry_eval_executor = None
    runner._runtime_readiness_recompute_callback = None
    runner._last_eval_ts = {}
    runner._last_periodic_eval_at_by_symbol = {}
    runner.start = lambda: None

    scheduled: list[bool] = []
    runner._schedule_entry_eval_drain = lambda: (scheduled.append(True), True)[1]

    runner._recover_strategy_eval_stall_once(time.monotonic())

    assert scheduled == []
    assert runner._logger.warning.call_args[1]["extra"]["drain_rescheduled"] is False
