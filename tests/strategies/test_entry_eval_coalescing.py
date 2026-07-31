"""Regression tests for bounded new-entry evaluation coalescing.

Verifies StrategyRunner._on_tick_safe no longer runs the heavy phase9
_on_tick body inline for OPTION_CANDIDATE/UNDERLYING routes, and that the
bounded async drain (_drain_pending_entry_evaluations) preserves latest-state
evaluation, fairness across symbols, mid-evaluation reschedule, exception
isolation, and never delays protective/position-management routes.

The underlying-trigger route (NSE:NIFTY registered in
_trigger_candidate_symbols) is used as the "real evaluation reaches
strategy_manager.generate_signal" fixture, since it is the one already
proven to work end-to-end by test_nifty_underlying_reaches_strategy_manager
in test_runner_symbol_role_gate.py. Tests that only need to verify the
coalescing contract itself (fairness, reschedule-once, exception isolation)
spy on _evaluate_entry_from_latest_state directly instead of depending on
deeper phase9 gating internals, which are out of scope for this PR.
"""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nifty_scalper_bot.strategies.runner import EntryEvaluationRoute
from nifty_scalper_bot.strategies.signal_generator import Signal
from tests.strategies.test_runner_symbol_role_gate import _build_phase9_runner

UNDERLYING_SYMBOL = "NSE:NIFTY"


def _run_loop_in_thread():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    return loop, thread


def _stop_loop(loop, thread):
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=2.0)
    loop.close()


def _wait_until(predicate, *, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def _underlying_runner(monkeypatch):
    runner_obj, strategy_manager, risk_manager, order_manager, selected_ce = (
        _build_phase9_runner(monkeypatch)
    )
    runner_obj._trigger_candidate_symbols = {UNDERLYING_SYMBOL}
    assert (
        runner_obj._entry_evaluation_route(UNDERLYING_SYMBOL)
        == EntryEvaluationRoute.UNDERLYING
    )
    return runner_obj, strategy_manager, risk_manager, order_manager, selected_ce


def test_runner_tick_notification_does_not_run_strategy_inline(monkeypatch):
    """Test A: the tick callback (_on_tick_safe) must return quickly for an
    UNDERLYING-routed candidate and must not run the heavy evaluator inline."""
    runner_obj, strategy_manager, _risk, _order, selected_ce = _underlying_runner(
        monkeypatch
    )
    started = threading.Event()
    finished = threading.Event()

    def _slow_generate_signal(symbol, price, trace_id=None):
        started.set()
        time.sleep(0.12)
        finished.set()
        return Signal(
            "BUY", selected_ce, 75, 0.9, "slow_signal", 90.0, 120.0, metadata={}
        )

    strategy_manager.generate_signal = Mock(side_effect=_slow_generate_signal)

    loop, thread = _run_loop_in_thread()
    runner_obj._main_loop = loop
    try:
        callback_start = time.perf_counter()
        runner_obj._on_tick_safe(
            {
                "symbol": UNDERLYING_SYMBOL,
                "last_price": 24000.0,
                "timestamp": time.time(),
            }
        )
        callback_duration_ms = (time.perf_counter() - callback_start) * 1000.0

        assert callback_duration_ms < 40.0
        assert not finished.is_set()

        assert _wait_until(lambda: started.is_set())
        assert _wait_until(lambda: finished.is_set())
    finally:
        _stop_loop(loop, thread)


def test_multiple_ticks_coalesce_to_one_latest_entry_evaluation(monkeypatch):
    """Test B: a burst of ticks for the same evaluation key must not queue
    one evaluation per tick -- pending stays bounded and the drain reads the
    latest generation rather than replaying every intermediate tick."""
    runner_obj, _strategy_manager, _risk, _order, _selected_ce = _underlying_runner(
        monkeypatch
    )
    seen_generations: list[int] = []
    original_eval = runner_obj._evaluate_entry_from_latest_state
    release = threading.Event()

    def _capture(symbol, *, trace_id=None):
        with runner_obj._eval_gate_lock:
            seen_generations.append(
                runner_obj._entry_eval_generation_by_symbol[symbol]
            )
        release.wait(timeout=2.0)
        return None

    runner_obj._evaluate_entry_from_latest_state = _capture

    loop, thread = _run_loop_in_thread()
    runner_obj._main_loop = loop
    try:
        for i in range(30):
            runner_obj._last_eval_ts[UNDERLYING_SYMBOL] = 0.0
            runner_obj._on_tick_safe(
                {
                    "symbol": UNDERLYING_SYMBOL,
                    "last_price": 24000.0 + i,
                    "timestamp": time.time(),
                }
            )
        assert _wait_until(lambda: len(seen_generations) >= 1)
        with runner_obj._eval_gate_lock:
            pending_count = len(runner_obj._pending_entry_eval_symbols)
            latest_generation = runner_obj._entry_eval_generation_by_symbol[
                UNDERLYING_SYMBOL
            ]
        # 30 ticks must not produce 30 queued evaluations.
        assert pending_count <= 1
        assert latest_generation == 30
        release.set()
        # The drain may run once (burst fully coalesced before it started) or
        # a small bounded number of times (if it started mid-burst and had to
        # pick up newer generations) -- both are correct. What must hold
        # regardless of this scheduling race: nowhere near 30 replayed
        # evaluations, and the final evaluation reflects the latest state.
        assert _wait_until(
            lambda: len(seen_generations) >= 1
            and seen_generations[-1] == 30,
            timeout=2.0,
        )
        assert len(seen_generations) <= 5
    finally:
        runner_obj._evaluate_entry_from_latest_state = original_eval
        _stop_loop(loop, thread)


def test_coalesced_entry_drain_processes_each_pending_symbol_once(monkeypatch):
    """Test C: two independent pending evaluation keys are each evaluated
    exactly once by the drain; one busy key does not starve the other."""
    runner_obj, _strategy_manager, _risk, _order, _selected_ce = _underlying_runner(
        monkeypatch
    )
    other_symbol = "NFO:NIFTY26JUNFUT"
    runner_obj._trigger_candidate_symbols.add(other_symbol)
    assert (
        runner_obj._entry_evaluation_route(other_symbol)
        == EntryEvaluationRoute.UNDERLYING
    )
    runner_obj._last_tick[other_symbol] = {
        "symbol": other_symbol,
        "last_price": 24010.0,
        "timestamp": time.time(),
    }

    seen: list[str] = []

    def _capture(symbol, *, trace_id=None):
        seen.append(symbol)
        return None

    runner_obj._evaluate_entry_from_latest_state = _capture

    loop, thread = _run_loop_in_thread()
    runner_obj._main_loop = loop
    try:
        runner_obj._on_tick_safe(
            {
                "symbol": UNDERLYING_SYMBOL,
                "last_price": 24000.0,
                "timestamp": time.time(),
            }
        )
        runner_obj._on_tick_safe(
            {"symbol": other_symbol, "last_price": 24010.0, "timestamp": time.time()}
        )
        assert _wait_until(lambda: len(seen) >= 2, timeout=2.0)
        assert seen.count(UNDERLYING_SYMBOL) == 1
        assert seen.count(other_symbol) == 1
    finally:
        _stop_loop(loop, thread)


def test_new_generation_during_evaluation_reschedules_once(monkeypatch):
    """Test D: a newer tick arriving mid-evaluation must be evaluated once
    more after the in-flight evaluation completes -- no unbounded recursion,
    and the pending set drains to empty."""
    runner_obj, _strategy_manager, _risk, _order, _selected_ce = _underlying_runner(
        monkeypatch
    )
    call_count = {"n": 0}
    in_first_call = threading.Event()
    release_first_call = threading.Event()

    def _capture(symbol, *, trace_id=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            in_first_call.set()
            release_first_call.wait(timeout=2.0)
        return None

    runner_obj._evaluate_entry_from_latest_state = _capture

    loop, thread = _run_loop_in_thread()
    runner_obj._main_loop = loop
    try:
        runner_obj._on_tick_safe(
            {
                "symbol": UNDERLYING_SYMBOL,
                "last_price": 24000.0,
                "timestamp": time.time(),
            }
        )
        assert _wait_until(lambda: in_first_call.is_set())
        # Inject a newer tick while the first evaluation is still in flight.
        runner_obj._last_eval_ts[UNDERLYING_SYMBOL] = 0.0
        runner_obj._on_tick_safe(
            {
                "symbol": UNDERLYING_SYMBOL,
                "last_price": 24001.0,
                "timestamp": time.time(),
            }
        )
        release_first_call.set()
        assert _wait_until(lambda: call_count["n"] >= 2, timeout=2.0)
        assert call_count["n"] == 2
        assert _wait_until(
            lambda: not runner_obj._pending_entry_eval_symbols, timeout=2.0
        )
        with runner_obj._eval_gate_lock:
            assert not runner_obj._entry_eval_active
    finally:
        _stop_loop(loop, thread)


def test_position_management_route_runs_only_protection_inline(monkeypatch):
    """Test F: a POSITION_MANAGEMENT-routed symbol (open position) must run
    ONLY the extracted protection helper, synchronously and immediately, even
    while an entry evaluation is busy on the worker. The full heavy _on_tick
    must not be invoked, and no entry-eval work may be scheduled for it."""
    runner_obj, strategy_manager, _risk, _order, _selected_ce = _underlying_runner(
        monkeypatch
    )
    busy = threading.Event()
    release_busy = threading.Event()

    def _slow_generate_signal(symbol, price, trace_id=None):
        busy.set()
        release_busy.wait(timeout=2.0)
        return None

    strategy_manager.generate_signal = Mock(side_effect=_slow_generate_signal)

    open_symbol = "NFO:NIFTY26JUN24100CE"
    runner_obj._position_manager.has_open_position = lambda s: s == open_symbol
    assert (
        runner_obj._entry_evaluation_route(open_symbol)
        == EntryEvaluationRoute.POSITION_MANAGEMENT
    )

    protection_calls: list[str] = []

    def _must_not_run(symbol, tick):
        raise AssertionError(
            f"heavy _on_tick must not run inline for position route: {symbol}"
        )

    loop, thread = _run_loop_in_thread()
    runner_obj._main_loop = loop
    try:
        runner_obj._on_tick_safe(
            {
                "symbol": UNDERLYING_SYMBOL,
                "last_price": 24000.0,
                "timestamp": time.time(),
            }
        )
        assert _wait_until(lambda: busy.is_set())

        # Only now forbid the heavy body and start recording protection, so
        # the underlying evaluation above (already on the worker) is
        # unaffected and we isolate the position-route call.
        runner_obj._on_tick = _must_not_run
        runner_obj._handle_position_tick_protection = (
            lambda symbol, tick: protection_calls.append(symbol)
        )
        pending_before = set(runner_obj._pending_entry_eval_symbols)

        exit_start = time.perf_counter()
        runner_obj._on_tick_safe(
            {"symbol": open_symbol, "last_price": 50.0, "timestamp": time.time()}
        )
        exit_duration_ms = (time.perf_counter() - exit_start) * 1000.0

        # Protection ran inline, immediately, despite the busy worker.
        assert protection_calls == [open_symbol]
        assert exit_duration_ms < 100.0
        # The position route must not enqueue any entry-evaluation work.
        with runner_obj._eval_gate_lock:
            assert open_symbol not in runner_obj._pending_entry_eval_symbols
            assert set(runner_obj._pending_entry_eval_symbols) == pending_before
        release_busy.set()
    finally:
        release_busy.set()
        _stop_loop(loop, thread)


def test_position_tick_protection_not_executed_twice(monkeypatch):
    """Test G: one tick must produce exactly one protective lifecycle update,
    even though protection runs at ingestion and the heavy body later runs on
    the worker for the same tick."""
    runner_obj, _strategy_manager, _risk, _order, _selected_ce = _underlying_runner(
        monkeypatch
    )
    bracket_ticks: list[tuple[str, float, bool]] = []
    runner_obj._bracket_manager = SimpleNamespace(
        on_tick=lambda sym, ltp, ts, **kwargs: bracket_ticks.append(
            (sym, ltp, kwargs.get("defer_submission") is True)
        )
    )

    loop, thread = _run_loop_in_thread()
    runner_obj._main_loop = loop
    try:
        runner_obj._on_tick_safe(
            {
                "symbol": UNDERLYING_SYMBOL,
                "last_price": 24000.0,
                "timestamp": time.time(),
            }
        )
        # Let the deferred heavy evaluation complete on the worker.
        assert _wait_until(
            lambda: not runner_obj._pending_entry_eval_symbols
            and not runner_obj._entry_eval_active,
            timeout=3.0,
        )
        time.sleep(0.05)
        assert len(bracket_ticks) == 1
        assert bracket_ticks[0][0] == UNDERLYING_SYMBOL
        assert bracket_ticks[0][2] is True
    finally:
        _stop_loop(loop, thread)


def test_slow_entry_evaluation_does_not_block_main_event_loop(monkeypatch):
    """Test A (decisive): while the heavy evaluation is running, the main
    event loop must keep making progress, and the evaluation must execute on
    a dedicated worker thread -- not the loop thread."""
    runner_obj, _strategy_manager, _risk, _order, _selected_ce = _underlying_runner(
        monkeypatch
    )
    eval_thread_names: list[str] = []
    eval_started = threading.Event()
    eval_done = threading.Event()

    def _slow_eval(symbol, *, trace_id=None):
        eval_thread_names.append(threading.current_thread().name)
        eval_started.set()
        time.sleep(0.2)
        eval_done.set()

    runner_obj._evaluate_entry_from_latest_state = _slow_eval

    loop, thread = _run_loop_in_thread()
    runner_obj._main_loop = loop
    loop_thread_name = None
    heartbeat = {"n": 0}

    async def _heartbeat():
        nonlocal loop_thread_name
        loop_thread_name = threading.current_thread().name
        while not eval_done.is_set():
            heartbeat["n"] += 1
            await asyncio.sleep(0.01)

    try:
        asyncio.run_coroutine_threadsafe(_heartbeat(), loop)
        runner_obj._on_tick_safe(
            {
                "symbol": UNDERLYING_SYMBOL,
                "last_price": 24000.0,
                "timestamp": time.time(),
            }
        )
        assert _wait_until(lambda: eval_started.is_set(), timeout=2.0)
        beats_at_start = heartbeat["n"]
        assert _wait_until(lambda: eval_done.is_set(), timeout=3.0)

        # The loop kept running throughout the 200 ms evaluation.
        assert heartbeat["n"] - beats_at_start >= 5

        # And the evaluation ran on the dedicated worker, not the loop thread.
        assert eval_thread_names
        assert all(n.startswith("nifty-entry-eval") for n in eval_thread_names)
        assert loop_thread_name not in eval_thread_names
        # Exactly one worker thread was used.
        assert len(set(eval_thread_names)) == 1
    finally:
        eval_done.set()
        _stop_loop(loop, thread)


def test_entry_eval_drain_processes_one_snapshot_per_invocation(monkeypatch):
    """Test C: each drain invocation handles exactly one captured batch. A
    newer generation arriving mid-batch is handled by a second drain, not by
    an unbounded `while True` inside the first."""
    runner_obj, _strategy_manager, _risk, _order, _selected_ce = _underlying_runner(
        monkeypatch
    )
    in_first = threading.Event()
    release_first = threading.Event()
    calls = {"n": 0}

    def _capture(symbol, *, trace_id=None):
        calls["n"] += 1
        if calls["n"] == 1:
            in_first.set()
            release_first.wait(timeout=2.0)

    runner_obj._evaluate_entry_from_latest_state = _capture

    loop, thread = _run_loop_in_thread()
    runner_obj._main_loop = loop
    try:
        drains_before = runner_obj._entry_eval_drain_count
        runner_obj._on_tick_safe(
            {
                "symbol": UNDERLYING_SYMBOL,
                "last_price": 24000.0,
                "timestamp": time.time(),
            }
        )
        assert _wait_until(lambda: in_first.is_set(), timeout=2.0)
        runner_obj._last_eval_ts[UNDERLYING_SYMBOL] = 0.0
        runner_obj._on_tick_safe(
            {
                "symbol": UNDERLYING_SYMBOL,
                "last_price": 24001.0,
                "timestamp": time.time(),
            }
        )
        release_first.set()
        assert _wait_until(lambda: calls["n"] >= 2, timeout=3.0)
        assert _wait_until(
            lambda: not runner_obj._pending_entry_eval_symbols
            and not runner_obj._entry_eval_active,
            timeout=3.0,
        )
        # Exactly two bounded drain invocations, not one monopolising loop.
        assert runner_obj._entry_eval_drain_count - drains_before == 2
        assert calls["n"] == 2
    finally:
        release_first.set()
        _stop_loop(loop, thread)


def test_continuous_busy_symbol_does_not_starve_other_pending_symbol(monkeypatch):
    """Test D: a continuously-ticking symbol must not starve another pending
    symbol -- B is evaluated within a bounded number of drain batches."""
    runner_obj, _strategy_manager, _risk, _order, _selected_ce = _underlying_runner(
        monkeypatch
    )
    busy_symbol = UNDERLYING_SYMBOL
    other_symbol = "NFO:NIFTY26JUNFUT"
    runner_obj._trigger_candidate_symbols.add(other_symbol)
    runner_obj._last_tick[other_symbol] = {
        "symbol": other_symbol,
        "last_price": 24010.0,
        "timestamp": time.time(),
    }
    seen: list[str] = []
    stop_feeding = threading.Event()

    def _capture(symbol, *, trace_id=None):
        seen.append(symbol)
        time.sleep(0.002)

    runner_obj._evaluate_entry_from_latest_state = _capture

    loop, thread = _run_loop_in_thread()
    runner_obj._main_loop = loop
    try:
        runner_obj._on_tick_safe(
            {"symbol": other_symbol, "last_price": 24010.0, "timestamp": time.time()}
        )
        deadline = time.time() + 1.5
        while time.time() < deadline and not stop_feeding.is_set():
            runner_obj._last_eval_ts[busy_symbol] = 0.0
            runner_obj._on_tick_safe(
                {
                    "symbol": busy_symbol,
                    "last_price": 24000.0,
                    "timestamp": time.time(),
                }
            )
            if other_symbol in seen:
                stop_feeding.set()
            time.sleep(0.002)

        assert other_symbol in seen, "continuously busy symbol starved the other"
        # The busy symbol stayed coalesced: far fewer evaluations than ticks.
        assert seen.count(busy_symbol) < 200
    finally:
        _stop_loop(loop, thread)


def test_context_only_tick_does_not_run_heavy_on_tick_inline(monkeypatch):
    """Test E: a CONTEXT_ONLY symbol must not run the heavy _on_tick body
    inline on the ingestion path -- it is coalesced onto the worker."""
    runner_obj, _strategy_manager, _risk, _order, _selected_ce = _underlying_runner(
        monkeypatch
    )
    context_symbol = "NFO:NIFTY26JUN24100CE"
    assert (
        runner_obj._entry_evaluation_route(context_symbol)
        == EntryEvaluationRoute.CONTEXT_ONLY
    )
    runner_obj._last_tick[context_symbol] = {
        "symbol": context_symbol,
        "last_price": 55.0,
        "timestamp": time.time(),
    }
    ingress_thread = threading.current_thread().name
    eval_threads: list[str] = []

    def _capture(symbol, *, trace_id=None):
        eval_threads.append(threading.current_thread().name)

    runner_obj._evaluate_entry_from_latest_state = _capture

    def _must_not_run_inline(symbol, tick):
        raise AssertionError("heavy _on_tick ran inline for CONTEXT_ONLY")

    runner_obj._on_tick = _must_not_run_inline

    loop, thread = _run_loop_in_thread()
    runner_obj._main_loop = loop
    try:
        start = time.perf_counter()
        runner_obj._on_tick_safe(
            {"symbol": context_symbol, "last_price": 55.0, "timestamp": time.time()}
        )
        duration_ms = (time.perf_counter() - start) * 1000.0
        assert duration_ms < 40.0
        assert _wait_until(lambda: bool(eval_threads), timeout=2.0)
        assert all(n.startswith("nifty-entry-eval") for n in eval_threads)
        assert ingress_thread not in eval_threads
    finally:
        _stop_loop(loop, thread)


def test_runner_shutdown_stops_entry_eval_executor(monkeypatch):
    """Test I: shutdown terminates the worker, clears pending work, and no
    evaluation runs afterwards. No leaked nifty-entry-eval thread."""
    runner_obj, _strategy_manager, _risk, _order, _selected_ce = _underlying_runner(
        monkeypatch
    )
    calls: list[str] = []
    runner_obj._evaluate_entry_from_latest_state = lambda symbol, **_kw: calls.append(
        symbol
    )

    loop, thread = _run_loop_in_thread()
    runner_obj._main_loop = loop
    try:
        runner_obj._on_tick_safe(
            {
                "symbol": UNDERLYING_SYMBOL,
                "last_price": 24000.0,
                "timestamp": time.time(),
            }
        )
        assert _wait_until(lambda: bool(calls), timeout=2.0)

        runner_obj._shutdown_entry_eval_worker()
        calls.clear()

        # No new work may be accepted or evaluated after shutdown.
        runner_obj._last_eval_ts[UNDERLYING_SYMBOL] = 0.0
        runner_obj._on_tick_safe(
            {
                "symbol": UNDERLYING_SYMBOL,
                "last_price": 24002.0,
                "timestamp": time.time(),
            }
        )
        time.sleep(0.15)
        assert calls == []
        with runner_obj._eval_gate_lock:
            assert not runner_obj._pending_entry_eval_symbols
            assert not runner_obj._entry_eval_active
    finally:
        _stop_loop(loop, thread)

    # This runner's own worker thread(s) terminated -- no leak from shutdown.
    for worker in getattr(runner_obj._entry_eval_executor, "_threads", ()):
        worker.join(timeout=2.0)
        assert not worker.is_alive()


def test_entry_evaluation_exception_does_not_stop_future_drains(monkeypatch):
    """Test G: one symbol's evaluation raising must not stop other pending
    keys, and must not leave the drain permanently active/stuck."""
    runner_obj, _strategy_manager, _risk, _order, _selected_ce = _underlying_runner(
        monkeypatch
    )
    other_symbol = "NFO:NIFTY26JUNFUT"
    runner_obj._trigger_candidate_symbols.add(other_symbol)
    runner_obj._last_tick[other_symbol] = {
        "symbol": other_symbol,
        "last_price": 24010.0,
        "timestamp": time.time(),
    }

    seen: list[str] = []

    def _capture(symbol, *, trace_id=None):
        seen.append(symbol)
        if symbol == UNDERLYING_SYMBOL:
            raise RuntimeError("boom")
        return None

    runner_obj._evaluate_entry_from_latest_state = _capture

    loop, thread = _run_loop_in_thread()
    runner_obj._main_loop = loop
    try:
        runner_obj._on_tick_safe(
            {
                "symbol": UNDERLYING_SYMBOL,
                "last_price": 24000.0,
                "timestamp": time.time(),
            }
        )
        runner_obj._on_tick_safe(
            {"symbol": other_symbol, "last_price": 24010.0, "timestamp": time.time()}
        )
        assert _wait_until(lambda: other_symbol in seen, timeout=2.0)
        assert UNDERLYING_SYMBOL in seen

        with runner_obj._eval_gate_lock:
            assert not runner_obj._entry_eval_active

        # A later tick for the failed symbol must still be evaluable.
        seen.clear()

        def _capture_ok(symbol, *, trace_id=None):
            seen.append(symbol)
            return None

        runner_obj._evaluate_entry_from_latest_state = _capture_ok
        runner_obj._last_eval_ts[UNDERLYING_SYMBOL] = 0.0
        runner_obj._on_tick_safe(
            {
                "symbol": UNDERLYING_SYMBOL,
                "last_price": 24005.0,
                "timestamp": time.time(),
            }
        )
        assert _wait_until(lambda: UNDERLYING_SYMBOL in seen, timeout=2.0)
    finally:
        _stop_loop(loop, thread)


def test_entry_eval_burst_invariants_bounded_and_no_duplicates(monkeypatch):
    """Test J (CI-sized burst harness): a large burst of ticks for one
    evaluation key must produce materially fewer evaluations than ticks,
    drain to an empty pending set, and never run two drains concurrently.
    """
    runner_obj, strategy_manager, _risk, _order, _selected_ce = _underlying_runner(
        monkeypatch
    )
    n_ticks = 2000
    evaluation_count = {"n": 0}
    max_concurrent = {"n": 0}
    concurrent_now = {"n": 0}
    concurrency_lock = threading.Lock()

    def _capture(symbol, *, trace_id=None):
        with concurrency_lock:
            concurrent_now["n"] += 1
            max_concurrent["n"] = max(max_concurrent["n"], concurrent_now["n"])
        evaluation_count["n"] += 1
        time.sleep(0.001)
        with concurrency_lock:
            concurrent_now["n"] -= 1

    runner_obj._evaluate_entry_from_latest_state = _capture

    loop, thread = _run_loop_in_thread()
    runner_obj._main_loop = loop
    try:
        start = time.perf_counter()
        for i in range(n_ticks):
            runner_obj._last_eval_ts[UNDERLYING_SYMBOL] = 0.0
            runner_obj._on_tick_safe(
                {
                    "symbol": UNDERLYING_SYMBOL,
                    "last_price": 24000.0 + (i % 7),
                    "timestamp": time.time(),
                }
            )
        ingest_duration_s = time.perf_counter() - start

        assert _wait_until(
            lambda: not runner_obj._pending_entry_eval_symbols, timeout=5.0
        )
        with runner_obj._eval_gate_lock:
            assert not runner_obj._entry_eval_active

        # Materially fewer evaluations than ticks submitted (coalescing did
        # its job), never concurrent (single-flight drain), and pending
        # drains to empty (no unbounded backlog).
        assert evaluation_count["n"] < n_ticks // 10
        assert max_concurrent["n"] == 1
        assert ingest_duration_s < 2.0
    finally:
        _stop_loop(loop, thread)


def test_subscribe_symbol_registers_single_lightweight_callback(monkeypatch):
    """Wiring test: production subscribe path registers exactly one
    tick-notification callback per symbol (on_datahub_tick), and it is not
    the heavy evaluator (_on_tick) registered directly."""
    runner_obj, _strategy_manager, _risk, _order, _selected_ce = _underlying_runner(
        monkeypatch
    )

    class _FakeDataHub:
        def __init__(self):
            self.subscriptions: list[tuple[str, object]] = []

        def subscribe_ticks(self, symbol, callback):
            self.subscriptions.append((symbol, callback))

    fake_hub = _FakeDataHub()
    runner_obj._data_hub = fake_hub
    runner_obj._datahub_registered_symbols = set()

    runner_obj._subscribe_symbol(UNDERLYING_SYMBOL)
    runner_obj._subscribe_symbol(UNDERLYING_SYMBOL)  # idempotent re-subscribe

    matching = [
        (sym, cb) for sym, cb in fake_hub.subscriptions if sym == UNDERLYING_SYMBOL
    ]
    assert len(matching) == 1
    registered_symbol, registered_callback = matching[0]
    assert registered_callback == runner_obj.on_datahub_tick
    assert registered_callback != runner_obj._on_tick


def test_entry_eval_scheduler_unavailable_never_runs_inline(monkeypatch):
    """With no running loop the callback must still return immediately, must
    NOT run the evaluation inline (no asyncio.run fallback), and must keep the
    symbol pending so a later tick retries scheduling."""
    runner_obj, _strategy_manager, _risk, _order, _selected_ce = _underlying_runner(
        monkeypatch
    )

    def _must_not_run(symbol, **_kw):
        raise AssertionError("evaluation must not run inline without a loop")

    runner_obj._evaluate_entry_from_latest_state = _must_not_run
    runner_obj._on_tick = lambda symbol, tick: (_ for _ in ()).throw(
        AssertionError("heavy _on_tick must not run inline without a loop")
    )
    runner_obj._main_loop = None
    with runner_obj._eval_gate_lock:
        runner_obj._entry_eval_active = False
        runner_obj._entry_eval_drain_scheduled = False
        runner_obj._pending_entry_eval_symbols.clear()

    start = time.perf_counter()
    runner_obj._on_tick_safe(
        {"symbol": UNDERLYING_SYMBOL, "last_price": 24000.0, "timestamp": time.time()}
    )
    duration_ms = (time.perf_counter() - start) * 1000.0

    assert duration_ms < 40.0
    with runner_obj._eval_gate_lock:
        # Pending state preserved for a later retry; nothing left active.
        assert UNDERLYING_SYMBOL in runner_obj._pending_entry_eval_symbols
        assert not runner_obj._entry_eval_active
        assert not runner_obj._entry_eval_drain_scheduled


def test_entry_eval_worker_skips_symbol_that_became_position_managed(monkeypatch):
    """1H: if a symbol acquires an open position between being marked pending
    and the worker reaching it, entry evaluation must be skipped -- protection
    owns that symbol, and an entry must never race an open position."""
    runner_obj, _strategy_manager, _risk, _order, selected_ce = _underlying_runner(
        monkeypatch
    )
    runner_obj._last_tick[selected_ce] = {
        "symbol": selected_ce,
        "last_price": 100.0,
        "timestamp": time.time(),
    }
    on_tick_calls: list[str] = []
    runner_obj._on_tick = lambda symbol, tick: on_tick_calls.append(symbol)

    # Sanity: it would normally be evaluated as an entry candidate.
    assert (
        runner_obj._entry_evaluation_route(selected_ce)
        == EntryEvaluationRoute.OPTION_CANDIDATE
    )

    # A position opens before the worker gets to it.
    runner_obj._position_manager.has_open_position = lambda s: s == selected_ce
    runner_obj._evaluate_entry_from_latest_state(selected_ce)

    assert on_tick_calls == []


# ==================== EXPLICIT RUNTIME-LOOP ATTACHMENT ====================


def test_background_tick_uses_explicit_runtime_loop(monkeypatch):
    """Test 1: a tick delivered from a background thread must be drained on
    the explicitly attached live loop B, never on a dormant loop A."""
    runner_obj, _sm, _r, _o, _ce = _underlying_runner(monkeypatch)

    dormant_loop = asyncio.new_event_loop()
    dormant_calls: list[object] = []
    dormant_loop.call_soon_threadsafe = lambda cb, *a: dormant_calls.append(cb)
    runner_obj._main_loop = dormant_loop

    loop_b, thread_b = _run_loop_in_thread()
    drain_loops: list[object] = []
    eval_threads: list[str] = []

    def _capture(symbol, *, trace_id=None):
        drain_loops.append(asyncio.get_event_loop_policy())
        eval_threads.append(threading.current_thread().name)

    runner_obj._evaluate_entry_from_latest_state = _capture

    try:
        fut = asyncio.run_coroutine_threadsafe(
            _attach_from_loop(runner_obj), loop_b
        )
        fut.result(timeout=2.0)
        assert runner_obj._main_loop is loop_b

        # Deliver the tick from a separate background thread.
        feeder = threading.Thread(
            target=runner_obj._on_tick_safe,
            args=(
                {
                    "symbol": UNDERLYING_SYMBOL,
                    "last_price": 24000.0,
                    "timestamp": time.time(),
                },
            ),
        )
        feeder.start()
        feeder.join(timeout=2.0)

        assert _wait_until(lambda: len(eval_threads) >= 1, timeout=3.0)
        assert _wait_until(
            lambda: not runner_obj._pending_entry_eval_symbols, timeout=3.0
        )
        assert len(eval_threads) == 1
        assert eval_threads[0].startswith("nifty-entry-eval")
        assert dormant_calls == []
        assert runner_obj._main_loop is loop_b
    finally:
        _stop_loop(loop_b, thread_b)
        dormant_loop.close()


async def _attach_from_loop(runner_obj) -> None:
    runner_obj.attach_runtime_loop(asyncio.get_running_loop())


def test_nonrunning_runtime_loop_cannot_be_attached(monkeypatch):
    """Test 2: an open but dormant loop must be rejected."""
    runner_obj, _sm, _r, _o, _ce = _underlying_runner(monkeypatch)
    previous = runner_obj._main_loop
    dormant = asyncio.new_event_loop()
    try:
        with pytest.raises(RuntimeError, match="running"):
            runner_obj.attach_runtime_loop(dormant)
        assert runner_obj._main_loop is previous
        assert not runner_obj._runtime_loop_attached
    finally:
        dormant.close()


def test_closed_runtime_loop_cannot_be_attached(monkeypatch):
    """Test 3: a closed loop must be rejected."""
    runner_obj, _sm, _r, _o, _ce = _underlying_runner(monkeypatch)
    previous = runner_obj._main_loop
    closed = asyncio.new_event_loop()
    closed.close()
    with pytest.raises(RuntimeError, match="closed"):
        runner_obj.attach_runtime_loop(closed)
    assert runner_obj._main_loop is previous
    assert not runner_obj._runtime_loop_attached


def test_schedule_entry_eval_drain_returns_false_for_nonrunning_loop(monkeypatch):
    """Test 4: a directly-assigned (never attached) dormant loop must not be
    scheduled onto, and pending work must survive."""
    runner_obj, _sm, _r, _o, _ce = _underlying_runner(monkeypatch)

    def _must_not_run(symbol, **_kw):
        raise AssertionError("evaluator must not run for a dormant loop")

    runner_obj._evaluate_entry_from_latest_state = _must_not_run

    dormant = asyncio.new_event_loop()
    queued: list[object] = []
    dormant.call_soon_threadsafe = lambda cb, *a: queued.append(cb)
    runner_obj._main_loop = dormant
    runner_obj._runtime_loop_attached = False
    with runner_obj._eval_gate_lock:
        runner_obj._pending_entry_eval_symbols.add(UNDERLYING_SYMBOL)
    try:
        assert runner_obj._schedule_entry_eval_drain() is False
        assert queued == []
        with runner_obj._eval_gate_lock:
            assert UNDERLYING_SYMBOL in runner_obj._pending_entry_eval_symbols
    finally:
        dormant.close()


def test_attach_runtime_loop_recovers_existing_pending_entry_eval(monkeypatch):
    """Test 5: work marked pending before any loop existed must be recovered
    exactly once when the authoritative loop is attached."""
    runner_obj, _sm, _r, _o, _ce = _underlying_runner(monkeypatch)
    seen: list[str] = []
    runner_obj._evaluate_entry_from_latest_state = lambda symbol, **_kw: seen.append(
        symbol
    )
    runner_obj._main_loop = None
    runner_obj._runtime_loop_attached = False

    runner_obj._on_tick_safe(
        {"symbol": UNDERLYING_SYMBOL, "last_price": 24000.0, "timestamp": time.time()}
    )
    with runner_obj._eval_gate_lock:
        assert UNDERLYING_SYMBOL in runner_obj._pending_entry_eval_symbols
    assert seen == []

    loop_b, thread_b = _run_loop_in_thread()
    try:
        fut = asyncio.run_coroutine_threadsafe(
            _attach_from_loop(runner_obj), loop_b
        )
        fut.result(timeout=2.0)
        assert _wait_until(lambda: seen == [UNDERLYING_SYMBOL], timeout=3.0)
        assert _wait_until(
            lambda: not runner_obj._pending_entry_eval_symbols, timeout=3.0
        )
        assert seen == [UNDERLYING_SYMBOL]
    finally:
        _stop_loop(loop_b, thread_b)
