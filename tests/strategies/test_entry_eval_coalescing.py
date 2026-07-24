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
from unittest.mock import Mock

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


def test_protective_exit_not_blocked_by_busy_entry_evaluation(monkeypatch):
    """Test F: a POSITION_MANAGEMENT-routed symbol (open position) must run
    synchronously via _on_tick_safe even while an UNDERLYING evaluation is
    busy in the coalesced drain -- exits are never routed through the
    entry-eval queue."""
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

    on_tick_calls: list[str] = []
    original_on_tick = runner_obj._on_tick

    def _spy_on_tick(symbol, tick):
        on_tick_calls.append(symbol)
        return original_on_tick(symbol, tick)

    runner_obj._on_tick = _spy_on_tick

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

        exit_start = time.perf_counter()
        runner_obj._on_tick_safe(
            {"symbol": open_symbol, "last_price": 50.0, "timestamp": time.time()}
        )
        exit_duration_ms = (time.perf_counter() - exit_start) * 1000.0

        # The protective/position-management route must have run inline,
        # immediately, regardless of the busy entry-eval drain.
        assert exit_duration_ms < 100.0
        assert open_symbol in on_tick_calls
        release_busy.set()
    finally:
        _stop_loop(loop, thread)


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


def test_entry_eval_schedule_failure_falls_back_inline(monkeypatch):
    """A scheduling error (e.g. no running loop reachable) must not crash
    ingestion and must not silently drop the pending evaluation -- it runs
    inline as a bounded fallback instead."""
    runner_obj, _strategy_manager, _risk, _order, _selected_ce = _underlying_runner(
        monkeypatch
    )
    seen: list[str] = []
    runner_obj._evaluate_entry_from_latest_state = lambda symbol, **_kw: seen.append(
        symbol
    )
    runner_obj._main_loop = None

    runner_obj._on_tick_safe(
        {"symbol": UNDERLYING_SYMBOL, "last_price": 24000.0, "timestamp": time.time()}
    )

    assert UNDERLYING_SYMBOL in seen
    with runner_obj._eval_gate_lock:
        assert not runner_obj._entry_eval_active
        assert not runner_obj._entry_eval_drain_scheduled
