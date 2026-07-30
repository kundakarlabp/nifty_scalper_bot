"""Health watchdog must not run per-tick on the synchronous tick path.

Post-#943 audit, section 4.2 (P0). _on_tick_safe() called _health_watchdog()
for every accepted tick. The watchdog is control plane, not data plane: it scans
active symbols, hydrates missing bars, repairs history and assesses WebSocket
recovery. Running it inside the synchronous callback made those costs
indivisible from tick ingestion -- measured 176-1088 ms single-callback stalls,
which the 50 ms drain budget cannot pre-empt because it is only checked BETWEEN
processed ticks.

Covers the audit's TDD cases 2 (watchdog cadence) and 5 (bounded synchronous
duration, asserted by call graph rather than wall clock).
"""

from __future__ import annotations

import asyncio
import threading
import time
from unittest.mock import MagicMock

from nifty_scalper_bot.strategies.runner import StrategyRunner


def _runner(*, interval: float = 5.0, loop=None) -> StrategyRunner:
    r = StrategyRunner.__new__(StrategyRunner)
    r._logger = MagicMock()
    r._main_loop = loop
    r._last_health_watchdog_ts = 0.0
    r._health_watchdog_inflight = False
    r._health_watchdog_interval_s = interval
    return r


def _run_loop_in_thread():
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    return loop, t


def _stop_loop(loop, thread):
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=2.0)
    loop.close()


def test_many_ticks_in_one_interval_cause_one_scan() -> None:
    """Audit TDD case 2: cadence, not tick frequency, drives scans."""
    r = _runner(interval=5.0)
    calls: list[int] = []
    r._health_watchdog = lambda: calls.append(1)

    for _ in range(200):
        r._dispatch_health_watchdog()

    assert len(calls) == 1, f"expected one scan per interval, got {len(calls)}"


def test_scan_runs_again_after_the_interval_elapses() -> None:
    """Cadence must not become a one-shot."""
    r = _runner(interval=0.01)
    calls: list[int] = []
    r._health_watchdog = lambda: calls.append(1)

    r._dispatch_health_watchdog()
    time.sleep(0.02)
    r._dispatch_health_watchdog()

    assert len(calls) == 2


def test_slow_watchdog_does_not_block_the_tick_caller() -> None:
    """Audit TDD case 5: the caller must not absorb watchdog duration."""
    loop, thread = _run_loop_in_thread()
    r = _runner(interval=0.0, loop=loop)
    started = threading.Event()
    finished = threading.Event()

    def _slow() -> None:
        started.set()
        time.sleep(0.4)
        finished.set()

    r._health_watchdog = _slow
    try:
        t0 = time.perf_counter()
        r._dispatch_health_watchdog()
        elapsed = time.perf_counter() - t0

        assert elapsed < 0.1, f"tick caller blocked for {elapsed:.3f}s"
        assert not finished.is_set(), "watchdog ran synchronously on the caller"

        deadline = time.time() + 3.0
        while not started.is_set() and time.time() < deadline:
            time.sleep(0.005)
        assert started.is_set(), "watchdog never dispatched to the loop"
    finally:
        finished.wait(timeout=2.0)
        _stop_loop(loop, thread)


def test_watchdog_runs_off_the_calling_thread() -> None:
    """Control-plane work must leave the market-data thread."""
    loop, thread = _run_loop_in_thread()
    r = _runner(interval=0.0, loop=loop)
    caller = threading.get_ident()
    seen: list[int] = []
    done = threading.Event()

    def _record() -> None:
        seen.append(threading.get_ident())
        done.set()

    r._health_watchdog = _record
    try:
        r._dispatch_health_watchdog()
        assert done.wait(timeout=3.0)
        assert seen and seen[0] != caller
    finally:
        _stop_loop(loop, thread)


def test_only_one_watchdog_pass_is_in_flight() -> None:
    """A long pass must not be re-entered by subsequent ticks."""
    loop, thread = _run_loop_in_thread()
    r = _runner(interval=0.0, loop=loop)
    entered = threading.Event()
    release = threading.Event()
    concurrent = {"now": 0, "max": 0}
    lock = threading.Lock()

    def _slow() -> None:
        with lock:
            concurrent["now"] += 1
            concurrent["max"] = max(concurrent["max"], concurrent["now"])
        entered.set()
        release.wait(timeout=3.0)
        with lock:
            concurrent["now"] -= 1

    r._health_watchdog = _slow
    try:
        r._dispatch_health_watchdog()
        assert entered.wait(timeout=3.0)
        for _ in range(20):
            r._dispatch_health_watchdog()
        release.set()
        time.sleep(0.1)
        assert concurrent["max"] == 1
    finally:
        release.set()
        _stop_loop(loop, thread)


def test_watchdog_failure_is_isolated_from_tick_ingestion() -> None:
    """Audit TDD case 6: a failing watchdog must not propagate to the caller."""
    loop, thread = _run_loop_in_thread()
    r = _runner(interval=0.0, loop=loop)
    done = threading.Event()

    def _boom() -> None:
        done.set()
        raise RuntimeError("watchdog exploded")

    r._health_watchdog = _boom
    try:
        r._dispatch_health_watchdog()   # must not raise
        assert done.wait(timeout=3.0)
        time.sleep(0.05)
        # In-flight flag released so later ticks are not permanently blocked.
        assert r._health_watchdog_inflight is False
    finally:
        _stop_loop(loop, thread)


def test_no_runtime_loop_runs_inline_rather_than_skipping_health() -> None:
    """Health checks must not be silently dropped when no loop is attached."""
    r = _runner(interval=0.0, loop=None)
    calls: list[int] = []
    r._health_watchdog = lambda: calls.append(1)

    r._dispatch_health_watchdog()

    assert calls == [1]


def test_tick_path_calls_the_dispatcher_not_the_watchdog_directly() -> None:
    """Guards against a regression back to per-tick synchronous scanning."""
    import inspect

    src = inspect.getsource(StrategyRunner._on_tick_safe)
    assert "self._dispatch_health_watchdog()" in src
    assert "self._health_watchdog()" not in src
