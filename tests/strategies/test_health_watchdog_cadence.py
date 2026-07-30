"""Health scans must run on cadence, not once per tick.

Post-#943 audit, section 4.2. _health_watchdog() was invoked for EVERY accepted
tick from inside _on_tick_safe, i.e. synchronously on the market-data tick
thread. It scans all active symbols and can perform bar hydration, historical
repair, MDM history ingestion and stale-symbol recovery assessment, producing
indivisible 176-1,088 ms tick callbacks.

The tick drain budget cannot pre-empt that: it is only checked BETWEEN
processed ticks, never inside a single callback.

Audit acceptance criterion covered here: "Health scans occur on configured
cadence, independent of tick frequency." Relocating the blocking history work
to the asynchronous control path is the remaining part of that correction and
is NOT covered by these tests.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

from nifty_scalper_bot.strategies.runner import StrategyRunner


def _runner(interval: float = 5.0) -> StrategyRunner:
    r = StrategyRunner.__new__(StrategyRunner)
    r._logger = MagicMock()
    r._health_watchdog_lock = threading.Lock()
    r._health_watchdog_last_run = 0.0
    r._health_watchdog_running = False
    r._health_watchdog_skipped = 0
    r._health_watchdog_interval_s = interval
    return r


def test_many_ticks_in_one_interval_produce_one_scan() -> None:
    """AUDIT TDD CASE 2: many ticks must not cause many full scans."""
    r = _runner(interval=60.0)
    calls: list[int] = []
    r._health_watchdog = lambda: calls.append(1)

    for _ in range(50):
        r._run_health_watchdog_on_cadence()

    assert len(calls) == 1, f"expected one scan per interval, got {len(calls)}"


def test_scan_runs_again_after_the_interval_elapses() -> None:
    """Cadence must not become a permanent suppression."""
    r = _runner(interval=0.05)
    calls: list[int] = []
    r._health_watchdog = lambda: calls.append(1)

    r._run_health_watchdog_on_cadence()
    time.sleep(0.06)
    r._run_health_watchdog_on_cadence()

    assert len(calls) == 2


def test_overlapping_invocation_is_refused() -> None:
    """A long scan must not be re-entered by a subsequent tick."""
    r = _runner(interval=0.0)
    entered = threading.Event()
    release = threading.Event()
    calls: list[int] = []

    def _slow() -> None:
        calls.append(1)
        entered.set()
        release.wait(timeout=2.0)

    r._health_watchdog = _slow

    t = threading.Thread(target=r._run_health_watchdog_on_cadence)
    t.start()
    assert entered.wait(timeout=2.0)

    # A second tick arrives while the first scan is still running.
    r._run_health_watchdog_on_cadence()
    assert len(calls) == 1, "overlapping scan was not refused"

    release.set()
    t.join(timeout=2.0)


def test_running_flag_is_cleared_even_when_the_scan_raises() -> None:
    """A failed scan must not permanently wedge health scanning."""
    r = _runner(interval=0.0)

    def _boom() -> None:
        raise RuntimeError("scan failed")

    r._health_watchdog = _boom

    try:
        r._run_health_watchdog_on_cadence()
    except RuntimeError:
        pass

    assert r._health_watchdog_running is False

    calls: list[int] = []
    r._health_watchdog = lambda: calls.append(1)
    r._run_health_watchdog_on_cadence()
    assert calls == [1]


def test_skipped_tick_count_is_reported_for_slow_scans(caplog) -> None:
    """Operators need to see how many ticks a slow scan displaced."""
    import logging

    r = _runner(interval=0.0)

    def _slow() -> None:
        time.sleep(0.12)

    r._health_watchdog = _slow
    with caplog.at_level(logging.WARNING):
        r._run_health_watchdog_on_cadence()

    recs = [
        rec for rec in caplog.records
        if getattr(rec, "event", None) == "HEALTH_WATCHDOG_SLOW"
    ]
    assert recs, "slow scan was not reported"
    assert recs[-1].duration_ms >= 100.0
    assert hasattr(recs[-1], "ticks_skipped")


def test_tick_path_calls_the_cadence_wrapper_not_the_scan_directly() -> None:
    """_on_tick_safe must not invoke the full scan per tick."""
    import inspect

    src = inspect.getsource(StrategyRunner._on_tick_safe)
    assert "_run_health_watchdog_on_cadence()" in src
    assert "self._health_watchdog()" not in src
