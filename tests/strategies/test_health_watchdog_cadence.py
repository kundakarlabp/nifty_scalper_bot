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
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock

from nifty_scalper_bot.strategies.runner import (
    StrategyRunner,
    _signal_frequency_window_elapsed,
)
from nifty_scalper_bot.utils.market_hours import IST, SAFE_END, SAFE_START


def _runner(interval: float = 5.0) -> StrategyRunner:
    r = StrategyRunner.__new__(StrategyRunner)
    r._logger = MagicMock()
    r._health_watchdog_lock = threading.Lock()
    r._health_watchdog_last_run = 0.0
    r._health_watchdog_running = False
    r._health_watchdog_skipped = 0
    r._health_watchdog_interval_s = interval
    r._main_loop = None
    return r


def test_many_ticks_in_one_interval_produce_one_scan() -> None:
    """AUDIT TDD CASE 2: many ticks must not cause many full scans."""
    r = _runner(interval=60.0)
    calls: list[int] = []
    done = threading.Event()

    def _scan() -> None:
        calls.append(1)
        done.set()

    r._health_watchdog = _scan

    for _ in range(50):
        r._run_health_watchdog_on_cadence()

    assert done.wait(timeout=2.0)
    assert len(calls) == 1, f"expected one scan per interval, got {len(calls)}"


def test_scan_runs_again_after_the_interval_elapses() -> None:
    """Cadence must not become a permanent suppression."""
    r = _runner(interval=0.05)
    calls: list[int] = []
    done = threading.Event()

    def _scan() -> None:
        calls.append(1)
        done.set()

    r._health_watchdog = _scan

    r._run_health_watchdog_on_cadence()
    assert done.wait(timeout=2.0)
    time.sleep(0.06)
    done.clear()
    r._run_health_watchdog_on_cadence()

    assert done.wait(timeout=2.0)
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

    r._run_health_watchdog_on_cadence()
    assert entered.wait(timeout=2.0)

    # A second tick arrives while the first scan is still running.
    r._run_health_watchdog_on_cadence()
    assert len(calls) == 1, "overlapping scan was not refused"

    release.set()
    deadline = time.time() + 2.0
    while r._health_watchdog_running and time.time() < deadline:
        time.sleep(0.01)


def test_running_flag_is_cleared_even_when_the_scan_raises() -> None:
    """A failed scan must not permanently wedge health scanning."""
    r = _runner(interval=0.0)

    def _boom() -> None:
        raise RuntimeError("scan failed")

    r._health_watchdog = _boom

    r._run_health_watchdog_on_cadence()
    deadline = time.time() + 2.0
    while r._health_watchdog_running and time.time() < deadline:
        time.sleep(0.01)

    assert r._health_watchdog_running is False

    calls: list[int] = []
    done = threading.Event()

    def _scan() -> None:
        calls.append(1)
        done.set()

    r._health_watchdog = _scan
    r._run_health_watchdog_on_cadence()
    assert done.wait(timeout=2.0)
    assert calls == [1]


def test_skipped_tick_count_is_reported_for_slow_scans(monkeypatch) -> None:
    """A slow scan reports elapsed time and displaced ticks, without log-capture races."""
    import logging

    r = _runner(interval=0.0)
    entered = threading.Event()
    release = threading.Event()
    logged = threading.Event()
    reports: list[dict[str, object]] = []

    def _slow() -> None:
        entered.set()
        release.wait(timeout=2.0)

    def _capture_watchdog_log(
        _logger, _key: str, message: str, **kwargs: object
    ) -> None:
        extra = kwargs.get("extra")
        if isinstance(extra, dict) and extra.get("event") == "HEALTH_WATCHDOG_SLOW":
            reports.append({"message": message, "level": kwargs.get("level"), **extra})
            logged.set()

    # The production formatter can replace pytest's logging capture handler.
    # Observe the structured event at the call boundary instead of caplog,
    # and synchronize with the worker rather than racing its completion.
    monkeypatch.setattr(
        "nifty_scalper_bot.strategies.runner.log_throttled",
        _capture_watchdog_log,
    )
    r._health_watchdog = _slow
    try:
        r._run_health_watchdog_on_cadence()
        assert entered.wait(timeout=2.0)
        r._run_health_watchdog_on_cadence()  # Must count one displaced scan.
        time.sleep(0.12)
    finally:
        release.set()

    assert logged.wait(timeout=2.0), "slow-scan warning was not emitted"
    assert len(reports) == 1
    report = reports[0]
    assert report["level"] == logging.WARNING
    assert float(report["duration_ms"]) >= 100.0
    assert report["ticks_skipped"] == 1
    assert "HEALTH_WATCHDOG_SLOW" in str(report["message"])
    assert "ticks_skipped=1" in str(report["message"])
    assert r._health_watchdog_running is False

def test_slow_scan_does_not_block_tick_caller() -> None:
    r = _runner(interval=0.0)
    started = threading.Event()
    release = threading.Event()
    caller_thread = threading.get_ident()
    scan_thread: list[int] = []

    def _slow() -> None:
        scan_thread.append(threading.get_ident())
        started.set()
        release.wait(timeout=2.0)

    r._health_watchdog = _slow
    started_at = time.perf_counter()
    r._run_health_watchdog_on_cadence()
    elapsed = time.perf_counter() - started_at

    try:
        assert elapsed < 0.1
        assert started.wait(timeout=2.0)
        assert scan_thread[0] != caller_thread
    finally:
        release.set()
        deadline = time.time() + 2.0
        while r._health_watchdog_running and time.time() < deadline:
            time.sleep(0.01)


def test_tick_path_calls_the_cadence_wrapper_not_the_scan_directly() -> None:
    """_on_tick_safe must not invoke the full scan per tick."""
    import inspect

    src = inspect.getsource(StrategyRunner._on_tick_safe)
    assert "_run_health_watchdog_on_cadence()" in src
    assert "self._health_watchdog()" not in src


def test_zero_signal_alert_requires_full_live_entry_hour() -> None:
    """Premarket and the opening hour are not a complete one-hour sample."""
    start = datetime.combine(date(2026, 10, 9), SAFE_START, tzinfo=IST)
    end = datetime.combine(date(2026, 10, 9), SAFE_END, tzinfo=IST)

    assert not _signal_frequency_window_elapsed(
        (start + timedelta(minutes=59, seconds=59)).timestamp()
    )
    assert _signal_frequency_window_elapsed((start + timedelta(hours=1)).timestamp())
    assert not _signal_frequency_window_elapsed(
        (end + timedelta(seconds=1)).timestamp()
    )


def test_zero_signal_alert_waits_for_full_post_restart_observation_hour() -> None:
    """A midday restart cannot report one full observed hour of zero signals."""
    entry_start = datetime.combine(date(2026, 10, 9), SAFE_START, tzinfo=IST)
    restarted_at = entry_start + timedelta(hours=3)
    assert not _signal_frequency_window_elapsed(
        (restarted_at + timedelta(minutes=20)).timestamp(),
        observation_started_at=restarted_at.timestamp(),
    )
    assert not _signal_frequency_window_elapsed(
        (restarted_at + timedelta(minutes=59, seconds=59)).timestamp(),
        observation_started_at=restarted_at.timestamp(),
    )
    assert _signal_frequency_window_elapsed(
        (restarted_at + timedelta(hours=1)).timestamp(),
        observation_started_at=restarted_at.timestamp(),
    )
