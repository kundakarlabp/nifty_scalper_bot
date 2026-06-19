"""Dashboard is super-lite: logs served from an in-memory ring, never a
subprocess on the polled request path.

Regression guard for the dashboard-hang root cause: ``_gather_logs`` used to
run ``journalctl`` (15s timeout) on every cache miss, blocking the shared
threadpool. It now reads a background-filled ring buffer with zero subprocess
on the hot path.
"""

from __future__ import annotations

import pytest

import nifty_scalper_bot.admin_dashboard as dash


@pytest.fixture(autouse=True)
def _prime_ring(monkeypatch: pytest.MonkeyPatch):
    # Pretend the follower already runs so _gather_logs never tries to spawn it.
    monkeypatch.setattr(dash, "_LOG_FOLLOWER_STARTED", True, raising=False)
    dash._LOG_RING.clear()
    for i in range(1, 201):
        dash._LOG_RING.append(f"[2026-06-19 11:13:00 IST] line {i} EXIT done")
    # Any subprocess on the hot path must fail the test loudly.
    def _boom(*a, **k):
        raise AssertionError("subprocess invoked on the hot log path")
    monkeypatch.setattr(dash.subprocess, "run", _boom)
    monkeypatch.setattr(dash.subprocess, "Popen", _boom)
    yield
    dash._LOG_RING.clear()


async def test_gather_logs_reads_ring_without_subprocess() -> None:
    out = dash._gather_logs(50, clean=True)
    rows = out.splitlines()
    assert len(rows) == 50  # last 50 of 200
    assert "line 200" in rows[-1]
    assert "line 151" in rows[0]


async def test_gather_logs_contains_filter() -> None:
    dash._LOG_RING.append("[2026-06-19 11:14:00 IST] ORDER_REJECTED foo")
    out = dash._gather_logs(400, contains="ORDER_REJECTED", clean=True)
    assert out.splitlines() == ["2026-06-19 11:14:00 IST  ORDER_REJECTED foo"]


async def test_gather_logs_empty_ring_message(monkeypatch: pytest.MonkeyPatch) -> None:
    dash._LOG_RING.clear()
    out = dash._gather_logs(100)
    assert out == "Waiting for logs…"


async def test_ensure_follower_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    # Already-started flag means no new thread is spawned (and no subprocess).
    started = {"n": 0}

    class _FakeThread:
        def __init__(self, *a, **k):
            started["n"] += 1

        def start(self):
            pass

    monkeypatch.setattr(dash.threading, "Thread", _FakeThread)
    monkeypatch.setattr(dash, "_LOG_FOLLOWER_STARTED", True, raising=False)
    dash._ensure_log_follower()
    assert started["n"] == 0
