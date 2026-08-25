"""Regression coverage for runtime self-test escalation.

The loop intends to alert after three consecutive failing checks, but the
edge-trigger dedup (``current_ok == previous_ok`` -> continue) ran *before* the
counter was maintained. A persistently failing check therefore pinned
``_self_test_failure_count`` at 1 and could never reach the threshold, so a
permanent silent failure never escalated.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from nifty_scalper_bot.core import app


class _Checker:
    interval_seconds = 60.0

    def run_full_check(self) -> dict[str, dict[str, object]]:
        return {"data_freshness": {"ok": False, "detail": "stale", "meta": {}}}


def _app_with_failing_checker(iterations: int) -> tuple[object, list[str]]:
    instance = app.NiftyScalperApp.__new__(app.NiftyScalperApp)
    instance._ctx = SimpleNamespace(selfchecker=_Checker())
    instance._self_test_interval = 60.0
    instance._last_self_check_ok = None
    instance._self_test_failure_count = 0
    alerts: list[str] = []

    async def _alert(name: str, detail: str, meta: object = None) -> None:
        alerts.append(f"{name}:{detail}")

    instance._send_self_test_alert = _alert

    remaining = {"n": iterations}

    class _Shutdown:
        def is_set(self) -> bool:
            if remaining["n"] <= 0:
                return True
            remaining["n"] -= 1
            return False

        async def wait(self) -> None:
            await asyncio.sleep(3600)

    instance._shutdown_event = _Shutdown()
    return instance, alerts


@pytest.mark.asyncio
async def test_persistent_failure_reaches_escalation_threshold(monkeypatch) -> None:
    """Three consecutive failures must escalate, not stall at one."""

    async def _instant_timeout(_awaitable: object, timeout: float) -> None:
        raise asyncio.TimeoutError

    monkeypatch.setattr(app.asyncio, "wait_for", _instant_timeout)
    instance, alerts = _app_with_failing_checker(3)

    await instance._self_test_loop()

    assert instance._self_test_failure_count == 3
    assert alerts == ["data_freshness:stale"]


@pytest.mark.asyncio
async def test_escalation_alerts_once_while_failure_persists(monkeypatch) -> None:
    """A steady failure must not re-alert on every later iteration."""

    async def _instant_timeout(_awaitable: object, timeout: float) -> None:
        raise asyncio.TimeoutError

    monkeypatch.setattr(app.asyncio, "wait_for", _instant_timeout)
    instance, alerts = _app_with_failing_checker(6)

    await instance._self_test_loop()

    assert len(alerts) == 1
