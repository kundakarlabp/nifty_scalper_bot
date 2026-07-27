"""send_alert() must never block a trading thread.

Regression for the 2026-07-27 production incident: send_alert() is reached
from the order/bracket path, which runs on the market-data tick thread. The
previous implementation fell back to `asyncio.run(_dispatch())` whenever no
loop was running on the calling thread, executing the full retry chain
(backoff min(2**attempt, 30) -> ~35s) synchronously. One tick callback
blocked for 34.3s with 548 pending ticks and 34s of event-loop lag.
"""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

from nifty_scalper_bot.notifications.telegram_webhook_enhanced import (
    TelegramEnhancedNotifier,
)


class _SlowNotifier(TelegramEnhancedNotifier):
    """Real send_alert(); only the network leaf is slowed."""

    dispatch_seconds = 1.0

    async def _send_single(self, chat_id, text, *, alert_id, event_type):
        await asyncio.sleep(self.dispatch_seconds)
        return SimpleNamespace(alert_id=alert_id)

    async def _acquire_token(self) -> None:
        return None

    def _log_provider_health(self) -> None:
        return None


def _notifier(monkeypatch, *, dispatch_seconds: float):
    """Notifier whose single-send path is deliberately slow."""
    n = _SlowNotifier.__new__(_SlowNotifier)
    object.__setattr__(
        n,
        "_logger",
        SimpleNamespace(
            debug=lambda *a, **k: None,
            info=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            error=lambda *a, **k: None,
        ),
    )
    object.__setattr__(n, "_chat_whitelist", {123})
    object.__setattr__(n, "_runtime_loop", None)
    object.__setattr__(n, "_telegram_degraded", False)
    object.__setattr__(n, "_telegram_degraded_until", 0.0)
    object.__setattr__(n, "dispatch_seconds", dispatch_seconds)
    return n


def _run_loop_in_thread():
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    return loop, t


def _stop_loop(loop, thread):
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=2.0)
    loop.close()


def test_send_alert_from_worker_thread_does_not_block(monkeypatch) -> None:
    """With a runtime loop attached, a slow dispatch must not block the caller."""
    n = _notifier(monkeypatch, dispatch_seconds=1.0)
    loop, thread = _run_loop_in_thread()
    n.attach_runtime_loop(loop)
    try:
        started = time.perf_counter()
        n.send_alert("order filled")
        elapsed = time.perf_counter() - started

        # Caller returns immediately; the 1s dispatch continues on the loop.
        assert elapsed < 0.2, f"send_alert blocked caller for {elapsed:.3f}s"
    finally:
        _stop_loop(loop, thread)


def test_send_alert_without_runtime_loop_drops_instead_of_blocking() -> None:
    """No usable loop -> drop the alert rather than freeze a trading thread."""
    n = _notifier(None, dispatch_seconds=1.0)
    n._runtime_loop = None

    started = time.perf_counter()
    n.send_alert("order filled")
    elapsed = time.perf_counter() - started

    assert elapsed < 0.2, f"send_alert blocked caller for {elapsed:.3f}s"


def test_send_alert_on_loop_thread_still_schedules_task() -> None:
    """Existing behaviour on the loop thread is preserved (fire-and-forget)."""
    n = _notifier(None, dispatch_seconds=0.05)

    async def _main():
        started = time.perf_counter()
        n.send_alert("order filled")
        elapsed = time.perf_counter() - started
        assert elapsed < 0.2
        await asyncio.sleep(0.1)

    asyncio.run(_main())
