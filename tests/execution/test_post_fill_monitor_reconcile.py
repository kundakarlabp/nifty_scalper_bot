"""Post-fill monitor reconciliation startup behaviour tests."""

from __future__ import annotations

import asyncio

from nifty_scalper_bot.execution.post_fill_monitor import PostFillMonitor


class _Broker:
    async def get_positions(self) -> list[dict[str, object]]:
        return []


class _Tracker:
    def get_open_positions(self) -> list[dict[str, object]]:
        return []


def test_start_runs_immediate_reconciliation() -> None:
    """Ensure startup executes one reconciliation cycle before background loop."""

    async def _run() -> None:
        monitor = PostFillMonitor(
            broker_client=_Broker(),
            state_tracker=_Tracker(),
            interval_sec=30,
        )
        await monitor.start()
        await asyncio.sleep(0.01)
        stats = monitor.get_stats()
        await monitor.stop()
        assert stats['interval_sec'] == 30
        assert stats['total_mismatches'] == 0

    asyncio.run(_run())
