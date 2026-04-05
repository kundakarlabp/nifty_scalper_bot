"""APScheduler background job scheduler."""

from __future__ import annotations

import logging
from typing import Callable, Awaitable

from apscheduler.schedulers.asyncio import AsyncIOScheduler  # type: ignore[import]

log = logging.getLogger(__name__)


class BotScheduler:
    def __init__(self) -> None:
        self._scheduler = AsyncIOScheduler()

    def schedule_daily_summary(self, ist_time: str, callback: Callable[[], Awaitable[None]]) -> None:
        """Schedule daily summary at HH:MM IST."""
        parts = ist_time.split(":")
        hour, minute = int(parts[0]), int(parts[1])
        self._scheduler.add_job(
            callback,
            trigger="cron",
            hour=hour,
            minute=minute,
            timezone="Asia/Kolkata",
            id="daily_summary",
        )

    def schedule_session_reset(self, callback: Callable[[], Awaitable[None]]) -> None:
        """Reset daily counters at 09:15 IST."""
        self._scheduler.add_job(
            callback,
            trigger="cron",
            hour=9,
            minute=15,
            timezone="Asia/Kolkata",
            id="session_reset",
        )

    def schedule_cleanup(self, callback: Callable[[], Awaitable[None]]) -> None:
        """Weekly DB cleanup at Sunday 02:00 IST."""
        self._scheduler.add_job(
            callback,
            trigger="cron",
            day_of_week="sun",
            hour=2,
            timezone="Asia/Kolkata",
            id="db_cleanup",
        )

    def start(self) -> None:
        self._scheduler.start()
        log.info("Scheduler started")

    def stop(self) -> None:
        self._scheduler.shutdown(wait=False)
