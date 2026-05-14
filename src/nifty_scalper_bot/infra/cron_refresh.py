"""Cron-style refresh helper for the instrument universe (InstrumentManager)."""

from __future__ import annotations

import asyncio
from datetime import datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from nifty_scalper_bot.data.instrument_loader import parse_kite_csv
from nifty_scalper_bot.infra.metrics import METRICS
from nifty_scalper_bot.utils.async_helpers import safe_task
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)
_IST = ZoneInfo("Asia/Kolkata")
_TARGET_TIME = time(hour=6, minute=30)


def _seconds_until_run(target_time: time, tz: ZoneInfo) -> float:
    """Return seconds until the next occurrence of *target_time* in *tz*.

    Args:
        target_time: Daily execution time in local clock units.
        tz: Timezone used to compute the next trigger instant.

    Returns:
        Float seconds remaining until the next trigger time.

    Raises:
        None.
    """
    now = datetime.now(tz)
    scheduled = now.replace(
        hour=target_time.hour,
        minute=target_time.minute,
        second=0,
        microsecond=0,
    )
    if scheduled <= now:
        scheduled = scheduled + timedelta(days=1)
    return max((scheduled - now).total_seconds(), 0.0)


def _refresh_interval_seconds(settings: Any) -> float:
    """Compute instrument refresh cadence from settings.

    Args: settings.
    Returns: Refresh interval in seconds (minimum 15 minutes).
    Raises: None.
    """
    instrument_cfg = getattr(settings, "instruments", None)
    raw_hours = getattr(instrument_cfg, "refresh_interval_hours", 24.0)
    try:
        hours = max(float(raw_hours), 0.25)
    except (TypeError, ValueError):
        hours = 24.0
    return hours * 3600.0


def schedule_instrument_refresh(
    settings: Any,
    instrument_manager: Any,
    *,
    state: Any | None = None,
) -> asyncio.Task[Any] | None:
    """Schedule InstrumentManager reload at 06:30 IST and on a periodic interval.

    Replaces the old CSV/SQLite-based InstrumentResolver refresh loop.
    InstrumentManager.load() fetches NFO instruments directly from the broker,
    so no CSV path or SQLite connection is needed.

    Args:
        settings: Settings object exposing ``instruments`` configuration.
        instrument_manager: InstrumentManager instance to reload periodically.
        state: Optional universe status object updated after each reload.

    Returns:
        Created asyncio Task when scheduling succeeds; ``None`` otherwise.

    Raises:
        None.
    """
    LOGGER.debug(
        "Entered schedule_instrument_refresh",
        extra={"event": "instrument_refresh_schedule_enter"},
    )

    instrument_cfg = getattr(settings, "instruments", None)
    if instrument_cfg is None:
        LOGGER.info(
            "instrument_refresh_disabled: no instruments config",
            extra={"event": "instrument_refresh_disabled", "reason": "no_config"},
        )
        return None
    if not getattr(instrument_cfg, "refresh_cron_enabled", True):
        LOGGER.info(
            "instrument_refresh_disabled: env toggle off",
            extra={"event": "instrument_refresh_disabled", "reason": "env_toggle"},
        )
        return None
    if instrument_manager is None:
        LOGGER.warning(
            "instrument_refresh_skipped: no InstrumentManager",
            extra={"event": "instrument_refresh_skipped", "reason": "no_manager"},
        )
        return None

    async def _refresh_loop() -> None:
        """Execute the InstrumentManager reload loop until cancelled."""
        LOGGER.debug(
            "Entered instrument refresh loop",
            extra={"event": "instrument_refresh_loop_enter"},
        )
        delay = _seconds_until_run(_TARGET_TIME, _IST)
        while True:
            try:
                await asyncio.sleep(delay)
                LOGGER.info(
                    "instrument_refresh_start",
                    extra={"event": "instrument_refresh_start"},
                )
                count = 0
                if hasattr(instrument_manager, 'load'):
                    await asyncio.to_thread(instrument_manager.load)
                    size_fn = getattr(instrument_manager, 'size', None)
                    if callable(size_fn):
                        count = int(size_fn())
                elif hasattr(instrument_manager, 'warm_from_broker_dump'):
                    csv_path = getattr(instrument_cfg, 'csv_path', None)
                    if csv_path is None:
                        rows = []
                    else:
                        with open(csv_path, 'rb') as csv_file:
                            rows = list(parse_kite_csv(csv_file))
                    await asyncio.to_thread(instrument_manager.warm_from_broker_dump, rows)
                    count = len(rows)
                else:
                    raise AttributeError('instrument manager has no supported refresh method')
                if state is not None:
                    try:
                        state.record_refresh(
                            tokens=count,
                            options=count,
                            source="cron",
                            path=str(getattr(instrument_cfg, 'db_path', '') or ''),
                            timestamp=datetime.now(timezone.utc),
                        )
                    except Exception as exc:
                        LOGGER.error('Failure in schedule_instrument_refresh: %s', exc, exc_info=exc)
                METRICS.record_resolver_tokens(source="cron_broker", count=count)
                LOGGER.info(
                    "instrument_refresh_complete tokens=%s source=broker",
                    count,
                    extra={
                        "event": "instrument_refresh_complete",
                        "tokens": count,
                        "options": count,
                        "source": "broker",
                    },
                )
                delay = _refresh_interval_seconds(settings)
            except asyncio.CancelledError:
                LOGGER.info(
                    "instrument_refresh_task_cancelled",
                    extra={"event": "instrument_refresh_task_cancelled"},
                )
                raise
            except Exception as exc:  # noqa: BLE001
                LOGGER.error(
                    "instrument_refresh_task_failed: %s",
                    exc,
                    extra={"event": "instrument_refresh_task_failed", "error": str(exc)},
                    exc_info=exc,
                )
                delay = 300.0  # Back off 5 minutes on error

    try:
        task = safe_task(_refresh_loop())
        LOGGER.info(
            "instrument_refresh_task_scheduled",
            extra={"event": "instrument_refresh_task_scheduled"},
        )
        return task
    except Exception as exc:  # noqa: BLE001
        LOGGER.error(
            "Failed to schedule instrument refresh: %s",
            exc,
            extra={"event": "instrument_refresh_schedule_error", "error": str(exc)},
            exc_info=exc,
        )
        return None


__all__ = ["schedule_instrument_refresh"]
