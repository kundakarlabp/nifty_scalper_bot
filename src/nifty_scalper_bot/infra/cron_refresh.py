"""Cron-style refresh helpers for the instrument universe cache."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from nifty_scalper_bot.data import (
    InstrumentResolver,
    InstrumentUniverseStatus,
    ensure_sqlite,
    load_rows_for_resolver,
    refresh_from_csv,
    sync_instrument_csv_from_broker,
)
from nifty_scalper_bot.infra.metrics import METRICS
from nifty_scalper_bot.utils.logging import get_logger
from nifty_scalper_bot.utils.async_helpers import safe_task

LOGGER = get_logger(__name__)
_IST = ZoneInfo("Asia/Kolkata")
_TARGET_TIME = time(hour=6, minute=30)


def _csv_refresh_interval_seconds(settings: Any) -> float:
    """Compute CSV refresh cadence.

    Args: settings.
    Returns: Refresh seconds.
    Raises: None.
    """

    instrument_cfg = getattr(settings, "instruments", None)
    raw_hours = getattr(instrument_cfg, "refresh_interval_hours", 24.0)
    try:
        hours = max(float(raw_hours), 0.25)
    except (TypeError, ValueError):
        hours = 24.0
    return hours * 3600.0


def _sync_csv_from_broker(
    resolver: InstrumentResolver, csv_path: Path
) -> dict[str, Any]:
    """Sync instrument CSV from resolver broker.

    Args: resolver,csv_path.
    Returns: Sync summary.
    Raises: Exception.
    """

    broker = getattr(resolver, "_broker", None)
    if broker is None:
        return {"written": 0, "path": str(csv_path), "source": "none"}
    return sync_instrument_csv_from_broker(broker, str(csv_path), exchange="NFO")


def _seconds_until_run(target_time: time, tz: ZoneInfo) -> float:
    """Return seconds until the next occurrence of *target_time* in *tz*.

    Args:
        target_time: Daily execution time expressed in local clock units.
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


def schedule_instrument_refresh(
    settings: Any,
    resolver: InstrumentResolver,
    *,
    state: InstrumentUniverseStatus | None = None,
) -> asyncio.Task[Any] | None:
    """Schedule resolver refresh with initial 06:30 IST trigger and periodic updates.

    Args:
        settings: Settings object exposing ``instruments`` configuration.
        resolver: Resolver instance warmed with cached instruments.
        state: Optional universe status snapshot updated after refresh.

    Returns:
        Created asyncio task when scheduling succeeds; ``None`` otherwise.

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
            "Condition met: instrument_refresh_disabled",
            extra={"event": "instrument_refresh_disabled", "reason": "no_config"},
        )
        return None
    if not getattr(instrument_cfg, "refresh_cron_enabled", True):
        LOGGER.info(
            "Condition met: instrument_refresh_disabled",
            extra={"event": "instrument_refresh_disabled", "reason": "env_toggle"},
        )
        return None
    csv_path: Path | None = getattr(instrument_cfg, "csv_path", None)
    if csv_path is None:
        LOGGER.info(
            "Condition met: instrument_refresh_skipped",
            extra={"event": "instrument_refresh_skipped", "reason": "csv_missing"},
        )
        return None

    async def _refresh_loop() -> None:
        """Execute the instrument refresh loop until cancelled."""

        LOGGER.debug(
            "Entered instrument refresh loop",
            extra={"event": "instrument_refresh_loop_enter"},
        )
        delay = _seconds_until_run(_TARGET_TIME, _IST)
        while True:
            try:
                await asyncio.sleep(delay)
                csv_sync_summary = _sync_csv_from_broker(resolver, csv_path)
                conn = ensure_sqlite(str(instrument_cfg.db_path))
                try:
                    summary = refresh_from_csv(conn, str(csv_path))
                    rows = load_rows_for_resolver(conn)
                    resolver.warm_from_broker_dump(rows)
                    tokens = len(rows)
                    options = int(summary.get("stored", tokens)) if summary else tokens
                    timestamp = datetime.now(timezone.utc)
                    if state is not None:
                        state.record_refresh(
                            tokens=tokens,
                            options=options,
                            source="cron",
                            path=str(instrument_cfg.db_path),
                            timestamp=timestamp,
                        )
                    METRICS.record_resolver_tokens(source="cron", count=tokens)
                    LOGGER.info(
                        "instrument_refresh_complete tokens=%s options=%s source=cron",
                        tokens,
                        options,
                        extra={
                            "event": "instrument_refresh_complete",
                            "tokens": tokens,
                            "options": options,
                            "path": str(csv_path),
                            "csv_written": int(csv_sync_summary.get("written") or 0),
                        },
                    )
                finally:
                    with suppress(Exception):
                        conn.close()
                delay = _csv_refresh_interval_seconds(settings)
            except asyncio.CancelledError:
                LOGGER.info(
                    "instrument_refresh_task_cancelled",
                    extra={"event": "instrument_refresh_task_cancelled"},
                )
                raise
            except Exception as exc:  # noqa: BLE001
                LOGGER.error(
                    "instrument_refresh_task_failed",
                    extra={
                        "event": "instrument_refresh_task_failed",
                        "error": str(exc),
                    },
                    exc_info=exc,
                )

    try:
        task = safe_task(_refresh_loop())
        LOGGER.info(
            "instrument_refresh_task_scheduled",
            extra={
                "event": "instrument_refresh_task_scheduled",
                "csv_path": str(csv_path),
            },
        )
        return task
    except Exception as exc:  # noqa: BLE001
        LOGGER.error(
            "Failed to schedule instrument refresh",
            extra={"event": "instrument_refresh_schedule_error", "error": str(exc)},
            exc_info=exc,
        )
        return None


__all__ = ["schedule_instrument_refresh"]
