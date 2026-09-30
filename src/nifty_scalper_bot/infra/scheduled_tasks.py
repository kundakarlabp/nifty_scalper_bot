"""Background maintenance tasks for order history persistence."""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from typing import Any, Awaitable, Callable

from time import time

from nifty_scalper_bot.infra.daily_log_archive import (
    archive_interval_seconds,
    build_daily_log_archiver,
)
from nifty_scalper_bot.infra.log_rotation import rotate_order_history_archive
from nifty_scalper_bot.infra.supabase_trade_replication import (
    build_supabase_trade_replicator,
    replication_interval_seconds,
)
from nifty_scalper_bot.utils.async_helpers import safe_task
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)

_TRADE_REPLICATION_STATUS: dict[str, Any] = {
    "enabled": False,
    "task_state": "not_started",
    "last_attempt_at": None,
    "last_success_at": None,
    "last_error": None,
    "events": 0,
    "ledger": 0,
    "checkpoint": 0,
}


def get_trade_replication_status() -> dict[str, Any]:
    """Return a read-only snapshot of process-owned replication health."""
    return dict(_TRADE_REPLICATION_STATUS)


def _set_trade_replication_status(**updates: Any) -> None:
    _TRADE_REPLICATION_STATUS.update(updates)


async def run_periodic_task(
    task_fn: Callable[[], Any] | Callable[[], Awaitable[Any]],
    interval_sec: float,
    task_name: str,
    *,
    run_immediately: bool = False,
    on_attempt: Callable[[], None] | None = None,
    on_success: Callable[[Any], None] | None = None,
    on_error: Callable[[Exception], None] | None = None,
) -> None:
    """Run *task_fn* periodically with defensive error handling.

    Args:
        task_fn: Zero-argument callable to execute on each interval.
        interval_sec: Seconds to wait between executions.
        task_name: Human-readable task identifier for logging.

    Returns:
        None.

    Raises:
        None.
    """

    LOGGER.debug(
        "Entered run_periodic_task",
        extra={"event": f"task.{task_name}.enter", "interval_sec": interval_sec},
    )
    first_run = True
    while True:
        try:
            if not (first_run and run_immediately):
                await asyncio.sleep(interval_sec)
            first_run = False
            if on_attempt is not None:
                on_attempt()
            LOGGER.debug(
                "Executing periodic task",
                extra={"event": f"task.{task_name}.start"},
            )
            result = task_fn()
            if inspect.isawaitable(result):
                result = await result
            if on_success is not None:
                on_success(result)
            LOGGER.debug(
                "Periodic task completed",
                extra={"event": f"task.{task_name}.success"},
            )
        except asyncio.CancelledError:
            LOGGER.info(
                "Periodic task cancelled",
                extra={"event": f"task.{task_name}.cancelled"},
            )
            break
        except Exception as exc:  # noqa: BLE001
            if on_error is not None:
                on_error(exc)
            LOGGER.error(
                "Periodic task failed name=%s error=%s",
                task_name,
                exc,
                extra={"event": f"task.{task_name}.error", "error": str(exc)},
                exc_info=exc,
            )


def start_daily_log_archive_task() -> asyncio.Task[Any] | None:
    """Start optional full-session log archival off the trading hot path."""
    archiver = build_daily_log_archiver()
    if archiver is None:
        return None
    return safe_task(
        run_periodic_task(
            task_fn=lambda: asyncio.to_thread(archiver.archive_once),
            interval_sec=archive_interval_seconds(),
            task_name="archive_daily_market_logs",
        )
    )


def start_trade_replication_task(
    db_path: str | Path,
) -> asyncio.Task[Any] | None:
    """Start optional trade replication independently of broker startup."""
    replicator = build_supabase_trade_replicator(db_path)
    if replicator is None:
        _set_trade_replication_status(
            enabled=False,
            task_state="disabled",
            last_attempt_at=None,
            last_success_at=None,
            last_error=None,
            events=0,
            ledger=0,
            checkpoint=0,
        )
        return None

    def on_attempt() -> None:
        _set_trade_replication_status(
            enabled=True,
            task_state="running",
            last_attempt_at=time(),
            last_error=None,
        )

    def on_success(result: Any) -> None:
        payload = result if isinstance(result, dict) else {}
        _set_trade_replication_status(
            enabled=True,
            task_state="running",
            last_success_at=time(),
            last_error=None,
            events=int(payload.get("events", 0) or 0),
            ledger=int(payload.get("ledger", 0) or 0),
            checkpoint=int(payload.get("checkpoint", 0) or 0),
        )

    def on_error(exc: Exception) -> None:
        _set_trade_replication_status(
            enabled=True,
            task_state="error",
            last_error=str(exc),
        )

    _set_trade_replication_status(
        enabled=True,
        task_state="scheduled",
        last_error=None,
    )
    task = safe_task(
        run_periodic_task(
            task_fn=lambda: asyncio.to_thread(replicator.replicate_once),
            interval_sec=replication_interval_seconds(),
            task_name="replicate_trade_observability",
            run_immediately=True,
            on_attempt=on_attempt,
            on_success=on_success,
            on_error=on_error,
        )
    )

    def task_done(done: asyncio.Task[Any]) -> None:
        if done.cancelled():
            _set_trade_replication_status(task_state="cancelled")
        elif done.exception() is not None:
            _set_trade_replication_status(
                task_state="dead",
                last_error=str(done.exception()),
            )
        else:
            _set_trade_replication_status(task_state="stopped")

    task.add_done_callback(task_done)
    return task


async def run_archive_rotation(order_manager: Any, max_age_days: int = 90) -> None:
    """Rotate order history archives for *order_manager*.

    Args:
        order_manager: Order manager exposing ``_history_persist_path``.
        max_age_days: Maximum age in days before rotation triggers.

    Returns:
        None.

    Raises:
        None.
    """

    LOGGER.debug(
        "Entered run_archive_rotation",
        extra={
            "event": "task.rotate_order_archive.enter",
            "max_age_days": max_age_days,
        },
    )
    try:
        archive_path = getattr(order_manager, "_history_persist_path", None)
        if archive_path is None:
            LOGGER.debug(
                "Archive rotation skipped (path unavailable)",
                extra={"event": "task.rotate_order_archive.missing_path"},
            )
            return
        rotate_order_history_archive(archive_path, max_age_days=max_age_days)
        LOGGER.info(
            "Condition met: archive rotation executed",
            extra={
                "event": "task.rotate_order_archive.success",
                "path": str(archive_path),
            },
        )
    except Exception as exc:  # noqa: BLE001 - defensive rotation guard
        LOGGER.error(
            "Archive rotation failed: %s",
            exc,
            extra={"event": "task.rotate_order_archive.error"},
            exc_info=exc,
        )


def start_background_tasks(
    order_manager: Any,
    logger: Any,
    *,
    trade_journal: Any | None = None,
) -> list[asyncio.Task[Any]]:
    """Start background maintenance tasks for the execution stack.

    Args:
        order_manager: Order manager instance responsible for persistence.
        logger: Logger compatible interface for lifecycle messages.
        trade_journal: Retained for call-site compatibility; replication is
            process-owned.

    Returns:
        List of created asyncio tasks.

    Raises:
        None.
    """

    LOGGER.debug(
        "Entered start_background_tasks",
        extra={"event": "tasks.start.enter"},
    )
    tasks: list[asyncio.Task[Any]] = []
    tasks.append(
        safe_task(
            run_periodic_task(
                task_fn=order_manager.persist_history_batch,
                interval_sec=300.0,
                task_name="persist_order_history",
            )
        )
    )
    tasks.append(
        safe_task(
            run_periodic_task(
                task_fn=lambda: run_archive_rotation(order_manager, max_age_days=90),
                interval_sec=86_400.0,
                task_name="rotate_order_archive",
            )
        )
    )
    logger.info(
        "Scheduled maintenance tasks created",
        extra={"event": "tasks.created", "count": len(tasks)},
    )

    return tasks
