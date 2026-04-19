"""Background maintenance tasks for order history persistence."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Awaitable, Callable

from nifty_scalper_bot.infra.log_rotation import rotate_order_history_archive
from nifty_scalper_bot.utils.async_helpers import safe_task
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


async def run_periodic_task(
    task_fn: Callable[[], Any] | Callable[[], Awaitable[Any]],
    interval_sec: float,
    task_name: str,
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
    while True:
        try:
            await asyncio.sleep(interval_sec)
            LOGGER.debug(
                "Executing periodic task",
                extra={"event": f"task.{task_name}.start"},
            )
            result = task_fn()
            if inspect.isawaitable(result):
                await result
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
            LOGGER.error(
                "Periodic task failed",
                extra={"event": f"task.{task_name}.error", "error": str(exc)},
                exc_info=exc,
            )


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


def start_background_tasks(order_manager: Any, logger: Any) -> list[asyncio.Task[Any]]:
    """Start background maintenance tasks for the execution stack.

    Args:
        order_manager: Order manager instance responsible for persistence.
        logger: Logger compatible interface for lifecycle messages.

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
