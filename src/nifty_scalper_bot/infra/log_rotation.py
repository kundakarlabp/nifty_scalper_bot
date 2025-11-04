"""Log rotation utilities for order history archives."""

from __future__ import annotations

import gzip
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


def rotate_order_history_archive(
    archive_path: Path,
    max_age_days: int = 90,
    compress: bool = True,
) -> None:
    """Rotate *archive_path* when its age exceeds *max_age_days*.

    Args:
        archive_path: Path to the order history archive file.
        max_age_days: Maximum age in days before rotation triggers.
        compress: Whether to gzip the rotated archive.

    Returns:
        None.

    Raises:
        None.
    """

    LOGGER.debug(
        "Entered rotate_order_history_archive",
        extra={
            "event": "order_archive.rotate.enter",
            "path": str(archive_path),
            "max_age_days": max_age_days,
        },
    )
    try:
        if not archive_path.exists():
            LOGGER.debug(
                "Archive path missing, skipping rotation",
                extra={
                    "event": "order_archive.rotate.missing",
                    "path": str(archive_path),
                },
            )
            return

        stat = archive_path.stat()
        modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        age = datetime.now(timezone.utc) - modified
        if age < timedelta(days=max_age_days):
            LOGGER.debug(
                "Archive not old enough for rotation",
                extra={
                    "event": "order_archive.rotate.skip",
                    "age_days": age.days,
                    "max_age_days": max_age_days,
                },
            )
            return

        timestamp = modified.strftime("%Y%m%d_%H%M%S")
        rotation_name = f"order_history_archive_{timestamp}.jsonl"
        if compress:
            rotation_name += ".gz"
        rotation_path = archive_path.parent / rotation_name

        if compress:
            with (
                archive_path.open("rb") as source,
                gzip.open(rotation_path, "wb") as target,
            ):
                shutil.copyfileobj(source, target)
            archive_path.unlink()
        else:
            shutil.move(str(archive_path), str(rotation_path))

        LOGGER.info(
            "Condition met: order archive rotated",
            extra={
                "event": "order_archive.rotate.success",
                "path": str(archive_path),
                "rotation_path": str(rotation_path),
                "age_days": age.days,
                "compressed": compress,
            },
        )
    except Exception as exc:  # noqa: BLE001 - defensive rotation guard
        LOGGER.error(
            "Failure in rotate_order_history_archive: %s",
            exc,
            extra={
                "event": "order_archive.rotate.error",
                "path": str(archive_path),
            },
            exc_info=exc,
        )
