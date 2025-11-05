from __future__ import annotations

from datetime import datetime, timedelta, timezone
import gzip
import os
from pathlib import Path

from nifty_scalper_bot.infra.log_rotation import rotate_order_history_archive


def _set_file_age(path: Path, age_days: int) -> None:
    target_time = datetime.now(timezone.utc) - timedelta(days=age_days)
    timestamp = target_time.timestamp()
    os.utime(path, (timestamp, timestamp))


def test_rotate_order_history_archive_noop_for_missing_path(tmp_path: Path) -> None:
    archive_path = tmp_path / "orders.jsonl"
    rotate_order_history_archive(archive_path, max_age_days=1)
    assert not archive_path.exists()


def test_rotate_order_history_archive_rotates_and_compresses(tmp_path: Path) -> None:
    archive_path = tmp_path / "orders.jsonl"
    archive_path.write_text("line-one\n", encoding="utf-8")
    _set_file_age(archive_path, age_days=120)

    rotate_order_history_archive(archive_path, max_age_days=30, compress=True)

    rotated = list(tmp_path.glob("order_history_archive_*.jsonl.gz"))
    assert len(rotated) == 1
    with gzip.open(rotated[0], "rt", encoding="utf-8") as handle:
        assert handle.read() == "line-one\n"
    assert not archive_path.exists()
