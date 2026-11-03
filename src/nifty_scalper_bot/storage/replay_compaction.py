"""Lossless verified compaction of completed production replay sessions.

Responsibilities: Validate replay writer counters and raw-byte SHA, compress
prior-day feeds atomically, and preserve the original session/run identities.
Safety: Never touch the current trading day; never delete unverified input,
repair corrupt captures, or run disk-heavy compaction during market hours.
"""

from __future__ import annotations

import gzip
import hashlib
import os
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from nifty_scalper_bot.storage.replay_archive import iter_session

IST = ZoneInfo("Asia/Kolkata")
_NAME = re.compile(r"^\d{4}-\d{2}-\d{2}\.jsonl$")


def _plain_fingerprint(path: Path, *, compressed: bool = False) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if compressed or path.name.endswith(".gz") else Path.open
    with opener(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def compact_completed_session(
    path: Path,
    *,
    now: datetime | None = None,
    remove_original: bool = True,
) -> dict[str, object]:
    """Write atomic gzip; require exact uncompressed SHA and archive integrity."""
    path = Path(path)
    if not _NAME.fullmatch(path.name):
        raise ValueError("replay_compaction_requires_daily_jsonl")
    current = now or datetime.now(IST)
    if current.tzinfo is None:
        raise ValueError("replay_compaction_requires_timezone")
    if path.stem >= current.astimezone(IST).date().isoformat():
        raise ValueError("replay_compaction_live_session_forbidden")
    if not path.is_file():
        raise FileNotFoundError(path)

    event_count = sum(1 for _ in iter_session(path))
    expected_sha = _plain_fingerprint(path)
    target = path.with_name(path.name + ".gz")
    if target.is_file():
        if (
            _plain_fingerprint(target) != expected_sha
            or sum(1 for _ in iter_session(target)) != event_count
        ):
            raise ValueError("replay_compaction_existing_target_mismatch")
    else:
        temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            with path.open("rb") as source, temporary.open("wb") as raw:
                with gzip.GzipFile(
                    filename="", fileobj=raw, mode="wb", compresslevel=6, mtime=0
                ) as compressed:
                    shutil.copyfileobj(source, compressed, length=1024 * 1024)
                raw.flush()
                os.fsync(raw.fileno())
            if _plain_fingerprint(temporary, compressed=True) != expected_sha:
                raise ValueError("replay_compaction_sha_mismatch")
            # Only the final name has the recognized gzip suffix for replay.
            temporary.replace(target)
            if sum(1 for _ in iter_session(target)) != event_count:
                raise ValueError("replay_compaction_integrity_mismatch")
        finally:
            temporary.unlink(missing_ok=True)
    if _plain_fingerprint(target) != expected_sha:
        raise ValueError("replay_compaction_sha_mismatch")
    if remove_original:
        # Recheck before removal, including a possible unexpected live append.
        if _plain_fingerprint(path) != expected_sha:
            raise ValueError("replay_compaction_source_changed")
        path.unlink()
    return {
        "session_date": path.stem,
        "events": event_count,
        "uncompressed_sha256": expected_sha,
        "compressed_path": str(target),
        "compressed_bytes": target.stat().st_size,
        "original_removed": remove_original,
        "complete_integrity_verified": True,
    }


def compact_prior_sessions(
    directory: Path, *, now: datetime | None = None
) -> list[dict[str, object]]:
    """Compact only during a safe off-market window, without broker I/O."""
    current = (now or datetime.now(IST)).astimezone(IST)
    # Avoid CPU/disk contention in the pre-open and active trading windows.
    if 7 <= current.hour < 17:
        return []
    directory = Path(directory)
    if not directory.is_dir():
        return []
    reports = []
    for path in sorted(directory.glob("????-??-??.jsonl")):
        if path.stem >= current.date().isoformat():
            continue
        try:
            reports.append(compact_completed_session(path, now=current))
        except (OSError, ValueError) as exc:
            # A corrupt day must not prevent verified newer days from compacting.
            reports.append(
                {
                    "session_date": path.stem,
                    "state": "invalid_input_retained",
                    "error_type": type(exc).__name__,
                    "original_removed": False,
                }
            )
    return reports
