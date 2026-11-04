"""Purpose: Copy completed, verified replay archives to a private Drive remote.

Responsibilities: backup_prior_sessions validates compressed tick tapes and
integrity sidecars, uploads the exact bytes with rclone, and persists a local
receipt only after remote MD5 verification. Research JSON is included if present.
Ownership: ReplayArchive and replay_compaction own local canonical evidence;
this optional, off-market copier never alters their files or live market data.
Safety: No credentials in code, shell commands, network I/O during market hours,
automatic deletes, broker access, or success claims before remote verification.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

from nifty_scalper_bot.storage.replay_archive import iter_session

_IST = ZoneInfo("Asia/Kolkata")
_REMOTE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,39}:$")
_MD5 = re.compile(r"^([0-9a-fA-F]{32})\s+")


def _hashes(path: Path) -> dict[str, Any]:
    sha = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            sha.update(chunk)
            md5.update(chunk)
            size += len(chunk)
    return {
        "sha256": sha.hexdigest(),
        "md5": md5.hexdigest(),
        "size": size,
    }


def _rclone(
    args: list[str], *, runner: Callable[..., Any] = subprocess.run
) -> str:
    try:
        result = runner(
            ["rclone", *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=1800,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("replay_drive_transfer_failed") from exc
    if result.returncode != 0:
        # Do not log provider error bodies: they can contain identifiers.
        raise RuntimeError(f"replay_drive_{args[0]}_failed")
    return str(result.stdout)


def _sidecars(directory: Path, day: str) -> list[Path]:
    matched = []
    for path in sorted(directory.glob("*.integrity.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            count = int(value.get("written_by_day", {}).get(day, 0))
        except (OSError, ValueError, TypeError, AttributeError):
            continue
        if count > 0:
            matched.append(path)
    if not matched:
        raise ValueError("replay_drive_integrity_sidecars_missing")
    return matched


def _one_day(
    data_dir: Path,
    tape: Path,
    remote: str,
    *,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    day = tape.name.removesuffix(".jsonl.gz")
    # Strictly validate the original event sequence and per-run sidecars.
    events = sum(1 for _ in iter_session(tape))
    sources: list[tuple[str, Path]] = [(tape.name, tape)]
    sources.extend((p.name, p) for p in _sidecars(tape.parent, day))
    for label, folder in (
        ("capture-fidelity.json", data_dir / "research" / "fidelity"),
        ("market-research.json", data_dir / "research" / "markouts"),
    ):
        candidate = folder / f"{day}.json"
        if candidate.is_file():
            sources.append((label, candidate))

    manifest_dir = data_dir / "replay_drive_backups"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{day}.manifest.json"
    receipt_path = manifest_dir / f"{day}.receipt.json"
    tracked = [
        {"name": name, **_hashes(source)}
        for name, source in sources
    ]
    manifest = {
        "schema_version": 1,
        "session_date_ist": day,
        "source": "production_recorded_websocket_feed",
        "event_count": events,
        "archive_integrity_verified": True,
        "exchange_message_completeness_proven": False,
        "files": tracked,
    }
    if receipt_path.exists():
        old = json.loads(receipt_path.read_text(encoding="utf-8"))
        if old.get("files") == tracked and old.get("remote") == remote:
            return {"day": day, "status": "previously_verified"}
        raise ValueError("replay_drive_existing_receipt_conflict")

    # A completed archive must never change after packaging or while uploading.
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    sources.append((manifest_path.name, manifest_path))
    remote_prefix = f"{remote}{day[:4]}/{day}/"
    for name, source in sources:
        expected = _hashes(source)
        destination = remote_prefix + name
        _rclone(
            ["copyto", str(source), destination, "--checksum", "--immutable"],
            runner=runner,
        )
        md5_lines = _rclone(["md5sum", destination], runner=runner)
        match = _MD5.match(md5_lines.strip())
        if match is None or match.group(1).lower() != expected["md5"]:
            raise ValueError("replay_drive_remote_checksum_mismatch")
        if _hashes(source) != expected:
            raise ValueError("replay_drive_source_changed_during_upload")

    receipt = {
        "schema_version": 1,
        "session_date_ist": day,
        "remote": remote,
        "files": tracked,
        "remote_md5_verified": True,
        "verified_at": datetime.now(_IST).isoformat(),
    }
    temporary = receipt_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(receipt_path)
    return {"day": day, "status": "uploaded_verified", "files": len(sources)}


def backup_prior_sessions(
    data_dir: Path,
    *,
    remote: str,
    now: datetime | None = None,
    runner: Callable[..., Any] = subprocess.run,
    limit: int = 1,
) -> list[dict[str, Any]]:
    """Upload at most one sealed prior-day capture per off-market invocation."""
    moment = now or datetime.now(_IST)
    if moment.tzinfo is None:
        raise ValueError("replay_drive_requires_timezone")
    local = moment.astimezone(_IST)
    # Reserve 07:00–17:00 IST for pre-open and live trading without disk pressure.
    if 7 <= local.hour < 17 or not remote:
        return []
    if not _REMOTE.fullmatch(remote):
        raise ValueError("replay_drive_remote_name_invalid")
    if shutil.which("rclone") is None:
        raise RuntimeError("replay_drive_rclone_not_installed")
    if limit < 1:
        raise ValueError("replay_drive_limit_invalid")

    reports = []
    archive = data_dir / "replay_archive"
    for tape in sorted(archive.glob("????-??-??.jsonl.gz")):
        day = tape.name.removesuffix(".jsonl.gz")
        if day >= local.date().isoformat():
            continue
        receipt = data_dir / "replay_drive_backups" / f"{day}.receipt.json"
        if receipt.is_file():
            continue
        reports.append(_one_day(data_dir, tape, remote, runner=runner))
        if len(reports) >= limit:
            break
    return reports
