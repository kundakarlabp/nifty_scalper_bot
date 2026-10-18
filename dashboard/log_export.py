"""Bounded, timezone-correct journal exports for the read-only console."""
from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

from dashboard.superlite_events import (
    SECRET,
    csv_bytes as csv_bytes,
    deduplicate_events,
    filter_events as filter_events,
    parse_event,
)

IST = ZoneInfo("Asia/Kolkata")
MAX_EXPORT_BYTES = 24 * 1024 * 1024


@dataclass(frozen=True)
class JournalResult:
    data: bytes
    count: int
    truncated: bool
    error: str | None = None


def window_epochs(selected_date: date, start_at: time, end_at: time) -> tuple[int, int]:
    if start_at >= end_at:
        raise ValueError("From time must be earlier than To time")
    start_dt = datetime.combine(selected_date, start_at, tzinfo=IST)
    end_dt = datetime.combine(selected_date, end_at, tzinfo=IST)
    return int(start_dt.timestamp()), int(end_dt.timestamp())


def journal_command(service: str, selected_date: date, start_at: time, end_at: time, output: str) -> list[str]:
    since_epoch, until_epoch = window_epochs(selected_date, start_at, end_at)
    return [
        "journalctl", "-u", service,
        "--since", f"@{since_epoch}",
        "--until", f"@{until_epoch}",
        "--no-pager", "-o", output,
    ]


def run_journal(command: list[str], max_bytes: int = MAX_EXPORT_BYTES) -> JournalResult:
    max_bytes = max(1_000_000, min(int(max_bytes), 100 * 1024 * 1024))
    try:
        with tempfile.TemporaryFile() as target:
            completed = subprocess.run(
                command,
                stdout=target,
                stderr=subprocess.PIPE,
                timeout=35,
                check=False,
            )
            if completed.returncode:
                error = (completed.stderr or b"").decode("utf-8", errors="replace").strip()
                return JournalResult(b"", 0, False, error or "journal query failed")
            size = target.tell()
            truncated = size > max_bytes
            target.seek(max(0, size - max_bytes) if truncated else 0)
            payload = target.read(max_bytes)
    except subprocess.TimeoutExpired:
        return JournalResult(b"", 0, False, "journal query exceeded 35 seconds")
    except Exception as exc:
        return JournalResult(b"", 0, False, f"{type(exc).__name__}: {exc}")
    return JournalResult(payload, payload.count(b"\n"), truncated)


def read_raw_logs(
    service: str,
    selected_date: date,
    start_at: time,
    end_at: time,
    contains: str = "",
) -> JournalResult:
    result = run_journal(journal_command(service, selected_date, start_at, end_at, "short-iso-precise"))
    if result.error:
        return result
    lines = result.data.decode("utf-8", errors="replace").splitlines()
    needle = contains.strip().lower()
    if needle:
        lines = [line for line in lines if needle in line.lower()]
    lines = [SECRET.sub(r"\1=[REDACTED]", line) for line in lines]
    payload = ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")
    return JournalResult(payload, len(lines), result.truncated)


def read_actionable_events(
    service: str,
    selected_date: date,
    start_at: time,
    end_at: time,
) -> tuple[list[dict[str, str]], JournalResult]:
    result = run_journal(journal_command(service, selected_date, start_at, end_at, "cat"))
    if result.error:
        return [], result
    rows = [
        event
        for line in result.data.decode("utf-8", errors="replace").splitlines()
        if (event := parse_event(line))
    ]
    rows = deduplicate_events(rows)
    return rows, JournalResult(b"", len(rows), result.truncated)
