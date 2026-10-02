"""Bounded, asynchronous archive of the observations delivered to the live runner.

Archive failures never block trading. Sequence gaps and writer failures invalidate
replay evidence; missing depth is preserved rather than manufactured.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
import time
import uuid
from dataclasses import fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Mapping
from zoneinfo import ZoneInfo

from nifty_scalper_bot.utils.serialization import to_json_safe

IST = ZoneInfo("Asia/Kolkata")
_PREFIXES = (
    "ORB_",
    "SMC_",
    "VWAP_",
    "STRATEGY_",
    "RISK_",
    "RUNNER_",
    "REGIME_",
    "READINESS_",
    "EXECUTION_",
    "ORDERS_",
    "SELECTOR_",
    "LIQUIDITY_",
    "OPTION_",
    "ATR_",
    "MIN_",
    "MAX_",
    "CONFIDENCE_",
    "GLOBAL_",
)


def capture_environment() -> dict[str, str]:
    """Capture decision settings only; never archive operator credentials."""
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith(_PREFIXES)
        and not any(
            word in key for word in ("SECRET", "PASSWORD", "API_KEY", "TOKEN", "URL")
        )
    }


def capture_settings(settings: Any) -> dict[str, Any]:
    """Serialize effective decision sections including nested Pydantic models."""

    def encode(value: Any) -> Any:
        if is_dataclass(value):
            return {
                field.name: encode(getattr(value, field.name))
                for field in fields(value)
            }
        if hasattr(value, "model_dump"):
            return encode(value.model_dump(mode="json"))
        if isinstance(value, dict):
            return {key: encode(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set, frozenset)):
            return [encode(item) for item in value]
        return to_json_safe(value)

    sections = (
        "risk",
        "orders",
        "elite",
        "execution",
        "regime",
        "selector",
        "liquidity",
        "option_universe",
    )
    return {key: encode(getattr(settings, key)) for key in sections}


class ReplayArchive:
    """Non-blocking append queue with explicit integrity counters."""

    def __init__(
        self,
        directory: Path,
        *,
        capacity: int = 8192,
        max_session_bytes: int = 512 * 1024 * 1024,
    ) -> None:
        self.directory = directory
        self.max_session_bytes = max_session_bytes
        self.run_id = uuid.uuid4().hex
        self.dropped = 0
        self.failed = 0
        self.written = 0
        self._written_by_day: dict[str, int] = {}
        self._last_integrity_write = 0.0
        self._sequence = 0
        self._closed = False
        self._lock = threading.Lock()
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=capacity)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._write, name="replay-archive", daemon=True
        )
        self._thread.start()

    def record(
        self, kind: str, payload: Mapping[str, Any], available_at: datetime
    ) -> None:
        """Preserve observation order, reception time and raw market quality."""
        if available_at.tzinfo is None:
            raise ValueError("replay_archive_requires_timezone")
        with self._lock:
            if self._closed:
                return
            self._sequence += 1
            event = {
                "schema_version": 1,
                "run_id": self.run_id,
                "sequence": self._sequence,
                "kind": kind,
                "available_at": available_at.isoformat(),
                "payload": to_json_safe(dict(payload)),
            }
            try:
                self._queue.put_nowait(event)
            except queue.Full:
                self.dropped += 1

    def stats(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "written": self.written,
            "dropped": self.dropped,
            "failed": self.failed,
            "pending": self._queue.qsize(),
            "written_by_day": dict(self._written_by_day),
        }

    def close(self) -> None:
        with self._lock:
            self._closed = True
        self._stop.set()
        self._thread.join(timeout=5)
        if not self._thread.is_alive():
            self._persist_integrity()

    def _persist_integrity(self) -> None:
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / f"{self.run_id}.integrity.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.stats()))
            temporary.replace(path)
            self._last_integrity_write = time.monotonic()
        except OSError:
            self.failed += 1

    def _write(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                event = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
                day = (
                    datetime.fromisoformat(event["available_at"]).astimezone(IST).date()
                )
                path = self.directory / f"{day}.jsonl"
                encoded = json.dumps(event, allow_nan=False, separators=(",", ":"))
                if (
                    path.exists()
                    and path.stat().st_size + len(encoded.encode())
                    > self.max_session_bytes
                ):
                    raise OSError("replay_archive_session_size_limit")
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(encoded + "\n")
                self.written += 1
                day_key = day.isoformat()
                self._written_by_day[day_key] = self._written_by_day.get(day_key, 0) + 1
            except (OSError, ValueError, TypeError):
                self.failed += 1
            finally:
                self._queue.task_done()
            if (
                self._queue.empty()
                or time.monotonic() - self._last_integrity_write >= 1.0
            ):
                self._persist_integrity()


def iter_session(path: Path) -> Iterator[dict[str, Any]]:
    """Reject partial writes, sequence gaps, failed capture and backwards clocks."""
    event_count = 0
    first = True
    sequences: dict[str, int] = {}
    counts: dict[str, int] = {}
    previous: datetime | None = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                event = json.loads(line)
                if event.get("schema_version") != 1 or event["kind"] not in {
                    "snapshot",
                    "tick",
                }:
                    raise ValueError
                timestamp = datetime.fromisoformat(event["available_at"])
                if timestamp.tzinfo is None or (
                    previous is not None and timestamp < previous
                ):
                    raise ValueError
                run = event["run_id"]
                sequence = event["sequence"]
                if type(sequence) is not int or sequence <= 0:
                    raise ValueError
                if run in sequences and sequence != sequences[run] + 1:
                    raise ValueError
                if not isinstance(event["payload"], dict):
                    raise ValueError
                if (first or run not in sequences) and event["kind"] != "snapshot":
                    raise ValueError("replay_initial_snapshot_missing")
                first = False
                sequences[run] = sequence
                counts[run] = counts.get(run, 0) + 1
                previous = timestamp
                event_count += 1
                yield event
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError("invalid_replay_event") from exc
    for run in sequences:
        integrity_path = path.parent / f"{run}.integrity.json"
        if not integrity_path.is_file():
            raise ValueError("replay_capture_integrity_missing")
        integrity = json.loads(integrity_path.read_text())
        if (
            integrity.get("dropped")
            or integrity.get("failed")
            or integrity.get("written_by_day", {}).get(path.stem) != counts[run]
        ):
            raise ValueError("replay_capture_incomplete")
    if not event_count:
        raise ValueError("replay_initial_snapshot_missing")


def load_session(path: Path) -> list[dict[str, Any]]:
    """Small-fixture convenience; production replay streams bounded records."""
    return list(iter_session(path))


def archive_fingerprint(path: Path) -> str:
    """Hash exact replay bytes for reproducible evidence."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
