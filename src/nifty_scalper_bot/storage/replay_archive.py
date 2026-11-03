"""Purpose: Capture ordered normalized market observations for replay without blocking trading.

Responsibilities: ReplayArchive.record queues delivered ticks and snapshots;
the background writer serializes JSON, reuses buffered day files, and writes
integrity sidecars with queue-pressure and optional upstream-ingress counters.
iter_session validates exact sequence, writer totals and reported upstream loss.

Ownership: ReplayArchive owns the authoritative local tick tape and writer
integrity; MarketDataManager owns WebSocket ingestion and source quality.
Safety: Writer failures, queue overflow and upstream drops/coalescing invalidate
research replay; never infer exchange delivery, depth, or fills from missing
observations. No broker actions, credentials or remote uploads occur here.
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
from typing import Any, Callable, Iterator, Mapping
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
    """Non-blocking tick intake with one background serialization/disk writer.

    The live path only copies a shallow, already-normalized observation into a
    bounded queue. The writer owns JSON conversion, file handles and sidecars.
    Capture loss fails replay integrity rather than silently sampling ticks.
    """

    def __init__(
        self,
        directory: Path,
        *,
        capacity: int = 8192,
        max_session_bytes: int = 512 * 1024 * 1024,
        upstream_provider: Callable[[], Mapping[str, int]] | None = None,
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
        self._max_pending = 0
        self._max_writer_lag_ms = 0.0
        self._upstream_provider = upstream_provider
        self._upstream_start = self._upstream_health()
        self._queue: queue.Queue[tuple[dict[str, Any], float]] = queue.Queue(
            maxsize=capacity
        )
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._write, name="replay-archive", daemon=True
        )
        self._thread.start()

    def _upstream_health(self) -> dict[str, int] | None:
        if self._upstream_provider is None:
            return None
        try:
            return {
                str(key): int(value)
                for key, value in self._upstream_provider().items()
                if type(value) is int
            }
        except Exception:  # noqa: BLE001 - observation only
            return None

    def record(
        self, kind: str, payload: Mapping[str, Any], available_at: datetime
    ) -> None:
        """Enqueue without JSON encoding or filesystem access on the tick path."""
        if available_at.tzinfo is None:
            raise ValueError("replay_archive_requires_timezone")
        # The caller owns an independent canonical tick mapping. Preserve
        # nested normalized quote fields unchanged; never mutate them here.
        shallow_payload = dict(payload)
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
                "payload": shallow_payload,
            }
            try:
                self._queue.put_nowait((event, time.monotonic()))
                self._max_pending = max(self._max_pending, self._queue.qsize())
            except queue.Full:
                self.dropped += 1

    def stats(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "written": self.written,
            "dropped": self.dropped,
            "failed": self.failed,
            "pending": self._queue.qsize(),
            "max_pending": self._max_pending,
            "queue_capacity": self._queue.maxsize,
            "max_writer_lag_ms": round(self._max_writer_lag_ms, 3),
            "written_by_day": dict(self._written_by_day),
            "upstream_start": self._upstream_start,
            "upstream_latest": self._upstream_health(),
        }

    def close(self) -> None:
        with self._lock:
            self._closed = True
        self._stop.set()
        self._thread.join(timeout=10)
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
        current_day: str | None = None
        current_bytes = 0
        handle: Any = None
        try:
            while not self._stop.is_set() or not self._queue.empty():
                try:
                    event, queued_at = self._queue.get(timeout=0.1)
                except queue.Empty:
                    if time.monotonic() - self._last_integrity_write >= 1:
                        try:
                            if handle is not None:
                                handle.flush()
                        except OSError:
                            self.failed += 1
                        self._persist_integrity()
                    continue
                try:
                    lag_ms = (time.monotonic() - queued_at) * 1000.0
                    self._max_writer_lag_ms = max(
                        self._max_writer_lag_ms, lag_ms
                    )
                    day = datetime.fromisoformat(
                        event["available_at"]
                    ).astimezone(IST).date().isoformat()
                    encoded = json.dumps(
                        {
                            **event,
                            "payload": to_json_safe(event["payload"]),
                        },
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                    row_bytes = len(encoded.encode("utf-8")) + 1
                    if day != current_day:
                        if handle is not None:
                            handle.flush()
                            handle.close()
                        handle = None
                        self.directory.mkdir(parents=True, exist_ok=True)
                        path = self.directory / f"{day}.jsonl"
                        current_bytes = path.stat().st_size if path.exists() else 0
                        handle = path.open(
                            "a", encoding="utf-8", buffering=1024 * 1024
                        )
                        current_day = day
                    if current_bytes + row_bytes > self.max_session_bytes:
                        raise OSError("replay_archive_session_size_limit")
                    handle.write(encoded + "\n")
                    current_bytes += row_bytes
                    self.written += 1
                    self._written_by_day[day] = self._written_by_day.get(day, 0) + 1
                except (OSError, ValueError, TypeError):
                    self.failed += 1
                finally:
                    self._queue.task_done()
                if time.monotonic() - self._last_integrity_write >= 1:
                    try:
                        if handle is not None:
                            handle.flush()
                    except OSError:
                        self.failed += 1
                    self._persist_integrity()
        finally:
            try:
                if handle is not None:
                    handle.flush()
                    handle.close()
            except OSError:
                self.failed += 1
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
        initial = integrity.get("upstream_start")
        ending = integrity.get("upstream_latest")
        if isinstance(initial, dict) and isinstance(ending, dict):
            for key in ("dropped", "coalesced", "legacy_queue_dropped"):
                if key in initial and key in ending and ending[key] > initial[key]:
                    raise ValueError("replay_capture_upstream_loss")
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
