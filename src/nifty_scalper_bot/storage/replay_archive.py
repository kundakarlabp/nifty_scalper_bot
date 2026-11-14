"""Purpose: Capture normalized market observations without blocking trading.

Responsibilities: Record ordered tick/snapshot events via the bounded
ReplayArchive queue; the background writer owns serialization, buffered
session files, and integrity/ingress accounting. Restore verified plain or
compacted gzip session tapes without changing record identities.
Ownership: MarketDataManager owns tick quality; ReplayArchive owns the local
capture and source-preserving integrity sidecars.
Safety: Do not block trading, manufacture quotes, or infer fills. Dropped,
coalesced, or corrupted records invalidate research evidence.
verify_raw_ingress_session checks exact per-contract/generation receipt sequences
and observed-versus-persisted per-day counts before certifying a wire tape.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import queue
import shutil
import threading
import time
import uuid
from copy import deepcopy
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
        self.requested_max_session_bytes = max_session_bytes
        self.max_session_bytes = max_session_bytes
        self.capacity_guard = "requested"
        # This is checked once when the recorder starts, not on the tick path.
        # Never raise the raw-tape limit unless the host can retain a 2 GiB
        # emergency disk reserve alongside the complete requested session.
        if max_session_bytes > 512 * 1024 * 1024:
            try:
                usage_path = directory if directory.exists() else directory.parent
                available = shutil.disk_usage(usage_path).free
                if available < max_session_bytes + 2 * 1024 * 1024 * 1024:
                    self.max_session_bytes = 512 * 1024 * 1024
                    self.capacity_guard = "insufficient_disk_headroom"
            except OSError:
                self.max_session_bytes = 512 * 1024 * 1024
                self.capacity_guard = "disk_headroom_unverified"
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
        # The queue owns its snapshot: depth/OHLC and other nested broker
        # fields must not change if the bus later mutates the caller's tick.
        # Avoid the expensive JSON conversion and filesystem I/O here.
        shallow_payload = {
            key: deepcopy(value) if isinstance(value, (dict, list, tuple)) else value
            for key, value in payload.items()
        }
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
            "requested_max_session_bytes": self.requested_max_session_bytes,
            "max_session_bytes": self.max_session_bytes,
            "capacity_guard": self.capacity_guard,
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
                    self._max_writer_lag_ms = max(self._max_writer_lag_ms, lag_ms)
                    day = (
                        datetime.fromisoformat(event["available_at"])
                        .astimezone(IST)
                        .date()
                        .isoformat()
                    )
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
                        handle = path.open("a", encoding="utf-8", buffering=1024 * 1024)
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
    reader = gzip.open if path.name.endswith(".jsonl.gz") else Path.open
    with reader(path, mode="rt", encoding="utf-8") as handle:
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
    session_day = (
        path.name.removesuffix(".jsonl.gz")
        if path.name.endswith(".jsonl.gz")
        else path.stem
    )
    for run in sequences:
        integrity_path = path.parent / f"{run}.integrity.json"
        if not integrity_path.is_file():
            raise ValueError("replay_capture_integrity_missing")
        integrity = json.loads(integrity_path.read_text())
        if (
            integrity.get("dropped")
            or integrity.get("failed")
            or integrity.get("written_by_day", {}).get(session_day) != counts[run]
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

def verify_raw_ingress_session(path: Path) -> dict[str, Any]:
    """Fail closed on missing pre-queue wire observations per symbol/epoch.

    This *wire* evidence is separate from the normalized decision-replay tape.
    It proves locally received observations were captured, not that Zerodha
    transmitted every exchange quote or that a strategy replay is executable.
    """
    day = (
        path.name.removesuffix(".jsonl.gz")
        if path.name.endswith(".jsonl.gz")
        else path.stem
    )
    seen: dict[str, dict[str, int]] = {}
    last_sequence: dict[tuple[str, str], int] = {}
    for event in iter_session(path):
        run = str(event["run_id"])
        payload = event["payload"]
        if payload.get("capture_stream") != "raw_ws_ingress":
            raise ValueError("raw_ingress_stream_identity_missing")
        if event["kind"] == "snapshot":
            continue
        receipt = payload.get("receipt")
        wire = payload.get("wire")
        if not isinstance(receipt, dict) or not isinstance(wire, dict):
            raise ValueError("raw_ingress_wire_or_receipt_missing")
        symbol = receipt.get("symbol")
        token = receipt.get("token")
        generation = receipt.get("generation")
        sequence = receipt.get("sequence")
        if (
            not isinstance(symbol, str)
            or not symbol
            or type(generation) is not int
            or type(sequence) is not int
            or sequence <= 0
        ):
            raise ValueError("raw_ingress_identity_invalid")
        identity = f"{symbol}|{token}|{generation}"
        prior = last_sequence.get((run, identity))
        if prior is not None and sequence != prior + 1:
            raise ValueError("raw_ingress_contract_epoch_sequence_gap")
        last_sequence[run, identity] = sequence
        bucket = seen.setdefault(run, {})
        bucket[identity] = bucket.get(identity, 0) + 1

    for run, by_identity in seen.items():
        integrity = json.loads(
            (path.parent / f"{run}.integrity.json").read_text()
        )
        starting = integrity.get("upstream_start")
        ending = integrity.get("upstream_latest")
        if not isinstance(starting, dict) or not isinstance(ending, dict):
            raise ValueError("raw_ingress_upstream_counts_missing")
        if int(ending.get("capture_errors", 0)) != 0:
            raise ValueError("raw_ingress_capture_error_observed")
        for identity, count in by_identity.items():
            key = f"day_receipt:{day}|{identity}"
            expected = int(ending.get(key, 0)) - int(starting.get(key, 0))
            if expected != count:
                raise ValueError("raw_ingress_contract_epoch_count_mismatch")
        for key, value in ending.items():
            prefix = f"day_receipt:{day}|"
            if not key.startswith(prefix):
                continue
            identity = key.removeprefix(prefix)
            expected = int(value) - int(starting.get(key, 0))
            if by_identity.get(identity, 0) != expected:
                raise ValueError("raw_ingress_contract_epoch_missing")
    if not seen:
        raise ValueError("raw_ingress_no_wire_observations")
    return {
        "verified": True,
        "wire_events": sum(sum(by_id.values()) for by_id in seen.values()),
        "counts_by_run_and_contract_epoch": seen,
        "sha256": archive_fingerprint(path),
    }

