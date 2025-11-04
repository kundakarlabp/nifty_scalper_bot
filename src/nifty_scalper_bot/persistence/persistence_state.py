"""Persistence and periodic flush worker for Nifty Scalper Bot (robust production-grade).

Features:
- Thread-safe in-memory `state` (documented) with context manager.
- Atomic writes to disk and backup rotation.
- Pydantic schema validation if pydantic available (fail-fast or warn depending on config).
- Optional event-driven flush, manual flush, background periodic worker.
- Health/status API (function-based; optional embedded HTTP server via `enable_http=True`).
- Emits throttled diagnostics via global `system_sampler`.
- Simple metrics counters and placeholder for Prometheus export.
- Encryption/compression stubs provided for secure storage integrations.
- Designed to be minimal dependency on external infra; easily extended for S3/DB later.

Usage (recommended):
- from nifty_scalper_bot.persistence.persistence_state import initialize, start_worker
- initialize(auto_start_worker=True)
- use update_state(...) to update app state
- call manual_flush_and_report() for on-demand persistence

Note:
- This file intentionally keeps business-model-agnostic snapshotting. If your
  domain objects are not JSON-serializable, adapt `snapshot_for_persistence`.
"""

from __future__ import annotations

import json
import os
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from nifty_scalper_bot.utils.logging import get_logger
from nifty_scalper_bot.utils.sampler_singleton import system_sampler, get_symbol_sampler

LOGGER = get_logger(__name__)

# ---------------------------------------------------------------------
# Config (env-driven)
# ---------------------------------------------------------------------
PNL_PERSIST_PATH = Path(os.getenv("PNL_PERSIST_PATH", "data/pnl_state.json"))
PERSIST_FLUSH_INTERVAL_S = float(os.getenv("PERSIST_FLUSH_INTERVAL_S", "10"))
PERSIST_RETENTION_BACKUPS = int(os.getenv("PERSIST_RETENTION_BACKUPS", "3"))
PERSIST_ENABLE_SCHEMA_VALIDATION = os.getenv("PERSIST_ENABLE_SCHEMA_VALIDATION", "true").lower() in ("1", "true", "yes")
PERSIST_ENABLE_EMBEDDED_HTTP = os.getenv("ENABLE_EMBEDDED_HTTP_SERVER", "false").lower() in ("1", "true", "yes")
PERSIST_HTTP_HOST = os.getenv("PERSIST_HTTP_HOST", "127.0.0.1")
PERSIST_HTTP_PORT = int(os.getenv("PERSIST_HTTP_PORT", "9234"))
PERSIST_COMPRESSION = os.getenv("PERSIST_COMPRESSION", "none")  # none, gzip (placeholder)
PERSIST_ENCRYPTION = os.getenv("PERSIST_ENCRYPTION", "none")    # none, kms (placeholder)
PERSIST_LOG_LEVEL_ON_SUCCESS = os.getenv("PERSIST_LOG_LEVEL_ON_SUCCESS", "info")  # info/debug


# ---------------------------------------------------------------------
# Optional schema model (pydantic if available)
# ---------------------------------------------------------------------
try:
    from pydantic import BaseModel, ValidationError

    class PersistModel(BaseModel):
        """Canonical minimal model for persisted state — extend as needed."""
        last_pnl: float = 0.0
        positions: Dict[str, Any] = {}
        orders: Dict[str, Any] = {}
        meta: Dict[str, Any] = {}

    def validate_snapshot(obj: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        try:
            PersistModel.parse_obj(obj)
            return True, None
        except ValidationError as exc:
            return False, str(exc)

except Exception:
    PersistModel = None  # type: ignore
    ValidationError = Exception  # type: ignore

    def validate_snapshot(obj: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        # Best-effort structural validation (lightweight)
        if not isinstance(obj, dict):
            return False, "snapshot must be a dict"
        # optional keys check - tolerate missing keys but warn
        # Return valid; caller may choose to warn instead of failing
        return True, None


# ---------------------------------------------------------------------
# In-memory state (documented)
# ---------------------------------------------------------------------
# What modules should keep in `state` (recommended minimal shape):
# {
#   "last_pnl": float,
#   "positions": {symbol: {"qty": int, "avg_price": float, ...}},
#   "orders": {order_id: {...}},
#   "meta": {"last_flush": epoch_float, "version": "1.0"}
# }
_state_lock = threading.RLock()
_state: Dict[str, Any] = {
    "last_pnl": 0.0,
    "positions": {},
    "orders": {},
    "meta": {"version": "1.0"},
}

# Worker control
_worker_thread: Optional[threading.Thread] = None
_worker_stop = threading.Event()

# Counters / metrics
_metrics_lock = threading.Lock()
_metrics = {
    "persist_emitted_total": 0,
    "persist_suppressed_total": 0,
    "persist_errors_total": 0,
    "last_flush_epoch": 0.0,
}


# ---------------------------------------------------------------------
# Helpers: atomic write / rotate / optional compression / encryption stub
# ---------------------------------------------------------------------
def _atomic_write(path: Path, data_bytes: bytes) -> None:
    """Atomic write: write to tmp file then replace target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data_bytes)
    tmp.replace(path)


def _rotate_backups(path: Path, keep: int) -> None:
    """Rotate simple numeric backups path -> path.1, path.1 -> path.2, ..."""
    try:
        for i in range(keep - 1, 0, -1):
            src = path.with_suffix(path.suffix + f".{i}")
            dst = path.with_suffix(path.suffix + f".{i+1}")
            if src.exists():
                src.replace(dst)
        if path.exists():
            path.replace(path.with_suffix(path.suffix + ".1"))
    except Exception:
        LOGGER.exception("persist: backup rotation failed", extra={"event": "persist_backup_rotation_failed"})


def _compress_bytes(data: bytes) -> bytes:
    """Placeholder: add gzip compression if configured."""
    if PERSIST_COMPRESSION.lower() == "gzip":
        import gzip

        return gzip.compress(data)
    return data


def _maybe_encrypt_bytes(data: bytes) -> bytes:
    """Placeholder for encryption. Integrate KMS / envelope encryption here."""
    if PERSIST_ENCRYPTION.lower() != "none":
        # No-op placeholder: implement encryption using your preferred KMS/service.
        # e.g., use AWS KMS to encrypt a data key then use AES-GCM to encrypt bytes.
        LOGGER.debug("persist: encryption requested but not implemented; storing plaintext", extra={"event": "persist_encrypt_stub"})
    return data


def _prepare_bytes_for_write(snapshot_json: str) -> bytes:
    raw = snapshot_json.encode("utf-8")
    raw = _compress_bytes(raw)
    raw = _maybe_encrypt_bytes(raw)
    return raw


# ---------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------
@contextmanager
def locked_state():
    """Context manager for safely accessing the global state dict."""
    with _state_lock:
        yield _state


def get_state_snapshot() -> Dict[str, Any]:
    """Return a deep-serializable snapshot of state. Copy under lock."""
    with locked_state():
        # shallow copy is often enough — adapt if you have nested non-serializable objects
        return json.loads(json.dumps(_state, default=str))  # convert to primitives


# ---------------------------------------------------------------------
# Persistence API
# ---------------------------------------------------------------------
def snapshot_for_persistence() -> Dict[str, Any]:
    """Prepare the state to persist. Override/extend in your app to customize."""
    return get_state_snapshot()


def dump_state_to_disk(path: Optional[Path | str] = None, *, validate: bool = True) -> None:
    """Persist the current in-memory state to disk (atomic)."""
    global _metrics
    p = Path(path or PNL_PERSIST_PATH)
    try:
        snapshot = snapshot_for_persistence()
        if validate and PERSIST_ENABLE_SCHEMA_VALIDATION:
            ok, err = validate_snapshot(snapshot)
            if not ok:
                LOGGER.warning("persist: snapshot validation failed: %s", err, extra={"event": "persist_validation_failed"})
                # Do not block persistence by default; record metric and continue.
                with _metrics_lock:
                    _metrics["persist_suppressed_total"] += 1

        snapshot_json = json.dumps(snapshot, ensure_ascii=False, indent=2)
        payload = _prepare_bytes_for_write(snapshot_json)

        _rotate_backups(p, PERSIST_RETENTION_BACKUPS)
        _atomic_write(p, payload)

        # update metrics & emit diagnostic via singleton sampler
        with _metrics_lock:
            _metrics["persist_emitted_total"] += 1
            _metrics["last_flush_epoch"] = time.time()

        system_sampler.maybe_log(
            LOGGER,
            event="persistent_state_flush_success",
            regime=None,
            multiplier=None,
            extra={"path": str(p), "size_bytes": len(payload)},
            level=PERSIST_LOG_LEVEL_ON_SUCCESS,
        )

        LOGGER.debug("persist: wrote snapshot to %s (bytes=%d)", str(p), len(payload), extra={"event": "persist_write_ok"})
    except Exception as exc:  # noqa: BLE001
        with _metrics_lock:
            _metrics["persist_errors_total"] += 1
        LOGGER.exception("persist: failed to write state: %s", exc, extra={"event": "persist_write_failed", "trace": traceback.format_exc()})
        system_sampler.maybe_log(
            LOGGER,
            event="persistent_flush_failed",
            regime=None,
            multiplier=None,
            extra={"path": str(p), "error": str(exc)},
            level="error",
        )


def load_state_from_disk(path: Optional[Path | str] = None, *, validate: bool = False) -> None:
    """Load state from disk into in-memory `_state`. Overrides in-memory state."""
    p = Path(path or PNL_PERSIST_PATH)
    if not p.exists():
        LOGGER.info("persist: no state file found at %s", p, extra={"event": "persist_load_missing"})
        return
    try:
        raw = p.read_bytes()
        # decrypt/decompress if implemented (no-op as placeholder)
        # currently assumes plaintext UTF-8 JSON
        try:
            data_text = raw.decode("utf-8")
        except Exception:
            # try gzip decompress if compressed
            import gzip

            data_text = gzip.decompress(raw).decode("utf-8")
        obj = json.loads(data_text)
        if validate and PERSIST_ENABLE_SCHEMA_VALIDATION:
            ok, err = validate_snapshot(obj)
            if not ok:
                LOGGER.warning("persist: loaded snapshot failed validation: %s", err, extra={"event": "persist_load_validation_failed"})
        with locked_state():
            _state.clear()
            if isinstance(obj, dict):
                _state.update(obj)
        system_sampler.maybe_log(
            LOGGER,
            event="persistent_state_loaded",
            regime=None,
            multiplier=None,
            extra={"path": str(p), "keys": list(_state.keys())},
        )
        LOGGER.debug("persist: loaded state from %s", p, extra={"event": "persist_load_ok"})
    except Exception as exc:  # noqa: BLE001
        with _metrics_lock:
            _metrics["persist_errors_total"] += 1
        LOGGER.exception("persist: failed to load state: %s", exc, extra={"event": "persist_load_failed"})
        system_sampler.maybe_log(
            LOGGER,
            event="persistent_load_failed",
            regime=None,
            multiplier=None,
            extra={"path": str(p), "error": str(exc)},
            level="error",
        )


def update_state(updates: Dict[str, Any]) -> None:
    """Merge updates into in-memory `_state` under lock. Caller ensures types/semantics."""
    with locked_state():
        _state.update(updates)


# ---------------------------------------------------------------------
# Event-driven flush API and queue (lightweight)
# ---------------------------------------------------------------------
_event_queue: "list[Tuple[str, Dict[str, Any]]]" = []
_event_queue_lock = threading.Lock()


def flush_on_event(event_name: str, extra: Optional[Dict[str, Any]] = None, *, immediate: bool = False) -> None:
    """
    Request a persistence flush for a specific event. If immediate=True, flush now.
    Otherwise the event is enqueued and the background worker will flush periodically.
    """
    extra = extra or {}
    if immediate:
        dump_state_to_disk()
        return
    with _event_queue_lock:
        _event_queue.append((event_name, extra))


def _drain_event_queue() -> Dict[str, Dict[str, Any]]:
    """Drain events into a dict of aggregated extras keyed by event name."""
    out: Dict[str, Dict[str, Any]] = {}
    with _event_queue_lock:
        while _event_queue:
            ev, ex = _event_queue.pop(0)
            if ev not in out:
                out[ev] = {"count": 0, "latest": ex}
            out[ev]["count"] += 1
            out[ev]["latest"] = ex
    return out


# ---------------------------------------------------------------------
# Background worker loop
# ---------------------------------------------------------------------
def _worker_loop(interval_s: float) -> None:
    LOGGER.debug("persist: worker loop starting", extra={"event": "persist_worker_start", "interval_s": interval_s})
    try:
        while not _worker_stop.wait(interval_s):
            # drain events (lightweight aggregation) and flush once
            events = _drain_event_queue()
            if events:
                # could include aggregated info into the snapshot meta
                with locked_state():
                    _state.setdefault("meta", {})["last_event_batch"] = {k: v["count"] for k, v in events.items()}
            dump_state_to_disk()
            # heartbeat diagnostic
            system_sampler.maybe_log(
                LOGGER,
                event="persistent_heartbeat_flush",
                regime=None,
                multiplier=None,
                extra={"interval_s": interval_s, "events": len(events)},
            )
    except Exception:
        LOGGER.exception("persist: worker crashed", extra={"event": "persist_worker_crash"})
    finally:
        LOGGER.debug("persist: worker loop exiting", extra={"event": "persist_worker_exit"})


def start_worker(interval_s: Optional[float] = None, *, force_restart: bool = False) -> None:
    """Start background flush worker; idempotent unless force_restart=True."""
    global _worker_thread
    if _worker_thread and _worker_thread.is_alive():
        if not force_restart:
            LOGGER.debug("persist: worker already running", extra={"event": "persist_worker_already"})
            return
        stop_worker()
    _worker_stop.clear()
    iv = float(interval_s or PERSIST_FLUSH_INTERVAL_S)
    _worker_thread = threading.Thread(target=_worker_loop, args=(iv,), daemon=True, name="persist-worker")
    _worker_thread.start()
    system_sampler.maybe_log(
        LOGGER,
        event="persistent_heartbeat_started",
        regime=None,
        multiplier=None,
        extra={"interval_s": iv},
    )


def stop_worker(timeout_s: float = 2.0) -> None:
    _worker_stop.set()
    if _worker_thread:
        _worker_thread.join(timeout_s)
    system_sampler.maybe_log(LOGGER, event="persistent_heartbeat_stopped", regime=None, multiplier=None)


# ---------------------------------------------------------------------
# Health / status API (callable)
# ---------------------------------------------------------------------
@dataclass
class PersistStatus:
    last_flush_epoch: float
    emitted_total: int
    suppressed_total: int
    errors_total: int
    loaded_keys: Tuple[str, ...]
    worker_running: bool


def get_status() -> PersistStatus:
    with _metrics_lock:
        emitted = _metrics.get("persist_emitted_total", 0)
        suppressed = _metrics.get("persist_suppressed_total", 0)
        errors = _metrics.get("persist_errors_total", 0)
        last_flush = _metrics.get("last_flush_epoch", 0.0)
    with _state_lock:
        keys = tuple(_state.keys())
    return PersistStatus(
        last_flush_epoch=last_flush,
        emitted_total=emitted,
        suppressed_total=suppressed,
        errors_total=errors,
        loaded_keys=keys,
        worker_running=bool(_worker_thread and _worker_thread.is_alive()),
    )


# Optional embedded HTTP server (small; opt-in)
if PERSIST_ENABLE_EMBEDDED_HTTP:

    def _make_simple_handler():
        from http.server import BaseHTTPRequestHandler, HTTPServer
        import urllib

        class Handler(BaseHTTPRequestHandler):  # type: ignore
            def do_GET(self):
                if self.path.startswith("/status"):
                    st = get_status()
                    payload = {
                        "last_flush_epoch": st.last_flush_epoch,
                        "emitted_total": st.emitted_total,
                        "suppressed_total": st.suppressed_total,
                        "errors_total": st.errors_total,
                        "loaded_keys": list(st.loaded_keys),
                        "worker_running": st.worker_running,
                    }
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(payload).encode("utf-8"))
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                # silence BaseHTTPRequestHandler default logging
                return

        return Handler

    def start_embedded_http(host: str = PERSIST_HTTP_HOST, port: int = PERSIST_HTTP_PORT) -> None:
        """Start a tiny HTTP server in background to serve /status for ops."""
        handler = _make_simple_handler()
        server = HTTPServer((host, port), handler)
        t = threading.Thread(target=server.serve_forever, daemon=True, name="persist-http")
        t.start()
        LOGGER.info("persist: embedded HTTP server started", extra={"event": "persist_http_started", "host": host, "port": port})
        # no direct stop hook here; server will exit with process

else:
    def start_embedded_http(*_, **__):
        LOGGER.debug("persist: embedded http disabled; set ENABLE_EMBEDDED_HTTP_SERVER=true to enable", extra={"event": "persist_http_disabled"})


# ---------------------------------------------------------------------
# Convenience / CLI helpers
# ---------------------------------------------------------------------
def manual_flush_and_report() -> Dict[str, Any]:
    start = time.time()
    dump_state_to_disk()
    elapsed = time.time() - start
    system_sampler.maybe_log(
        LOGGER,
        event="persistent_flush_recorded",
        regime=None,
        multiplier=None,
        extra={"elapsed_s": elapsed},
    )
    return {"path": str(PNL_PERSIST_PATH), "elapsed_s": elapsed}


def initialize(persist_path: Optional[str] = None, *, auto_start_worker: bool = False, start_http: bool = False) -> None:
    """Initialize persistence module (load state). Does not auto-start worker unless requested."""
    global PNL_PERSIST_PATH
    if persist_path:
        PNL_PERSIST_PATH = Path(persist_path)
    load_state_from_disk()
    if auto_start_worker:
        start_worker()
    if start_http:
        start_embedded_http()


# ---------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------
__all__ = [
    "snapshot_for_persistence",
    "dump_state_to_disk",
    "load_state_from_disk",
    "update_state",
    "flush_on_event",
    "manual_flush_and_report",
    "start_worker",
    "stop_worker",
    "initialize",
    "get_status",
    "locked_state",
]
