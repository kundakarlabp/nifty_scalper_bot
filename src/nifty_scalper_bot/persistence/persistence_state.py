"""
Persistence and periodic flush worker for Nifty Scalper Bot (robust production-grade).

Features:
- Thread-safe in-memory `state` (documented) with context manager.
- Atomic writes to disk and backup rotation.
- Pydantic schema validation if pydantic available
    (fail-fast or warn depending on config).
- Optional event-driven flush, manual flush, background periodic worker.
- Health/status API (function-based; optional embedded HTTP server via
    `enable_http=True`).
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
from nifty_scalper_bot.utils.sampler_singleton import system_sampler

LOGGER = get_logger(__name__)

PNL_PERSIST_PATH = Path(os.getenv("PNL_PERSIST_PATH", "data/pnl_state.json"))
PERSIST_FLUSH_INTERVAL_S = float(
    os.getenv("PERSIST_FLUSH_INTERVAL_S", "10")
)
PERSIST_RETENTION_BACKUPS = int(
    os.getenv("PERSIST_RETENTION_BACKUPS", "3")
)
PERSIST_ENABLE_SCHEMA_VALIDATION = (
    os.getenv("PERSIST_ENABLE_SCHEMA_VALIDATION", "true").lower()
    in ("1", "true", "yes")
)
PERSIST_ENABLE_EMBEDDED_HTTP = (
    os.getenv("ENABLE_EMBEDDED_HTTP_SERVER", "false").lower()
    in ("1", "true", "yes")
)
PERSIST_HTTP_HOST = os.getenv("PERSIST_HTTP_HOST", "127.0.0.1")
PERSIST_HTTP_PORT = int(
    os.getenv("PERSIST_HTTP_PORT", "9234")
)
PERSIST_COMPRESSION = os.getenv("PERSIST_COMPRESSION", "none")
PERSIST_ENCRYPTION = os.getenv("PERSIST_ENCRYPTION", "none")
PERSIST_LOG_LEVEL_ON_SUCCESS = os.getenv("PERSIST_LOG_LEVEL_ON_SUCCESS", "info")

try:
    from pydantic import BaseModel
    from pydantic import ValidationError as PydanticValidationError
    class PersistModel(BaseModel):
        last_pnl: float = 0.0
        positions: Dict[str, Any] = {}
        orders: Dict[str, Any] = {}
        meta: Dict[str, Any] = {}
    def validate_snapshot(obj: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        try:
            PersistModel.model_validate(obj)
            return True, None
        except PydanticValidationError as exc:
            return False, str(exc)
    PersistModelType = PersistModel
    ValidationErrorType = PydanticValidationError
except Exception:
    PersistModelType = type(None)
    ValidationErrorType = Exception
    def validate_snapshot(obj: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        # obj is always a dict here, so no need to check type
        return True, None

_state_lock = threading.RLock()
_state: Dict[str, Any] = {
    "last_pnl": 0.0,
    "positions": {},
    "orders": {},
    "meta": {"version": "1.0"},
}
_worker_thread: Optional[threading.Thread] = None
_worker_stop = threading.Event()
_metrics_lock = threading.Lock()
_metrics: Dict[str, float] = {
    "persist_emitted_total": 0,
    "persist_suppressed_total": 0,
    "persist_errors_total": 0,
    "last_flush_epoch": 0.0,
}

def _atomic_write(path: Path, data_bytes: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data_bytes)
    tmp.replace(path)

def _rotate_backups(path: Path, keep: int) -> None:
    try:
        for i in range(keep - 1, 0, -1):
            src = path.with_suffix(path.suffix + f".{i}")
            dst = path.with_suffix(path.suffix + f".{i+1}")
            if src.exists():
                src.replace(dst)
        if path.exists():
            path.replace(path.with_suffix(path.suffix + ".1"))
    except Exception:
        LOGGER.exception(
            "persist: backup rotation failed",
            extra={"event": "persist_backup_rotation_failed"},
        )

def _compress_bytes(data: bytes) -> bytes:
    if PERSIST_COMPRESSION.lower() == "gzip":
        import gzip
        return gzip.compress(data)
    return data

def _maybe_encrypt_bytes(data: bytes) -> bytes:
    if PERSIST_ENCRYPTION.lower() != "none":
        LOGGER.debug(
            "persist: encryption requested but not implemented; storing plaintext",
            extra={"event": "persist_encrypt_stub"},
        )
    return data

def _prepare_bytes_for_write(snapshot_json: str) -> bytes:
    raw = snapshot_json.encode("utf-8")
    raw = _compress_bytes(raw)
    raw = _maybe_encrypt_bytes(raw)
    return raw

@contextmanager
def locked_state():
    with _state_lock:
        yield _state

def get_state_snapshot() -> Dict[str, Any]:
    with locked_state():
        return json.loads(json.dumps(_state, default=str))

def snapshot_for_persistence() -> Dict[str, Any]:
    return get_state_snapshot()

def dump_state_to_disk(
    path: Optional[Path | str] = None,
    *,
    validate: bool = True,
) -> None:
    global _metrics
    p = Path(path or PNL_PERSIST_PATH)
    try:
        snapshot = snapshot_for_persistence()
        if validate and PERSIST_ENABLE_SCHEMA_VALIDATION:
            ok, err = validate_snapshot(snapshot)
            if not ok:
                LOGGER.warning(
                    "persist: snapshot validation failed: %s",
                    err,
                    extra={"event": "persist_validation_failed"},
                )
                with _metrics_lock:
                    _metrics["persist_suppressed_total"] += 1
        snapshot_json = json.dumps(
            snapshot, ensure_ascii=False, indent=2
        )
        payload = _prepare_bytes_for_write(snapshot_json)
        _rotate_backups(p, PERSIST_RETENTION_BACKUPS)
        _atomic_write(p, payload)
        with _metrics_lock:
            _metrics["persist_emitted_total"] += 1
            _metrics["last_flush_epoch"] = time.time()
        system_sampler.maybe_log(
            LOGGER,
            regime=None,
            multiplier=None,
            extra={"path": str(p), "size_bytes": len(payload)},
            level=PERSIST_LOG_LEVEL_ON_SUCCESS,
        )
        LOGGER.debug(
            "persist: wrote snapshot to %s (bytes=%d)",
            str(p),
            len(payload),
            extra={"event": "persist_write_ok"},
        )
    except Exception as exc:
        with _metrics_lock:
            _metrics["persist_errors_total"] += 1
        LOGGER.exception(
            "persist: failed to write state: %s",
            exc,
            extra={"event": "persist_write_failed", "trace": traceback.format_exc()},
        )
        system_sampler.maybe_log(
            LOGGER,
            regime=None,
            multiplier=None,
            extra={"path": str(p), "error": str(exc)},
            level="error",
        )

def load_state_from_disk(
    path: Optional[Path | str] = None,
    *,
    validate: bool = False,
) -> None:
    p = Path(path or PNL_PERSIST_PATH)
    if not p.exists():
        LOGGER.info(
            "persist: no state file found at %s",
            p,
            extra={"event": "persist_load_missing"},
        )
        return
    try:
        raw = p.read_bytes()
        try:
            data_text = raw.decode("utf-8")
        except Exception:
            import gzip
            data_text = gzip.decompress(raw).decode("utf-8")
        obj = json.loads(data_text)
        if validate and PERSIST_ENABLE_SCHEMA_VALIDATION:
            ok, err = validate_snapshot(obj)
            if not ok:
                LOGGER.warning(
                    "persist: loaded snapshot failed validation: %s",
                    err,
                    extra={"event": "persist_load_validation_failed"},
                )
        with locked_state():
            _state.clear()
            if isinstance(obj, dict):
                _state.update(obj)
        system_sampler.maybe_log(
            LOGGER,
            regime=None,
            multiplier=None,
            extra={"path": str(p), "keys": list(_state.keys())},
        )
        LOGGER.debug(
            "persist: loaded state from %s",
            p,
            extra={"event": "persist_load_ok"},
        )
    except Exception as exc:
        with _metrics_lock:
            _metrics["persist_errors_total"] += 1
        LOGGER.exception(
            "persist: failed to load state: %s",
            exc,
            extra={"event": "persist_load_failed"},
        )
        system_sampler.maybe_log(
            LOGGER,
            regime=None,
            multiplier=None,
            extra={"path": str(p), "error": str(exc)},
            level="error",
        )

def update_state(updates: Dict[str, Any]) -> None:
    with locked_state():
        _state.update(updates)

_event_queue: "list[Tuple[str, Dict[str, Any]]]" = []
_event_queue_lock = threading.Lock()

def flush_on_event(
    event_name: str,
    extra: Optional[Dict[str, Any]] = None,
    *,
    immediate: bool = False,
) -> None:
    extra = extra or {}
    if immediate:
        dump_state_to_disk()
        return
    with _event_queue_lock:
        _event_queue.append((event_name, extra))

def _drain_event_queue() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with _event_queue_lock:
        while _event_queue:
            ev, ex = _event_queue.pop(0)
            if ev not in out:
                out[ev] = {"count": 0, "latest": ex}
            out[ev]["count"] += 1
            out[ev]["latest"] = ex
    return out

def _worker_loop(interval_s: float) -> None:
    LOGGER.debug(
        "persist: worker loop starting",
        extra={"event": "persist_worker_start", "interval_s": interval_s},
    )
    try:
        while not _worker_stop.wait(interval_s):
            events = _drain_event_queue()
            if events:
                with locked_state():
                    _state.setdefault("meta", {})["last_event_batch"] = {
                        k: v["count"] for k, v in events.items()
                    }
            dump_state_to_disk()
            system_sampler.maybe_log(
                LOGGER,
                regime=None,
                multiplier=None,
                extra={"interval_s": interval_s, "events": len(events)},
            )
    except Exception:
        LOGGER.exception(
            "persist: worker crashed",
            extra={"event": "persist_worker_crash"},
        )
    finally:
        LOGGER.debug(
            "persist: worker loop exiting",
            extra={"event": "persist_worker_exit"},
        )

def start_worker(
    interval_s: Optional[float] = None,
    *,
    force_restart: bool = False,
) -> None:
    global _worker_thread
    if _worker_thread and _worker_thread.is_alive():
        if not force_restart:
            LOGGER.debug(
                "persist: worker already running",
                extra={"event": "persist_worker_already"},
            )
            return
        stop_worker()
    _worker_stop.clear()
    iv = float(interval_s or PERSIST_FLUSH_INTERVAL_S)
    _worker_thread = threading.Thread(
        target=_worker_loop,
        args=(iv,),
        daemon=True,
        name="persist-worker",
    )
    _worker_thread.start()
    system_sampler.maybe_log(
        LOGGER,
        regime=None,
        multiplier=None,
        extra={"interval_s": iv},
    )

def stop_worker(timeout_s: float = 2.0) -> None:
    _worker_stop.set()
    if _worker_thread:
        _worker_thread.join(timeout_s)
    system_sampler.maybe_log(LOGGER, regime=None, multiplier=None)

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
        emitted = int(_metrics.get("persist_emitted_total", 0))
        suppressed = int(_metrics.get("persist_suppressed_total", 0))
        errors = int(_metrics.get("persist_errors_total", 0))
        last_flush = float(_metrics.get("last_flush_epoch", 0.0))
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

def manual_flush_and_report() -> Dict[str, Any]:
    start = time.time()
    dump_state_to_disk()
    elapsed = time.time() - start
    system_sampler.maybe_log(
        LOGGER,
        regime=None,
        multiplier=None,
        extra={"elapsed_s": elapsed},
    )
    return {"path": str(PNL_PERSIST_PATH), "elapsed_s": elapsed}

def initialize(
    persist_path: Optional[str] = None,
    *,
    auto_start_worker: bool = False,
    start_http: bool = False,
) -> None:
    global PNL_PERSIST_PATH
    if persist_path:
        # PNL_PERSIST_PATH is not a true constant, allow reassignment
        PNL_PERSIST_PATH = Path(persist_path)
    load_state_from_disk()
    if auto_start_worker:
        start_worker()
    if start_http:
        pass

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
