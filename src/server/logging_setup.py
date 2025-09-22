"""Utilities for configuring consistent structured logging."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import threading
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

_LOG = logging.getLogger(__name__)

_SUPPRESS_CACHE: dict[str, float] = {}
_SUPPRESS_LOCK = threading.Lock()
_SUPPRESS_WINDOW: float = -1.0

_RESERVED_ATTRS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "message",
    "asctime",
}


def _iso_now() -> str:
    """Return an ISO-8601 timestamp for the current moment."""

    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _q(value: Any) -> str:
    """Return a logfmt-safe representation of ``value``."""

    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return f"{value}"
    escaped = str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
    if " " in escaped or "=" in escaped or escaped == "":
        return f'"{escaped}"'
    return escaped


def _collect_extra(record: logging.LogRecord) -> dict[str, Any]:
    """Collect custom fields from a :class:`logging.LogRecord`."""

    extra: dict[str, Any] = {}
    record_extra = getattr(record, "extra", None)
    if isinstance(record_extra, dict):
        extra.update(record_extra)
    for key, value in record.__dict__.items():
        if key in _RESERVED_ATTRS or key in {"extra", "ts", "tag"}:
            continue
        extra.setdefault(key, value)
    return extra


def _canonicalize(value: Any) -> Any:
    """Return a JSON-serializable representation of ``value``."""

    if isinstance(value, Mapping):
        return {str(k): _canonicalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(v) for v in value]
    if isinstance(value, set):
        return sorted(_canonicalize(v) for v in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    iso_formatter = getattr(value, "isoformat", None)
    if callable(iso_formatter):
        try:
            return iso_formatter()
        except Exception:  # pragma: no cover - defensive
            pass
    return str(value)


def _suppress_signature(tag: str, level: str, fields: Mapping[str, Any]) -> str:
    payload = {
        "tag": tag,
        "level": level.lower(),
        "fields": _canonicalize(dict(fields)),
    }
    try:
        serialized = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    except TypeError:  # pragma: no cover - fallback
        serialized = str(payload)
    return hashlib.md5(serialized.encode("utf-8")).hexdigest()


def _resolve_suppress_window() -> float:
    raw = os.getenv("LOG_SUPPRESS_WINDOW_SEC")
    if raw is None:
        raw = os.getenv("LOG_DEDUP_TTL_S")
    if not raw:
        return 0.0
    try:
        window = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return window if window > 0 else 0.0


def _should_suppress(tag: str, level: str, fields: Mapping[str, Any], window: float) -> bool:
    signature = _suppress_signature(tag, level, fields)
    now = time.monotonic()
    global _SUPPRESS_WINDOW
    with _SUPPRESS_LOCK:
        if _SUPPRESS_WINDOW != window:
            _SUPPRESS_CACHE.clear()
            _SUPPRESS_WINDOW = window
        expires_at = _SUPPRESS_CACHE.get(signature, 0.0)
        if expires_at > now:
            return True
        _SUPPRESS_CACHE[signature] = now + window
    return False


def _reset_log_event_suppressor() -> None:
    """Reset the cached structured log suppression state."""

    global _SUPPRESS_WINDOW
    with _SUPPRESS_LOCK:
        _SUPPRESS_CACHE.clear()
        _SUPPRESS_WINDOW = -1.0


class _LogfmtFormatter(logging.Formatter):
    """Format log records using a minimal logfmt schema."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401 - standard logging override
        ts = getattr(record, "ts", None) or _iso_now()
        level = record.levelname.lower()
        raw_msg = record.getMessage()
        tag = getattr(record, "tag", None) or (raw_msg.split(" ", 1)[0] if raw_msg else "log")

        parts = [f"ts={ts}", f"lvl={level}", f"tag={_q(tag)}"]
        for key, value in _collect_extra(record).items():
            parts.append(f"{key}={_q(value)}")
        if raw_msg:
            parts.append(f"msg={_q(raw_msg)}")
        return " ".join(parts)


class _JsonFormatter(logging.Formatter):
    """Format log records as JSON payloads."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401 - standard logging override
        payload: dict[str, Any] = {
            "ts": getattr(record, "ts", None) or _iso_now(),
            "level": record.levelname.lower(),
            "tag": getattr(record, "tag", None) or "log",
            "msg": record.getMessage() or "",
        }
        payload.update(_collect_extra(record))
        return json.dumps(payload, ensure_ascii=False)


def setup_root_logger() -> None:
    """Configure the root logger with a single stdout handler."""

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level_value = getattr(logging, level_name, None)
    if not isinstance(level_value, int):
        level_value = logging.INFO

    fmt = os.getenv("LOG_FORMAT", "logfmt").lower()
    root = logging.getLogger()
    root.setLevel(level_value)

    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    if fmt == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(_LogfmtFormatter())

    root.addHandler(handler)
    _LOG.info("logging.init", extra={"extra": {"level": level_name, "format": fmt}})


def log_event(tag: str, level: str = "info", **fields: Any) -> None:
    """Emit a structured log line with ``tag`` and arbitrary ``fields``."""

    window = _resolve_suppress_window()
    if window > 0.0 and _should_suppress(tag, level, fields, window):
        return
    logger = logging.getLogger(tag)
    msg = " ".join(f"{key}={_q(value)}" for key, value in fields.items()) if fields else ""
    record_extra = {"extra": fields, "ts": _iso_now(), "tag": tag}
    log_fn = getattr(logger, level.lower(), None)
    if not callable(log_fn):
        log_fn = logger.info
    log_fn(msg, extra=record_extra)

