"""Utilities for configuring consistent structured logging."""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from datetime import datetime, timezone
from typing import Any

_LOG = logging.getLogger(__name__)

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


class LogMemoSuppressor:
    """Remember once-only log combinations to avoid repeated noise."""

    def __init__(self) -> None:
        self._seen: set[str] = set()
        self._lock = threading.Lock()

    def should_log(self, *parts: object) -> bool:
        """Return ``True`` if the given ``parts`` combination hasn't been seen."""

        key = "|".join(str(part) for part in parts)
        with self._lock:
            if key in self._seen:
                return False
            self._seen.add(key)
            return True

    def reset(self) -> None:
        """Clear the tracked combinations."""

        with self._lock:
            self._seen.clear()


def log_event(tag: str, level: str = "info", **fields: Any) -> None:
    """Emit a structured log line compatible with JSON and logfmt handlers."""

    logger = logging.getLogger(tag)
    msg = " ".join(f"{key}={_q(value)}" for key, value in fields.items()) if fields else ""
    record_extra = {"extra": fields, "ts": _iso_now(), "tag": tag}
    log_fn = getattr(logger, level.lower(), None)
    if not callable(log_fn):
        log_fn = logger.info
    log_fn(msg, extra=record_extra)


class LogSuppressor:
    """Suppress bursts of repeated warnings/errors for a fixed window."""

    def __init__(self, window_sec: float | None = None) -> None:
        self._window = float(os.getenv("LOG_SUPPRESS_WINDOW_SEC", window_sec or 300))
        self._last: dict[tuple[str, str, str], float] = {}
        self._count: dict[tuple[str, str, str], int] = {}

    def should_log(self, kind: str, group: Any, *identity: Any) -> bool:
        """Return ``True`` if the call should emit a log for ``identity``."""

        import time

        group_text = str(group)
        if identity:
            ident_text = "|".join(str(part) for part in identity)
        else:
            ident_text = group_text
        key = (kind, group_text, ident_text)
        now = time.time()
        last = self._last.get(key, 0.0)
        if now - last >= self._window:
            repeats = self._count.pop(key, 0)
            if repeats:
                log_event(
                    "warn.suppressed",
                    "warning",
                    group=group_text,
                    ident=ident_text,
                    repeats=repeats,
                    window_s=self._window,
                )
            self._last[key] = now
            return True

        self._count[key] = self._count.get(key, 0) + 1
        return False

