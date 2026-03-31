"""Runtime diagnostics helpers for capturing log output."""

from __future__ import annotations

import logging
from collections import deque
from threading import RLock
from typing import Deque, Dict, List


def _coerce_level(level: str | int | None) -> int:
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        return getattr(logging, level.upper(), logging.INFO)
    return logging.INFO


def _redact(message: str) -> str:
    lowered = message.lower()
    if "token" not in lowered and "secret" not in lowered and "key" not in lowered:
        return message
    parts: List[str] = []
    for chunk in message.split():
        chk = chunk.lower()
        if "token" in chk or "secret" in chk or "key" in chk:
            parts.append("<redacted>")
        else:
            parts.append(chunk)
    return " ".join(parts)


class _TapHandler(logging.Handler):
    """Logging handler that records messages into a ring buffer."""

    def __init__(self, tap: "LogTap", name: str) -> None:
        super().__init__()
        self._tap = tap
        self._name = name

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - logging
        try:
            msg = self.format(record)
        except Exception:  # noqa: BLE001 - defensive
            msg = record.getMessage()
        self._tap._append(self._name, record, msg)


class LogTap:
    """Maintain per-logger ring buffers for diagnostics."""

    def __init__(self, max_lines: int = 5000) -> None:
        self._max_lines = max(1, int(max_lines))
        self._lock = RLock()
        self._buffers: Dict[str, Deque[str]] = {}
        self._handlers: Dict[str, _TapHandler] = {}

    def enable(self, logger_name: str, level: str | int | None = None) -> None:
        """Attach tap handler to *logger_name* at *level*."""

        name = logger_name or "root"
        logger = logging.getLogger(logger_name) if logger_name else logging.getLogger()
        with self._lock:
            if name in self._handlers:
                handler = self._handlers[name]
            else:
                handler = _TapHandler(self, name)
                handler.setFormatter(
                    logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
                )
                logger.addHandler(handler)
                self._handlers[name] = handler
                self._buffers.setdefault(name, deque(maxlen=self._max_lines))
        logger.setLevel(_coerce_level(level))

    def disable(self, logger_name: str) -> None:
        """Detach tap handler from *logger_name*."""

        name = logger_name or "root"
        logger = logging.getLogger(logger_name) if logger_name else logging.getLogger()
        with self._lock:
            handler = self._handlers.pop(name, None)
            if handler is not None:
                logger.removeHandler(handler)

    def tail(self, logger_name: str, lines: int = 10) -> list[str]:
        """Return last *lines* entries for *logger_name*."""

        name = logger_name or "root"
        count = max(1, int(lines))
        with self._lock:
            buffer = self._buffers.get(name)
            if not buffer:
                return []
            return list(buffer)[-count:]

    def _append(self, name: str, record: logging.LogRecord, message: str) -> None:
        clean = _redact(message)
        line = f"{record.asctime if hasattr(record, 'asctime') else ''} {clean}".strip()
        with self._lock:
            buf = self._buffers.setdefault(name, deque(maxlen=self._max_lines))
            buf.append(line)


LOG_TAP = LogTap()


__all__ = ["LOG_TAP", "LogTap"]
