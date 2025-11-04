"""Structured logging helpers for the bot."""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from typing import Any, Optional

_DEFAULT_LOGGER_NAME = "nifty_scalper_bot"
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
    "event",
}

_EVENT_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_.:-]+")

LOGGER = logging.getLogger(f"{_DEFAULT_LOGGER_NAME}.logging")

_THROTTLE_LOCK = threading.Lock()
_THROTTLE_STATE: dict[str, float] = {}
_STATE_CACHE_LOCK = threading.Lock()
_STATE_CACHE: dict[str, Any] = {}


def _normalise_event(value: object, default: str) -> str:
    """Return a sanitised event label suitable for metrics exporters.

    Args:
        value: Candidate event value supplied on the log record.
        default: Fallback label when *value* is falsy or invalid.

    Returns:
        str: Event label containing only safe characters for scraping.

    Raises:
        None.
    """

    text = str(value).strip() if value is not None else ""
    if not text:
        text = default
    normalised = _EVENT_SANITIZE_RE.sub("_", text.replace(" ", "_"))
    normalised = normalised.strip("._-")
    return normalised or default


class EventEnricher(logging.Filter):
    """Ensure every record has an ``event`` attribute for scraping."""

    def __init__(self, default_event: str = "app.log") -> None:
        """Initialise the filter with a default event value."""

        super().__init__()
        self._default_event = default_event

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401 - override
        current = getattr(record, "event", None)
        sanitized = _normalise_event(current, self._default_event)
        setattr(record, "event", sanitized)
        return True


class KeyValueFormatter(logging.Formatter):
    """Append structured ``key=value`` pairs to log messages."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401 - override
        message = super().format(record)
        extras: list[str] = []
        for key, value in sorted(record.__dict__.items()):
            if key in _RESERVED_ATTRS or key.startswith("_"):
                continue
            extras.append(f"{key}={self._coerce(value)}")
        if extras:
            return f"{message} | {' '.join(extras)}"
        return message

    @staticmethod
    def _coerce(value: object) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return f"{value}"
        return str(value)


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger for the application.

    Args:
        level: Log level name such as ``"INFO"`` or ``"DEBUG"``.

    Returns:
        None.

    Raises:
        None.
    """

    LOGGER.debug(
        "Entered setup_logging",
        extra={"event": "logging_setup_enter", "level": level},
    )
    try:
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        handler = logging.StreamHandler()
        handler.addFilter(EventEnricher())
        if _resolve_bool('LOG_DEDUP_ENABLED', True):
            handler.addFilter(_DedupFilter())
        handler.setFormatter(
            KeyValueFormatter(
                fmt="%(asctime)s %(levelname)s %(name)s event=%(event)s %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S%z",
            )
        )
        logging.basicConfig(handlers=[handler], level=numeric_level, force=True)
        LOGGER.info(
            "Condition met: logging_initialized",
            extra={
                "event": "logging_initialized",
                "level": logging.getLevelName(numeric_level),
            },
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.error(
            "Failure in setup_logging: %s",
            exc,
            extra={"event": "logging_setup_error"},
        )


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a named logger, defaulting to the package root.

    Args:
        name: Optional logger namespace override.

    Returns:
        logging.Logger: Configured logger instance.

    Raises:
        None.
    """

    target = name or _DEFAULT_LOGGER_NAME
    try:
        logger = logging.getLogger(target)
        if not any(isinstance(flt, EventEnricher) for flt in logger.filters):
            logger.addFilter(EventEnricher())
        return logger
    except Exception as exc:  # noqa: BLE001
        LOGGER.error(
            "Failure in get_logger for %s: %s",
            target,
            exc,
            extra={"event": "logging_get_logger_error"},
        )
        return LOGGER


def get_tracer_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger dedicated to trace correlation entries.

    Args:
        name: Optional logger namespace; defaults to package root.

    Returns:
        logging.Logger: Logger instance suitable for trace logging.

    Raises:
        None.
    """

    LOGGER.debug(
        "Entered get_tracer_logger",
        extra={"event": "logging_tracer_logger_enter", "logger_name": name or ""},
    )
    try:
        tracer = get_logger(name)
        LOGGER.info(
            "Condition met: tracer_logger_ready",
            extra={"event": "logging_tracer_logger_ready", "logger_name": tracer.name},
        )
        return tracer
    except Exception as exc:  # noqa: BLE001
        LOGGER.error(
            "Failure in get_tracer_logger: %s",
            exc,
            extra={"event": "logging_tracer_logger_error"},
        )
        return LOGGER


def log_throttled(
    logger: logging.Logger,
    key: str,
    msg: str,
    *,
    level: int = logging.INFO,
    interval_sec: float | None = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """Emit a log record at most once during the specified interval.

    Args:
        logger: Logger used for the emission.
        key: Cache key identifying the throttled message stream.
        msg: Human readable message to emit when allowed.
        level: Numeric logging level for the record.
        interval_sec: Override interval in seconds between emissions.
        extra: Optional mapping passed to ``logger.log``.

    Returns:
        None.

    Raises:
        None.
    """

    logger.debug(
        'Entered log_throttled',
        extra={
            'event': 'logging_log_throttled_enter',
            'log_key': key,
            'level': level,
            'interval': interval_sec,
        },
    )
    try:
        interval = (
            _resolve_float('LOG_THROTTLE_DEFAULT_SEC', 5.0)
            if interval_sec is None
            else float(interval_sec)
        )
        now = time.time()
        with _THROTTLE_LOCK:
            last_emit = _THROTTLE_STATE.get(key, 0.0)
            if now - last_emit < interval:
                return
            _THROTTLE_STATE[key] = now
        logger.log(level, msg, extra=extra or {})
    except Exception as exc:  # noqa: BLE001
        logger.error(
            'Failure in log_throttled: %s',
            exc,
            extra={'event': 'logging_log_throttled_error', 'log_key': key},
            exc_info=exc,
        )


def log_state_change(
    logger: logging.Logger,
    key: str,
    value: Any,
    *,
    level: int = logging.INFO,
    msg: str | None = None,
    extra: Optional[dict[str, Any]] = None,
) -> bool:
    """Log a message when the tracked value changes for the given key.

    Args:
        logger: Logger used for the emission.
        key: Identifier for the cached state value.
        value: New state value to compare with the cached value.
        level: Logging level for the resulting record.
        msg: Optional explicit message overriding the default format.
        extra: Optional mapping passed to ``logger.log``.

    Returns:
        bool: ``True`` when a record is emitted, ``False`` otherwise.

    Raises:
        None.
    """

    logger.debug(
        'Entered log_state_change',
        extra={'event': 'logging_log_state_change_enter', 'log_key': key},
    )
    try:
        with _STATE_CACHE_LOCK:
            previous = _STATE_CACHE.get(key, None)
            if previous == value:
                return False
            _STATE_CACHE[key] = value
        message = msg or f'{key} changed: {previous!r} -> {value!r}'
        payload = {'previous_value': previous, 'current_value': value}
        if extra:
            payload.update(extra)
        logger.log(level, message, extra=payload)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error(
            'Failure in log_state_change: %s',
            exc,
            extra={'event': 'logging_log_state_change_error', 'log_key': key},
            exc_info=exc,
        )
        return False


class _DedupFilter(logging.Filter):
    """Drop duplicate records for a configurable time window."""

    def __init__(self) -> None:
        """Initialise the filter using environment configuration."""

        super().__init__()
        self._window = _resolve_float('LOG_DEDUP_WINDOW_SEC', 2.0)
        self._cache: dict[tuple[str, int], float] = {}
        self._lock = threading.Lock()

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401 - override
        signature = (record.getMessage(), record.levelno)
        now = time.time()
        with self._lock:
            last_emit = self._cache.get(signature, 0.0)
            if now - last_emit < self._window:
                return False
            self._cache[signature] = now
        return True


def _resolve_bool(name: str, default: bool) -> bool:
    """Return an environment-controlled boolean flag.

    Args:
        name: Environment variable name.
        default: Fallback boolean value.

    Returns:
        bool: Resolved boolean flag.

    Raises:
        None.
    """

    try:
        raw = os.getenv(name)
        if raw is None:
            return default
        return str(raw).strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
    except Exception as exc:  # noqa: BLE001
        LOGGER.error(
            'Failure in _resolve_bool for %s: %s',
            name,
            exc,
            extra={'event': 'logging_resolve_bool_error', 'variable': name},
            exc_info=exc,
        )
        return default


def _resolve_float(name: str, default: float) -> float:
    """Return an environment-controlled floating point value.

    Args:
        name: Environment variable name.
        default: Fallback float value when parsing fails.

    Returns:
        float: Resolved float value.

    Raises:
        None.
    """

    try:
        raw = os.getenv(name)
        if raw is None:
            return float(default)
        return float(raw)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error(
            'Failure in _resolve_float for %s: %s',
            name,
            exc,
            extra={'event': 'logging_resolve_float_error', 'variable': name},
            exc_info=exc,
        )
        return float(default)


__all__ = [
    'get_logger',
    'get_tracer_logger',
    'log_state_change',
    'log_throttled',
    'setup_logging',
]
