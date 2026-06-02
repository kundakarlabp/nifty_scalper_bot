"""Structured logging helpers for the bot.

This module provides a production-grade logging infrastructure with:
- Structured key=value formatting.
- Automatic event enrichment for metrics.
- Rate limiting and deduplication to prevent log floods.
- Granular control over third-party library noise.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import sys
import threading
import time
from typing import Any, Optional

# =============================================================================
# 1. CONSTANTS & GLOBALS
# =============================================================================

_DEFAULT_LOGGER_NAME = "nifty_scalper_bot"
_DEFAULT_RATE_LIMIT_SEC = 3.0
_DEFAULT_DEDUP_WINDOW_SEC = 2.0

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

# Global State & Locks
LOGGER = logging.getLogger(f"{_DEFAULT_LOGGER_NAME}.logging")
_THROTTLE_LOCK = threading.Lock()
_THROTTLE_STATE: dict[str, float] = {}
_STATE_CACHE_LOCK = threading.Lock()
_STATE_CACHE: dict[str, Any] = {}
_FILTER_INSTALL_LOCK = threading.Lock()


# =============================================================================
# 2. PRIVATE UTILITIES (Helpers & Environment)
# =============================================================================


def _normalise_event(value: object, default: str) -> str:
    """Return a sanitised event label suitable for metrics exporters."""
    text = str(value).strip() if value is not None else ""
    if not text:
        text = default
    normalised = _EVENT_SANITIZE_RE.sub("_", text.replace(" ", "_"))
    normalised = normalised.strip("._-")
    return normalised or default


def _resolve_bool(name: str, default: bool) -> bool:
    """Return an environment-controlled boolean flag."""
    try:
        raw = os.getenv(name)
        if raw is None:
            return default
        return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}
    except Exception as exc:  # noqa: BLE001
        LOGGER.error(
            "Failure in _resolve_bool for %s: %s",
            name,
            exc,
            extra={"event": "logging_resolve_bool_error", "variable": name},
            exc_info=exc,
        )
        return default


def _resolve_float(name: str, default: float) -> float:
    """Return an environment-controlled floating point value."""
    try:
        raw = os.getenv(name)
        if raw is None:
            return float(default)
        return float(raw)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error(
            "Failure in _resolve_float for %s: %s",
            name,
            exc,
            extra={"event": "logging_resolve_float_error", "variable": name},
            exc_info=exc,
        )
        return float(default)


def _resolve_int(name: str, default: int) -> int:
    """Return an environment-controlled integer value."""
    try:
        raw = os.getenv(name)
        if raw is None:
            return int(default)
        return int(float(raw))
    except Exception as exc:  # noqa: BLE001
        LOGGER.error(
            "Failure in _resolve_int for %s: %s",
            name,
            exc,
            extra={"event": "logging_resolve_int_error", "variable": name},
            exc_info=exc,
        )
        return int(default)


class LogThrottle:
    """Per-key cooldown helper for repetitive logs."""

    def __init__(self) -> None:
        """Initialise throttle state. Args: none. Returns: None. Raises: None."""
        self._seen: dict[str, float] = {}
        self._lock = threading.Lock()

    def should_log(self, key: str, cooldown_seconds: float) -> bool:
        """Check if key can log now. Args: key/cooldown_seconds. Returns: bool. Raises: None."""
        now = time.monotonic()
        with self._lock:
            last = self._seen.get(key, 0.0)
            if now - last < max(0.0, float(cooldown_seconds)):
                return False
            self._seen[key] = now
        return True


# =============================================================================
# 3. FORMATTERS & FILTERS
# =============================================================================


class _MaxLevelFilter(logging.Filter):
    """Allow records below or equal to a maximum level."""

    def __init__(self, max_level: int) -> None:
        super().__init__()
        self._max_level = int(max_level)

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401 - override
        return int(record.levelno) <= self._max_level


class _MinLevelFilter(logging.Filter):
    """Allow records at or above a minimum level."""

    def __init__(self, min_level: int) -> None:
        super().__init__()
        self._min_level = int(min_level)

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401 - override
        return int(record.levelno) >= self._min_level


class EventEnricher(logging.Filter):
    """Ensure every record has an ``event`` attribute for scraping."""

    def __init__(self, default_event: str = "app.log") -> None:
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


class _BurstDedupFilter(logging.Filter):
    """Drop repeated log records seen during a short window."""

    def __init__(self, window_sec: float) -> None:
        super().__init__()
        self._window = max(float(window_sec), 0.0)
        self._last_seen: dict[str, float] = {}
        self._lock = threading.Lock()

    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
        if self._window <= 0:
            return True
        if getattr(record, "bypass_filters", False):
            return True
        try:
            signature = f"{record.name}|{record.levelno}|{record.getMessage()}"
            now = time.monotonic()
            with self._lock:
                last_emit = self._last_seen.get(signature, 0.0)
                if now - last_emit < self._window:
                    return False
                self._last_seen[signature] = now
            return True
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in _BurstDedupFilter.filter: %s",
                exc,
                extra={"event": "logging_burst_dedup_error", "bypass_filters": True},
                exc_info=exc,
            )
            return True


class _RateLimitFilter(logging.Filter):
    """Rate-limit repeated log records within a configured interval."""

    def __init__(self, interval_sec: float) -> None:
        super().__init__()
        self._interval = max(float(interval_sec), 0.0)
        self._last_seen: dict[tuple[str, int, str], float] = {}
        self._lock = threading.Lock()

    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
        if self._interval <= 0:
            return True
        if getattr(record, "bypass_filters", False):
            return True
        try:
            key = (
                record.name,
                record.levelno,
                record.msg if isinstance(record.msg, str) else str(record.msg),
            )
            now = time.monotonic()
            with self._lock:
                last_emit = self._last_seen.get(key, 0.0)
                if now - last_emit < self._interval:
                    return False
                self._last_seen[key] = now
            return True
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in _RateLimitFilter.filter: %s",
                exc,
                extra={
                    "event": "logging_rate_limit_filter_error",
                    "bypass_filters": True,
                },
                exc_info=exc,
            )
            return True


class _DedupFilter(logging.Filter):
    """Drop duplicate records for a configurable time window."""

    def __init__(self, window_sec: float) -> None:
        super().__init__()
        self._window = max(float(window_sec), 0.0)
        self._cache: dict[tuple[str, int, str], float] = {}
        self._lock = threading.Lock()

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401 - override
        if self._window <= 0:
            return True
        event = getattr(record, "event", "")
        signature = (record.getMessage(), record.levelno, str(event))
        now = time.time()
        with self._lock:
            last_emit = self._cache.get(signature, 0.0)
            if now - last_emit < self._window:
                return False
            self._cache[signature] = now
        return True


class _EventSamplingFilter(logging.Filter):
    """Sample noisy records based on event regex matching."""

    def __init__(self, pattern: str | None, every: int) -> None:
        super().__init__()
        self._regex = re.compile(pattern) if pattern else None
        self._every = max(int(every), 1)
        self._lock = threading.Lock()
        self._counts: dict[str, int] = {}

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401 - override
        if not self._regex:
            return True
        label = str(getattr(record, "event", None) or record.getMessage())
        if not self._regex.search(label):
            return True
        with self._lock:
            counter = self._counts.get(label, 0) + 1
            self._counts[label] = counter
            return (counter % self._every) == 0


class DedupLogger:
    """Suppress repeated identical INFO logs within a cooldown window."""

    def __init__(self, logger: logging.Logger, cooldown_seconds: float = 60.0) -> None:
        self._logger = logger
        self._cooldown = cooldown_seconds
        self._last_messages: dict[str, float] = {}

    def info(self, msg: object, *args: object, **kwargs: object) -> None:
        now = time.monotonic()
        key = str(msg % args if args else msg)

        last = self._last_messages.get(key)
        if last is not None and (now - last) < self._cooldown:
            return

        self._last_messages[key] = now
        self._logger.info(msg, *args, **kwargs)

    def debug(self, msg: object, *args: object, **kwargs: object) -> None:
        self._logger.debug(msg, *args, **kwargs)

    def warning(self, msg: object, *args: object, **kwargs: object) -> None:
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: object, *args: object, **kwargs: object) -> None:
        self._logger.error(msg, *args, **kwargs)

    def __getattr__(self, item: str) -> Any:
        """Args: attribute name; Returns: delegated logger attr; Raises: AttributeError."""

        return getattr(self._logger, item)


# =============================================================================
# 4. SETUP & CONFIGURATION LOGIC
# =============================================================================


def _apply_noisy_overrides() -> None:
    """Apply default log level overrides for noisy modules."""
    LOGGER.debug(
        "Entered _apply_noisy_overrides",
        extra={"event": "logging_noisy_overrides_enter", "bypass_filters": True},
    )
    try:
        defaults = {
            "nifty_scalper_bot.risk.regime_sizing": "WARNING",
            "nifty_scalper_bot.infra.metrics": "WARNING",
            "nifty_scalper_bot.data.market_data_manager": "INFO",
            "nifty_scalper_bot.data.rest.zerodha_client": "INFO",
        }
        for logger_name, default_level in defaults.items():
            env_key = f"LOG_OVERRIDE_{logger_name}"
            raw_level = os.getenv(env_key, default_level) or default_level
            override = raw_level.strip().upper()
            with contextlib.suppress(AttributeError, ValueError):
                level_value = getattr(logging, override or default_level, logging.INFO)
                logging.getLogger(logger_name).setLevel(level_value)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error(
            "Failure in _apply_noisy_overrides: %s",
            exc,
            extra={"event": "logging_noisy_overrides_error", "bypass_filters": True},
            exc_info=exc,
        )


def _install_filters_once() -> None:
    """Install burst dedup and rate-limit filters on the root handlers once."""
    LOGGER.debug(
        "Entered _install_filters_once",
        extra={"event": "logging_install_filters_enter", "bypass_filters": True},
    )
    try:
        with _FILTER_INSTALL_LOCK:
            root = logging.getLogger()
            if getattr(root, "_nifty_filters_installed", False):
                return
            if not root.handlers:
                return
            rate_limit = _resolve_float(
                "LOG_RATE_LIMIT_SEC",
                _DEFAULT_RATE_LIMIT_SEC,
            )
            dedup_window = _resolve_float(
                "LOG_DEDUP_WINDOW_SEC",
                _DEFAULT_DEDUP_WINDOW_SEC,
            )
            rate_filter = _RateLimitFilter(rate_limit)
            dedup_filter = _BurstDedupFilter(dedup_window)

            # Use default regex if environment variable not set
            sample_regex = os.getenv(
                "LOG_SAMPLE_EVENT_REGEX", r"^regime_multiplier_applied$"
            )
            sample_every = _resolve_int("LOG_SAMPLE_EVERY", 50)
            sampling_filter = _EventSamplingFilter(sample_regex, sample_every)

            for handler in root.handlers:
                existing_filters = list(getattr(handler, "filters", []))
                if not any(
                    isinstance(flt, _BurstDedupFilter) for flt in existing_filters
                ):
                    handler.addFilter(dedup_filter)
                if not any(
                    isinstance(flt, _RateLimitFilter) for flt in existing_filters
                ):
                    handler.addFilter(rate_filter)
                # Ensure sampling filter is also added if needed
                if not any(
                    isinstance(flt, _EventSamplingFilter) for flt in existing_filters
                ):
                    handler.addFilter(sampling_filter)

            _apply_noisy_overrides()
            root._nifty_filters_installed = True  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001
        LOGGER.error(
            "Failure in _install_filters_once: %s",
            exc,
            extra={"event": "logging_install_filters_error", "bypass_filters": True},
            exc_info=exc,
        )


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger for the application."""
    LOGGER.debug(
        "Entered setup_logging",
        extra={"event": "logging_setup_enter", "level": level},
    )
    try:
        resolved_level_name = os.getenv("LOG_LEVEL", level)
        numeric_level = getattr(logging, resolved_level_name.upper(), logging.INFO)
        stdout_handler = logging.StreamHandler(sys.stdout)
        stderr_handler = logging.StreamHandler(sys.stderr)
        stdout_handler.addFilter(_MaxLevelFilter(logging.WARNING))
        stderr_handler.addFilter(_MinLevelFilter(logging.ERROR))
        handlers = [stdout_handler, stderr_handler]
        for handler in handlers:
            handler.addFilter(EventEnricher())

            # Note: We add filters here for the new handlers, but _install_filters_once
            # attaches them to existing handlers on the root logger later as well.
            if _resolve_bool("LOG_DEDUP_ENABLED", True):
                handler.addFilter(_DedupFilter(_resolve_float("LOG_DEDUP_WINDOW_SEC", 2.0)))

            # Standard format vs JSON format
            if os.getenv("LOG_FORMAT", "").strip().lower() == "json":
                handler.setFormatter(
                    logging.Formatter(
                        '{"ts":"%(asctime)s","lvl":"%(levelname)s","logger":"%(name)s",'
                        '"msg":"%(message)s","event":"%(event)s"}',
                        datefmt="%Y-%m-%dT%H:%M:%S%z",
                    )
                )
            else:
                handler.setFormatter(
                    KeyValueFormatter(
                        fmt=(
                            "%(asctime)s %(levelname)s %(name)s "
                            "event=%(event)s %(message)s"
                        ),
                        datefmt="%Y-%m-%dT%H:%M:%S%z",
                    )
                )
        logging.basicConfig(handlers=handlers, level=numeric_level, force=True)
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


# =============================================================================
# 5. PUBLIC API (Helpers)
# =============================================================================


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a named logger, defaulting to the package root."""
    target = name or _DEFAULT_LOGGER_NAME
    try:
        _install_filters_once()
        base_logger = logging.getLogger(target)
        # Avoid duplicate filters if get_logger is called repeatedly
        if not any(isinstance(flt, EventEnricher) for flt in base_logger.filters):
            base_logger.addFilter(EventEnricher())
        return DedupLogger(base_logger)  # type: ignore[return-value]
    except Exception as exc:  # noqa: BLE001
        LOGGER.error(
            "Failure in get_logger for %s: %s",
            target,
            exc,
            extra={"event": "logging_get_logger_error"},
        )
        return LOGGER


def get_tracer_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger dedicated to trace correlation entries."""
    LOGGER.debug(
        "Entered get_tracer_logger",
        extra={"event": "logging_tracer_logger_enter", "logger_name": name or ""},
    )
    try:
        tracer = get_logger(name)
        LOGGER.debug(
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
    *fmt_args: Any,
    interval_sec: float = 60.0,
    level: int = logging.INFO,
    extra: Optional[dict[str, Any]] = None,
    exc_info: bool | BaseException | None = None,
) -> None:
    """Emit a log record at most once during the specified interval."""
    if not isinstance(logger, logging.Logger):
        logger = LOGGER
    logger.debug(
        "Entered log_throttled",
        extra={
            "event": "logging_log_throttled_enter",
            "log_key": key,
            "level": level,
            "interval": interval_sec,
        },
    )
    try:
        interval = float(interval_sec)
        now = time.time()
        with _THROTTLE_LOCK:
            last_emit = _THROTTLE_STATE.get(key, 0.0)
            if now - last_emit < interval:
                return
            _THROTTLE_STATE[key] = now
        if fmt_args:
            try:
                msg = msg % fmt_args
            except Exception:
                msg = f"{msg} | fmt_args={fmt_args!r}"
        logger.log(level, msg, extra=extra or {}, exc_info=exc_info)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Failure in log_throttled: %s",
            exc,
            extra={"event": "logging_log_throttled_error", "log_key": key},
            exc_info=exc,
        )


def log_once_or_throttled(
    logger: logging.Logger,
    key: str,
    msg: str,
    *,
    interval_sec: float = 60.0,
    level: int = logging.INFO,
    extra: Optional[dict[str, Any]] = None,
    exc_info: bool | BaseException | None = None,
) -> None:
    """Emit once immediately and then at most once per interval. Args: logger/key/msg/interval/level/extra/exc_info. Returns: None. Raises: None."""
    try:
        log_throttled(
            logger,
            key,
            msg,
            interval_sec=interval_sec,
            level=level,
            extra=extra,
            exc_info=exc_info,
        )
    except Exception:
        with contextlib.suppress(Exception):
            logger.log(level, msg, extra=extra or {}, exc_info=exc_info)

def log_state_change(
    logger: logging.Logger,
    key: str,
    value: Any,
    *,
    level: int = logging.INFO,
    msg: str | None = None,
    extra: Optional[dict[str, Any]] = None,
) -> bool:
    """Log a message when the tracked value changes for the given key."""
    logger.debug(
        "Entered log_state_change",
        extra={"event": "logging_log_state_change_enter", "log_key": key},
    )
    try:
        with _STATE_CACHE_LOCK:
            previous = _STATE_CACHE.get(key, None)
            if previous == value:
                return False
            _STATE_CACHE[key] = value
        message = msg or f"{key} changed: {previous!r} -> {value!r}"
        payload = {"previous_value": previous, "current_value": value}
        if extra:
            payload.update(extra)
        logger.log(level, message, extra=payload)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Failure in log_state_change: %s",
            exc,
            extra={"event": "logging_log_state_change_error", "log_key": key},
            exc_info=exc,
        )
        return False


# =============================================================================
# 6. CONVENIENCE: THIRD-PARTY SILENCING
# =============================================================================


def silence_third_party_loggers() -> None:
    """Suppress verbose third-party HTTP and async logging."""
    for logger_name in [
        "httpcore",
        "httpx",
        "urllib3",
        "asyncio",
        "concurrent.futures",
        "selector",
    ]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def enable_business_logic_logging() -> None:
    """Enable logs for Decisions, but silence the Loops."""
    # ❌ FIX: Commented out these lines so they don't override your Railway config.
    # Now, if you set LOG_LEVEL=INFO, these will actually stay at INFO.

    # logging.getLogger("nifty_scalper_bot.strategies").setLevel(logging.DEBUG)
    # logging.getLogger("nifty_scalper_bot.core").setLevel(logging.DEBUG)

    # 2. Low Visibility (Silence 'Entered OrderQueue', 'Circuit Breaker' loops)
    # You can keep these if you want to explicitly silence them, or comment them out too.
    # logging.getLogger("nifty_scalper_bot.execution").setLevel(logging.INFO)
    # logging.getLogger("nifty_scalper_bot.risk").setLevel(logging.INFO)
    # logging.getLogger("nifty_scalper_bot.lifecycle").setLevel(logging.INFO)
    pass


# =============================================================================
# 7. MODULE INITIALIZATION
# =============================================================================

# These functions modify global logging state immediately upon import.
# Note: This is an application-specific side effect.
silence_third_party_loggers()
enable_business_logic_logging()


__all__ = [
    "get_logger",
    "get_tracer_logger",
    "log_state_change",
    "log_throttled",
    "log_once_or_throttled",
    "setup_logging",
    "silence_third_party_loggers",
    "enable_business_logic_logging",
]
