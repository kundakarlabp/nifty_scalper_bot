"""
Structured logging setup for NiftyScalperBot v2.

Uses structlog with:
- JSON renderer in production (log_format=json)
- Console (colourised) renderer in development (log_format=console)
- IST timestamps on every event
- Automatic injection of module/function context

Call ``configure_logging()`` once at process startup (e.g. from main()).
Then use ``get_logger(__name__)`` in every module.
"""

from __future__ import annotations

import datetime
import logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict, Processor

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore[no-redef]

_IST = ZoneInfo("Asia/Kolkata")

# ---------------------------------------------------------------------------
# IST timestamp processor
# ---------------------------------------------------------------------------


def _add_ist_timestamp(
    logger: Any, method: str, event_dict: EventDict
) -> EventDict:
    """Replace structlog's default UTC timestamp with an IST ISO-8601 string."""
    event_dict["timestamp"] = (
        datetime.datetime.now(_IST).isoformat(timespec="milliseconds")
    )
    return event_dict


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_configured: bool = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def configure_logging(
    log_level: str | None = None,
    log_format: str | None = None,
) -> None:
    """
    Configure structlog and the stdlib root logger.

    Call **once** at process startup.  Safe to call again (e.g. in tests)
    — it will reconfigure from scratch each time.

    Args:
        log_level:  Override log level string (``"DEBUG"``, ``"INFO"``, …).
                    Defaults to the value from :func:`get_settings`.
        log_format: Override format (``"json"`` or ``"console"``).
                    Defaults to the value from :func:`get_settings`.
    """
    global _configured

    # Lazy import to avoid circular dependency at module import time
    from ..config.settings import LogFormat, LogLevel, get_settings

    cfg = get_settings()
    level_str: str = (log_level or cfg.infra.log_level.value).upper()
    fmt_str: str = (log_format or cfg.infra.log_format.value).lower()

    numeric_level = getattr(logging, level_str, logging.INFO)

    # --- stdlib root logger ---------------------------------------------------
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=numeric_level,
        force=True,
    )

    # Silence noisy third-party loggers
    for noisy in ("urllib3", "asyncio", "websocket", "httpx", "hpack"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # --- shared processors ----------------------------------------------------
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        _add_ist_timestamp,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    # --- renderer choice ------------------------------------------------------
    if fmt_str == "json":
        renderer: Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    # --- configure structlog --------------------------------------------------
    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        cache_logger_on_first_use=True,
    )

    # Wire structlog output through a ProcessorFormatter so that stdlib
    # loggers (from third-party libs) also get our processors applied.
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(numeric_level)

    _configured = True


def get_logger(name: str | None = None, **initial_values: Any) -> structlog.BoundLogger:
    """
    Return a structlog bound logger, auto-configuring if needed.

    Usage::

        log = get_logger(__name__, strategy="ORB_PRO")
        log.info("signal_generated", confidence=0.82, symbol="NIFTY24DEC23500CE")

    Args:
        name:           Logger name — pass ``__name__`` for module-based naming.
        **initial_values: Key-value pairs permanently bound to this logger instance.

    Returns:
        A ``structlog.BoundLogger`` (or its filtered variant).
    """
    if not _configured:
        configure_logging()

    logger = structlog.get_logger(name)
    if initial_values:
        logger = logger.bind(**initial_values)
    return logger  # type: ignore[return-value]
