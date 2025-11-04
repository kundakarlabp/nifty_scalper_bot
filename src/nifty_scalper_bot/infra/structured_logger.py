"""Human friendly logging helpers."""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from typing import cast
from zoneinfo import ZoneInfo

from nifty_scalper_bot.infra.diagnostics import LOG_TAP
from nifty_scalper_bot.infra.metrics import METRICS
from nifty_scalper_bot.utils.reasons import canonical

_LEVEL_SYMBOLS = {
    "DEBUG": "🔹",
    "INFO": "✅",
    "WARNING": "⚠️",
    "ERROR": "❌",
    "CRITICAL": "💥",
}


class HumanFormatter(logging.Formatter):
    """Render human centric single-line log messages."""

    def __init__(self, tz: str = "Asia/Kolkata", color: bool = True) -> None:
        """Initialize formatter with timezone and color preferences.

        Args:
            tz: IANA timezone name for timestamps.
            color: Whether to emit ANSI colors when supported.
        """

        super().__init__()
        self._tz = ZoneInfo(tz)
        self._use_color = color and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401 - override
        ts = datetime.fromtimestamp(record.created, self._tz).strftime(
            "%Y-%m-%d %H:%M:%S %Z"
        )
        level = record.levelname
        emoji = _LEVEL_SYMBOLS.get(level, "•")
        message = record.getMessage()

        if self._use_color:
            colors = {
                "DEBUG": "\033[36m",
                "INFO": "\033[32m",
                "WARNING": "\033[33m",
                "ERROR": "\033[31m",
                "CRITICAL": "\033[41m",
            }
            color_code = colors.get(level, "\033[0m")
            reset = "\033[0m"
            return f"{color_code}[{ts}] {emoji} {message}{reset}"

        return f"[{ts}] {emoji} {message}"


def setup_structured_logging(level: str = "INFO") -> None:
    """Configure root logger to use a concise human readable format."""

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    tz = os.getenv("TZ", "Asia/Kolkata")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(HumanFormatter(tz=tz, color=True))
    root.addHandler(handler)

    try:
        LOG_TAP.enable("", level)
    except Exception:  # pragma: no cover - diagnostics optional
        pass

    for noisy in ("httpx", "urllib3", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    root.info("🚀 Logging initialized | level=%s | timezone=%s", level.upper(), tz)


__all__ = ["HumanFormatter", "setup_structured_logging"]


def emit_diag(
    logger: logging.Logger,
    event: str,
    *,
    reason: str | None = None,
    severity: str = "info",
    alert: bool = False,
    **fields: object,
) -> None:
    """Log structured diagnostic event with optional canonical reason.

    Args:
        logger: Logger used for emitting the diagnostic record.
        event: Event name describing the diagnostic scenario.
        reason: Optional reason token to canonicalise for consistency.
        severity: Human readable severity level (info|warning|critical).
        alert: ``True`` when the event should surface as an alert payload.
        **fields: Additional structured metadata attached to the log entry.

    Returns:
        None.

    Raises:
        None.
    """

    payload = dict(fields)
    payload["event"] = event
    payload["severity"] = (severity or "info").strip().lower() or "info"
    payload["alert"] = bool(alert)
    reason_token = canonical(reason)
    payload["reason"] = reason_token
    logger.info(event, extra=payload)
    try:
        metadata = {
            key: value
            for key, value in payload.items()
            if key not in {"event", "reason"}
        }
        severity_value = cast(str, payload["severity"])
        METRICS.record_diagnostic_event(
            event=event,
            reason=reason_token,
            severity=severity_value,
            metadata=metadata,
        )
    except Exception as exc:  # noqa: BLE001 - defensive
        logger.error(
            "Failure in emit_diag metrics dispatch: %s",
            exc,
            extra={
                "event": f"{event}_diagnostic_metric_error",
                "reason": reason_token,
            },
        )


__all__.append("emit_diag")
