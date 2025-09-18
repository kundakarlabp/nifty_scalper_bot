from __future__ import annotations

import logging
import os
from collections import deque

from src.boot.validate_env import _log_cred_presence
from src.config import settings
from src.utils.log_filters import install_warmup_filters
from src.utils.logger_setup import setup_logging
from src.utils.logging_tools import RateLimitFilter, log_buffer_handler


def _setup_logging() -> None:  # pragma: no cover
    try:
        env_log_path = os.environ.get("LOG_PATH") or os.environ.get("LOG_FILE")
        log_path = settings.log_path
        if log_path is None and env_log_path:
            log_path = env_log_path
        setup_logging(
            level=settings.log_level,
            log_file=str(log_path) if log_path else None,
            json=settings.log_json,
        )
        install_warmup_filters()
        root = logging.getLogger()
        root.addFilter(RateLimitFilter(interval=120.0))
        if settings.log_ring_enabled:
            if log_buffer_handler not in root.handlers:
                root.addHandler(log_buffer_handler)
        else:
            if log_buffer_handler in root.handlers:
                root.removeHandler(log_buffer_handler)
        _log_cred_presence()
        logging.getLogger("urllib3").setLevel(logging.WARNING)
    except Exception as exc:
        logging.getLogger("main").warning(
            "Failed to initialize logging: %s", exc, exc_info=True
        )
        raise


def _import_telegram_class():  # pragma: no cover
    try:
        from src.notifications.telegram_controller import (
            TelegramController,  # type: ignore
        )

        return TelegramController
    except ImportError:
        logging.getLogger("main").exception("Failed to import Telegram controller")
        raise


def _tail_logs(path: str, n: int = 200) -> list[str]:  # pragma: no cover
    """Return last ``n`` lines from ``path`` without loading entire file."""
    try:
        lines: deque[str] = deque(maxlen=n)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                lines.append(line.rstrip("\n"))
        return list(lines)
    except Exception as exc:
        logging.getLogger("main").warning(
            "tail_logs failed for %s: %s", path, exc, exc_info=True
        )
        return []
