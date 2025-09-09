from __future__ import annotations

import logging
import os
from collections import deque

from src.boot.validate_env import _log_cred_presence
from src.config import settings
from src.utils.logging_tools import RateLimitFilter, log_buffer_handler
from src.utils.logger_setup import setup_logging
from src.utils.log_filters import install_warmup_filters


def _setup_logging() -> None:  # pragma: no cover
    try:
        setup_logging(level=settings.log_level, json=settings.log_json)
        install_warmup_filters()
        root = logging.getLogger()
        root.addFilter(RateLimitFilter(interval=120.0))
        if log_buffer_handler not in root.handlers:
            root.addHandler(log_buffer_handler)
        log_file = os.environ.get("LOG_FILE")
        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(
                logging.Formatter(
                    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    "%Y-%m-%d %H:%M:%S",
                )
            )
            root.addHandler(fh)
        _log_cred_presence()
        logging.getLogger("urllib3").setLevel(logging.WARNING)
    except Exception:
        logging.getLogger("main").exception("Failed to initialize logging")
        raise


def _import_telegram_class():  # pragma: no cover
    try:
        from src.notifications.telegram_controller import TelegramController  # type: ignore
        return TelegramController
    except ImportError:
        logging.getLogger("main").exception(
            "Failed to import Telegram controller"
        )
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
