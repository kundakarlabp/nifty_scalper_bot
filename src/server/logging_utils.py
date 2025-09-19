from __future__ import annotations

import logging
import os
import random
import time
from collections import deque
from typing import Callable

from src.boot.validate_env import _log_cred_presence
from src.config import LOG_LEVEL, settings
from src.utils.log_filters import install_warmup_filters
from src.utils.logger_setup import setup_logging
from src.utils.logging_tools import RateLimitFilter, log_buffer_handler


class LogGate:
    """Simple time/sample gate for noisy log statements."""

    def __init__(
        self,
        min_interval_seconds: float,
        sample_rate: float,
        *,
        clock: Callable[[], float] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        if min_interval_seconds < 0:
            raise ValueError("min_interval_seconds must be non-negative")
        if not 0.0 <= sample_rate <= 1.0:
            raise ValueError("sample_rate must be between 0 and 1")
        self._min_interval = float(min_interval_seconds)
        self._sample_rate = float(sample_rate)
        self._clock = clock or time.monotonic
        self._rng = rng or random.Random()
        self._last_emitted: float | None = None

    def allow(self) -> bool:
        """Return ``True`` when the caller should emit a log."""

        now = self._clock()
        if self._last_emitted is None or now - self._last_emitted >= self._min_interval:
            self._last_emitted = now
            return True
        return self._rng.random() < self._sample_rate

    def reset(self) -> None:
        """Reset the gate, forcing the next ``allow`` to pass."""

        self._last_emitted = None


def _setup_logging() -> None:  # pragma: no cover
    try:
        env_log_path = os.environ.get("LOG_PATH") or os.environ.get("LOG_FILE")
        log_path = settings.log_path
        if log_path is None and env_log_path:
            log_path = env_log_path
        setup_logging(
            level=LOG_LEVEL,
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
