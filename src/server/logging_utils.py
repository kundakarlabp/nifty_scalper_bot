from __future__ import annotations

import logging
import os
import random
import threading
import time
from collections import defaultdict, deque
from typing import Callable

from src.boot.validate_env import _log_cred_presence
from src.config import LOG_LEVEL, LOG_MIN_INTERVAL_SEC, LOG_SAMPLE_RATE, settings
from src.utils.log_filters import install_warmup_filters
from src.utils.logger_setup import setup_logging
from src.utils.logging_tools import RateLimitFilter, log_buffer_handler


class LogGate:
    """Keyed rate limiter with optional sampling for noisy log statements."""

    _DEFAULT_KEY = "__default__"

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
        self._last_emitted: defaultdict[str, float] = defaultdict(lambda: 0.0)

    @property
    def min_interval(self) -> float:
        return self._min_interval

    @property
    def sample_rate(self) -> float:
        return self._sample_rate

    def should_emit(self, key: str, *, force: bool = False) -> bool:
        """Return ``True`` when ``key`` should be emitted."""

        if force:
            self._last_emitted[key] = self._clock()
            return True
        if self._sample_rate < 1.0 and self._rng.random() > self._sample_rate:
            return False
        now = self._clock()
        last_emit = self._last_emitted[key]
        if (now - last_emit) < self._min_interval:
            return False
        self._last_emitted[key] = now
        return True

    def allow(self, *, force: bool = False) -> bool:
        """Backward-compatible wrapper for non-keyed callers."""

        return self.should_emit(self._DEFAULT_KEY, force=force)

    def reset(self, key: str | None = None) -> None:
        """Reset the stored timestamps for ``key`` or all keys."""

        if key is None:
            self._last_emitted.clear()
        else:
            self._last_emitted.pop(key, None)


class SimpleLogGate:
    """Thread-safe keyed rate limiter without sampling."""

    def __init__(
        self,
        default_interval_seconds: float = 0.0,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._default_interval = float(default_interval_seconds)
        self._clock = clock or time.monotonic
        self._last_emitted: dict[str, float] = {}
        self._key_intervals: dict[str, float] = {}
        self._lock = threading.Lock()

    def set_interval(self, key: str, seconds: float) -> None:
        """Configure a persistent interval for ``key``."""

        window = max(0.0, float(seconds))
        with self._lock:
            if window == 0.0:
                self._key_intervals.pop(key, None)
            else:
                self._key_intervals[key] = window

    def _interval_for(self, key: str, override: float | None) -> float:
        if override is not None:
            return float(override)
        with self._lock:
            return self._key_intervals.get(key, self._default_interval)

    def should_emit(
        self,
        key: str,
        *,
        interval: float | None = None,
        force: bool = False,
    ) -> bool:
        """Return ``True`` when ``key`` may be emitted again."""

        if force:
            with self._lock:
                self._last_emitted[key] = self._clock()
            return True

        window = self._interval_for(key, interval)
        if window <= 0.0:
            with self._lock:
                self._last_emitted[key] = self._clock()
            return True

        now = self._clock()
        with self._lock:
            last = self._last_emitted.get(key)
            if last is None or (now - last) >= window:
                self._last_emitted[key] = now
                return True
            return False

    def ok(
        self,
        key: str,
        *,
        interval: float | None = None,
        force: bool = False,
    ) -> bool:
        """Alias for :meth:`should_emit` for ergonomic call sites."""

        return self.should_emit(key, interval=interval, force=force)

    def reset(self, key: str | None = None) -> None:
        """Reset stored timestamps for ``key`` or all keys."""

        with self._lock:
            if key is None:
                self._last_emitted.clear()
            else:
                self._last_emitted.pop(key, None)


_GLOBAL_LOG_GATE = LogGate(LOG_MIN_INTERVAL_SEC, LOG_SAMPLE_RATE)


def should_emit_log(
    key: str,
    *,
    force: bool = False,
    gate: LogGate | None = None,
) -> bool:
    """Return ``True`` when a log for ``key`` should be emitted."""

    current_gate = gate or _GLOBAL_LOG_GATE
    return current_gate.should_emit(key, force=force)


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
        level_name = getattr(settings, "LOG_LEVEL", None) or settings.log_level
        try:
            level_value = getattr(logging, str(level_name).upper(), logging.INFO)
        except Exception:
            level_value = logging.INFO
        logging.getLogger().setLevel(level_value)
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
