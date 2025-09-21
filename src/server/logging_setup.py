"""Utilities for initializing structured logging for server processes."""

from __future__ import annotations

from threading import Lock

from src.server.logging_utils import _setup_logging

_init_lock = Lock()
_initialized = False


def setup_root_logger(*, force: bool = False) -> None:
    """Initialize the root logger if it has not been configured yet.

    Parameters
    ----------
    force:
        When ``True`` the logging configuration is applied even if a previous
        invocation succeeded. This is useful for tests that need to override
        handlers.
    """

    global _initialized
    with _init_lock:
        if _initialized and not force:
            return
        _setup_logging()
        _initialized = True
