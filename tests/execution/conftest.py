"""Execution-test resource cleanup fixtures."""

from __future__ import annotations

import threading
from collections.abc import Generator
from functools import wraps
from typing import Any

import pytest

from nifty_scalper_bot.execution.bracket_core import BracketManager as CoreBracketManager


@pytest.fixture(autouse=True)
def cleanup_bracket_watchdogs(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[None, None, None]:
    """Stop every bracket watchdog created by an execution test."""
    created: list[CoreBracketManager] = []
    original_init = CoreBracketManager.__init__

    @wraps(original_init)
    def tracked_init(self: CoreBracketManager, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        created.append(self)

    monkeypatch.setattr(CoreBracketManager, "__init__", tracked_init)
    yield

    current = threading.current_thread()
    for manager in reversed(created):
        manager.shutdown()
        watchdog = getattr(manager, "_watchdog_thread", None)
        if watchdog is not None and watchdog is not current and watchdog.is_alive():
            watchdog.join(timeout=1.0)
