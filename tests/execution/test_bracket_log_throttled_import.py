"""Regression: _log_throttled used `logging` without importing it -> NameError in
the watchdog exit loop (safety net), swallowed as 'Failure in _watchdog_exit_loop'."""
from __future__ import annotations

from typing import Any

from nifty_scalper_bot.execution.bracket_manager import BracketManager


class _OM:
    def __init__(self) -> None:
        self._broker = None
    def place_order(self, **k: Any) -> str:
        return "x"
    def cancel_order(self, _o: str) -> bool:
        return True


async def test_log_throttled_does_not_raise_nameerror() -> None:
    mgr = BracketManager(order_manager=_OM())
    # Must not raise NameError: name 'logging' is not defined
    mgr._log_throttled("info", "TEST:key", 0.0, "msg %s", "arg")
    mgr._log_throttled("warning", "TEST:key2", 0.0, "msg2")
