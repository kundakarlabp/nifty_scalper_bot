"""IST-aware market session gate."""

from __future__ import annotations

from datetime import datetime, timedelta

from ..utils.time_utils import IST, SESSION_END, SESSION_START, is_trading_hours


class SessionGate:
    """Controls when the bot is allowed to trade."""

    def __init__(self, override: bool = False) -> None:
        self._override = override  # For testing

    def is_trading_now(self) -> bool:
        if self._override:
            return True
        return is_trading_hours()

    def time_to_open(self) -> timedelta:
        from ..utils.time_utils import time_to_open
        return time_to_open()

    def time_to_close(self) -> timedelta:
        from ..utils.time_utils import time_to_close
        return time_to_close()
