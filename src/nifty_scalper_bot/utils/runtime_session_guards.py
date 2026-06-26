"""Install cross-cutting market-session guards at package startup.

This module keeps transport code independent of contract-selection logic while
ensuring its private clock-window check also respects the canonical NSE holiday
calendar. The installation is idempotent and does not alter custom intraday
start/end times or off-hours test mode.
"""

from __future__ import annotations

from datetime import datetime
from threading import Lock
from typing import Any, Callable

from nifty_scalper_bot.utils.smart_symbol import is_nse_trading_day

_INSTALL_LOCK = Lock()
_INSTALLED = False


def install_websocket_market_calendar_guard() -> bool:
    """Make WebSocket watchdog/reconnect windows holiday-aware.

    Returns ``True`` when the class is guarded after this call. The original
    method remains authoritative for custom time-window and disabled-window
    behavior; the wrapper adds only the missing exchange-calendar decision.
    """
    global _INSTALLED
    with _INSTALL_LOCK:
        if _INSTALLED:
            return True
        try:
            from nifty_scalper_bot.streaming.websocket_manager import WebSocketManager
        except Exception:
            return False

        original: Callable[[Any], bool] = WebSocketManager._is_within_trading_window
        if bool(getattr(original, "_nse_calendar_guard", False)):
            _INSTALLED = True
            return True

        def holiday_aware_window(self: Any) -> bool:
            if not bool(getattr(self, "_trading_window_enabled", True)):
                return original(self)
            timezone = getattr(self, "_trading_tz", None)
            try:
                now = datetime.now(timezone) if timezone is not None else datetime.now()
                if not is_nse_trading_day(now.date()):
                    return False
            except Exception:
                # Fail back to the existing transport window logic. Trading and
                # order arming retain their independent fail-closed gates.
                pass
            return original(self)

        setattr(holiday_aware_window, "_nse_calendar_guard", True)
        setattr(holiday_aware_window, "_unguarded_window", original)
        WebSocketManager._is_within_trading_window = holiday_aware_window
        _INSTALLED = True
        return True


__all__ = ["install_websocket_market_calendar_guard"]
