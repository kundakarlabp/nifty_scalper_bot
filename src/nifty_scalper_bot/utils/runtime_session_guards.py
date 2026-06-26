"""Install cross-cutting market-session guards for streaming transport."""

from __future__ import annotations

import time
from datetime import datetime
from threading import Lock
from typing import Any, Callable

from nifty_scalper_bot.utils.smart_symbol import is_nse_trading_day

_INSTALL_LOCK = Lock()
_INSTALLED = False


def install_websocket_market_calendar_guard() -> bool:
    """Make WebSocket liveness, reconnect and close handling session-aware.

    The original transport time-window logic remains authoritative for custom
    intraday start/end times. The wrapper adds the missing NSE holiday decision
    and prevents normal off-session code-1006 closes from being escalated or
    repeatedly reconnected.
    """
    global _INSTALLED
    with _INSTALL_LOCK:
        if _INSTALLED:
            return True
        try:
            from nifty_scalper_bot.streaming.websocket_manager import (
                ConnectionState,
                WebSocketManager,
            )
        except Exception:
            return False

        original_window: Callable[[Any], bool] = (
            WebSocketManager._is_within_trading_window
        )
        if bool(getattr(original_window, "_nse_calendar_guard", False)):
            _INSTALLED = True
            return True
        original_close: Callable[[Any, Any, int, str], None] = WebSocketManager._on_close

        def holiday_aware_window(self: Any) -> bool:
            if not bool(getattr(self, "_trading_window_enabled", True)):
                return original_window(self)
            timezone = getattr(self, "_trading_tz", None)
            try:
                now = datetime.now(timezone) if timezone is not None else datetime.now()
                if not is_nse_trading_day(now.date()):
                    return False
            except Exception:
                # Fall back to the existing transport window logic. Order arming
                # retains independent fail-closed session and readiness gates.
                pass
            return original_window(self)

        def session_aware_close(self: Any, ws: Any, code: int, reason: str) -> None:
            if int(code or 0) == 1006 and not holiday_aware_window(self):
                try:
                    self._connected.clear()
                    self._state = ConnectionState.DISCONNECTED
                    self._last_disconnect_at = time.time()
                    self._stream_health = "idle"
                    callback = getattr(self, "_on_disconnect_callback", None)
                    if callable(callback):
                        callback()
                    self._logger.info(
                        "WEBSOCKET_IDLE_CLOSED code=1006 reason=%s action=no_reconnect",
                        reason or "closing_handshake_timeout",
                        extra={
                            "event": "WEBSOCKET_IDLE_CLOSED",
                            "code": 1006,
                            "reason": reason or "closing_handshake_timeout",
                            "action": "no_reconnect",
                        },
                    )
                    return
                except Exception:
                    # Preserve the original handler if the defensive quiet-close
                    # path cannot update a legacy manager instance safely.
                    pass
            original_close(self, ws, code, reason)

        setattr(holiday_aware_window, "_nse_calendar_guard", True)
        setattr(holiday_aware_window, "_unguarded_window", original_window)
        setattr(session_aware_close, "_nse_session_close_guard", True)
        setattr(session_aware_close, "_unguarded_close", original_close)
        WebSocketManager._is_within_trading_window = holiday_aware_window
        WebSocketManager._on_close = session_aware_close
        _INSTALLED = True
        return True


__all__ = ["install_websocket_market_calendar_guard"]
