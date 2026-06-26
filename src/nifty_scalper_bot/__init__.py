"""Nifty scalper bot package."""

from nifty_scalper_bot.utils.runtime_session_guards import (
    install_websocket_market_calendar_guard,
)

# Idempotent cross-cutting guard: WebSocket liveness/reconnect watchdogs must use
# the same NSE trading calendar as strategy and execution session gates.
install_websocket_market_calendar_guard()

__all__: list[str] = []
