"""Deterministic market-data hardening bootstrap.

This module verifies that candle, manager, and WebSocket hardening hooks are
installed before the trading application is built.
"""

from __future__ import annotations

from typing import Any

_MDM_HARDENING_ATTR = "_freshness_hardening_installed"
_WS_HARDENING_ATTR = "_market_data_hardening_installed"
_CANDLE_HARDENING_ATTR = "_candle_state_hardening_installed"
_CLOCK_FLUSH_HARDENING_ATTR = "_candle_clock_flush_hardening_installed"


def install_market_data_hardening_or_raise(logger: Any | None = None) -> dict[str, bool]:
    """Install all hardening layers while preserving the public return contract."""

    from nifty_scalper_bot.data.candle_clock_flush_hardening import (
        install_candle_clock_flush_hardening,
    )
    from nifty_scalper_bot.data.candle_engine import CandleEngine
    from nifty_scalper_bot.data.candle_state_hardening import (
        install_candle_state_hardening,
    )
    from nifty_scalper_bot.data.market_data_hardening import (
        install_market_data_manager_hardening,
    )
    from nifty_scalper_bot.data.market_data_manager import MarketDataManager
    from nifty_scalper_bot.streaming.market_data_hardening import (
        install_websocket_market_data_hardening,
    )
    from nifty_scalper_bot.streaming.websocket_manager import WebSocketManager

    install_candle_state_hardening(CandleEngine)
    install_market_data_manager_hardening(MarketDataManager)
    install_candle_clock_flush_hardening(MarketDataManager)
    install_websocket_market_data_hardening(WebSocketManager)

    full_state = {
        "candle": bool(getattr(CandleEngine, _CANDLE_HARDENING_ATTR, False)),
        "clock_flush": bool(
            getattr(MarketDataManager, _CLOCK_FLUSH_HARDENING_ATTR, False)
        ),
        "mdm": bool(getattr(MarketDataManager, _MDM_HARDENING_ATTR, False)),
        "websocket": bool(getattr(WebSocketManager, _WS_HARDENING_ATTR, False)),
    }
    # Preserve the established API used by startup callers and existing tests.
    state = {"mdm": full_state["mdm"], "websocket": full_state["websocket"]}

    if logger is not None:
        logger.info(
            "MARKET_DATA_HARDENING_INSTALLED mdm=%s websocket=%s candle=%s clock_flush=%s",
            full_state["mdm"],
            full_state["websocket"],
            full_state["candle"],
            full_state["clock_flush"],
            extra={"event": "MARKET_DATA_HARDENING_INSTALLED", **full_state},
        )

    if not all(full_state.values()):
        raise RuntimeError(f"market_data_hardening_not_installed state={full_state}")

    return state


__all__ = ["install_market_data_hardening_or_raise"]
