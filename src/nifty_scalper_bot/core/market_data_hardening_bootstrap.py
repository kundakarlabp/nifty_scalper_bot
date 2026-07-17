"""Deterministic market-data hardening bootstrap.

This module is intentionally narrow: it verifies that the live market-data
classes expose their runtime hardening hooks before the trading app is built.
The class definition files still own the actual hook installation; this layer
makes startup visibility explicit and fails loudly in real live mode if the
hooks cannot be confirmed.
"""

from __future__ import annotations

from typing import Any

_MDM_HARDENING_ATTR = "_freshness_hardening_installed"
_WS_HARDENING_ATTR = "_market_data_hardening_installed"
_CANDLE_HARDENING_ATTR = "_candle_state_hardening_installed"


def install_market_data_hardening_or_raise(logger: Any | None = None) -> dict[str, bool]:
    """Install and verify candle, MDM, and WebSocket hardening layers."""

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

    # Install candle hardening before MarketDataManager instances are created so
    # every engine uses the same per-engine serialization and closed-minute
    # watermark contract from its first tick.
    install_candle_state_hardening(CandleEngine)
    install_market_data_manager_hardening(MarketDataManager)
    install_websocket_market_data_hardening(WebSocketManager)

    state = {
        "candle": bool(getattr(CandleEngine, _CANDLE_HARDENING_ATTR, False)),
        "mdm": bool(getattr(MarketDataManager, _MDM_HARDENING_ATTR, False)),
        "websocket": bool(getattr(WebSocketManager, _WS_HARDENING_ATTR, False)),
    }

    if logger is not None:
        logger.info(
            "MARKET_DATA_HARDENING_INSTALLED candle=%s mdm=%s websocket=%s",
            state["candle"],
            state["mdm"],
            state["websocket"],
            extra={
                "event": "MARKET_DATA_HARDENING_INSTALLED",
                "candle": state["candle"],
                "mdm": state["mdm"],
                "websocket": state["websocket"],
            },
        )

    if not all(state.values()):
        raise RuntimeError(f"market_data_hardening_not_installed state={state}")

    return state


__all__ = ["install_market_data_hardening_or_raise"]
