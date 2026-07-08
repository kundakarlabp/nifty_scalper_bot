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


def install_market_data_hardening_or_raise(logger: Any | None = None) -> dict[str, bool]:
    """Install and verify MDM/WebSocket market-data hardening.

    Args:
        logger: Optional structured logger used for the startup confirmation.

    Returns:
        Mapping with ``mdm`` and ``websocket`` installation states.

    Raises:
        RuntimeError: If either hardening layer is unavailable after explicit
            idempotent installation.
    """

    from nifty_scalper_bot.data.market_data_hardening import (
        install_market_data_manager_hardening,
    )
    from nifty_scalper_bot.data.market_data_manager import MarketDataManager
    from nifty_scalper_bot.streaming.market_data_hardening import (
        install_websocket_market_data_hardening,
    )
    from nifty_scalper_bot.streaming.websocket_manager import WebSocketManager

    install_market_data_manager_hardening(MarketDataManager)
    install_websocket_market_data_hardening(WebSocketManager)

    state = {
        "mdm": bool(getattr(MarketDataManager, _MDM_HARDENING_ATTR, False)),
        "websocket": bool(getattr(WebSocketManager, _WS_HARDENING_ATTR, False)),
    }

    if logger is not None:
        logger.info(
            "MARKET_DATA_HARDENING_INSTALLED mdm=%s websocket=%s",
            state["mdm"],
            state["websocket"],
            extra={
                "event": "MARKET_DATA_HARDENING_INSTALLED",
                "mdm": state["mdm"],
                "websocket": state["websocket"],
            },
        )

    if not all(state.values()):
        raise RuntimeError(f"market_data_hardening_not_installed state={state}")

    return state


__all__ = ["install_market_data_hardening_or_raise"]
