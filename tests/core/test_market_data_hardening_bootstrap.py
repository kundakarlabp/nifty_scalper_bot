from __future__ import annotations

import logging

from nifty_scalper_bot.core.market_data_hardening_bootstrap import (
    install_market_data_hardening_or_raise,
)
from nifty_scalper_bot.data.market_data_manager import MarketDataManager
from nifty_scalper_bot.streaming.websocket_manager import WebSocketManager


def test_market_data_hardening_bootstrap_confirms_mdm_and_websocket_hooks(caplog) -> None:
    caplog.set_level(logging.INFO)

    state = install_market_data_hardening_or_raise(logging.getLogger("test.market_data_hardening"))

    assert state == {"mdm": True, "websocket": True}
    assert getattr(MarketDataManager, "_freshness_hardening_installed", False) is True
    assert getattr(WebSocketManager, "_market_data_hardening_installed", False) is True
    assert "MARKET_DATA_HARDENING_INSTALLED mdm=True websocket=True" in caplog.text
