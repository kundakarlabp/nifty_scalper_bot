from __future__ import annotations

import importlib


def test_production_app_import_installs_all_market_data_hardening() -> None:
    """The real core.app import must install hardening without manual calls."""
    app_module = importlib.import_module("nifty_scalper_bot.core.app")
    assert app_module is not None

    from nifty_scalper_bot.data.candle_engine import CandleEngine
    from nifty_scalper_bot.data.market_data_manager import MarketDataManager
    from nifty_scalper_bot.streaming.websocket_manager import WebSocketManager

    assert getattr(CandleEngine, "_candle_state_hardening_installed", False) is True
    assert getattr(MarketDataManager, "_freshness_hardening_installed", False) is True
    assert (
        getattr(MarketDataManager, "_candle_clock_flush_hardening_installed", False)
        is True
    )
    assert getattr(WebSocketManager, "_market_data_hardening_installed", False) is True


def test_production_app_import_is_idempotent() -> None:
    import nifty_scalper_bot.core.app as app_module

    before = {
        "app": id(app_module),
    }
    reloaded = importlib.reload(app_module)
    assert id(reloaded) == before["app"]

    from nifty_scalper_bot.data.candle_engine import CandleEngine
    from nifty_scalper_bot.data.market_data_manager import MarketDataManager

    assert getattr(CandleEngine, "_candle_state_hardening_installed", False) is True
    assert (
        getattr(MarketDataManager, "_candle_clock_flush_hardening_installed", False)
        is True
    )
