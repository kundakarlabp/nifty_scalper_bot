from nifty_scalper_bot.data.market_data_manager import MarketDataManager


def test_ws_close_does_not_reset_tick_cache_storage() -> None:
    mdm = MarketDataManager(broker=None, websocket=None, settings={})
    mdm._last_tick_time['NSE:NIFTY'] = 1.0
    assert 'NSE:NIFTY' in mdm._last_tick_time
