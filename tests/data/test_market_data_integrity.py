from nifty_scalper_bot.data.market_data_manager import MarketDataManager


def test_duplicate_tick_is_dropped() -> None:
    manager = MarketDataManager()
    tick = {'timestamp': 100.0, 'ltp': 200.0}
    assert manager._is_duplicate('NSE:NIFTY', tick) is False
    assert manager._is_duplicate('NSE:NIFTY', tick) is True
