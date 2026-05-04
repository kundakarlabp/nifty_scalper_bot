from nifty_scalper_bot.data.market_data_manager import MarketDataManager


def test_history_serialization_primitives_exist() -> None:
    mdm = MarketDataManager(broker=None, websocket=None, settings={})
    assert hasattr(mdm, '_history_lock')
    assert hasattr(mdm, '_last_history_request_ts')
    assert hasattr(mdm, '_history_min_interval_sec')
