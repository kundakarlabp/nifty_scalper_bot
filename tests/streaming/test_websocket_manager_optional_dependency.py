from nifty_scalper_bot.streaming.websocket_manager import WebSocketManager


def test_build_ticker_raises_clear_error_when_sdk_missing(monkeypatch):
    monkeypatch.setattr('nifty_scalper_bot.streaming.websocket_manager.KiteTicker', None)
    manager = WebSocketManager(api_key='k', access_token='t')
    try:
        manager._build_ticker()
    except RuntimeError as exc:
        assert 'kiteconnect is not installed' in str(exc)
    else:
        raise AssertionError('expected RuntimeError')
