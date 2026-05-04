from types import SimpleNamespace
from nifty_scalper_bot.data.market_data_manager import MarketDataManager


def test_start_websocket_pending_connection_logs(caplog):
    mdm = MarketDataManager.__new__(MarketDataManager)
    mdm._ws = SimpleNamespace(start=lambda: None)
    mdm._desired_tokens = {1, 2}
    mdm._ws_start_requested = False
    mdm._ws_connect_task_started = False
    mdm._ws_connected = False
    mdm._logger = __import__('logging').getLogger('test.mdm')
    mdm._reconcile_ws_subscriptions = lambda: None
    mdm.ws_status_snapshot = lambda: {"connect_task_alive": True, "connected": False, "tokens": 1}
    with caplog.at_level('INFO'):
        MarketDataManager.start_websocket(mdm)
    assert mdm._ws_start_requested is True
    assert mdm._ws_connected is False
    assert "MDM_WS_START_PENDING_CONNECTION" in caplog.text
