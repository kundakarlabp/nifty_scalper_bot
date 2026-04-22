from __future__ import annotations

import logging
from types import SimpleNamespace

from nifty_scalper_bot.streaming.websocket_manager import WebSocketManager


def test_logs_ws_subscriptions_for_futures_and_options(caplog) -> None:
    manager = WebSocketManager(api_key='k', access_token='t')
    manager._market_data_manager = SimpleNamespace(
        _symbol_by_token={101: 'NFO:NIFTY26MAYFUT', 202: 'NFO:NIFTY26MAY24500CE'}
    )

    with caplog.at_level(logging.INFO):
        manager._log_ws_subscriptions([101, 202, 101])

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        'ws_transport_subscribed symbol=NFO:NIFTY26MAYFUT token=101' in msg
        for msg in messages
    )
    assert any(
        'ws_transport_subscribed symbol=NFO:NIFTY26MAY24500CE token=202' in msg
        for msg in messages
    )
