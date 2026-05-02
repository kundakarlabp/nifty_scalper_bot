from __future__ import annotations

from nifty_scalper_bot.core.app import _polling_fallback_degraded
from nifty_scalper_bot.streaming.websocket_manager import WebSocketManager


def test_1006_marks_degraded_not_failed(monkeypatch) -> None:
    ws = WebSocketManager('k', 't', tokens=[1])
    called: list[str] = []
    monkeypatch.setattr(ws, '_schedule_reconnect', lambda reason: called.append(reason))
    ws._on_close(None, 1006, 'abnormal')
    assert ws._stream_health == 'degraded'
    assert ws._last_disconnect_at > 0
    assert called == ['close:1006']


def test_reconnect_replays_subscriptions() -> None:
    ws = WebSocketManager('k', 't', tokens=[1, 2])

    class Ticker:
        MODE_FULL = 'full'

        def __init__(self) -> None:
            self.subscribed = []
            self.mode_set = []

        def subscribe(self, payload):
            self.subscribed.extend(payload)

        def set_mode(self, mode, payload):
            self.mode_set.append((mode, list(payload)))

    ticker = Ticker()
    ws._ticker = ticker
    ws._connected.set()
    import asyncio
    asyncio.run(ws._resubscribe_if_connected())
    assert ticker.subscribed == [1, 2]


def test_live_execution_blocked_when_stream_stale_or_degraded() -> None:
    assert _polling_fallback_degraded(ws_ok=False, lagging=False, futures_fresh=True, options_fresh=True) is True
