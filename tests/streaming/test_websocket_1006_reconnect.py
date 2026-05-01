from __future__ import annotations

from nifty_scalper_bot.streaming.websocket_manager import WebSocketManager


def test_1006_marks_degraded_and_schedules_reconnect(monkeypatch) -> None:
    ws = WebSocketManager('k', 't', tokens=[1, 2])
    called: list[str] = []
    monkeypatch.setattr(ws, '_schedule_reconnect', lambda reason: called.append(reason))

    ws._on_close(None, 1006, 'abnormal')

    assert ws._stream_health == 'degraded'
    assert ws._last_disconnect_at > 0
    assert called and called[0] == 'close:1006'
