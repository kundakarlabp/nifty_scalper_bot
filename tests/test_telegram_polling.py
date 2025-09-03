from types import SimpleNamespace
import threading
import time
import requests

from src.notifications.telegram_controller import TelegramController
from src.config import settings


def _basic_settings(monkeypatch) -> None:
    monkeypatch.setattr(
        settings,
        "telegram",
        SimpleNamespace(bot_token="t", chat_id=1, enabled=True, extra_admin_ids=[]),
        raising=False,
    )


def test_stop_polling_closes_session(monkeypatch) -> None:
    _basic_settings(monkeypatch)
    tc = TelegramController(status_provider=lambda: {})
    calls: list[str] = []

    class DummySession:
        def close(self) -> None:
            calls.append("close")

    class DummyThread:
        def join(self, timeout: float | None = None) -> None:
            calls.append("join")

    tc._session = DummySession()
    tc._poll_thread = DummyThread()
    tc._started = True
    tc.stop_polling()
    assert calls == ["close", "join"]


def test_poll_loop_recovers_after_connection_error(monkeypatch) -> None:
    _basic_settings(monkeypatch)
    session_created: list[int] = []
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    error = {"raised": False}

    class DummySession:
        def __init__(self) -> None:
            session_created.append(1)

        def get(self, *args, **kwargs):
            if not error["raised"]:
                error["raised"] = True
                raise requests.exceptions.ConnectionError("boom")
            tc._stop.set()
            return SimpleNamespace(json=lambda: {"ok": True, "result": []})

        def close(self) -> None:
            pass

    monkeypatch.setattr(requests, "Session", DummySession)
    tc = TelegramController(status_provider=lambda: {})
    thread = threading.Thread(target=tc._poll_loop)
    thread.start()
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert len(session_created) >= 2
