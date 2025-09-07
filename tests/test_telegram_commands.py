from types import SimpleNamespace
import requests
import time

from src.notifications.telegram_commands import TelegramCommands


def test_loop_invokes_callback(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    tc = TelegramCommands("t", "123", lambda cmd, arg: calls.append((cmd, arg)))
    tc._running = True

    monkeypatch.setattr(time, "sleep", lambda *_: None)

    def fake_get(url: str, params: dict, timeout: int):
        tc._running = False
        return SimpleNamespace(
            status_code=200,
            json=lambda: {
                "result": [
                    {
                        "update_id": 1,
                        "message": {
                            "text": "/hello world",
                            "chat": {"id": "123"},
                        },
                    },
                    {
                        "update_id": 2,
                        "message": {
                            "text": "/ping",
                            "chat": {"id": "123"},
                        },
                    },
                ]
            },
        )

    monkeypatch.setattr(requests, "get", fake_get)
    tc._loop()
    assert calls == [("/hello", "world"), ("/ping", "")]


def test_loop_ignores_unknown_chat(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    tc = TelegramCommands("t", "123", lambda cmd, arg: calls.append((cmd, arg)))
    tc._running = True

    monkeypatch.setattr(time, "sleep", lambda *_: None)

    def fake_get(url: str, params: dict, timeout: int):
        tc._running = False
        return SimpleNamespace(
            status_code=200,
            json=lambda: {
                "result": [
                    {
                        "update_id": 1,
                        "message": {
                            "text": "/hello world",
                            "chat": {"id": "999"},
                        },
                    }
                ]
            },
        )

    monkeypatch.setattr(requests, "get", fake_get)
    tc._loop()
    assert calls == []


def test_loop_ignores_empty_message(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    tc = TelegramCommands("t", "123", lambda cmd, arg: calls.append((cmd, arg)))
    tc._running = True

    monkeypatch.setattr(time, "sleep", lambda *_: None)

    def fake_get(url: str, params: dict, timeout: int):
        tc._running = False
        return SimpleNamespace(
            status_code=200,
            json=lambda: {
                "result": [
                    {
                        "update_id": 1,
                        "message": {
                            "text": "",
                            "chat": {"id": "123"},
                        },
                    }
                ]
            },
        )

    monkeypatch.setattr(requests, "get", fake_get)
    tc._loop()
    assert calls == []


def test_loop_recovers_from_network_error(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    tc = TelegramCommands("t", "123", lambda cmd, arg: calls.append((cmd, arg)))
    tc._running = True

    monkeypatch.setattr(time, "sleep", lambda *_: None)

    responses = [
        requests.exceptions.ConnectionError("boom"),
        SimpleNamespace(
            status_code=200,
            json=lambda: {
                "result": [
                    {
                        "update_id": 1,
                        "message": {
                            "text": "/start",
                            "chat": {"id": "123"},
                        },
                    }
                ]
            },
        ),
    ]

    def fake_get(url: str, params: dict, timeout: int):
        res = responses.pop(0)
        if isinstance(res, Exception):
            raise res
        tc._running = False
        return res

    monkeypatch.setattr(requests, "get", fake_get)
    tc._loop()
    assert calls == [("/start", "")]
