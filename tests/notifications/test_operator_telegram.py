from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from telegram.ext import CommandHandler

from nifty_scalper_bot.notifications.operator_telegram import (
    OPERATOR_COMMAND_NAMES,
    register_operator_commands,
    registered_command_names,
)


class FakeApp:
    def __init__(self) -> None:
        self.handlers: dict[int, list[object]] = {0: []}

    def add_handler(self, handler: object, group: int = 0) -> None:
        self.handlers.setdefault(group, []).append(handler)


class DummyMessage:
    def __init__(self) -> None:
        self.replies: list[str] = []
        self.chat = SimpleNamespace(id=12345)

    async def reply_text(self, text: str, **_: Any) -> None:
        self.replies.append(text)


class DummyUpdate:
    def __init__(self, chat_id: int = 12345) -> None:
        self.effective_chat = SimpleNamespace(id=chat_id)
        self.effective_message = DummyMessage()
        self.effective_message.chat = self.effective_chat
        self.message = self.effective_message


class DummyContext:
    pass


def _handler(app: FakeApp, command: str) -> CommandHandler:
    for item in app.handlers.get(0, []):
        if isinstance(item, CommandHandler) and command in item.commands:
            return item
    raise AssertionError(f"missing command {command}")


@pytest.mark.asyncio
async def test_operator_registry_keeps_only_production_commands(caplog: pytest.LogCaptureFixture) -> None:
    app = FakeApp()
    app.add_handler(CommandHandler("legacy", lambda *_: None))
    service = SimpleNamespace(chat_id=12345)

    with caplog.at_level("INFO"):
        commands = register_operator_commands(app, service)  # type: ignore[arg-type]

    assert commands == sorted(OPERATOR_COMMAND_NAMES)
    assert registered_command_names(app) == sorted(OPERATOR_COMMAND_NAMES)  # type: ignore[arg-type]
    assert "legacy" not in commands
    assert any("TELEGRAM_COMMAND_REGISTRY_FINAL" in rec.getMessage() for rec in caplog.records)


@pytest.mark.asyncio
async def test_help_lists_exactly_kept_commands() -> None:
    app = FakeApp()
    service = SimpleNamespace(chat_id=12345)
    register_operator_commands(app, service)  # type: ignore[arg-type]
    update = DummyUpdate()

    await _handler(app, "help").callback(update, DummyContext())  # type: ignore[arg-type]

    lines = update.effective_message.replies[-1].splitlines()
    assert [line.split(" - ", 1)[0] for line in lines] == list(OPERATOR_COMMAND_NAMES)
    assert "mode -" not in update.effective_message.replies[-1]
    assert "emergencystop -" not in update.effective_message.replies[-1]


@pytest.mark.asyncio
async def test_ping_replies_pong_with_missing_subsystems() -> None:
    app = FakeApp()
    service = SimpleNamespace(chat_id=12345)
    register_operator_commands(app, service)  # type: ignore[arg-type]
    update = DummyUpdate()

    await _handler(app, "ping").callback(update, DummyContext())  # type: ignore[arg-type]

    assert update.effective_message.replies == ["pong"]


@pytest.mark.asyncio
async def test_unauthorized_chat_is_rejected_and_logged(caplog: pytest.LogCaptureFixture) -> None:
    app = FakeApp()
    service = SimpleNamespace(chat_id=12345)
    register_operator_commands(app, service)  # type: ignore[arg-type]
    update = DummyUpdate(chat_id=999)

    with caplog.at_level("WARNING"):
        await _handler(app, "ping").callback(update, DummyContext())  # type: ignore[arg-type]

    assert update.effective_message.replies == []
    assert "received_chat_id=999 expected_chat_id=12345" in caplog.text


@pytest.mark.asyncio
async def test_missing_subsystem_warns_not_silent() -> None:
    app = FakeApp()
    service = SimpleNamespace(chat_id=12345)
    register_operator_commands(app, service)  # type: ignore[arg-type]
    update = DummyUpdate()

    await _handler(app, "check_core").callback(update, DummyContext())  # type: ignore[arg-type]

    assert "WARN: StrategyRunner not attached" in update.effective_message.replies[-1]


@pytest.mark.asyncio
async def test_selftest_is_read_only() -> None:
    app = FakeApp()
    broker = SimpleNamespace(place_order=lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not trade")))
    order = SimpleNamespace(cancel_order=lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not cancel")))
    service = SimpleNamespace(chat_id=12345, deps=SimpleNamespace(broker_client=broker, order_manager=order))
    register_operator_commands(app, service)  # type: ignore[arg-type]
    update = DummyUpdate()

    await _handler(app, "selftest").callback(update, DummyContext())  # type: ignore[arg-type]

    assert "Selftest (read-only)" in update.effective_message.replies[-1]


@pytest.mark.asyncio
async def test_emergency_calls_kill_switch_only_when_available() -> None:
    app = FakeApp()
    called: list[str] = []

    def kill_switch(reason: str) -> str:
        called.append(reason)
        return "armed"

    service = SimpleNamespace(
        chat_id=12345,
        deps=SimpleNamespace(order_manager=SimpleNamespace(engage_kill_switch=kill_switch)),
    )
    register_operator_commands(app, service)  # type: ignore[arg-type]
    update = DummyUpdate()

    await _handler(app, "emergency").callback(update, DummyContext())  # type: ignore[arg-type]

    assert called == ["telegram_emergency"]
    assert "Emergency triggered" in update.effective_message.replies[-1]


@pytest.mark.asyncio
async def test_emergency_reports_unwired_without_side_effect() -> None:
    app = FakeApp()
    service = SimpleNamespace(chat_id=12345)
    register_operator_commands(app, service)  # type: ignore[arg-type]
    update = DummyUpdate()

    await _handler(app, "emergency").callback(update, DummyContext())  # type: ignore[arg-type]

    assert update.effective_message.replies == ["Emergency handler not wired."]
