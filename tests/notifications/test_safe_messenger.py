import types

import pytest
from telegram.constants import ParseMode
from telegram.error import BadRequest

from nifty_scalper_bot.notifications.safe_messenger import MAX, SUFFIX, SafeMessenger


class _StubBot:
    def __init__(self) -> None:
        self.messages: list[tuple[int, str, dict[str, object]]] = []

    async def send_message(self, chat_id: int, text: str, **kwargs: object):
        self.messages.append((chat_id, text, dict(kwargs)))
        return types.SimpleNamespace(text=text)


@pytest.mark.asyncio
async def test_safe_messenger_escapes_plain_text() -> None:
    bot = _StubBot()
    messenger = SafeMessenger(bot)

    await messenger.send_text(1, "<b>danger</b>")

    assert bot.messages
    _, text, kwargs = bot.messages[-1]
    assert text == "&lt;b&gt;danger&lt;/b&gt;"
    assert kwargs.get("parse_mode") == ParseMode.HTML


class _PlainBot:
    def __init__(self) -> None:
        self.messages: list[tuple[int, str, dict[str, object]]] = []

    async def send_message(self, chat_id: int, text: str, **kwargs: object):
        self.messages.append((chat_id, text, dict(kwargs)))
        return types.SimpleNamespace(text=text)


@pytest.mark.asyncio
async def test_safe_messenger_plain_mode_uses_parse_mode_none() -> None:
    bot = _PlainBot()
    messenger = SafeMessenger(bot)

    await messenger.send_text(7, "<b>tag</b> & more", parse_mode=None)

    assert bot.messages
    _, text, kwargs = bot.messages[-1]
    assert text == "<b>tag</b> & more"
    assert kwargs.get("parse_mode") is None


class _FailingBot:
    def __init__(self) -> None:
        self.messages: list[str] = []
        self.plain_messages: list[str] = []
        self.documents: list[tuple[int, bytes, str | None]] = []

    async def send_message(
        self, chat_id: int, text: str, parse_mode=None, **kwargs: object
    ):
        if parse_mode is None:
            self.plain_messages.append(text)
        else:
            self.messages.append(text)
        raise BadRequest("invalid")

    async def send_document(
        self, chat_id: int, document, caption: str | None = None, **kwargs: object
    ):
        payload = document.read()
        document.seek(0)
        self.documents.append((chat_id, payload, caption))
        return types.SimpleNamespace(document=document, caption=caption)


@pytest.mark.asyncio
async def test_safe_messenger_fallbacks_to_document() -> None:
    bot = _FailingBot()
    messenger = SafeMessenger(bot)
    text = "X" * (MAX + 100)

    result = await messenger.send_text(42, text)

    assert bot.messages
    assert bot.messages[-1].endswith(SUFFIX)
    assert bot.plain_messages
    assert bot.documents
    chat_id, payload, caption = bot.documents[-1]
    assert chat_id == 42
    assert caption == "Output too long"
    assert payload.decode("utf-8") == text
    assert result.caption == "Output too long"
