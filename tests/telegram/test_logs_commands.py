"""Telegram /logs (inline tail) and /dumplogs (downloadable file) commands.

These give an Android-friendly, nil-cost way to read and download logs without
SSH. Both read the in-process RING buffer. cmd_dump_logs existed but was never
registered; these are the wired, tested entry points.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nifty_scalper_bot.notifications import operator_telegram as ot
from nifty_scalper_bot.notifications.telegram_controller import RING


@pytest.fixture(autouse=True)
def _seed_ring():
    RING.buf.clear()
    for i in range(1, 251):
        RING.add(f"12:00:0{i % 10} [INFO] root: event line {i}")
    yield
    RING.buf.clear()


async def test_cmd_logs_replies_with_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    reply = AsyncMock()
    monkeypatch.setattr(ot, "safe_reply", reply)
    update = SimpleNamespace(effective_message=SimpleNamespace(text="/logs 20"))

    await ot.cmd_logs(update, SimpleNamespace(), service=None)

    reply.assert_awaited_once()
    text = reply.await_args.args[1]
    assert "line 250" in text  # newest present
    assert "line 231" in text  # 20 lines back
    assert "line 230" not in text  # only the last 20


async def test_cmd_logs_caps_message_length(monkeypatch: pytest.MonkeyPatch) -> None:
    reply = AsyncMock()
    monkeypatch.setattr(ot, "safe_reply", reply)
    update = SimpleNamespace(effective_message=SimpleNamespace(text="/logs 400"))

    await ot.cmd_logs(update, SimpleNamespace(), service=None)

    text = reply.await_args.args[1]
    assert len(text) <= 3520  # 3500 tail + truncation marker


async def test_cmd_dumplogs_sends_document() -> None:
    sent = {}

    async def _send_document(doc):
        sent["doc"] = doc

    chat = SimpleNamespace(send_document=_send_document)
    update = SimpleNamespace(
        effective_message=SimpleNamespace(text="/dumplogs 100"),
        effective_chat=chat,
    )

    await ot.cmd_dumplogs(update, SimpleNamespace(), service=None)

    assert "doc" in sent  # a document was sent
    # InputFile wraps the bytes; confirm our newest line is in the payload.
    payload = sent["doc"].input_file_content
    body = payload.decode() if isinstance(payload, (bytes, bytearray)) else str(payload)
    assert "line 250" in body


async def test_cmd_dumplogs_no_chat_falls_back_to_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reply = AsyncMock()
    monkeypatch.setattr(ot, "safe_reply", reply)
    update = SimpleNamespace(
        effective_message=SimpleNamespace(text="/dumplogs"), effective_chat=None
    )

    await ot.cmd_dumplogs(update, SimpleNamespace(), service=None)

    reply.assert_awaited_once()


async def test_logs_commands_registered() -> None:
    names = {spec.name for spec in ot.OPERATOR_COMMANDS}
    assert {"logs", "dumplogs"} <= names
