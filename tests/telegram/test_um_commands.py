import html
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nifty_scalper_bot.core.unified_manager import UMInstrument, UMPrice
from nifty_scalper_bot.data.instruments import InstrumentResolver
from nifty_scalper_bot.notifications.telegram_controller import (
    TelegramBot,
    TelegramDeps,
)


def _extract_payload(mock_call: AsyncMock) -> dict[str, object]:
    """Return JSON payload from the last Telegram reply mock."""

    args, _kwargs = mock_call.await_args
    message = args[2]
    sanitized = (
        message.strip()
        .removeprefix("<pre>")
        .removesuffix("</pre>")
    )
    decoded = html.unescape(sanitized)
    return json.loads(decoded)


@pytest.mark.asyncio
async def test_cmd_resolve_known_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    resolver = InstrumentResolver(MagicMock())
    resolver.upsert("NIFTY25OCT25900CE", 123, exchange="NFO")

    deps = TelegramDeps(token="x", chat_id=1, app_version="test", resolver=resolver)
    bot = TelegramBot(deps)
    chat = SimpleNamespace(id=1)
    update = SimpleNamespace()
    ctx = SimpleNamespace(args=["NIFTY25OCT25900CE"])

    instrument = UMInstrument(
        symbol="NIFTY25OCT25900CE",
        exchange="NFO",
        token=123,
        lot_size=50,
        kind="OPT",
        expiry="2024-10-25",
    )
    manager = SimpleNamespace(resolve=MagicMock(return_value=instrument))

    monkeypatch.setattr(bot, "_get_unified_manager", lambda: manager)
    monkeypatch.setattr(bot, "_guard", AsyncMock(return_value=chat))
    reply = AsyncMock()
    monkeypatch.setattr(bot, "_reply", reply)

    await bot.cmd_resolve(update, ctx)

    payload = _extract_payload(reply)
    assert payload["symbol"] == "NIFTY25OCT25900CE"
    assert payload["exchange"] == "NFO"
    assert payload["token"] == 123
    assert payload["lot_size"] == 50
    assert payload["expiry"] == "2024-10-25"


@pytest.mark.asyncio
async def test_cmd_resolve_unknown_symbol_logs_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    resolver = InstrumentResolver(MagicMock())
    deps = TelegramDeps(token="x", chat_id=1, app_version="test", resolver=resolver)
    bot = TelegramBot(deps)
    chat = SimpleNamespace(id=1)
    update = SimpleNamespace()
    ctx = SimpleNamespace(args=["UNKNOWN"])

    instrument = UMInstrument(
        symbol="UNKNOWN",
        exchange=None,
        token=None,
        lot_size=1,
        kind="OPT",
        expiry=None,
    )
    manager = SimpleNamespace(resolve=MagicMock(return_value=instrument))

    monkeypatch.setattr(bot, "_get_unified_manager", lambda: manager)
    monkeypatch.setattr(bot, "_guard", AsyncMock(return_value=chat))
    reply = AsyncMock()
    monkeypatch.setattr(bot, "_reply", reply)

    caplog.set_level(logging.DEBUG, logger="nifty_scalper_bot.data.instruments")

    await bot.cmd_resolve(update, ctx)
    first_warnings = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING and "no token" in record.getMessage()
    ]
    assert len(first_warnings) == 1

    caplog.clear()

    await bot.cmd_resolve(update, ctx)
    second_warnings = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING and "no token" in record.getMessage()
    ]
    debug_records = [
        record
        for record in caplog.records
        if record.levelno == logging.DEBUG and "suppressed" in record.getMessage()
    ]
    assert not second_warnings
    assert debug_records


@pytest.mark.asyncio
async def test_cmd_umlearn_updates_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    resolver = InstrumentResolver(MagicMock())
    deps = TelegramDeps(token="x", chat_id=1, app_version="test", resolver=resolver)
    bot = TelegramBot(deps)
    chat = SimpleNamespace(id=1)
    update = SimpleNamespace()
    ctx = SimpleNamespace(args=["NIFTY25OCT25900CE"])

    class _StubManager:
        def __init__(self) -> None:
            self.price_calls = 0
            self.resolve_calls = 0

        def get_price(self, symbol: str) -> UMPrice:
            self.price_calls += 1
            return UMPrice(
                symbol=symbol,
                ltp=100.0,
                bid=99.5,
                ask=100.5,
                ts=0.0,
                source="broker",
            )

        def resolve(self, symbol: str) -> UMInstrument:
            self.resolve_calls += 1
            return UMInstrument(
                symbol=symbol,
                exchange="NFO",
                token=789,
                lot_size=50,
                kind="OPT",
                expiry="2024-10-25",
            )

    manager = _StubManager()
    monkeypatch.setattr(bot, "_get_unified_manager", lambda: manager)
    monkeypatch.setattr(bot, "_guard", AsyncMock(return_value=chat))
    reply = AsyncMock()
    monkeypatch.setattr(bot, "_reply", reply)

    await bot.cmd_umlearn(update, ctx)

    assert manager.price_calls == 1
    assert manager.resolve_calls == 1
    assert resolver.resolve("NIFTY25OCT25900CE") == 789
    payload = _extract_payload(reply)
    assert payload["token"] == 789
    assert payload["exchange"] == "NFO"
    assert payload["lot"] == 50
