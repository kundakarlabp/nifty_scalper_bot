from __future__ import annotations

from types import SimpleNamespace

import pytest

from nifty_scalper_bot.notifications.telegram_controller import TelegramBot, TelegramDeps


class _DummyMdm:
    def __init__(self) -> None:
        self.desired: set[int] = set()
        self.sub_calls = 0
        self.unsub_calls = 0

    def request_token_subscriptions(self, tokens: list[int]) -> int:
        self.sub_calls += 1
        before = len(self.desired)
        self.desired.update(int(token) for token in tokens)
        return len(self.desired) - before

    def request_token_unsubscriptions(self, tokens: list[int]) -> int:
        self.unsub_calls += 1
        before = len(self.desired)
        self.desired.difference_update(int(token) for token in tokens)
        return before - len(self.desired)

    def desired_token_count(self) -> int:
        return len(self.desired)


class _DummySupervisor:
    def __init__(self) -> None:
        self.ensure_started_calls = 0

    def resolve_symbols(self, symbols: list[str]) -> tuple[list[int], list[str], dict[str, int]]:
        mapping = {'NIFTY': 256265}
        resolved: list[int] = []
        unresolved: list[str] = []
        pairs: dict[str, int] = {}
        for symbol in symbols:
            token = mapping.get(symbol)
            if token is None:
                unresolved.append(symbol)
            else:
                resolved.append(token)
                pairs[symbol] = token
        return resolved, unresolved, pairs

    def ensure_started(self) -> bool:
        self.ensure_started_calls += 1
        return True


class _ForbiddenWs:
    def __getattr__(self, name: str):
        if name in {'subscribe', 'unsubscribe', 'subscribe_tokens', 'unsubscribe_tokens'}:
            raise AssertionError(f'unexpected websocket call: {name}')
        raise AttributeError(name)


class _DummyChat:
    def __init__(self, chat_id: int) -> None:
        self.id = chat_id


@pytest.mark.asyncio
async def test_subscribe_routes_through_mdm(monkeypatch: pytest.MonkeyPatch) -> None:
    deps = TelegramDeps(token='x', chat_id=1, app_version='test')
    deps.market_data_manager = _DummyMdm()
    deps.stream_supervisor = _DummySupervisor()
    deps.websocket_manager = _ForbiddenWs()
    bot = TelegramBot(deps)

    sent: list[str] = []
    async def _guard(_update):  # noqa: ANN001
        return _DummyChat(1)

    monkeypatch.setattr(bot, '_guard', _guard)

    async def _reply(_chat, _ctx, text: str, parse_mode=None):  # noqa: ANN001
        del parse_mode
        sent.append(text)
        return None

    monkeypatch.setattr(bot, '_reply', _reply)

    ctx = SimpleNamespace(args=['NIFTY', '123'])
    await bot.cmd_subscribe(SimpleNamespace(), ctx)

    mdm = deps.market_data_manager
    assert mdm.sub_calls == 1
    assert mdm.desired == {123, 256265}
    assert deps.stream_supervisor.ensure_started_calls == 1
    assert any('Tracked via MarketDataManager' in line for line in sent)


@pytest.mark.asyncio
async def test_unsubscribe_routes_through_mdm(monkeypatch: pytest.MonkeyPatch) -> None:
    deps = TelegramDeps(token='x', chat_id=1, app_version='test')
    mdm = _DummyMdm()
    mdm.desired = {123, 256265}
    deps.market_data_manager = mdm
    deps.websocket_manager = _ForbiddenWs()
    bot = TelegramBot(deps)

    sent: list[str] = []
    async def _guard(_update):  # noqa: ANN001
        return _DummyChat(1)

    monkeypatch.setattr(bot, '_guard', _guard)

    async def _reply(_chat, _ctx, text: str, parse_mode=None):  # noqa: ANN001
        del parse_mode
        sent.append(text)
        return None

    monkeypatch.setattr(bot, '_reply', _reply)

    ctx = SimpleNamespace(args=['123'])
    await bot.cmd_unsubscribe(SimpleNamespace(), ctx)

    assert mdm.unsub_calls == 1
    assert mdm.desired == {256265}
    assert any('Removed from desired subscription set' in line for line in sent)
