"""Tests for Telegram regime command wiring."""

from __future__ import annotations

import pytest

from nifty_scalper_bot.notifications.telegram_controller import (
    TelegramBot,
    TelegramDeps,
)


class ManagerStub:
    def __init__(self) -> None:
        self.bypass = False
        self.calls: list[object] = []

    def get_regime_filter_bypass(self) -> bool:
        self.calls.append("get_bypass")
        return self.bypass

    def get_regime_filter_stats(self) -> dict[str, int]:
        self.calls.append("get_stats")
        return {"pass": 1}

    def get_filter_reasons(self) -> tuple[str, ...]:
        self.calls.append("get_reasons")
        return tuple()

    def get_decision_history(self, limit: int = 1) -> list[object]:
        self.calls.append(("get_decision_history", limit))
        return []

    def set_regime_filter_bypass(self, state: bool) -> bool:
        self.calls.append(("set_bypass", state))
        self.bypass = bool(state)
        return self.bypass


class StrategyStub:
    def get_regime_filter_bypass(self) -> bool:  # noqa: D401 - test stub
        raise AssertionError("strategy fallback should not be invoked")

    def set_regime_filter_bypass(self, _state: bool) -> bool:  # noqa: D401 - test stub
        raise AssertionError("strategy fallback should not be invoked")


@pytest.mark.asyncio
async def test_regime_bypass_uses_central_manager(
    monkeypatch: "pytest.MonkeyPatch",
) -> None:
    manager = ManagerStub()
    deps = TelegramDeps(
        token="x",
        chat_id=1,
        app_version="test",
        regime_manager=manager,
        strategy_manager=StrategyStub(),
    )
    bot = TelegramBot(deps)

    async def fake_guard(_update: object) -> object:
        return object()

    async def fake_reply(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(bot, "_guard", fake_guard)
    monkeypatch.setattr(bot, "_reply", fake_reply)

    class DummyCtx:
        args = ["toggle"]

    await bot.cmd_regime_bypass(object(), DummyCtx())

    assert ("set_bypass", True) in manager.calls
    assert "get_stats" in manager.calls
    assert any(
        entry[0] == "get_decision_history"
        for entry in manager.calls
        if isinstance(entry, tuple)
    )
