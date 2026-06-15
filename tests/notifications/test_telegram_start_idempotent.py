"""TelegramBot.start() must be idempotent so it can be started early (before the
data-hydration pipeline) and the later in-pipeline call becomes a safe no-op.
This underpins the fix that makes operator commands live regardless of market
data readiness. Async so it executes under the repo conftest hook.
"""

from __future__ import annotations

from nifty_scalper_bot.notifications.telegram_controller import TelegramBot, TelegramDeps


async def test_start_is_idempotent_when_already_started() -> None:
    deps = TelegramDeps(token="dummy", chat_id=12345, app_version="test")
    bot = TelegramBot(deps)
    # Simulate a successful first start without touching the network.
    bot._started = True
    bot._running = True

    register_calls = {"n": 0}
    original = bot._register_handlers

    def _counting_register() -> None:
        register_calls["n"] += 1
        return original()

    bot._register_handlers = _counting_register  # type: ignore[method-assign]

    # A second start() (the later in-pipeline call) must no-op: no handler
    # re-registration, no polling attempt, no exception.
    await bot.start()
    assert register_calls["n"] == 0
    assert bot._started is True


async def test_handlers_registered_flag_guards_double_registration() -> None:
    deps = TelegramDeps(token="dummy", chat_id=12345, app_version="test")
    bot = TelegramBot(deps)
    # _register_handlers must be safe to call twice (guard on _handlers_registered).
    assert bot._handlers_registered is False
    bot._register_handlers()
    assert bot._handlers_registered is True
    # second call is a no-op (would raise if it tried to re-add on a real app twice)
    bot._register_handlers()
    assert bot._handlers_registered is True
