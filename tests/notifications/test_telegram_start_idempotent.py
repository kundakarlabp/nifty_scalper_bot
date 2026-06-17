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


async def test_start_clears_webhook_and_drops_pending_before_polling() -> None:
    # Regression: operator commands were unresponsive because start_polling ran
    # without first clearing a possible webhook / stale backlog, causing a 409
    # Conflict on getUpdates. start() must delete the webhook (drop_pending=True)
    # and start polling with drop_pending_updates=True.
    from unittest.mock import AsyncMock, MagicMock

    deps = TelegramDeps(token="dummy", chat_id=12345, app_version="test")
    bot = TelegramBot(deps)

    app = MagicMock()
    app.initialize = AsyncMock()
    app.start = AsyncMock()
    app.bot.delete_webhook = AsyncMock()
    app.updater = MagicMock()
    app.updater.start_polling = AsyncMock()
    bot.application = app
    bot._register_handlers = lambda: None  # type: ignore[method-assign]
    bot._safe_send = AsyncMock()  # type: ignore[method-assign]
    bot._ensure_alert_worker = lambda: None  # type: ignore[method-assign]
    bot._bg_task_started = True  # skip heartbeat task creation

    await bot.start()

    app.bot.delete_webhook.assert_awaited_once_with(drop_pending_updates=True)
    app.updater.start_polling.assert_awaited_once_with(drop_pending_updates=True)


async def test_dispatch_alert_timeout_is_quiet_not_error(caplog) -> None:
    # A transient timeout while sending a best-effort alert must be logged quietly
    # (debug), not as a scary ❌ ERROR. Alerts are best-effort; a timeout just means
    # the message didn't go out.
    import logging
    from unittest.mock import AsyncMock, MagicMock

    deps = TelegramDeps(token="dummy", chat_id=12345, app_version="test")
    bot = TelegramBot(deps)
    app = MagicMock()
    app.bot = MagicMock()
    bot._app = app
    bot._telegram_hold_until = None
    bot._telegram_last_dispatch_at = None

    messenger = MagicMock()
    messenger.send_text = AsyncMock(side_effect=TimeoutError("Timed out"))
    bot._ensure_messenger = lambda *_a, **_k: messenger

    with caplog.at_level(logging.DEBUG):
        await bot._dispatch_alert("hello", "info")

    msgs = [r.getMessage() for r in caplog.records]
    assert not any("Failure in _dispatch_alert send" in m for m in msgs), "timeout must not log as ERROR"
    assert any("transient failure" in m for m in msgs), "timeout should log a quiet transient line"
