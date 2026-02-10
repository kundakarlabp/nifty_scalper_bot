from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram.error import TimedOut

from nifty_scalper_bot.notifications.telegram_webhook_enhanced import (
    TelegramEnhancedNotifier,
)


@pytest.mark.asyncio
async def test_send_single_latches_degraded_after_retry_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify notifier enters degraded mode after repeated transport failures.

    Args:
        monkeypatch: Fixture used to freeze loop time and async sleeps.

    Returns:
        None.

    Raises:
        Exception: If notifier execution fails unexpectedly.
    """

    fake_time = 100.0
    monkeypatch.setattr(
        'nifty_scalper_bot.notifications.telegram_webhook_enhanced._current_loop_time',
        lambda: fake_time,
    )

    async def _noop_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(
        'nifty_scalper_bot.notifications.telegram_webhook_enhanced.asyncio.sleep',
        _noop_sleep,
    )

    bot = MagicMock()
    bot.send_message = AsyncMock(side_effect=TimedOut('timeout'))
    settings = SimpleNamespace(
        rate_per_second=20.0,
        burst_capacity=20.0,
        whitelist_chat_ids=None,
        whitelist_user_ids=None,
    )
    notifier = TelegramEnhancedNotifier(bot=bot, settings=settings)

    await notifier._send_single(chat_id=1, text='x')

    assert notifier._telegram_degraded is True
    assert notifier._telegram_degraded_until == pytest.approx(400.0)
    assert bot.send_message.await_count == 5

    bot.send_message.reset_mock()
    await notifier._send_single(chat_id=1, text='x')
    assert bot.send_message.await_count == 0


@pytest.mark.asyncio
async def test_send_single_recovers_after_degraded_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify degraded latch is cleared once cooldown expires.

    Args:
        monkeypatch: Fixture used to set deterministic loop time.

    Returns:
        None.

    Raises:
        Exception: If notifier execution fails unexpectedly.
    """

    fake_time = 500.0
    monkeypatch.setattr(
        'nifty_scalper_bot.notifications.telegram_webhook_enhanced._current_loop_time',
        lambda: fake_time,
    )

    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=None)
    settings = SimpleNamespace(
        rate_per_second=20.0,
        burst_capacity=20.0,
        whitelist_chat_ids=None,
        whitelist_user_ids=None,
    )
    notifier = TelegramEnhancedNotifier(bot=bot, settings=settings)
    notifier._telegram_degraded = True
    notifier._telegram_degraded_logged = True
    notifier._telegram_degraded_until = 450.0

    await notifier._send_single(chat_id=1, text='ok')

    assert bot.send_message.await_count == 1
    assert notifier._telegram_degraded is False
    assert notifier._telegram_degraded_until == pytest.approx(0.0)
