from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram.error import Forbidden, TimedOut

from nifty_scalper_bot.notifications.telegram_webhook_enhanced import (
    NotificationDispatchResult,
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

    result = await notifier._send_single(
        chat_id=1, text='x', alert_id='a1', event_type='alert'
    )

    assert notifier._telegram_degraded is True
    assert notifier._telegram_degraded_until == pytest.approx(400.0)
    assert bot.send_message.await_count == 5
    assert result.status == 'failed_timeout'

    bot.send_message.reset_mock()
    result_2 = await notifier._send_single(
        chat_id=1, text='x', alert_id='a2', event_type='alert'
    )
    assert bot.send_message.await_count == 0
    assert result_2.status == 'dropped_degraded'


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

    result = await notifier._send_single(
        chat_id=1, text='ok', alert_id='a3', event_type='alert'
    )

    assert bot.send_message.await_count == 1
    assert notifier._telegram_degraded is False
    assert notifier._telegram_degraded_until == pytest.approx(0.0)
    assert result.status == 'sent'


@pytest.mark.asyncio
async def test_send_single_success_returns_sent(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level('INFO')
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=None)
    settings = SimpleNamespace(rate_per_second=20.0, burst_capacity=20.0, whitelist_chat_ids=[1], whitelist_user_ids=None)
    notifier = TelegramEnhancedNotifier(bot=bot, settings=settings)

    result = await notifier._send_single(chat_id=1, text='ok', alert_id='a4', event_type='alert')
    assert result.status == 'sent'
    assert 'NOTIFICATION_DISPATCH_RESULT' in caplog.text


@pytest.mark.asyncio
async def test_send_event_no_whitelisted_chats_terminal_result() -> None:
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=None)
    settings = SimpleNamespace(rate_per_second=20.0, burst_capacity=20.0, whitelist_chat_ids=None, whitelist_user_ids=None)
    notifier = TelegramEnhancedNotifier(bot=bot, settings=settings)
    results = await notifier.send_event('event_x', {'a': 1})
    assert len(results) == 1
    assert results[0].status == 'no_whitelisted_chats'


@pytest.mark.asyncio
async def test_send_alert_running_loop_logs_queued_not_sent(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level('INFO')
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=None)
    settings = SimpleNamespace(rate_per_second=20.0, burst_capacity=20.0, whitelist_chat_ids=[1], whitelist_user_ids=None)
    notifier = TelegramEnhancedNotifier(bot=bot, settings=settings)
    monkeypatch.setattr(
        'nifty_scalper_bot.notifications.telegram_webhook_enhanced.safe_task',
        lambda coro: asyncio.create_task(coro),
    )
    notifier.send_alert('hello')
    await asyncio.sleep(0)
    assert 'NOTIFICATION_DISPATCH_QUEUED' in caplog.text
    assert 'NOTIFICATION_DISPATCH_RESULT' in caplog.text


@pytest.mark.asyncio
async def test_forbidden_returns_failed_forbidden() -> None:
    bot = MagicMock()
    bot.send_message = AsyncMock(side_effect=Forbidden('forbidden'))
    settings = SimpleNamespace(rate_per_second=20.0, burst_capacity=20.0, whitelist_chat_ids=[1], whitelist_user_ids=None)
    notifier = TelegramEnhancedNotifier(bot=bot, settings=settings)
    result = await notifier._send_single(chat_id=1, text='x', alert_id='a5', event_type='alert')
    assert result.status == 'failed_forbidden'


@pytest.mark.asyncio
async def test_send_single_retry_window_exceeded_returns_terminal_result(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level('INFO')
    current = 100.0
    monkeypatch.setattr(
        'nifty_scalper_bot.notifications.telegram_webhook_enhanced._current_loop_time',
        lambda: current,
    )
    bot = MagicMock()
    bot.send_message = AsyncMock(side_effect=TimedOut('timeout'))
    settings = SimpleNamespace(rate_per_second=20.0, burst_capacity=20.0, whitelist_chat_ids=[1], whitelist_user_ids=None)
    notifier = TelegramEnhancedNotifier(bot=bot, settings=settings)
    notifier._telegram_backoff_active = True
    notifier._telegram_backoff_until = 99.0

    async def _advance_sleep(_seconds: float) -> None:
        nonlocal current
        current += 301.0

    monkeypatch.setattr(
        'nifty_scalper_bot.notifications.telegram_webhook_enhanced.asyncio.sleep',
        _advance_sleep,
    )
    result = await notifier._send_single(chat_id=1, text='x', alert_id='a6', event_type='alert')
    assert result is not None
    assert result.status == 'dropped_degraded'
    assert result.error_type == 'RetryWindowExceeded'
    assert notifier._telegram_degraded is True
    assert 'NOTIFICATION_DISPATCH_RESULT' in caplog.text


@pytest.mark.asyncio
async def test_send_event_never_returns_none_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = 100.0
    monkeypatch.setattr(
        'nifty_scalper_bot.notifications.telegram_webhook_enhanced._current_loop_time',
        lambda: current,
    )
    bot = MagicMock()
    bot.send_message = AsyncMock(side_effect=TimedOut('timeout'))
    settings = SimpleNamespace(rate_per_second=20.0, burst_capacity=20.0, whitelist_chat_ids=[1], whitelist_user_ids=None)
    notifier = TelegramEnhancedNotifier(bot=bot, settings=settings)
    notifier._telegram_backoff_active = True
    notifier._telegram_backoff_until = 99.0

    async def _advance_sleep(_seconds: float) -> None:
        nonlocal current
        current += 301.0

    monkeypatch.setattr(
        'nifty_scalper_bot.notifications.telegram_webhook_enhanced.asyncio.sleep',
        _advance_sleep,
    )
    results = await notifier.send_event('event_y', {'k': 'v'})
    assert results
    assert all(isinstance(r, NotificationDispatchResult) for r in results)
    assert all(r is not None for r in results)


@pytest.mark.asyncio
async def test_telegram_logs_do_not_emit_raw_chat_id(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level('INFO')
    chat_id = 123456789
    bot = MagicMock()
    bot.send_message = AsyncMock(side_effect=Forbidden('forbidden'))
    settings = SimpleNamespace(rate_per_second=20.0, burst_capacity=20.0, whitelist_chat_ids=[chat_id], whitelist_user_ids=None)
    notifier = TelegramEnhancedNotifier(bot=bot, settings=settings)
    await notifier._send_single(chat_id=chat_id, text='x', alert_id='a7', event_type='alert')
    assert str(chat_id) not in caplog.text
    assert any(hasattr(record, 'chat_id_hash') for record in caplog.records)


def test_send_alert_no_whitelist_logs_terminal_result(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level('INFO')
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=None)
    settings = SimpleNamespace(rate_per_second=20.0, burst_capacity=20.0, whitelist_chat_ids=None, whitelist_user_ids=None)
    notifier = TelegramEnhancedNotifier(bot=bot, settings=settings)
    notifier.send_alert('hello')
    assert 'NOTIFICATION_DISPATCH_RESULT' in caplog.text
    assert 'no_whitelisted_chats' in caplog.text
