"""Webhook-first Telegram integration with rate-limited notifications
and robust command replies (/ping, /status, /start).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from logging import Logger
from time import monotonic
from typing import Any, Callable, Iterable, Mapping, Sequence, cast

from fastapi import APIRouter, Request, Response, status
from telegram import Bot, Message, Update
from telegram.error import (
    Forbidden,
    NetworkError,
    RetryAfter,
    TelegramError,
    TimedOut,
)
from telegram.ext import Application

from nifty_scalper_bot.config.settings import NotificationSettings
from nifty_scalper_bot.utils.async_helpers import safe_task
from nifty_scalper_bot.utils.logging import get_logger


def _ensure_set(values: Iterable[int] | None) -> set[int]:
    """Normalise iterable *values* into a set of integers."""

    if not values:
        return set()
    return {int(v) for v in values}


def _current_loop_time() -> float:
    """Return a loop-agnostic monotonic clock safe during wiring."""

    return monotonic()


def _truncate(text: str, limit: int = 3500) -> str:
    """Return ``text`` truncated to ``limit`` characters with ellipsis."""

    if len(text) <= limit:
        return text
    return f"{text[: limit - 1]}…"


def _hash_chat_id(chat_id: int) -> str:
    digest = hashlib.sha256(str(chat_id).encode("utf-8")).hexdigest()
    return digest[:12]


_TELEGRAM_MAX_CHARS = 3500
_TRUNCATION_SUFFIX = "\n…(truncated)"


@dataclass(slots=True)
class TelegramPollingMetrics:
    """Track fallback polling activity for the lightweight controller."""

    fallback_activations: int = 0
    polling_updates: int = 0
    polling_errors: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "fallback_activations": self.fallback_activations,
            "polling_updates": self.polling_updates,
            "polling_errors": self.polling_errors,
        }


def _clamp(text: str, *, limit: int = _TELEGRAM_MAX_CHARS) -> str:
    """Clamp ``text`` to ``limit`` characters with an ellipsis suffix."""

    if len(text) <= limit:
        return text
    if limit <= len(_TRUNCATION_SUFFIX):
        return _TRUNCATION_SUFFIX[:limit]
    return text[: limit - len(_TRUNCATION_SUFFIX)] + _TRUNCATION_SUFFIX


def iter_chunks(text: str, size: int = _TELEGRAM_MAX_CHARS) -> Iterable[str]:
    """Yield ``text`` in chunks no larger than ``size`` characters."""

    if size <= 0:
        raise ValueError("size must be positive")
    clamped = _clamp(text, limit=size)
    if not clamped:
        yield clamped
        return
    for start in range(0, len(clamped), size):
        yield clamped[start : start + size]


async def safe_reply_text(
    bot: Bot,
    chat_id: int,
    text: str,
    **kwargs: Any,
) -> Message | None:
    """Send ``text`` to ``chat_id`` in Telegram-safe chunks.

    Telegram rejects messages over 4096 characters; this helper splits long
    responses and returns the final :class:`telegram.Message` object so callers
    retain backwards compatibility with ``send_message``.
    """

    last_message: Message | None = None
    for chunk in iter_chunks(text):
        last_message = await bot.send_message(
            chat_id=chat_id,
            text=chunk,
            **kwargs,
        )
    return last_message


@dataclass(slots=True)
class NotificationDispatchResult:
    alert_id: str
    event_type: str
    provider: str
    status: str
    attempts: int
    latency_ms: float | None = None
    error_type: str | None = None
    error: str | None = None
    degraded: bool = False
    backoff_until: float | None = None


@dataclass(slots=True)
class TelegramEnhancedNotifier:
    """Asynchronous notifier with token-bucket rate limiting and retries."""

    bot: Bot
    settings: NotificationSettings
    _logger: Logger = field(init=False, repr=False)
    _rate_lock: asyncio.Lock = field(init=False, repr=False)
    _rate: float = field(init=False, repr=False)
    _capacity: float = field(init=False, repr=False)
    _tokens: float = field(init=False, repr=False)
    _last_refill: float = field(init=False, repr=False)
    _chat_whitelist: set[int] = field(init=False, repr=False)
    _user_whitelist: set[int] = field(init=False, repr=False)
    _telegram_degraded: bool = field(init=False, repr=False, default=False)
    _telegram_degraded_until: float = field(init=False, repr=False, default=0.0)
    _telegram_degraded_logged: bool = field(init=False, repr=False, default=False)
    _telegram_backoff_active: bool = field(init=False, repr=False, default=False)
    _telegram_backoff_until: float = field(init=False, repr=False, default=0.0)
    _consecutive_failures: int = field(init=False, repr=False, default=0)
    _last_success_ts: float = field(init=False, repr=False, default=0.0)
    _last_error_type: str | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        self._logger = get_logger(__name__)
        self._rate_lock = asyncio.Lock()
        self._rate = max(
            float(getattr(self.settings, "rate_per_second", 20.0) or 20.0), 0.1
        )
        burst = getattr(self.settings, "burst_capacity", None)
        self._capacity = float(burst if burst is not None else self._rate)
        if self._capacity <= 0:
            self._capacity = self._rate
        self._tokens = self._capacity
        self._last_refill = _current_loop_time()

        self._chat_whitelist = _ensure_set(
            getattr(self.settings, "whitelist_chat_ids", None)
        )
        self._user_whitelist = _ensure_set(
            getattr(self.settings, "whitelist_user_ids", None)
        )

    @classmethod
    def from_settings(
        cls, settings: NotificationSettings | None
    ) -> "TelegramEnhancedNotifier | None":
        """Build a notifier from :class:`NotificationSettings` when enabled."""

        if settings is None or not getattr(settings, "enabled", False):
            return None
        token = getattr(settings, "token", None)
        if not token:
            return None
        bot = Bot(token=token)
        return cls(bot=bot, settings=settings)

    async def send_event(
        self, event: str, payload: Mapping[str, object] | None = None
    ) -> list[NotificationDispatchResult]:
        """Send ``event`` with optional ``payload`` to authorised chats."""

        alert_id = f"evt-{int(_current_loop_time() * 1000)}-{random.randint(1000, 9999)}"
        if not self._chat_whitelist:
            result = NotificationDispatchResult(
                alert_id=alert_id,
                event_type=event,
                provider="telegram",
                status="no_whitelisted_chats",
                attempts=0,
            )
            self._logger.info(
                "NOTIFICATION_DISPATCH_RESULT",
                extra=asdict(result),
            )
            self._log_provider_health()
            return [result]

        message = self._format_message(event, payload)
        results: list[NotificationDispatchResult] = []

        for chat_id in self._chat_whitelist:
            await self._acquire_token()
            results.append(
                await self._send_single(
                    chat_id,
                    message,
                    alert_id=alert_id,
                    event_type=event,
                )
            )
        return results

    def send_alert(self, message: str) -> None:
        """Send alert synchronously with proper event loop handling.

        Args:
            message: Human-readable text to deliver to whitelisted chats.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered send_alert",
            extra={"event": "notifier.send_alert.enter"},
        )

        if not self._chat_whitelist:
            return

        alert_id = f"alert-{int(_current_loop_time() * 1000)}-{random.randint(1000, 9999)}"

        async def _dispatch() -> list[NotificationDispatchResult]:
            """Deliver alert to all whitelisted chats."""
            results: list[NotificationDispatchResult] = []
            for chat_id in self._chat_whitelist:
                await self._acquire_token()
                results.append(
                    await self._send_single(
                        chat_id,
                        _clamp(str(message)),
                        alert_id=alert_id,
                        event_type="alert",
                    )
                )
            return results

        try:
            # Check if we're already in an event loop
            try:
                loop = asyncio.get_running_loop()
                # We're in a running loop - schedule as task
                task = safe_task(_dispatch())
                self._logger.info("NOTIFICATION_DISPATCH_QUEUED", extra={"status": "queued", "alert_id": alert_id, "provider": "telegram", "event_type": "alert"})
                def _on_done(done_task: asyncio.Task[list[NotificationDispatchResult]]) -> None:
                    with suppress(asyncio.CancelledError):
                        exc = done_task.exception()
                        if exc is not None:
                            self._logger.error("NOTIFICATION_DISPATCH_FAILED", extra={"alert_id": alert_id, "provider": "telegram", "event_type": "alert", "error_type": type(exc).__name__, "error": str(exc)})
                task.add_done_callback(_on_done)
            except RuntimeError:
                # No running loop - create one
                asyncio.run(_dispatch())
        except Exception as exc:
            self._logger.error(
                f"Alert send failed: {exc}",
                extra={"event": "notifier.send_alert.error"},
                exc_info=exc,
            )

    async def _acquire_token(self) -> None:
        async with self._rate_lock:
            while True:
                self._refill_tokens()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self._rate if self._rate > 0 else 1.0
                wait = min(max(wait, 0.01), 5.0)
                await asyncio.sleep(wait)

    def _refill_tokens(self) -> None:
        now = _current_loop_time()
        elapsed = now - self._last_refill
        if elapsed <= 0:
            return
        refill = elapsed * self._rate
        self._tokens = min(self._capacity, self._tokens + refill)
        self._last_refill = now

    def _log_provider_health(self) -> None:
        now = _current_loop_time()
        last_success_age_s = (now - self._last_success_ts) if self._last_success_ts > 0 else None
        self._logger.info("TELEGRAM_PROVIDER_HEALTH", extra={"provider": "telegram", "degraded": self._telegram_degraded, "degraded_until": self._telegram_degraded_until, "backoff_active": self._telegram_backoff_active and now < self._telegram_backoff_until, "backoff_until": self._telegram_backoff_until, "consecutive_failures": self._consecutive_failures, "last_success_age_s": last_success_age_s, "last_error_type": self._last_error_type})

    async def _send_single(
        self, chat_id: int, text: str, *, alert_id: str, event_type: str
    ) -> NotificationDispatchResult:
        attempt = 1
        max_attempts = 5
        retry_window_s = 300.0
        retry_window_start = _current_loop_time()

        # Circuit-breaker latch prevents unbounded retry storms during outages.
        now = _current_loop_time()
        if self._telegram_backoff_active and now < self._telegram_backoff_until:
            result = NotificationDispatchResult(alert_id=alert_id, event_type=event_type, provider="telegram", status="dropped_degraded", attempts=0, degraded=True, backoff_until=self._telegram_backoff_until)
            self._logger.info("NOTIFICATION_DISPATCH_RESULT", extra=asdict(result))
            self._log_provider_health()
            return result
        if self._telegram_backoff_active and now >= self._telegram_backoff_until:
            self._telegram_backoff_active = False
            self._telegram_backoff_until = 0.0
        if self._telegram_degraded and now < self._telegram_degraded_until:
            result = NotificationDispatchResult(alert_id=alert_id, event_type=event_type, provider="telegram", status="dropped_degraded", attempts=0, degraded=True, backoff_until=self._telegram_degraded_until)
            self._logger.info("NOTIFICATION_DISPATCH_RESULT", extra=asdict(result))
            self._log_provider_health()
            return result
        if self._telegram_degraded and now >= self._telegram_degraded_until:
            self._telegram_degraded = False
            self._telegram_degraded_logged = False

        while True:
            elapsed = _current_loop_time() - retry_window_start
            if elapsed >= retry_window_s:
                # Latch degraded state so callers stop retrying until cooldown expires.
                self._telegram_degraded = True
                self._telegram_degraded_until = _current_loop_time() + retry_window_s
                if not self._telegram_degraded_logged:
                    self._telegram_degraded_logged = True
                    self._logger.error(
                        "telegram_notifier_degraded",
                        extra={
                            "event": "telegram_notifier_degraded",
                            "chat_id": chat_id,
                            "cooldown_s": retry_window_s,
                        },
                    )
                return
            try:
                self._logger.info("NOTIFICATION_DISPATCH_ATTEMPT", extra={"alert_id": alert_id, "event_type": event_type, "provider": "telegram", "chat_id_hash": _hash_chat_id(chat_id), "attempt": attempt})
                started = _current_loop_time()
                await self.bot.send_message(
                    chat_id=chat_id,
                    text=text,
                    disable_web_page_preview=True,
                )
                self._logger.debug(
                    "telegram_send_success",
                    extra={"event": "send", "chat_id": chat_id, "attempt": attempt},
                )
                self._telegram_degraded = False
                self._telegram_degraded_logged = False
                self._telegram_degraded_until = 0.0
                self._telegram_backoff_active = False
                self._telegram_backoff_until = 0.0
                self._last_success_ts = _current_loop_time()
                self._consecutive_failures = 0
                self._last_error_type = None
                result = NotificationDispatchResult(alert_id=alert_id, event_type=event_type, provider="telegram", status="sent", attempts=attempt, latency_ms=(_current_loop_time() - started) * 1000.0, degraded=False)
                self._logger.info("NOTIFICATION_DISPATCH_RESULT", extra=asdict(result))
                self._log_provider_health()
                return result
            except RetryAfter as exc:  # pragma: no cover - depends on API
                delay = float(getattr(exc, "retry_after", 1.0) or 1.0)
                delay = max(delay, 1.0)
                self._logger.warning(
                    "telegram_retry_after",
                    extra={
                        "event": "retry_after",
                        "chat_id": chat_id,
                        "attempt": attempt,
                        "delay": delay,
                    },
                )
                self._telegram_backoff_active = True
                self._telegram_backoff_until = _current_loop_time() + delay
                await asyncio.sleep(delay)
                attempt += 1
                if attempt > max_attempts:
                    self._consecutive_failures += 1
                    self._last_error_type = "RetryAfter"
                    result = NotificationDispatchResult(alert_id=alert_id, event_type=event_type, provider="telegram", status="queued_retry", attempts=max_attempts, error_type="RetryAfter", error=str(exc), degraded=self._telegram_degraded, backoff_until=self._telegram_backoff_until)
                    self._logger.info("NOTIFICATION_DISPATCH_RESULT", extra=asdict(result))
                    self._log_provider_health()
                    return result
            except (TimedOut, NetworkError) as exc:  # pragma: no cover - network
                if attempt >= max_attempts:
                    self._logger.error(
                        "telegram_network_error",
                        extra={
                            "event": "network_error",
                            "chat_id": chat_id,
                            "attempt": attempt,
                            "err": str(exc),
                        },
                    )
                    # Repeated hard failures indicate transport degradation; latch
                    # to avoid retry spam from every caller until cooldown expires.
                    self._telegram_degraded = True
                    self._telegram_degraded_until = _current_loop_time() + retry_window_s
                    if not self._telegram_degraded_logged:
                        self._telegram_degraded_logged = True
                        self._logger.error(
                            "telegram_notifier_degraded",
                            extra={
                                "event": "telegram_notifier_degraded",
                                "chat_id": chat_id,
                                "cooldown_s": retry_window_s,
                            },
                        )
                    self._consecutive_failures += 1
                    err_type = type(exc).__name__
                    self._last_error_type = err_type
                    result = NotificationDispatchResult(alert_id=alert_id, event_type=event_type, provider="telegram", status="failed_timeout" if isinstance(exc, TimedOut) else "failed_network", attempts=attempt, error_type=err_type, error=str(exc), degraded=True, backoff_until=self._telegram_degraded_until)
                    self._logger.info("NOTIFICATION_DISPATCH_RESULT", extra=asdict(result))
                    self._log_provider_health()
                    return result
                delay = min(2.0**attempt, 30.0)
                self._logger.warning(
                    "telegram_network_retry",
                    extra={
                        "event": "network_retry",
                        "chat_id": chat_id,
                        "attempt": attempt,
                        "delay": delay,
                        "err": str(exc),
                    },
                )
                self._telegram_backoff_active = True
                self._telegram_backoff_until = _current_loop_time() + delay
                await asyncio.sleep(delay)
                attempt += 1
            except Forbidden as exc:
                self._logger.warning(
                    "telegram_forbidden",
                    extra={
                        "event": "forbidden",
                        "chat_id": chat_id,
                        "attempt": attempt,
                        "err": str(exc),
                    },
                )
                self._consecutive_failures += 1
                self._last_error_type = "Forbidden"
                result = NotificationDispatchResult(alert_id=alert_id, event_type=event_type, provider="telegram", status="failed_forbidden", attempts=attempt, error_type="Forbidden", error=str(exc), degraded=self._telegram_degraded, backoff_until=self._telegram_backoff_until)
                self._logger.info("NOTIFICATION_DISPATCH_RESULT", extra=asdict(result))
                self._log_provider_health()
                return result
            except TelegramError as exc:  # pragma: no cover - network
                if attempt >= max_attempts:
                    self._logger.error(
                        "telegram_send_failed",
                        extra={
                            "event": "send_failed",
                            "chat_id": chat_id,
                            "attempt": attempt,
                            "err": str(exc),
                        },
                    )
                    # Generic Telegram transport failures are latched as degraded so
                    # recovery waits for cooldown instead of immediate retry storms.
                    self._telegram_degraded = True
                    self._telegram_degraded_until = _current_loop_time() + retry_window_s
                    if not self._telegram_degraded_logged:
                        self._telegram_degraded_logged = True
                        self._logger.error(
                            "telegram_notifier_degraded",
                            extra={
                                "event": "telegram_notifier_degraded",
                                "chat_id": chat_id,
                                "cooldown_s": retry_window_s,
                            },
                        )
                    self._consecutive_failures += 1
                    self._last_error_type = "TelegramError"
                    result = NotificationDispatchResult(alert_id=alert_id, event_type=event_type, provider="telegram", status="failed_telegram", attempts=attempt, error_type="TelegramError", error=str(exc), degraded=True, backoff_until=self._telegram_degraded_until)
                    self._logger.info("NOTIFICATION_DISPATCH_RESULT", extra=asdict(result))
                    self._log_provider_health()
                    return result
                delay = min(2.0**attempt, 30.0)
                self._logger.warning(
                    "telegram_send_retry",
                    extra={
                        "event": "send_retry",
                        "chat_id": chat_id,
                        "attempt": attempt,
                        "delay": delay,
                        "err": str(exc),
                    },
                )
                self._telegram_backoff_active = True
                self._telegram_backoff_until = _current_loop_time() + delay
                await asyncio.sleep(delay)
                attempt += 1

    def _format_message(self, event: str, payload: Mapping[str, object] | None) -> str:
        fields: list[str] = [f"event: {event}"]
        if payload:
            for key, value in payload.items():
                try:
                    if isinstance(value, (dict, list, tuple, set)):
                        serialised = json.dumps(value, default=str)
                    else:
                        serialised = str(value)
                except (TypeError, ValueError):
                    serialised = str(value)
                fields.append(f"{key}: {_truncate(serialised, 500)}")
        message = "\n".join(fields)
        return _truncate(message)


class TelegramWebhookController:
    """FastAPI webhook controller that dispatches updates asynchronously.

    Includes a minimal command router and safe reply helper so that basic ops
    commands work out of the box without pulling in the legacy console.
    """

    def __init__(
        self,
        bot: Bot,
        settings: NotificationSettings,
        *,
        status_provider: Callable[[], Mapping[str, object]] | None = None,
        application: Application | None = None,
    ) -> None:
        self.bot = bot
        self.settings = settings
        self.logger = get_logger(__name__)
        # Optional dependency injection from core to enrich /status
        self._status_provider = status_provider
        # The full PTB Application (if provided) with all command handlers
        self._application: Application | None = None
        self._application_ready = asyncio.Event()
        self._application_active = False
        if application is not None:
            self.attach_application(application)
        self._fallback_lock = asyncio.Lock()
        self._fallback_active = False
        self._polling_task: asyncio.Task[None] | None = None
        self._polling_metrics = TelegramPollingMetrics()
        self.router = APIRouter()
        self.router.add_api_route(
            "/telegram/webhook",
            self._handle_webhook,
            methods=["POST"],
            status_code=status.HTTP_200_OK,
        )
        self._chat_whitelist = _ensure_set(
            getattr(settings, "whitelist_chat_ids", None)
        )
        self._user_whitelist = _ensure_set(
            getattr(settings, "whitelist_user_ids", None)
        )

    async def _handle_webhook(self, request: Request) -> Response:
        try:
            payload = await request.json()
        except Exception:  # noqa: BLE001
            self.logger.warning(
                "telegram_webhook_bad_payload",
                extra={"event": "webhook_bad_payload"},
            )
            return Response(status_code=status.HTTP_200_OK)

        try:
            update = Update.de_json(data=payload, bot=self.bot)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(
                "telegram_update_decode_failed",
                extra={"event": "decode_failed", "err": str(exc)},
            )
            return Response(status_code=status.HTTP_200_OK)

        safe_task(self.dispatch_update(update))
        return Response(status_code=status.HTTP_200_OK)

    def attach_application(self, application: Application) -> None:
        """Attach a PTB :class:`Application` for command dispatching."""

        self._application = application
        self._application_active = False
        self._application_ready.clear()

    def is_polling_active(self) -> bool:
        """Return ``True`` if the polling fallback task is running."""

        return self._fallback_active

    def polling_metrics(self) -> Mapping[str, int]:
        """Expose telemetry for webhook-to-polling fallback usage."""

        return self._polling_metrics.as_dict()

    async def activate_polling_fallback(
        self,
        reason: str,
        *,
        interval: float | None = None,
    ) -> None:
        """Activate the controller-managed polling fallback loop."""

        await self._activate_polling_fallback(reason, interval=interval)

    async def _activate_polling_fallback(
        self,
        reason: str,
        *,
        interval: float | None = None,
    ) -> None:
        """Internal helper to start polling fallback and track telemetry."""

        async with self._fallback_lock:
            if self._fallback_active:
                return
            default_interval = cast(
                float, getattr(self.settings, "polling_interval_seconds", 10.0)
            )
            base_interval = interval if interval is not None else default_interval
            poll_interval = max(min(float(base_interval), 60.0), 1.0)
            self._fallback_active = True
            self._polling_metrics.fallback_activations += 1
            self.logger.warning(
                "telegram_polling_fallback_start",
                extra={
                    "event": "polling_start",
                    "reason": reason,
                    "interval": poll_interval,
                },
            )
            with suppress(TelegramError):
                await self.bot.delete_webhook(drop_pending_updates=False)
            self._polling_task = safe_task(
                self._polling_loop(poll_interval),
            )

    async def _polling_loop(self, interval: float) -> None:
        """Continuously poll Telegram updates with retry handling."""

        offset = 0
        try:
            while True:
                try:
                    updates = await self.bot.get_updates(offset=offset, timeout=30)
                except TelegramError as exc:  # pragma: no cover - network
                    self._polling_metrics.polling_errors += 1
                    self.logger.error(
                        "telegram_polling_error",
                        extra={"event": "polling_error", "err": str(exc)},
                    )
                    await asyncio.sleep(min(interval * 2.0, 60.0))
                    continue

                for update in updates:
                    offset = update.update_id + 1
                    self._polling_metrics.polling_updates += 1
                    try:
                        await self.dispatch_update(update)
                    except Exception as exc:  # noqa: BLE001
                        self._polling_metrics.polling_errors += 1
                        self.logger.exception(
                            "telegram_polling_dispatch_failed",
                            extra={"event": "polling_dispatch_failed", "err": str(exc)},
                        )

                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.exception(
                "telegram_polling_loop_failed",
                extra={"event": "polling_loop_failed", "err": str(exc)},
            )
        finally:
            self._polling_task = None
            self._fallback_active = False

    def notify_application_ready(self, *, ready: bool = True) -> None:
        """Toggle readiness of the attached PTB application."""

        if not ready:
            self._application_active = False
            self._application_ready.set()
            return
        self._application_active = True
        self._application_ready.set()

    async def dispatch_update(self, update: Update) -> None:
        """Dispatch ``update`` through PTB or the built-in console."""

        application = self._application
        if application is not None:
            try:
                if not self._is_authorised(update):
                    return
                if not self._application_ready.is_set():
                    await self._application_ready.wait()
                if not self._application_active:
                    return
                self._queue_application_update(application, update)
            except Exception as exc:  # noqa: BLE001
                self.logger.exception(
                    "telegram_ptb_process_failed", extra={"err": str(exc)}
                )
            return

        await self.process_update(update)

    def _queue_application_update(
        self, application: Application, update: Update
    ) -> None:
        async def _runner() -> None:
            try:
                await application.process_update(update)
            except Exception as exc:  # noqa: BLE001
                self.logger.exception(
                    "telegram_ptb_process_failed", extra={"err": str(exc)}
                )

        safe_task(_runner())

    async def process_update(self, update: Update) -> None:
        """Process ``update`` applying whitelist checks and dispatching."""

        if self._application is not None:
            return

        try:
            if not self._is_authorised(update):
                return

            message = update.effective_message
            if message and message.text:
                text = (message.text or "").strip()
                chat_id = message.chat_id
                args: Sequence[str] = []
                if text.startswith("/"):
                    parts = text.split()
                    # Support /cmd and /cmd@BotUserName
                    head = parts[0][1:]
                    command = head.split("@", 1)[0].lower()
                    args = parts[1:]
                    await self.on_command(command, chat_id, list(args), update)
                else:
                    await self.on_message(text, chat_id, update)
            elif update.callback_query:
                await self.on_callback(update)
            else:
                await self.on_other(update)
        except Exception as exc:  # noqa: BLE001
            self.logger.exception(
                "telegram_update_failed",
                extra={"event": "update_failed", "err": str(exc)},
            )

    def _is_authorised(self, update: Update) -> bool:
        message = update.effective_message
        chat_id = message.chat_id if message else None
        user_id = message.from_user.id if message and message.from_user else None

        # When both whitelists are configured we require membership in both sets.
        if chat_id is not None and self._chat_whitelist:
            if chat_id not in self._chat_whitelist:
                self.logger.info(
                    "telegram_update_rejected",
                    extra={
                        "event": "update_rejected",
                        "chat_id": chat_id,
                        "user_id": user_id,
                        "reason": "chat_not_whitelisted",
                    },
                )
                return False

        if user_id is not None and self._user_whitelist:
            if user_id not in self._user_whitelist:
                self.logger.info(
                    "telegram_update_rejected",
                    extra={
                        "event": "update_rejected",
                        "chat_id": chat_id,
                        "user_id": user_id,
                        "reason": "user_not_whitelisted",
                    },
                )
                return False

        if self._chat_whitelist and chat_id is None:
            self.logger.info(
                "telegram_update_rejected",
                extra={"event": "update_rejected", "reason": "missing_chat"},
            )
            return False

        return True

    async def on_command(
        self, command: str, chat_id: int, args: Sequence[str], update: Update
    ) -> None:
        """Handle ``/command`` invocations with safe defaults."""

        if self._application is not None:
            return

        self.logger.info(
            "telegram_command",
            extra={
                "event": "command",
                "chat_id": chat_id,
                "command": command,
                # ``args`` is a reserved LogRecord attribute; avoid KeyError
                "cmd_args": list(args),
            },
        )

        # This method is ONLY used when no Application is injected (fallback mode)

        # --- Built-in commands ---
        if command in ("ping", "p"):
            await self._reply(chat_id, await self._ping_text())
            return

        if command in ("status", "st"):
            await self._reply(chat_id, await self._status_text())
            return

        if command in ("start", "help"):
            await self._reply(chat_id, self._help_text())
            return

        # Unknown command – acknowledge to confirm webhook is alive
        await self._reply(chat_id, f"Unknown command: /{command}")

    async def on_message(self, text: str, chat_id: int, update: Update) -> None:
        """Handle text messages that are not commands."""

        self.logger.info(
            "telegram_message",
            extra={
                "event": "message",
                "chat_id": chat_id,
                "text": _truncate(text, 120),
            },
        )
        # Keep console clean by default (no echo). Uncomment to guide users:
        # await self._reply(chat_id, "Send /help for commands.")

    async def on_callback(self, update: Update) -> None:
        """Handle callback queries (default: log and ignore)."""

        callback = update.callback_query
        data = callback.data if callback else None
        chat_id = (
            callback.message.chat.id
            if callback and callback.message and callback.message.chat
            else None
        )
        self.logger.info(
            "telegram_callback",
            extra={
                "event": "callback",
                "chat_id": chat_id,
                "data": _truncate(str(data), 200) if data else None,
            },
        )
        if callback:
            await callback.answer()

    async def on_other(self, update: Update) -> None:
        """Handle non-message updates."""

        self.logger.debug(
            "telegram_update_ignored",
            extra={"event": "ignored", "update_id": update.update_id},
        )

    # ---------- Reply helpers ----------
    async def _reply(self, chat_id: int, text: str) -> None:
        """Send a reply with a small retry policy (non-blocking caller)."""
        attempt = 1
        while attempt <= 3:
            try:
                await safe_reply_text(
                    self.bot,
                    chat_id,
                    text,
                    disable_web_page_preview=True,
                )
                self.logger.debug(
                    "telegram_reply_sent",
                    extra={"event": "reply", "chat_id": chat_id, "attempt": attempt},
                )
                return
            except RetryAfter as exc:
                delay = max(float(getattr(exc, "retry_after", 1.0) or 1.0), 1.0)
                await asyncio.sleep(delay)
            except (TimedOut, NetworkError):
                await asyncio.sleep(min(2**attempt, 8.0))
            except TelegramError as exc:
                self.logger.warning(
                    "telegram_reply_failed",
                    extra={
                        "event": "reply_failed",
                        "chat_id": chat_id,
                        "err": str(exc),
                    },
                )
                return
            attempt += 1

    async def _ping_text(self) -> str:
        return "pong"

    async def _status_text(self) -> str:
        base: dict[str, str] = {"webhook": "ok"}
        if callable(self._status_provider):
            try:
                extra = self._status_provider() or {}
                for key, value in extra.items():
                    base[str(key)] = str(value)
            except Exception as exc:  # noqa: BLE001
                self.logger.warning(
                    "status_provider_error",
                    extra={"event": "status_provider_error", "err": str(exc)},
                )
        lines = [f"{key}: {value}" for key, value in base.items()]
        return "status\n" + "\n".join(lines)

    def _help_text(self) -> str:
        # Fallback console help
        return (
            "Commands:\n"
            "/ping – liveness check\n"
            "/status – basic status\n"
            "/start – this help\n"
        )


_WEBHOOK_LAST_OK: str | None = None
_POLLING_TASK_STARTED = False


async def register_webhook(
    bot: Bot, public_url: str, logger: Logger | None = None
) -> bool:
    """Register a webhook for ``bot`` at ``public_url``."""

    logger = logger or get_logger(__name__)

    if not public_url:
        logger.info(
            "telegram_webhook_skipped_or_not_configured",
            extra={"event": "webhook_skipped", "public_url": public_url},
        )
        return False

    if not public_url.startswith("https://"):
        logger.error(
            "telegram_webhook_invalid",
            extra={"event": "webhook_invalid", "public_url": public_url},
        )
        return False

    endpoint = public_url.rstrip("/") + "/telegram/webhook"

    global _WEBHOOK_LAST_OK
    if _WEBHOOK_LAST_OK == endpoint:
        logger.info(
            "telegram_webhook_already_set",
            extra={"event": "webhook_ok", "url": endpoint},
        )
        return True

    try:
        info = await bot.get_webhook_info()
    except Exception:
        info = None

    current = (getattr(info, "url", "") or "").rstrip("/") if info else ""
    if current == endpoint:
        _WEBHOOK_LAST_OK = endpoint
        logger.info(
            "telegram_webhook_already_set",
            extra={"event": "webhook_ok", "url": endpoint},
        )
        return True

    max_attempts = 5
    backoff_base = 0.8

    for attempt in range(1, max_attempts + 1):
        try:
            drop_pending = bool(current and current != endpoint)
            await bot.set_webhook(
                url=endpoint,
                allowed_updates=None,
                drop_pending_updates=drop_pending,
            )
            _WEBHOOK_LAST_OK = endpoint
            logger.info(
                "telegram_set_webhook",
                extra={
                    "event": "set_webhook",
                    "endpoint": endpoint,
                    "drop_pending": drop_pending,
                },
            )
            return True
        except RetryAfter as exc:
            delay = float(getattr(exc, "retry_after", 1.0) or 1.0)
            logger.warning(
                "telegram_set_webhook_retry_after",
                extra={
                    "event": "set_webhook_retry_after",
                    "delay": delay,
                    "attempt": attempt,
                },
            )
            await asyncio.sleep(delay)
        except TelegramError as exc:  # pragma: no cover - network
            sleep_for = backoff_base * (2 ** (attempt - 1))
            sleep_for += random.uniform(0.0, 0.4)
            logger.warning(
                "telegram_set_webhook_failed",
                extra={
                    "event": "set_webhook_failed",
                    "err": str(exc),
                    "attempt": attempt,
                    "sleep": round(sleep_for, 2),
                },
            )
            if attempt >= max_attempts:
                break
            await asyncio.sleep(sleep_for)

    logger.error(
        "telegram_webhook_registration_failed",
        extra={"event": "webhook_failed", "url": endpoint},
    )
    return False


def start_fallback_polling(
    bot: Bot, handler: TelegramWebhookController, interval_sec: float = 10.0
) -> None:
    """Start lightweight polling fallback when webhook registration fails."""

    global _POLLING_TASK_STARTED
    if _POLLING_TASK_STARTED:
        return
    _POLLING_TASK_STARTED = True

    async def _starter() -> None:
        try:
            await handler.activate_polling_fallback(
                "legacy helper invoked",
                interval=max(interval_sec, 1.0),
            )
        finally:
            # Allow future retries if the controller stops polling.
            global _POLLING_TASK_STARTED
            _POLLING_TASK_STARTED = False

    safe_task(_starter())


__all__ = [
    "TelegramEnhancedNotifier",
    "TelegramWebhookController",
    "register_webhook",
    "safe_reply_text",
    "start_fallback_polling",
]


# --- FastAPI integration snippet (documentation only) ---
#
# from fastapi import FastAPI
# from telegram import Bot
#
# from nifty_scalper_bot.config.settings import get_settings
# from nifty_scalper_bot.notifications.telegram_webhook_enhanced import (
#     TelegramEnhancedNotifier,
#     TelegramWebhookController,
#     register_webhook,
#     start_fallback_polling,
# )
#
# settings = get_settings()
# app = FastAPI()
#
# notifier = TelegramEnhancedNotifier.from_settings(settings.notifications)
# base_token = settings.notifications.token
# shared_bot = (
#     notifier.bot if notifier is not None else Bot(base_token) if base_token else None
# )
# controller: TelegramWebhookController | None = None
# if shared_bot is not None:
#     controller = TelegramWebhookController(
#         bot=shared_bot,
#         settings=settings.notifications,
#     )
#     app.include_router(controller.router)
#
# @app.on_event("startup")
# async def _setup_telegram_webhook() -> None:
#     if (
#         not settings.notifications.enabled
#         or shared_bot is None
#         or controller is None
#         or not settings.notifications.public_base_url
#     ):
#         app.logger.info(  # or use get_logger(__name__)
#             "telegram_webhook_skipped",
#             extra={"event": "webhook_skipped", "reason": "not_configured"},
#         )
#         return
#     success = await register_webhook(
#         shared_bot,
#         settings.notifications.public_base_url,
#     )
#     if not success and getattr(settings.notifications, "allow_poll_fallback", False):
#         start_fallback_polling(shared_bot, controller)
#
# --- Optional python-telegram-bot Application rate limiter example ---
#
# from telegram.ext import AIORateLimiter, ApplicationBuilder
#
# limiter = AIORateLimiter(
#     overall_max_rate=int(settings.notifications.rate_per_second or 25)
# )
# application = (
#     ApplicationBuilder()
#     .token(settings.notifications.token)
#     .rate_limiter(limiter)
#     .build()
# )
