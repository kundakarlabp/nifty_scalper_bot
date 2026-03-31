"""Resilient Telegram service for coordinating bot notifications."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Callable, Coroutine, Optional

from telegram import Message, Update
from telegram.error import Conflict, NetworkError, RetryAfter, TimedOut
from telegram.ext import Application, ApplicationBuilder, CommandHandler, ContextTypes

from nifty_scalper_bot.utils.async_helpers import safe_task

_LOG = logging.getLogger("nifty_scalper_bot.telegram")


CommandCallback = Callable[
    [Update, ContextTypes.DEFAULT_TYPE], Coroutine[object, object, None]
]


def _iter_command_handlers(app: Application) -> list[CommandHandler]:
    """Return command handlers already registered on ``app``.

    Args:
        app: Telegram application whose handlers should be inspected.

    Returns:
        A list of command handlers discovered on the application.

    Raises:
        None.
    """

    try:
        handlers = getattr(app, "handlers", {})
        layer = handlers.get(0, []) if isinstance(handlers, dict) else []
        # PTB v20+: CommandHandler exposes `.commands` (set[str])
        return [h for h in layer if getattr(h, "commands", None)]
    except Exception as exc:  # noqa: BLE001
        _LOG.error("Failure in _iter_command_handlers: %s", exc)
        return []


def _list_existing_commands(app: Application) -> list[str]:
    """Return a sorted list of unique command names on ``app``.

    Args:
        app: Telegram application whose command registry is inspected.

    Returns:
        Sorted command names currently wired on the application.

    Raises:
        None.
    """

    commands: set[str] = set()
    for handler in _iter_command_handlers(app):
        try:
            entries = getattr(handler, "commands", None)
            if entries:
                commands.update(entries)
        except Exception as exc:  # noqa: BLE001
            _LOG.error("Failure while collating commands: %s", exc)
    return sorted(commands)


def _has_command(app: Application, name: str) -> bool:
    """Return ``True`` when ``name`` is registered on ``app``.

    Args:
        app: Telegram application inspected for command handlers.
        name: Command name searched in the registry.

    Returns:
        ``True`` when the command already exists, ``False`` otherwise.

    Raises:
        None.
    """

    try:
        for handler in _iter_command_handlers(app):
            commands = getattr(handler, "commands", None)
            if commands and name in commands:
                return True
    except Exception as exc:  # noqa: BLE001
        _LOG.error("Failure in _has_command: %s", exc)
    return False


def safe_add(app: Application, name: str, handler: CommandCallback) -> None:
    """Register ``handler`` for ``name`` without overriding existing wiring.

    Args:
        app: Telegram application receiving the handler registration.
        name: Command name associated with the handler.
        handler: Coroutine callback executed when the command triggers.

    Returns:
        None.

    Raises:
        None.
    """

    try:
        if _has_command(app, name):
            _LOG.info("telegram.cmd.skip exists name=%s", name)
            return
        app.add_handler(CommandHandler(name, handler))
        _LOG.info("telegram.cmd.add name=%s", name)
    except Exception as exc:  # noqa: BLE001
        _LOG.error("Failure in safe_add: %s", exc)


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Return environment variable *name* or *default* when unset."""

    value = os.getenv(name)
    if value is None:
        return default
    stripped = value.strip()
    return stripped if stripped else default


class TelegramService:
    """Manage Telegram polling lifecycle with retries and graceful shutdown."""

    def __init__(
        self,
        *,
        token: str,
        chat_id: int,
        get_status: Callable[[], str],
        force_ws_reconnect: Optional[Callable[[], None]] = None,
        close_all_positions: Optional[Callable[[], None]] = None,
    ) -> None:
        self.token = token
        self.chat_id = int(chat_id)
        self.get_status = get_status
        self.force_ws_reconnect = force_ws_reconnect
        self.close_all_positions = close_all_positions

        self._app: Application | None = None
        self._task: asyncio.Task[None] | None = None
        self._stop_evt = asyncio.Event()
        self._app_initialized = False

    async def _reply(self, update: Update, text: str) -> None:
        """Send *text* to the effective message when available."""

        message: Message | None = update.effective_message
        if message is None:
            _LOG.warning("Cannot reply to update without message: %s", text)
            return
        await message.reply_text(text)

    def _is_allowed(self, update: Update) -> bool:
        chat = update.effective_chat.id if update and update.effective_chat else None
        return chat == self.chat_id

    def _is_admin(self, update: Update) -> bool:
        return self._is_allowed(update)

    async def _ensure_auth(self, update: Update, *, admin: bool = False) -> bool:
        if not self._is_allowed(update):
            chat = (
                update.effective_chat.id
                if update and update.effective_chat
                else "unknown"
            )
            _LOG.warning("Rejected Telegram update from unauthorized chat: %s", chat)
            return False
        if update.effective_message is None:
            _LOG.warning("Authorized update missing message payload")
            return False
        if admin and not self._is_admin(update):
            await self._reply(update, "Unauthorized (admin only).")
            return False
        return True

    async def _cmd_start(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        """Acknowledge bot availability to authorized chats.

        Args:
            update: Incoming Telegram update payload.
            _: Callback context provided by PTB; unused.

        Returns:
            None.

        Raises:
            None.
        """

        _LOG.debug("Entered _cmd_start")
        if not await self._ensure_auth(update):
            return
        try:
            await self._reply(update, "✅ Nifty Scalper Bot: online.")
            _LOG.info("Condition met: responded to /start")
        except Exception as exc:  # noqa: BLE001
            _LOG.error("Failure in _cmd_start: %s", exc)

    async def _cmd_ping(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        """Respond with a pong to confirm bot reachability.

        Args:
            update: Incoming Telegram update payload.
            _: Callback context provided by PTB; unused.

        Returns:
            None.

        Raises:
            None.
        """

        _LOG.debug("Entered _cmd_ping")
        if not await self._ensure_auth(update):
            return
        try:
            await self._reply(update, "pong")
            _LOG.info("Condition met: replied to /ping")
        except Exception as exc:  # noqa: BLE001
            _LOG.error("Failure in _cmd_ping: %s", exc)

    async def _cmd_status(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        """Return the current status summary from the trading stack.

        Args:
            update: Incoming Telegram update payload.
            _: Callback context provided by PTB; unused.

        Returns:
            None.

        Raises:
            None.
        """

        _LOG.debug("Entered _cmd_status")
        if not await self._ensure_auth(update):
            return
        try:
            snapshot = self.get_status()
            _LOG.info("Condition met: retrieved status snapshot")
            await self._reply(update, snapshot)
        except Exception as exc:  # noqa: BLE001
            _LOG.error("Failure in _cmd_status: %s", exc)
            try:
                await self._reply(update, f"status error: {exc!s}")
            except Exception as nested_exc:  # noqa: BLE001
                _LOG.error("Failure while replying with status error: %s", nested_exc)

    async def _cmd_status_verbose(
        self, update: Update, _: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Return a verbose status summary without altering existing wiring.

        Args:
            update: Incoming Telegram update payload.
            _: Callback context provided by PTB; unused.

        Returns:
            None.

        Raises:
            None.
        """

        _LOG.debug("Entered _cmd_status_verbose")
        if not await self._ensure_auth(update):
            return
        try:
            snapshot = self.get_status()
        except Exception as exc:  # noqa: BLE001
            _LOG.error("Failure in _cmd_status_verbose: %s", exc)
            try:
                await self._reply(update, f"status verbose error: {exc!s}")
            except Exception as nested_exc:  # noqa: BLE001
                _LOG.error(
                    "Failure while replying with status verbose error: %s", nested_exc
                )
            return
        message = f"# Status\n{snapshot}" if snapshot else "# Status\n(n/a)"
        _LOG.info("Condition met: produced verbose status snapshot")
        try:
            await self._reply(update, message)
        except Exception as exc:  # noqa: BLE001
            _LOG.error("Failure while replying with verbose status: %s", exc)

    async def _cmd_ws(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        """Trigger websocket reconnection when supported.

        Args:
            update: Incoming Telegram update payload.
            _: Callback context provided by PTB; unused.

        Returns:
            None.

        Raises:
            None.
        """

        _LOG.debug("Entered _cmd_ws")
        if not await self._ensure_auth(update, admin=True):
            return
        if self.force_ws_reconnect is None:
            _LOG.info("Condition met: websocket control missing")
            await self._reply(update, "ws control not wired.")
            return
        try:
            self.force_ws_reconnect()
            await self._reply(update, "♻️ WS reconnect triggered.")
            _LOG.info("Condition met: websocket reconnect invoked")
        except Exception as exc:  # noqa: BLE001
            _LOG.error("Failure in _cmd_ws: %s", exc)
            try:
                await self._reply(update, f"ws error: {exc!s}")
            except Exception as nested_exc:  # noqa: BLE001
                _LOG.error("Failure while replying with ws error: %s", nested_exc)

    async def _cmd_emergency(
        self, update: Update, _: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Invoke the configured emergency close handler when available.

        Args:
            update: Incoming Telegram update payload.
            _: Callback context provided by PTB; unused.

        Returns:
            None.

        Raises:
            None.
        """

        _LOG.debug("Entered _cmd_emergency")
        if not await self._ensure_auth(update, admin=True):
            return
        if self.close_all_positions is None:
            _LOG.info("Condition met: emergency close not configured")
            await self._reply(update, "emergency close not wired.")
            return
        try:
            self.close_all_positions()
            await self._reply(update, "🛑 Emergency close triggered.")
            _LOG.info("Condition met: emergency close invoked")
        except Exception as exc:  # noqa: BLE001
            _LOG.error("Failure in _cmd_emergency: %s", exc)
            try:
                await self._reply(update, f"emergency error: {exc!s}")
            except Exception as nested_exc:  # noqa: BLE001
                _LOG.error(
                    "Failure while replying with emergency error: %s", nested_exc
                )

    async def _cmd_selftest(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        """Surface readiness checks for core integrations.

        Args:
            update: Incoming Telegram update payload.
            _: Callback context provided by PTB; unused.

        Returns:
            None.

        Raises:
            None.
        """

        _LOG.debug("Entered _cmd_selftest")
        if not await self._ensure_auth(update, admin=True):
            return
        checks: list[str] = []
        try:
            snapshot = self.get_status()
            checks.append("status=OK" if snapshot else "status=EMPTY")
        except Exception as exc:  # noqa: BLE001
            _LOG.error("Failure in _cmd_selftest status retrieval: %s", exc)
            checks.append("status=ERROR")
        ws_ready = self.force_ws_reconnect is not None
        checks.append(f"ws={'OK' if ws_ready else 'MISSING'}")
        emergency_ready = self.close_all_positions is not None
        checks.append(f"emergency={'OK' if emergency_ready else 'MISSING'}")
        response = "selftest: " + " | ".join(checks)
        _LOG.info("Condition met: compiled selftest summary")
        try:
            await self._reply(update, response)
        except Exception as exc:  # noqa: BLE001
            _LOG.error("Failure while replying with selftest summary: %s", exc)

    async def _cmd_help_admin(
        self, update: Update, _: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """List present and missing administrative commands.

        Args:
            update: Incoming Telegram update payload.
            _: Callback context provided by PTB; unused.

        Returns:
            None.

        Raises:
            None.
        """

        _LOG.debug("Entered _cmd_help_admin")
        if not await self._ensure_auth(update, admin=True):
            return
        planned = {
            "start",
            "ping",
            "status",
            "statusv",
            "ws",
            "emergency",
            "selftest",
            "help_admin",
        }
        app = self._app
        if app is None:
            _LOG.info("Condition met: telegram application not ready for help_admin")
            await self._reply(update, "telegram application not ready")
            return
        try:
            existing = set(_list_existing_commands(app))
            missing = sorted(planned - existing)
            present = sorted(planned & existing)
            message = (
                "present: "
                + ", ".join(present)
                + "\n"
                + "missing: "
                + ", ".join(missing)
            )
            _LOG.info("Condition met: reported admin command inventory")
            await self._reply(update, message)
        except Exception as exc:  # noqa: BLE001
            _LOG.error("Failure in _cmd_help_admin: %s", exc)
            try:
                await self._reply(update, f"help_admin error: {exc!s}")
            except Exception as nested_exc:  # noqa: BLE001
                _LOG.error(
                    "Failure while replying with help_admin error: %s", nested_exc
                )

    async def _on_error(self, _: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        error = context.error
        if isinstance(error, RetryAfter):
            _LOG.warning("Telegram RetryAfter (%.1fs)", error.retry_after)
        elif isinstance(error, TimedOut):
            _LOG.warning("Telegram timed out; continuing.")
        elif isinstance(error, NetworkError):
            _LOG.warning("Telegram network error; continuing.")
        else:
            _LOG.exception("Telegram handler error")

    def _build_application(self) -> Application:
        app = (
            ApplicationBuilder()
            .token(self.token)
            .concurrent_updates(True)
            .read_timeout(35)
            .write_timeout(35)
            .connect_timeout(15)
            .pool_timeout(15)
            .connection_pool_size(100)
            .build()
        )
        app.add_error_handler(self._on_error)
        _LOG.info(
            "telegram.cmd.inventory.initial",
            extra={"commands": _list_existing_commands(app)},
        )
        command_specs: tuple[tuple[str, CommandCallback, tuple[str, ...]], ...] = (
            ("help", self._cmd_help_admin, ("opshelp", "help_admin")),
            ("status", self._cmd_status, ("session",)),
            ("statusv", self._cmd_status_verbose, ()),
            ("health", self._cmd_selftest, ("selftest", "diag")),
            ("ping", self._cmd_ping, ("net",)),
            ("start", self._cmd_start, ()),
            ("ws_status", self._cmd_ws, ("ws",)),
            ("emergency", self._cmd_emergency, ()),
        )
        for primary, handler, aliases in command_specs:
            safe_add(app, primary, handler)
            for alias in aliases:
                safe_add(app, alias, handler)
        _LOG.info(
            "telegram.cmd.inventory.final",
            extra={"commands": _list_existing_commands(app)},
        )
        return app

    async def _start_polling(self) -> None:
        """Start Telegram polling safely.

        Raises:
            RuntimeError: If the PTB updater cannot be retrieved.
        """

        app = self._app
        if app is None:
            raise RuntimeError("Telegram application is not configured")
        if not self._app_initialized:
            await app.initialize()
            self._app_initialized = True
        await app.start()
        updater = app.updater
        if updater is None:
            raise RuntimeError("Application updater is not available")
        await updater.start_polling(
            allowed_updates=Update.ALL_TYPES,
            timeout=30,
            drop_pending_updates=True,
        )
        _LOG.info("✅ telegram_polling_started")

    async def _stop_polling(self) -> None:
        """Gracefully stop Telegram polling."""

        app = self._app
        if app is None:
            return
        updater = app.updater
        success = True
        try:
            if updater is not None:
                await updater.stop()
        except Exception as exc:  # noqa: BLE001
            success = False
            _LOG.warning("telegram_polling_shutdown_failed", exc_info=exc)
        try:
            await app.stop()
        except Exception as exc:  # noqa: BLE001
            success = False
            _LOG.warning("telegram_polling_shutdown_failed", exc_info=exc)
        try:
            await app.shutdown()
        except Exception as exc:  # noqa: BLE001
            success = False
            _LOG.warning("telegram_polling_shutdown_failed", exc_info=exc)
        if success:
            _LOG.info("🛑 telegram_polling_stopped")
        self._app_initialized = False

    async def start(self, ready_evt: asyncio.Event) -> None:
        """Start polling once *ready_evt* has been triggered."""

        self._app = self._build_application()
        self._app_initialized = False
        await self._app.initialize()
        self._app_initialized = True

        try:
            await self._app.bot.delete_webhook(drop_pending_updates=True)
        except Conflict:
            pass

        _LOG.info("Telegram waiting for core readiness...")
        await ready_evt.wait()
        _LOG.info("Telegram starting polling.")

        async def _runner() -> None:
            assert self._app is not None
            while not self._stop_evt.is_set():
                try:
                    await self._start_polling()
                    await self._stop_evt.wait()
                except (RetryAfter, TimedOut, NetworkError) as exc:
                    _LOG.warning("Polling transient error: %s; retrying soon", exc)
                    await asyncio.sleep(3)
                    continue
                finally:
                    await self._stop_polling()
            _LOG.info("Telegram runner stopped.")

        self._task = safe_task(_runner())

    async def stop(self) -> None:
        """Stop polling and dispose of the application."""

        _LOG.info("Telegram stopping...")
        self._stop_evt.set()
        if self._task:
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._app and self._app_initialized:
            await self._stop_polling()
        self._app = None


def telegram_from_env(
    *,
    get_status: Callable[[], str],
    force_ws_reconnect: Optional[Callable[[], None]] = None,
    close_all_positions: Optional[Callable[[], None]] = None,
) -> TelegramService | None:
    """Factory that wires configuration from environment variables."""

    token = _env("TELEGRAM__BOT_TOKEN") or _env("TELEGRAM_BOT_TOKEN")
    chat_raw = _env("TELEGRAM__CHAT_ID") or _env("TELEGRAM_CHAT_ID")

    if not token and not chat_raw:
        _LOG.warning(
            "Telegram disabled (set TELEGRAM__BOT_TOKEN and TELEGRAM__CHAT_ID to "
            "enable)"
        )
        return None
    if not token or not chat_raw:
        _LOG.error(
            "Telegram misconfigured: TELEGRAM__BOT_TOKEN and TELEGRAM__CHAT_ID must "
            "both be set."
        )
        return None

    try:
        chat_id = int(chat_raw)
    except ValueError:
        _LOG.error("Invalid TELEGRAM__CHAT_ID %r (must be an integer)", chat_raw)
        return None

    _LOG.info("Telegram enabled for chat_id=%s", chat_id)
    return TelegramService(
        token=token,
        chat_id=chat_id,
        get_status=get_status,
        force_ws_reconnect=force_ws_reconnect,
        close_all_positions=close_all_positions,
    )


__all__ = ["TelegramService", "telegram_from_env"]
