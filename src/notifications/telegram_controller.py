# src/notifications/telegram_controller.py
from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any, Callable, Dict, Optional

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

logger = logging.getLogger(__name__)


class TelegramController:
    """
    Minimal, production-safe Telegram bot.

    - Start in a background thread: start()
    - Safe cross-thread send_message()
    - Commands:
        /start
        /status
        /summary
        /health
        /config get
        /config set <key> <value>
        /emergency   (flatten positions & cancel orders)

    Inject runtime context with set_context(...).
    Expose config bridges by setting _config_getter/_config_setter from the runner.
    """

    def __init__(self, bot_token: str, chat_id: Optional[int] = None) -> None:
        self.bot_token = bot_token.strip()
        self.chat_id = int(chat_id or 0)

        self._application: Optional[Application] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

        # Optional context set by runner
        self._session = None          # TradingSession
        self._executor = None         # OrderExecutor
        self._health_getter: Optional[Callable[[], Dict[str, Any]]] = None

        # Config bridges (wired by StrategyRunner)
        self._config_getter: Optional[Callable[[], Dict[str, Any]]] = None
        self._config_setter: Optional[Callable[[str, str], str]] = None

    # ---------------- public lifecycle ----------------

    def set_context(
        self,
        session=None,
        executor=None,
        health_getter: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> None:
        self._session = session
        self._executor = executor
        self._health_getter = health_getter

    def start(self) -> None:
        if not self.bot_token:
            logger.warning("TelegramController: missing bot token; not starting.")
            return
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run_polling, name="tg-bot", daemon=True)
        self._thread.start()
        logger.info("Telegram bot started.")

    def stop(self) -> None:
        try:
            if self._application:
                self._application.stop()
        except Exception:
            pass

    # ---------------- messaging (thread-safe) ----------------

    def send_message(self, text: str, parse_mode: Optional[str] = None) -> bool:
        """
        Thread-safe send. No-op if bot not yet running.
        """
        app = self._application
        loop = self._loop
        if not app or not loop or not loop.is_running():
            return False
        try:
            fut = asyncio.run_coroutine_threadsafe(
                app.bot.send_message(
                    chat_id=self.chat_id if self.chat_id else None,
                    text=text,
                    parse_mode=parse_mode or ParseMode.HTML,
                ),
                loop,
            )
            fut.result(timeout=5.0)
            return True
        except Exception as e:
            logger.warning("Telegram send failed: %s", e)
            return False

    # ---------------- internals ----------------

    def _run_polling(self) -> None:
        try:
            # Create a fresh loop in this thread
            asyncio.set_event_loop(asyncio.new_event_loop())
            self._loop = asyncio.get_event_loop()

            app = ApplicationBuilder().token(self.bot_token).build()
            self._application = app

            # Handlers
            app.add_handler(CommandHandler("start", self._cmd_start))
            app.add_handler(CommandHandler("status", self._cmd_status))
            app.add_handler(CommandHandler("summary", self._cmd_summary))
            app.add_handler(CommandHandler("health", self._cmd_health))
            app.add_handler(CommandHandler("config", self._cmd_config, block=False))
            app.add_handler(CommandHandler("emergency", self._cmd_emergency))
            # Fallback: treat plain text as /config set key value or help
            app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_text))

            # Run (non-blocking loop kept open)
            app.run_polling(close_loop=False, allowed_updates=Update.ALL_TYPES)
        except Exception as e:
            logger.error("Telegram polling error: %s", e, exc_info=True)

    # ---------------- auth / helpers ----------------

    def _authorized(self, update: Update) -> bool:
        if not self.chat_id:
            # First contact: latch to this chat for safety
            try:
                self.chat_id = update.effective_chat.id  # type: ignore[assignment]
                logger.info("TelegramController: latched chat_id=%s", self.chat_id)
                return True
            except Exception:
                return False
        return update.effective_chat and update.effective_chat.id == self.chat_id

    async def _reply(self, update: Update, text: str) -> None:
        if not self._authorized(update):
            return
        try:
            await update.effective_chat.send_message(text, parse_mode=ParseMode.HTML)  # type: ignore[union-attr]
        except Exception as e:
            logger.debug("Telegram reply failed: %s", e)

    @staticmethod
    def _fmt_json(obj: Any, limit: int = 3500) -> str:
        s = json.dumps(obj, indent=2, default=str)
        if len(s) > limit:
            return s[:limit] + "\n… (truncated)"
        return s

    # ---------------- command handlers ----------------

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:  # noqa: ARG002
        if not self._authorized(update):
            return
        await self._reply(update, "🤖 Bot online.\nUse /status, /summary, /health, /config get, /config set <k> <v>, /emergency")

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:  # noqa: ARG002
        if not self._authorized(update):
            return
        s = self._session
        ex = self._executor
        if not s:
            await self._reply(update, "No session context.")
            return
        lines = [
            "<b>Status</b>",
            f"Equity: <code>{s.equity:.2f}</code>",
            f"PnL (today): <code>{s.daily_pnl:.2f}</code>",
            f"Trades (today): <code>{s.trades_today}</code>",
            f"Consec losses: <code>{s.consecutive_losses}</code>",
        ]
        if ex:
            active = getattr(ex, "get_active_orders", lambda: [])()
            lines.append(f"Open orders: <code>{len(active)}</code>")
            for r in active[:5]:
                lines.append(f"• {r.transaction_type} {r.symbol} qty={r.qty} SL={r.hard_stop_price}")
        await self._reply(update, "\n".join(lines))

    async def _cmd_summary(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:  # noqa: ARG002
        if not self._authorized(update):
            return
        s = self._session
        if not s or not s.trade_history:
            await self._reply(update, "No closed trades yet.")
            return
        # Last 5 trades
        last = s.trade_history[-5:]
        lines = ["<b>Recent trades</b>"]
        for t in last:
            pnl = f"{t.pnl:.2f}"
            lines.append(f"• {t.side} {t.symbol} qty={t.qty} @ {t.entry_price:.2f} → {t.exit_price:.2f}  PnL=<code>{pnl}</code>")
        await self._reply(update, "\n".join(lines))

    async def _cmd_health(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:  # noqa: ARG002
        if not self._authorized(update):
            return
        if self._health_getter:
            try:
                h = self._health_getter()
            except Exception as e:
                await self._reply(update, f"health error: <code>{e}</code>")
                return
            await self._reply(update, self._fmt_json(h))
        else:
            await self._reply(update, '{"status":"unknown"}')

    async def _cmd_config(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._authorized(update):
            return
        args = context.args or []
        if not args:
            await self._reply(update, "Usage:\n/config get\n/config set <key> <value>")
            return

        sub = args[0].lower()
        if sub == "get":
            getter = self._config_getter
            if not getter:
                await self._reply(update, "Config getter not wired.")
                return
            cfg = getter()
            await self._reply(update, self._fmt_json(cfg))
            return

        if sub == "set":
            if len(args) < 3:
                await self._reply(update, "Usage: /config set <key> <value>")
                return
            key = args[1]
            val = " ".join(args[2:])
            setter = self._config_setter
            if not setter:
                await self._reply(update, "Config setter not wired.")
                return
            try:
                msg = setter(key, val)
            except Exception as e:
                msg = f"error: {e}"
            await self._reply(update, f"<code>{msg}</code>")
            return

        await self._reply(update, "Usage:\n/config get\n/config set <key> <value>")

    async def _cmd_emergency(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:  # noqa: ARG002
        if not self._authorized(update):
            return
        ex = self._executor
        s = self._session
        if not ex:
            await self._reply(update, "No executor context.")
            return

        try:
            # Flatten: best-effort using executor’s market exit + cancel pending legs
            active = list(getattr(ex, "get_active_orders", lambda: [])())
            for r in active:
                try:
                    ex.exit_order(r.order_id, exit_reason="emergency")
                except Exception:
                    pass
            try:
                ex.cancel_all_orders()
            except Exception:
                pass
            if s and callable(getattr(s, "flatten_all", None)):
                # If a price provider is needed, use executor.get_last_price
                price_fn = lambda sym: getattr(ex, "get_last_price")(sym, exchange=getattr(r, "exchange", None))  # type: ignore[name-defined]
                s.flatten_all(price_fn)
            await self._reply(update, "❗ Emergency flatten executed.")
        except Exception as e:
            logger.error("Emergency handler failed: %s", e, exc_info=True)
            await self._reply(update, f"Emergency failed: <code>{e}</code>")

    async def _on_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        # Allow quick "/config set k v" style without slash
        if not self._authorized(update):
            return
        text = (update.message.text or "").strip().lower()
        if text.startswith("config set "):
            parts = update.message.text.split(maxsplit=3)  # keep case/value
            if len(parts) >= 4:
                _, _, key, val = parts
                setter = self._config_setter
                if setter:
                    try:
                        msg = setter(key, val)
                    except Exception as e:
                        msg = f"error: {e}"
                    await self._reply(update, f"<code>{msg}</code>")
                    return
        await self._reply(update, "Unknown. Try: /status /summary /health /config get /config set <k> <v> /emergency")
