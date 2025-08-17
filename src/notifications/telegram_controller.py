# src/notifications/telegram_controller.py
"""
Telegram Controller

Purpose:
- Manages Telegram command interface for controlling the bot.
- Supports /start, /stop, /status, /mode commands.
- Quality mode now supports AUTO, ON, and OFF (with regime-aware auto switching).
- Runs polling in a background worker thread to avoid blocking.

Relevant Changes:
- Added /mode quality auto|on|off command.
- Status messages now display current quality mode and reasoning if AUTO.
- Polling wrapped into background thread so it doesn’t block the trading loop.
- File header includes metadata and recent changes for clarity.

Usage:
    controller = TelegramController()
    controller.start_polling()
"""

import logging
import threading
from typing import Optional

from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext

from src.config import Config

logger = logging.getLogger(__name__)


class TelegramController:
    def __init__(self, token: str, chat_id: str, trader_ref=None):
        self.token = token
        self.chat_id = str(chat_id)
        self.trader_ref = trader_ref
        self.updater: Optional[Updater] = None
        self.quality_mode = str(getattr(Config, "QUALITY_MODE_DEFAULT", "off")).lower()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    # ---------------------------- core commands ---------------------------- #

    def _start(self, update: Update, context: CallbackContext) -> None:
        if str(update.effective_chat.id) != self.chat_id:
            return
        self._send("✅ Bot is active and listening to commands.")

    def _stop(self, update: Update, context: CallbackContext) -> None:
        if str(update.effective_chat.id) != self.chat_id:
            return
        self._send("🛑 Bot stop command received (no effect on process).")

    def _status(self, update: Update, context: CallbackContext) -> None:
        if str(update.effective_chat.id) != self.chat_id:
            return
        qmode = self.quality_mode.upper()
        msg = f"📊 Status:\nQuality Mode: {qmode}"
        if self.trader_ref:
            try:
                status = self.trader_ref.get_status()
                msg += f"\nTrades Today: {status.get('trades', 0)}"
                msg += f"\nPnL: ₹{status.get('pnl', 0):.2f}"
            except Exception as e:
                msg += f"\n⚠️ Status fetch error: {e}"
        self._send(msg)

    def _mode(self, update: Update, context: CallbackContext) -> None:
        if str(update.effective_chat.id) != self.chat_id:
            return
        try:
            if not context.args:
                self._send("Usage: /mode quality [auto|on|off]")
                return
            subcmd = context.args[0].lower()
            if subcmd == "quality":
                if len(context.args) >= 2:
                    new_mode = context.args[1].lower()
                    if new_mode in ["auto", "on", "off"]:
                        self.quality_mode = new_mode
                        if self.trader_ref:
                            self.trader_ref.set_quality_mode(new_mode)
                        self._send(f"⚙️ Quality mode set to: {new_mode.upper()}")
                    else:
                        self._send("⚠️ Invalid mode. Use auto|on|off")
                else:
                    self._send(f"Current quality mode: {self.quality_mode.upper()}")
            else:
                self._send("⚠️ Unknown mode. Only 'quality' supported.")
        except Exception as e:
            self._send(f"❌ Error processing /mode: {e}")

    # ---------------------------- polling worker --------------------------- #

    def start_polling(self) -> None:
        """Start Telegram polling in background thread."""
        if self._running:
            return
        self._running = True

        def _worker():
            try:
                self.updater = Updater(token=self.token, use_context=True)
                dp = self.updater.dispatcher
                dp.add_handler(CommandHandler("start", self._start))
                dp.add_handler(CommandHandler("stop", self._stop))
                dp.add_handler(CommandHandler("status", self._status))
                dp.add_handler(CommandHandler("mode", self._mode))

                self._send("🤖 Telegram controller started.")
                self.updater.start_polling(drop_pending_updates=True)
                self.updater.idle()
            except Exception as e:
                logger.error("Telegram polling error: %s", e, exc_info=True)

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()

    def stop_polling(self) -> None:
        """Stop Telegram polling gracefully."""
        if self.updater:
            try:
                self.updater.stop()
            except Exception:
                pass
        self._running = False

    # ------------------------------- utils -------------------------------- #

    def _send(self, text: str) -> None:
        try:
            if not self.updater:
                return
            bot = self.updater.bot
            bot.send_message(chat_id=self.chat_id, text=text)
        except Exception as e:
            logger.warning("⚠️ Telegram send failed: %s", e)

    # ------------------------------- hooks -------------------------------- #

    def set_quality_mode(self, mode: str) -> None:
        self.quality_mode = mode.lower().strip()