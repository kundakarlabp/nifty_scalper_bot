# src/notifications/telegram_controller.py
"""
Telegram Controller for trading bot.

Features
--------
- Command handling (/start, /stop, /mode, /quality, /regime, /pause, /resume, etc.)
- Status & summary reporting with pinned mini-status
- Supports LIVE/SHADOW mode, Quality ON|OFF|AUTO, Regime AUTO|TREND|RANGE|OFF
- Default modes: LIVE, QUALITY=AUTO, REGIME=AUTO
- Polling-based background worker to avoid webhook errors
"""

import logging
import threading
import time
from typing import Any, Callable, Dict, Optional

from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext

from src.config import Config

logger = logging.getLogger(__name__)


class TelegramController:
    def __init__(
        self,
        status_callback: Callable[[], Dict[str, Any]],
        control_callback: Callable[[str, str], bool],
        summary_callback: Callable[[], str],
    ) -> None:
        self.token = getattr(Config, "TELEGRAM_BOT_TOKEN", "")
        self.chat_id = int(getattr(Config, "TELEGRAM_CHAT_ID", 0))
        self.status_callback = status_callback
        self.control_callback = control_callback
        self.summary_callback = summary_callback
        self.updater: Optional[Updater] = None
        self.dispatcher = None
        self._pause_until: float = 0.0

    # ---------------- Polling Worker ---------------- #

    def start_polling(self) -> None:
        if not self.token:
            logger.warning("No TELEGRAM_BOT_TOKEN provided, skipping Telegram init.")
            return
        try:
            self.updater = Updater(token=self.token, use_context=True)
            self.dispatcher = self.updater.dispatcher

            # Commands
            self.dispatcher.add_handler(CommandHandler("start", self._cmd_start))
            self.dispatcher.add_handler(CommandHandler("stop", self._cmd_stop))
            self.dispatcher.add_handler(CommandHandler("mode", self._cmd_mode))
            self.dispatcher.add_handler(CommandHandler("quality", self._cmd_quality))
            self.dispatcher.add_handler(CommandHandler("regime", self._cmd_regime))
            self.dispatcher.add_handler(CommandHandler("risk", self._cmd_risk))
            self.dispatcher.add_handler(CommandHandler("pause", self._cmd_pause))
            self.dispatcher.add_handler(CommandHandler("resume", self._cmd_resume))
            self.dispatcher.add_handler(CommandHandler("status", self._cmd_status))
            self.dispatcher.add_handler(CommandHandler("summary", self._cmd_summary))
            self.dispatcher.add_handler(CommandHandler("refresh", self._cmd_refresh))
            self.dispatcher.add_handler(CommandHandler("health", self._cmd_health))
            self.dispatcher.add_handler(CommandHandler("emergency", self._cmd_emergency))

            threading.Thread(target=self.updater.start_polling, daemon=True).start()
            logger.info("📡 Telegram polling started (polling mode).")

        except Exception as e:
            logger.error("Telegram polling failed: %s", e, exc_info=True)

    # ---------------- Command Handlers ---------------- #

    def _cmd_start(self, update: Update, ctx: CallbackContext) -> None:
        self.control_callback("start", "")
        self._reply(update, "✅ Trading started.")

    def _cmd_stop(self, update: Update, ctx: CallbackContext) -> None:
        self.control_callback("stop", "")
        self._reply(update, "🛑 Trading stopped.")

    def _cmd_mode(self, update: Update, ctx: CallbackContext) -> None:
        arg = " ".join(ctx.args) if ctx.args else "live"
        self.control_callback("mode", arg)
        self._reply(update, f"Mode set to {arg.upper()}")

    def _cmd_quality(self, update: Update, ctx: CallbackContext) -> None:
        arg = " ".join(ctx.args) if ctx.args else "auto"
        self.control_callback("mode", f"quality {arg}")
        self._reply(update, f"✨ Quality mode set to {arg.upper()}")

    def _cmd_regime(self, update: Update, ctx: CallbackContext) -> None:
        arg = " ".join(ctx.args) if ctx.args else "auto"
        self.control_callback("regime", arg)
        self._reply(update, f"🧭 Regime set to {arg.upper()}")

    def _cmd_risk(self, update: Update, ctx: CallbackContext) -> None:
        if not ctx.args:
            self._reply(update, "Usage: /risk <pct>")
            return
        self.control_callback("risk", ctx.args[0])
        self._reply(update, f"Risk updated to {ctx.args[0]}%")

    def _cmd_pause(self, update: Update, ctx: CallbackContext) -> None:
        mins = float(ctx.args[0]) if ctx.args else 1.0  # default 1 min
        self._pause_until = time.time() + mins * 60
        self._reply(update, f"⏸️ Entries paused for {mins} min")

    def _cmd_resume(self, update: Update, ctx: CallbackContext) -> None:
        self._pause_until = 0.0
        self._reply(update, "▶️ Entries resumed")

    def _cmd_status(self, update: Update, ctx: CallbackContext) -> None:
        st = self.status_callback()
        msg = (
            "<b>📊 Bot Status</b>\n"
            f"🔁 Trading: {'🟢 Running' if st['is_trading'] else '🔴 Stopped'}\n"
            f"🌐 Mode: {'🟢 LIVE' if st['live_mode'] else '⚪ SHADOW'}\n"
            f"✨ Quality: {str(st.get('quality_mode', 'AUTO')).upper()}\n"
            f"🧭 Regime: {str(st.get('regime_mode', 'AUTO')).upper()}\n"
            f"📦 Open Positions: {st['open_positions']}\n"
            f"📈 Closed Today: {st['closed_today']}\n"
            f"💰 Daily P&L: {st['daily_pnl']:.2f}\n"
            f"🏦 Acct Size: ₹{st['account_size']}\n"
            f"📅 Session: {st['session_date']}\n"
        )
        self._reply(update, msg, parse_mode="HTML")

    def _cmd_summary(self, update: Update, ctx: CallbackContext) -> None:
        self._reply(update, self.summary_callback(), parse_mode="HTML")

    def _cmd_refresh(self, update: Update, ctx: CallbackContext) -> None:
        self.control_callback("refresh", "")
        self._reply(update, "🔄 Balance/instruments refreshed.")

    def _cmd_health(self, update: Update, ctx: CallbackContext) -> None:
        self.control_callback("health", "")
        self._reply(update, "✅ Health check done.")

    def _cmd_emergency(self, update: Update, ctx: CallbackContext) -> None:
        self.control_callback("emergency", "")
        self._reply(update, "🛑 Emergency: All cleared.")

    # ---------------- Helpers ---------------- #

    def _reply(self, update: Update, text: str, parse_mode: Optional[str] = None) -> None:
        try:
            update.message.reply_text(text, parse_mode=parse_mode)
        except Exception as e:
            logger.error("Reply failed: %s", e)

    def send_message(self, text: str, parse_mode: Optional[str] = None) -> None:
        try:
            if not self.updater or not self.chat_id:
                return
            self.updater.bot.send_message(chat_id=self.chat_id, text=text, parse_mode=parse_mode)
        except Exception as e:
            logger.error("Send_message error: %s", e)

    def send_startup_alert(self) -> None:
        self.send_message("🤖 Bot is now online (LIVE mode, Quality=AUTO, Regime=AUTO).")