from __future__ import annotations

import json
import logging
import threading
from typing import Any, Callable, Dict, Optional

from src.config import settings

# We use python-telegram-bot v13 style imports (common in many Railway images)
try:
    from telegram import Update, ParseMode
    from telegram.ext import Updater, CommandHandler, CallbackContext, Filters
except Exception as e:  # pragma: no cover
    raise RuntimeError(f"python-telegram-bot not installed or incompatible: {e}")

log = logging.getLogger(__name__)


def _fmt_bool(b: bool) -> str:
    return "🟢" if b else "🔴"


def _is_admin(update: Update) -> bool:
    chat_id = update.effective_chat.id if update and update.effective_chat else None
    wanted = {settings.telegram.chat_id}
    extra = set(settings.telegram.extra_admin_ids or [])
    return (chat_id in wanted) or (chat_id in extra)


class TelegramController:
    """
    Thin wrapper around python-telegram-bot. Starts polling in a daemon thread.
    Exposes commands to control runner and query diagnostics.
    """

    def __init__(
        self,
        *,
        token: str,
        chat_id: int,
        runner,
        log_provider: Optional[Callable[[int], str]] = None,
    ) -> None:
        self.token = token
        self.chat_id = int(chat_id)
        self.runner = runner
        self.log_provider = log_provider or (lambda n: "log tail unavailable")

        self.updater: Optional[Updater] = None
        self._thread: Optional[threading.Thread] = None

    # ---------- public notify helpers (used by runner/executor) ----------

    def notify_text(self, text: str) -> None:
        try:
            if self.updater:
                self.updater.bot.send_message(chat_id=self.chat_id, text=text)
        except Exception as e:
            log.warning("notify_text failed: %s", e)

    def notify_entry(self, *, symbol: str, side: str, qty: int, price: float, record_id: str = "") -> None:
        msg = f"🟦 ENTRY {side} {symbol}\nQty: {qty} @ {price:.2f}\nRef: {record_id}"
        self.notify_text(msg)

    def notify_fills(self, text: str) -> None:
        self.notify_text(text)

    # ---------- lifecycle ----------

    def start(self) -> None:
        if self.updater:  # already started
            return
        self.updater = Updater(self.token, use_context=True)

        dp = self.updater.dispatcher

        dp.add_handler(CommandHandler("help", self._cmd_help))
        dp.add_handler(CommandHandler("id", self._cmd_id))
        dp.add_handler(CommandHandler("status", self._cmd_status))
        dp.add_handler(CommandHandler("flow", self._cmd_flow))
        dp.add_handler(CommandHandler("diag", self._cmd_diag))
        dp.add_handler(CommandHandler("mode", self._cmd_mode, Filters.user(user_id=[self.chat_id])))
        dp.add_handler(CommandHandler("pause", self._cmd_pause, Filters.user(user_id=[self.chat_id])))
        dp.add_handler(CommandHandler("resume", self._cmd_resume, Filters.user(user_id=[self.chat_id])))
        dp.add_handler(CommandHandler("risk", self._cmd_risk, Filters.user(user_id=[self.chat_id])))
        dp.add_handler(CommandHandler("quality", self._cmd_quality, Filters.user(user_id=[self.chat_id])))
        dp.add_handler(CommandHandler("regime", self._cmd_regime, Filters.user(user_id=[self.chat_id])))
        dp.add_handler(CommandHandler("positions", self._cmd_positions))
        dp.add_handler(CommandHandler("active", self._cmd_active))
        dp.add_handler(CommandHandler("logs", self._cmd_logs))
        dp.add_handler(CommandHandler("ping", self._cmd_ping))
        dp.add_handler(CommandHandler("emergency", self._cmd_emergency, Filters.user(user_id=[self.chat_id])))

        # Unknown commands go to /help
        dp.add_handler(CommandHandler(None, self._cmd_help))

        # Run on a daemon thread
        self._thread = threading.Thread(target=self.updater.start_polling, daemon=True)
        self._thread.start()
        log.info("Telegram polling started.")

    # ---------- command handlers ----------

    def _cmd_help(self, update: Update, context: CallbackContext) -> None:
        if not _is_admin(update):
            return
        text = (
            "🤖 *Nifty Scalper Bot*\n\n"
            "/status – bot status\n"
            "/flow – green/red flow summary\n"
            "/diag – full JSON diagnostics\n"
            "/mode live|shadow – toggle live trading\n"
            "/pause [minutes] – pause entries\n"
            "/resume – resume entries\n"
            "/risk <pct> – set risk per trade (e.g. /risk 0.5)\n"
            "/quality auto|on|off – indicator quality gates\n"
            "/regime auto|trend|range – regime override\n"
            "/positions – broker day positions\n"
            "/active – active orders\n"
            "/logs [n] – tail n log lines\n"
            "/emergency – flatten & cancel all\n"
            "/id – show chat id\n"
            "/ping – heartbeat\n"
        )
        update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)

    def _cmd_id(self, update: Update, context: CallbackContext) -> None:
        update.message.reply_text(f"Chat ID: `{update.effective_chat.id}`", parse_mode=ParseMode.MARKDOWN)

    def _cmd_ping(self, update: Update, context: CallbackContext) -> None:
        sp = self.runner.status_provider() if hasattr(self.runner, "status_provider") else {}
        ok = _fmt_bool(True)
        update.message.reply_text(f"{ok} heartbeat | live={int(sp.get('live_trading', False))} "
                                  f"paused={sp.get('paused', False)} active={sp.get('active_orders', 0)}")

    def _cmd_status(self, update: Update, context: CallbackContext) -> None:
        sp = self.runner.status_provider()
        msg = (
            f"📊 {sp.get('time_ist')}\n"
            f"{'🟢' if sp.get('live_trading') else '🟡'} {'LIVE' if sp.get('live_trading') else 'DRY'} | Kite\n"
            f"📦 Active: {sp.get('active_orders', 0)} | ⏸ {sp.get('paused')}\n"
            f"🗜 Quality: {sp.get('quality')} | Regime: {sp.get('regime_mode')}"
        )
        update.message.reply_text(msg)

    def _cmd_flow(self, update: Update, context: CallbackContext) -> None:
        d = self.runner.diagnostics()
        checks = d.get("checks", [])
        parts = []
        ok_all = True
        for c in checks:
            ok = bool(c.get("ok"))
            ok_all = ok_all and ok
            parts.append(f"{_fmt_bool(ok)} {c.get('name')}")
        head = "✅ Flow OK" if ok_all else "❗ Flow has issues"
        update.message.reply_text(f"{head}\n" + " · ".join(parts))

    def _cmd_diag(self, update: Update, context: CallbackContext) -> None:
        d = self.runner.diagnostics()
        pretty = json.dumps(d, indent=2, default=str)
        update.message.reply_text(f"```\n{pretty}\n```", parse_mode=ParseMode.MARKDOWN)

    def _cmd_mode(self, update: Update, context: CallbackContext) -> None:
        if not _is_admin(update):
            return
        args = (context.args or [])
        if not args:
            update.message.reply_text("Usage: /mode live|shadow")
            return
        val = args[0].lower()
        if val == "live":
            self.runner.live = True
            update.message.reply_text("Mode set to LIVE.")
        elif val in ("dry", "shadow"):
            self.runner.live = False
            update.message.reply_text("Mode set to SHADOW.")
        else:
            update.message.reply_text("Usage: /mode live|shadow")

    def _cmd_pause(self, update: Update, context: CallbackContext) -> None:
        if not _is_admin(update):
            return
        minutes = None
        if context.args:
            try:
                minutes = float(context.args[0])
            except Exception:
                minutes = None
        self.runner.pause(minutes=minutes)
        if minutes:
            update.message.reply_text(f"Entries paused for {minutes} min.")
        else:
            update.message.reply_text("Entries paused.")

    def _cmd_resume(self, update: Update, context: CallbackContext) -> None:
        if not _is_admin(update):
            return
        self.runner.resume()
        update.message.reply_text("Entries resumed.")

    def _cmd_risk(self, update: Update, context: CallbackContext) -> None:
        if not _is_admin(update):
            return
        if not context.args:
            update.message.reply_text(f"Current risk %: {settings.risk.risk_per_trade * 100:.2f}")
            return
        try:
            pct = float(context.args[0])
            settings.risk.risk_per_trade = pct / 100.0
            update.message.reply_text(f"Risk per trade set to {pct:.2f}%")
        except Exception:
            update.message.reply_text("Usage: /risk <percent> (e.g., /risk 0.5)")

    def _cmd_quality(self, update: Update, context: CallbackContext) -> None:
        if not _is_admin(update):
            return
        mode = (context.args[0].lower() if context.args else "auto")
        if hasattr(self.runner.strategy, "set_quality_mode"):
            self.runner.strategy.set_quality_mode(mode)
            update.message.reply_text(f"Quality mode set to {mode}.")
        else:
            update.message.reply_text("Strategy does not support quality mode.")

    def _cmd_regime(self, update: Update, context: CallbackContext) -> None:
        if not _is_admin(update):
            return
        mode = (context.args[0].lower() if context.args else "auto")
        if hasattr(self.runner.strategy, "set_regime_mode"):
            self.runner.strategy.set_regime_mode(mode)
            update.message.reply_text(f"Regime mode set to {mode}.")
        else:
            update.message.reply_text("Strategy does not support regime mode.")

    def _cmd_positions(self, update: Update, context: CallbackContext) -> None:
        pos = self.runner.positions_provider()
        pretty = json.dumps(pos, indent=2, default=str)
        update.message.reply_text(f"```\n{pretty}\n```", parse_mode=ParseMode.MARKDOWN)

    def _cmd_active(self, update: Update, context: CallbackContext) -> None:
        acts = self.runner.actives_provider()
        pretty = json.dumps(acts, indent=2, default=str)
        update.message.reply_text(f"```\n{pretty}\n```", parse_mode=ParseMode.MARKDOWN)

    def _cmd_logs(self, update: Update, context: CallbackContext) -> None:
        n = 200
        if context.args:
            try:
                n = int(context.args[0])
            except Exception:
                pass
        text = self.log_provider(n)
        if not text:
            text = "<no logs>"
        # Telegram message size limit: split if needed
        chunks = [text[i:i+3500] for i in range(0, len(text), 3500)]
        for ch in chunks:
            update.message.reply_text(f"```\n{ch}\n```", parse_mode=ParseMode.MARKDOWN)

    def _cmd_emergency(self, update: Update, context: CallbackContext) -> None:
        if not _is_admin(update):
            return
        ex = getattr(self.runner, "executor", None)
        if not ex:
            update.message.reply_text("No executor.")
            return
        try:
            if hasattr(ex, "flatten_and_cancel_all"):
                ok, msg = ex.flatten_and_cancel_all()
            elif hasattr(ex, "cancel_all"):
                ok, msg = ex.cancel_all()
            else:
                ok, msg = False, "executor has no emergency method"
        except Exception as e:
            ok, msg = False, str(e)
        update.message.reply_text(f"{'✅' if ok else '❌'} {msg}")