# src/notifications/telegram_controller.py
import logging
import threading
import time
from typing import Any, Callable, Dict, Optional
import requests

logger = logging.getLogger(__name__)


class TelegramController:
    """
    Telegram bot interface for receiving commands and sending alerts.
    Communicates with RealTimeTrader via callback functions.

    Public API (unchanged):
      - __init__(status_callback, control_callback, summary_callback)
      - send_message(text, parse_mode=None)
      - send_startup_alert()
      - send_realtime_session_alert(action)
      - send_signal_alert(token, signal, position)
      - start_polling()
      - stop_polling()
    """

    def __init__(
        self,
        status_callback: Callable[[], Dict[str, Any]],
        control_callback: Callable[[str, str], bool],
        summary_callback: Callable[[], str],
    ) -> None:
        """Initialize Telegram bot configuration and callbacks."""
        from src.config import Config

        self.bot_token = Config.TELEGRAM_BOT_TOKEN
        self.chat_id = Config.TELEGRAM_CHAT_ID

        if not self.bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN is required in Config")
        if not self.chat_id:
            raise ValueError("TELEGRAM_CHAT_ID is required in Config")

        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self._session = requests.Session()
        self._session.headers.update({"Connection": "keep-alive"})

        # Polling state
        self._polling = False
        self._polling_thread: Optional[threading.Thread] = None

        # Callbacks
        self.status_callback = status_callback
        self.control_callback = control_callback
        self.summary_callback = summary_callback

        # Internal
        self._offset: Optional[int] = None
        self._timeout_s: int = 30  # long-poll timeout
        self._backoff_s: int = 1   # grows up to 10s on errors

        logger.info("TelegramController initialized with bot token and chat ID.")

    # -------------------------- sending messages -------------------------- #

    def _send_message(self, text: str, parse_mode: Optional[str] = None) -> bool:
        """Internal helper to send a message to Telegram."""
        url = f"{self.base_url}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "disable_notification": False,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode

        try:
            resp = self._session.post(url, json=payload, timeout=10)
            if resp.status_code == 200 and resp.json().get("ok"):
                logger.debug("Telegram message sent.")
                return True
            logger.error("sendMessage failed [%s]: %s", resp.status_code, resp.text)
        except Exception as exc:
            logger.error("sendMessage exception: %s", exc, exc_info=True)
        return False

    def send_message(self, text: str, parse_mode: Optional[str] = None) -> bool:
        """Public wrapper to send a message."""
        return self._send_message(text, parse_mode=parse_mode)

    def send_startup_alert(self) -> None:
        self._send_message("🟢 Nifty Scalper Bot started.\nAwaiting commands...")

    def send_realtime_session_alert(self, action: str) -> None:
        if action == "START":
            text = "✅ Real-time trading session STARTED."
        elif action == "STOP":
            text = "🛑 Real-time trading session STOPPED."
        else:
            text = f"ℹ️ Session {action}."
        self._send_message(text)

    def send_signal_alert(self, token: int, signal: Dict[str, Any], position: Dict[str, Any]) -> None:
        direction = signal.get("signal") or signal.get("direction", "ENTRY")
        entry = signal.get("entry_price", "N/A")
        sl = signal.get("stop_loss", "N/A")
        target = signal.get("target", "N/A")
        conf = signal.get("confidence", 0.0)
        qty = position.get("quantity", "N/A")

        text = (
            f"🔥 <b>NEW SIGNAL #{token}</b>\n"
            f"🎯 Direction: <b>{direction}</b>\n"
            f"💰 Entry: <code>{entry}</code>\n"
            f"📉 Stop Loss: <code>{sl}</code>\n"
            f"🎯 Target: <code>{target}</code>\n"
            f"📊 Confidence: <code>{conf:.2f}</code>\n"
            f"🧮 Quantity: <code>{qty}</code>"
        )
        self._send_message(text, parse_mode="HTML")

    # --------------------------- command handling -------------------------- #

    def _send_status(self, status: Dict[str, Any]) -> None:
        lines = [
            "📊 <b>Bot Status</b>",
            f"🔁 <b>Trading:</b> {'🟢 Running' if status.get('is_trading') else '🔴 Stopped'}",
            f"🌐 <b>Mode:</b> {'🟢 LIVE' if status.get('live_mode') else '🛡️ Shadow'}",
            f"📦 <b>Open Orders:</b> {status.get('open_orders', 0)}",
            f"📈 <b>Trades Today:</b> {status.get('trades_today', 0)}",
            f"💰 <b>Daily P&L:</b> {status.get('total_pnl', status.get('daily_pnl', 0.0)):.2f}",
            f"🕒 <b>Last Update:</b> {status.get('last_update', 'N/A')}",
        ]
        self._send_message("\n".join(lines), parse_mode="HTML")

    def _send_summary(self, summary: str) -> None:
        self._send_message(summary, parse_mode="HTML")

    def _send_help(self) -> None:
        help_text = (
            "🤖 <b>Nifty Scalper Bot – Commands</b>\n"
            "/start – start trading\n"
            "/stop – stop trading\n"
            "/mode live|shadow – switch mode\n"
            "/status – show status\n"
            "/summary – recent trades\n"
            "/help – this help"
        )
        self._send_message(help_text, parse_mode="HTML")

    def _handle_command(self, command: str, arg: str = "") -> None:
        logger.info("📩 Command: '%s %s'", command, arg)

        if command == "status":
            try:
                status = self.status_callback()
            except Exception as e:
                logger.error("status_callback failed: %s", e, exc_info=True)
                self._send_message("❌ Failed to fetch status.")
                return
            self._send_status(status)

        elif command == "summary":
            try:
                summary = self.summary_callback()
            except Exception as e:
                logger.error("summary_callback failed: %s", e, exc_info=True)
                self._send_message("❌ Failed to fetch summary.")
                return
            self._send_summary(summary)

        elif command in ["start", "stop", "mode"]:
            try:
                ok = self.control_callback(command, arg)
                if not ok:
                    self._send_message(f"❌ Command '/{command} {arg}' failed.")
            except Exception as e:
                logger.error("control_callback failed: %s", e, exc_info=True)
                self._send_message(f"❌ Error handling '/{command} {arg}'.")
        elif command == "help":
            self._send_help()
        else:
            self._send_message(f"❌ Unknown command: `{command}`\nUse /help", parse_mode="Markdown")

    # ---------------------------- polling thread --------------------------- #

    def _poll_loop(self) -> None:
        """Long-poll loop (runs in a daemon thread)."""
        url = f"{self.base_url}/getUpdates"
        self._backoff_s = 1
        logger.info("📡 Telegram polling started. Awaiting commands...")

        while self._polling:
            try:
                payload = {"timeout": self._timeout_s}
                if self._offset is not None:
                    payload["offset"] = self._offset

                resp = self._session.get(url, params=payload, timeout=self._timeout_s + 5)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("ok"):
                        updates = data.get("result", [])
                        if updates:
                            for upd in updates:
                                self._offset = upd["update_id"] + 1
                                message = upd.get("message") or upd.get("edited_message") or {}
                                text = (message.get("text") or "").strip()
                                if text.startswith("/"):
                                    parts = text[1:].split(maxsplit=1)
                                    cmd = parts[0].lower()
                                    arg = parts[1] if len(parts) > 1 else ""
                                    self._handle_command(cmd, arg)
                        # reset backoff if we had a good cycle
                        self._backoff_s = 1
                elif resp.status_code == 409:
                    # Another webhook/poller active
                    logger.error("409 Conflict: webhook or another poller active. Stopping polling.")
                    break
                else:
                    logger.error("getUpdates failed [%s]: %s", resp.status_code, resp.text)
                    time.sleep(self._backoff_s)
                    self._backoff_s = min(self._backoff_s * 2, 10)

            except requests.exceptions.ReadTimeout:
                # benign: long-poll timeout
                continue
            except Exception as exc:
                logger.error("Polling error: %s", exc, exc_info=True)
                time.sleep(self._backoff_s)
                self._backoff_s = min(self._backoff_s * 2, 10)

        logger.info("🛑 Telegram polling stopped.")

    def start_polling(self) -> None:
        """Start long-polling in a background daemon thread (non-blocking)."""
        if self._polling:
            logger.warning("Telegram polling already active.")
            return
        self._polling = True
        self._polling_thread = threading.Thread(target=self._poll_loop, name="TG-Poller", daemon=True)
        self._polling_thread.start()

    def stop_polling(self) -> None:
        """Signal the polling loop to stop and join the thread."""
        if not self._polling:
            logger.info("Telegram polling already stopped.")
            return
        logger.info("🛑 Stopping Telegram polling...")
        self._polling = False
        if self._polling_thread and self._polling_thread.is_alive():
            # Wait briefly for graceful exit
            self._polling_thread.join(timeout=5)
        self._polling_thread = None
