import logging
import threading
import time
from typing import Any, Callable, Dict, Optional
import requests

from config import Config

logger = logging.getLogger(__name__)


class TelegramController:
    def __init__(
        self,
        status_callback: Optional[Callable[[], Dict[str, Any]]] = None,
        control_callback: Optional[Callable[[str, str], bool]] = None,
        summary_callback: Optional[Callable[[], str]] = None,
    ) -> None:
        self.bot_token = Config.TELEGRAM_BOT_TOKEN
        self.user_id = Config.TELEGRAM_USER_ID
        self.status_callback = status_callback
        self.control_callback = control_callback
        self.summary_callback = summary_callback
        self.polling_active = False
        self._polling_thread: Optional[threading.Thread] = None
        self._update_offset = 0
        self.awaiting_confirmation = False

    def _send_message(self, text: str) -> None:
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        data = {"chat_id": self.user_id, "text": text}
        try:
            requests.post(url, data=data)
        except Exception as e:
            logger.warning(f"❌ Failed to send message to Telegram: {e}")

    def send_startup_alert(self) -> None:
        """Send startup message when bot starts."""
        self._send_message("🚀 Nifty Scalper Bot has started and is online!")

    def _handle_command(self, command: str) -> None:
        command = command.strip().lower()

        if command == "/start":
            if self.control_callback:
                success = self.control_callback("start", "")
                self._send_message("✅ Bot started!" if success else "❌ Failed to start bot.")
            else:
                self._send_message("⚠️ Start control not configured.")

        elif command == "/stop":
            if self.control_callback:
                success = self.control_callback("stop", "")
                self._send_message("🛑 Bot stopped." if success else "❌ Failed to stop bot.")
            else:
                self._send_message("⚠️ Stop control not configured.")

        elif command == "/status":
            if self.status_callback:
                status = self.status_callback()
                status_msg = (
                    "📊 Status:\n"
                    f"is_trading: {status.get('is_trading')}\n"
                    f"open_orders: {status.get('open_orders')}\n"
                    f"trades_today: {status.get('trades_today')}\n"
                    f"live_mode: {status.get('live_mode')}\n"
                    f"equity: {status.get('equity')}\n"
                    f"equity_peak: {status.get('equity_peak')}\n"
                    f"daily_loss: {status.get('daily_loss')}\n"
                    f"consecutive_losses: {status.get('consecutive_losses')}"
                )
                self._send_message(status_msg)
            else:
                self._send_message("⚠️ Status callback not configured.")

        elif command == "/summary":
            if self.summary_callback:
                summary = self.summary_callback()
                self._send_message(summary)
            else:
                self._send_message("⚠️ Summary callback not configured.")

        elif command.startswith("/mode "):
            mode = command.split(" ", 1)[1]
            if mode in ["live", "shadow"]:
                if self.control_callback:
                    success = self.control_callback("mode", mode)
                    self._send_message(f"✅ Mode switched to: {mode}" if success else "❌ Failed to switch mode.")
                else:
                    self._send_message("⚠️ Mode control not configured.")
            else:
                self._send_message("❌ Invalid mode. Use /mode live or /mode shadow.")

        elif command == "/help":
            help_text = (
                "🤖 Available Commands:\n"
                "/start – begin trading\n"
                "/stop – halt trading\n"
                "/status – show current bot status\n"
                "/summary – show daily P&L summary\n"
                "/mode live|shadow – switch trading mode\n"
                "/help – show this help message"
            )
            self._send_message(help_text)

        else:
            self._send_message("❓ Unknown command. Send /help for the list of commands.")

    def _poll_updates(self) -> None:
        url = f"https://api.telegram.org/bot{self.bot_token}/getUpdates"
        while self.polling_active:
            try:
                response = requests.get(url, params={"offset": self._update_offset + 1, "timeout": 10})
                if response.status_code == 200:
                    updates = response.json()["result"]
                    for update in updates:
                        self._update_offset = update["update_id"]
                        message = update.get("message", {}).get("text")
                        user_id = update.get("message", {}).get("chat", {}).get("id")
                        if message and user_id == self.user_id:
                            logger.info(f"📩 Received command: {message}")
                            self._handle_command(message)
                else:
                    logger.warning(f"⚠️ Telegram API error: {response.status_code}")
            except Exception as e:
                logger.error(f"❌ Error while polling Telegram: {e}")
            time.sleep(2)

    def start_polling(self) -> None:
        if not self.polling_active:
            self.polling_active = True
            self._polling_thread = threading.Thread(target=self._poll_updates, daemon=True)
            self._polling_thread.start()
            logger.info("✅ Telegram polling started.")

    def stop_polling(self) -> None:
        self.polling_active = False
        if self._polling_thread and self._polling_thread.is_alive():
            self._polling_thread.join()
            logger.info("🛑 Telegram polling stopped.")