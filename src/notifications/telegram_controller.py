import logging
import threading
import time
import requests
from typing import Any, Callable, Dict, Optional

from config import Config

logger = logging.getLogger(__name__)


class TelegramController:
    def __init__(
        self,
        status_callback: Optional[Callable[[], Dict[str, Any]]] = None,
        control_callback: Optional[Callable[[str], bool]] = None,
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
            logger.error(f"Failed to send message: {e}")

    def send_startup_alert(self) -> None:
        """Send a startup message to confirm bot is running."""
        try:
            message = "✅ Nifty Scalper Bot is now active and monitoring trades!"
            self._send_message(message)
        except Exception as e:
            logger.error(f"Failed to send startup alert: {e}")

    def _process_command(self, command: str) -> None:
        if command == "/start":
            if self.control_callback:
                success = self.control_callback("start")
                self._send_message("🚀 Trading started!" if success else "❌ Failed to start trading.")
        elif command == "/stop":
            if self.control_callback:
                success = self.control_callback("stop")
                self._send_message("🛑 Trading stopped!" if success else "❌ Failed to stop trading.")
        elif command == "/status":
            if self.status_callback:
                status = self.status_callback()
                status_msg = "\n".join(f"{k}: {v}" for k, v in status.items())
                self._send_message(f"📊 Status:\n{status_msg}")
        elif command == "/summary":
            if self.summary_callback:
                summary = self.summary_callback()
                self._send_message(f"📈 Daily Summary:\n{summary}")
        elif command.startswith("/mode "):
            mode = command.split(" ", 1)[1]
            if self.control_callback:
                success = self.control_callback(f"mode {mode}")
                self._send_message(f"⚙️ Mode switched to: {mode}" if success else "❌ Failed to switch mode.")
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

    def _poll_commands(self) -> None:
        logger.info("📡 Telegram polling thread started.")
        while self.polling_active:
            try:
                url = f"https://api.telegram.org/bot{self.bot_token}/getUpdates"
                params = {"offset": self._update_offset + 1, "timeout": 10}
                response = requests.get(url, params=params, timeout=15)
                updates = response.json().get("result", [])

                for update in updates:
                    self._update_offset = update["update_id"]
                    message = update.get("message", {})
                    text = message.get("text", "")
                    user_id = message.get("from", {}).get("id")

                    if user_id == self.user_id and text.startswith("/"):
                        self._process_command(text.strip())
            except Exception as e:
                logger.warning(f"Polling error: {e}")
            time.sleep(2)

    def start_polling(self) -> None:
        if not self.polling_active:
            self.polling_active = True
            self._polling_thread = threading.Thread(target=self._poll_commands, daemon=True)
            self._polling_thread.start()

    def stop_polling(self) -> None:
        self.polling_active = False
        if self._polling_thread:
            self._polling_thread.join(timeout=5)
            self._polling_thread = None