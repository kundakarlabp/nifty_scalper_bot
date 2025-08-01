
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
            logger.error(f"Telegram send_message failed: {e}")

    def send_startup_alert(self) -> None:
    """Send a startup message to confirm bot is running."""
    try:
        message = "✅ Nifty Scalper Bot is now active and monitoring trades!"
        self._send_message(message)
    except Exception as e:
        logging.error(f"Failed to send startup alert: {e}")

    def _handle_command(self, text: str) -> None:
        logger.info(f"Telegram command received: {text}")
        cmd = text.strip().lower()

        if cmd == "/start":
            if self.control_callback and self.control_callback("start"):
                self._send_message("âœ… Trading started.")
            else:
                self._send_message("âš ï¸ Could not start trading.")
        elif cmd == "/stop":
            if self.control_callback and self.control_callback("stop"):
                self._send_message("ðŸ›‘ Trading stopped.")
            else:
                self._send_message("âš ï¸ Could not stop trading.")
        elif cmd == "/status":
            if self.status_callback:
                status = self.status_callback()
                formatted = "\n".join(f"{k}: {v}" for k, v in status.items())
                self._send_message(f"ðŸ“Š Bot Status:\n{formatted}")
            else:
                self._send_message("âš ï¸ Status not available.")
        elif cmd == "/summary":
            if self.summary_callback:
                summary = self.summary_callback()
                self._send_message(f"ðŸ“ˆ Daily Summary:\n{summary}")
            else:
                self._send_message("âš ï¸ No summary available.")
        elif cmd.startswith("/mode"):
            _, _, mode = cmd.partition(" ")
            if mode in ("live", "shadow"):
                if self.control_callback and self.control_callback(f"mode {mode}"):
                    self._send_message(f"âš™ï¸ Mode switched to {mode}.")
                else:
                    self._send_message("âŒ Could not change mode.")
            else:
                self._send_message("Usage: /mode live|shadow")
        elif cmd == "/help":
            self._send_message(
                "ðŸ¤– Available Commands:\n"
                "/start â€“ begin trading\n"
                "/stop â€“ halt trading\n"
                "/status â€“ show current bot status\n"
                "/summary â€“ show daily P&L summary\n"
                "/mode live|shadow â€“ switch trading mode\n"
                "/help â€“ show this help message"
            )
        else:
            self._send_message("â“ Unknown command. Use /help")

    def _poll_loop(self) -> None:
        logger.info("Telegram polling started.")
        self.polling_active = True
        while self.polling_active:
            url = f"https://api.telegram.org/bot{self.bot_token}/getUpdates"
            params = {"offset": self._update_offset + 1, "timeout": 30}
            try:
                response = requests.get(url, params=params, timeout=35)
                updates = response.json().get("result", [])
                for update in updates:
                    self._update_offset = update["update_id"]
                    msg = update.get("message", {})
                    if str(msg.get("chat", {}).get("id")) != str(self.user_id):
                        continue
                    if "text" in msg:
                        self._handle_command(msg["text"])
            except Exception as e:
                logger.warning(f"Polling error: {e}")
            time.sleep(1)

    def start_polling(self) -> None:
        if not self._polling_thread:
            self._polling_thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._polling_thread.start()

    def stop_polling(self) -> None:
        self.polling_active = False
        if self._polling_thread:
            self._polling_thread.join()
            self._polling_thread = None