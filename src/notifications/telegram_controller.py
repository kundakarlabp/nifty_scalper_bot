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
    """

    def __init__(
        self,
        status_callback: Callable[[], Dict[str, Any]],
        control_callback: Callable[[str, str], bool],
        summary_callback: Callable[[], str],
    ) -> None:
        """
        Initialize Telegram bot.
        :param status_callback: Function that returns bot status
        :param control_callback: Function to handle control commands
        :param summary_callback: Function that returns trade summary
        """
        from src.config import Config

        self.bot_token = Config.TELEGRAM_BOT_TOKEN
        self.chat_id = Config.TELEGRAM_CHAT_ID

        if not self.bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN is required in Config")
        if not self.chat_id:
            raise ValueError("TELEGRAM_CHAT_ID is required in Config")

        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.polling = False
        self.polling_thread: Optional[threading.Thread] = None

        # Callbacks from trader
        self.status_callback = status_callback
        self.control_callback = control_callback
        self.summary_callback = summary_callback

        logger.info("TelegramController initialized with bot token and chat ID.")

    def _send_message(self, text: str, parse_mode: Optional[str] = None) -> bool:
        """
        Internal method to send a message to Telegram.
        :param text: Message text
        :param parse_mode: "Markdown" or "HTML"
        :return: True if successful
        """
        url = f"{self.base_url}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "disable_notification": False,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode

        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                logger.debug("Telegram message sent successfully.")
                return True
            else:
                logger.error("Failed to send Telegram message: %s", response.text)
        except Exception as exc:
            logger.error("Exception sending Telegram message: %s", exc, exc_info=True)
        return False

    def send_message(self, text: str, parse_mode: Optional[str] = None) -> bool:
        """
        Public wrapper to send a message to the user.
        Safe to call from RealTimeTrader or other components.
        """
        return self._send_message(text, parse_mode=parse_mode)

    def send_startup_alert(self) -> None:
        """Send startup notification when bot starts."""
        text = "🟢 Nifty Scalper Bot started.\nAwaiting commands..."
        self._send_message(text)

    def send_realtime_session_alert(self, action: str) -> None:
        """Send START/STOP alert."""
        if action == "START":
            text = "✅ Real-time trading session STARTED."
        elif action == "STOP":
            text = "🛑 Real-time trading session STOPPED."
        else:
            text = f"ℹ️ Session {action}."
        self._send_message(text)

    def send_signal_alert(self, token: int, signal: Dict[str, Any], position: Dict[str, Any]) -> None:
        """Send a trading signal alert."""
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

    def _send_status(self, status: Dict[str, Any]) -> None:
        """Send formatted status message."""
        lines = [
            "📊 <b>Bot Status</b>",
            f"🔁 <b>Trading:</b> {'🟢 Running' if status['is_trading'] else '🔴 Stopped'}",
            f"🌐 <b>Mode:</b> {'🟢 LIVE' if status['live_mode'] else '🛡️ Shadow'}",
            f"📦 <b>Open Orders:</b> {status['open_orders']}",
            f"📈 <b>Trades Today:</b> {status['trades_today']}",
            f"💰 <b>Daily P&L:</b> {status.get('daily_pnl', 0.0):.2f}",
            f"⚖️ <b>Risk Level:</b> {status.get('risk_level', 'N/A')}",
        ]
        text = "\n".join(lines)
        self._send_message(text, parse_mode="HTML")

    def _send_summary(self, summary: str) -> None:
        """Send trade summary."""
        self._send_message(summary, parse_mode="HTML")

    def _handle_command(self, command: str, arg: str = "") -> None:
        """Process incoming Telegram command."""
        logger.info("📩 Received command: '%s %s'", command, arg)

        if command == "status":
            status = self.status_callback()
            self._send_status(status)

        elif command == "summary":
            summary = self.summary_callback()
            self._send_summary(summary)

        elif command in ["start", "stop", "mode"]:
            success = self.control_callback(command, arg)
            # Note: control_callback already sends user feedback
            if not success:
                logger.warning("Command '/%s %s' failed.", command, arg)

        else:
            self._send_message(f"❌ Unknown command: `{command}`\nUse /status or /help", parse_mode="Markdown")

    def _poll_updates(self) -> None:
        """Background thread function to poll for Telegram updates."""
        url = f"{self.base_url}/getUpdates"
        offset = None
        timeout = 30

        logger.info("📡 Telegram polling started. Awaiting commands...")

        while self.polling:
            try:
                payload = {"timeout": timeout, "offset": offset}
                response = requests.get(url, params=payload, timeout=timeout + 5)

                if response.status_code == 200:
                    data = response.json()
                    if data.get("ok") and len(data.get("result", [])) > 0:
                        for result in data["result"]:
                            offset = result["update_id"] + 1
                            message = result.get("message", {})
                            text = message.get("text", "").strip()

                            if text.startswith("/"):
                                cmd_parts = text[1:].split(maxsplit=1)
                                command = cmd_parts[0].lower()
                                arg = cmd_parts[1] if len(cmd_parts) > 1 else ""
                                self._handle_command(command, arg)

                elif response.status_code == 409:
                    logger.error("Conflict: Another webhook or polling instance is active. Stopping.")
                    self.polling = False
                else:
                    logger.error("Telegram getUpdates failed: %s", response.text)

            except requests.exceptions.ReadTimeout:
                logger.debug("Telegram polling timeout — continuing...")
            except Exception as exc:
                logger.error("Error in Telegram polling: %s", exc, exc_info=True)
                time.sleep(5)

        logger.info("🛑 Telegram polling stopped.")

    def start_polling(self) -> None:
        """Start long-polling for Telegram messages."""
        if self.polling:
            logger.warning("Polling already active.")
            return

        self.polling = True
        self._poll_updates()

    def stop_polling(self) -> None:
        """Stop the polling loop gracefully."""
        logger.info("🛑 Stopping Telegram polling...")
        self.polling = False
        if self.polling_thread and self.polling_thread.is_alive():
            self.polling_thread.join(timeout=5)
