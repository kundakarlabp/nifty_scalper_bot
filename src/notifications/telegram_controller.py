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

    Exposed commands:
      /start, /stop, /mode live|shadow, /status, /summary,
      /refresh, /health, /emergency, /help
    """

    def __init__(
        self,
        status_callback: Callable[[], Dict[str, Any]],
        control_callback: Callable[[str, str], bool],
        summary_callback: Callable[[], str],
    ) -> None:
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

    # ---------- low-level send ----------
    def _send_message(self, text: str, parse_mode: Optional[str] = None) -> bool:
        url = f"{self.base_url}/sendMessage"
        payload: Dict[str, Any] = {
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

    # ---------- public send helpers ----------
    def send_message(self, text: str, parse_mode: Optional[str] = None) -> bool:
        return self._send_message(text, parse_mode=parse_mode)

    def send_startup_alert(self) -> None:
        text = "🟢 Nifty Scalper Bot started.\nType /help for commands. Awaiting instructions…"
        self._send_message(text)

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

    # ---------- status/summary formatting ----------
    def _send_status(self, status: Dict[str, Any]) -> None:
        lines = [
            "📊 <b>Bot Status</b>",
            f"🔁 <b>Trading:</b> {'🟢 Running' if status.get('is_trading') else '🔴 Stopped'}",
            f"🌐 <b>Mode:</b> {'🟢 LIVE' if status.get('live_mode') else '🛡️ Shadow'}",
            f"📦 <b>Open Orders:</b> {status.get('open_orders', 0)}",
            f"📈 <b>Trades Today:</b> {status.get('trades_today', 0)}",
            f"💰 <b>Daily P&L:</b> {status.get('daily_pnl', 0.0):.2f}",
            f"⚖️ <b>Risk Level:</b> {status.get('risk_level', 'N/A')}",
        ]
        text = "\n".join(lines)
        self._send_message(text, parse_mode="HTML")

    def _send_summary(self, summary: str) -> None:
        self._send_message(summary, parse_mode="HTML")

    # ---------- command router ----------
    def _handle_command(self, command: str, arg: str = "") -> None:
        logger.info("📩 Received command: '%s %s'", command, arg)

        if command == "help":
            self._send_message(
                "🤖 <b>Commands</b>\n"
                "/start – start trading\n"
                "/stop – stop trading\n"
                "/mode live|shadow – switch mode\n"
                "/status – bot status\n"
                "/summary – daily summary\n"
                "/refresh – refresh instruments cache\n"
                "/health – system health\n"
                "/emergency – stop & cancel orders",
                parse_mode="HTML",
            )
            return

        if command == "status":
            status = self.status_callback()
            self._send_status(status)
            return

        if command == "summary":
            summary = self.summary_callback()
            self._send_summary(summary)
            return

        if command in ["start", "stop", "mode", "refresh", "health", "emergency"]:
            success = self.control_callback(command, arg)
            if not success:
                logger.warning("Command '/%s %s' failed.", command, arg)
                self._send_message(f"⚠️ Command '/{command} {arg}' failed.")
            return

        self._send_message(
            "❌ Unknown command.\n"
            "Try: /start, /stop, /mode live, /mode shadow, /status, /summary, /refresh, /health, /emergency, /help"
        )

    # ---------- polling loop ----------
    def _poll_updates(self) -> None:
        url = f"{self.base_url}/getUpdates"
        offset: Optional[int] = None
        timeout = 30

        logger.info("📡 Telegram polling started. Awaiting commands...")
        while self.polling:
            try:
                payload = {"timeout": timeout, "offset": offset}
                response = requests.get(url, params=payload, timeout=timeout + 5)

                if response.status_code == 200:
                    data = response.json()
                    results = data.get("result", []) if data.get("ok") else []
                    for result in results:
                        offset = result["update_id"] + 1
                        message = result.get("message") or result.get("edited_message") or {}
                        text = (message.get("text") or "").strip()
                        if not text.startswith("/"):
                            continue
                        cmd_parts = text[1:].split(maxsplit=1)
                        command = cmd_parts[0].lower()
                        arg = cmd_parts[1] if len(cmd_parts) > 1 else ""
                        self._handle_command(command, arg)

                elif response.status_code == 409:
                    logger.error("Conflict: Another webhook or polling instance is active. Stopping.")
                    self.polling = False

                else:
                    logger.error("Telegram getUpdates failed (%s): %s", response.status_code, response.text)

            except requests.exceptions.ReadTimeout:
                logger.debug("Telegram polling timeout — continuing...")
            except Exception as exc:
                logger.error("Error in Telegram polling: %s", exc, exc_info=True)
                time.sleep(5)

        logger.info("🛑 Telegram polling stopped.")

    # ---------- lifecycle ----------
    def start_polling(self) -> None:
        """Start long-polling for Telegram messages in a background thread."""
        if self.polling:
            logger.warning("Polling already active.")
            return

        self.polling = True
        self.polling_thread = threading.Thread(
            target=self._poll_updates,
            name="TelegramPolling",
            daemon=True,
        )
        self.polling_thread.start()

    def stop_polling(self) -> None:
        """Stop the polling loop gracefully."""
        logger.info("🛑 Stopping Telegram polling...")
        self.polling = False
        if self.polling_thread and self.polling_thread.is_alive():
            self.polling_thread.join(timeout=5)
        self.polling_thread = None