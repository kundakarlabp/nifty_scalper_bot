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
        self._update_offset = 0  # To avoid fetching old updates

    def _send_message(self, text: str) -> None:
        # ✅ Fixed: Removed space in URL
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        data = {
            "chat_id": self.user_id,
            "text": text,
            "parse_mode": "HTML",  # Optional: for bold/italic formatting
            "disable_web_page_preview": True,
        }
        try:
            response = requests.post(url, data=data, timeout=10)
            if not response.ok:
                logger.warning(
                    f"Telegram send failed: {response.status_code} - {response.text}"
                )
        except Exception as e:
            logger.warning(f"❌ Failed to send message to Telegram: {e}")

    def send_startup_alert(self) -> None:
        self._send_message("🚀 <b>Nifty Scalper Bot</b> has started and is online!")

    def send_realtime_session_alert(self, mode: str) -> None:
        if mode == "START":
            self._send_message("✅ Trading started.")
        elif mode == "STOP":
            self._send_message("🛑 Trading stopped.")
        elif mode == "ALREADY_RUNNING":
            self._send_message("⚠️ Trading is already running.")
        elif mode == "NOT_RUNNING":
            self._send_message("ℹ️ Trading is not currently running.")

    def send_signal_alert(self, token: int, signal: Dict[str, Any], position: Dict[str, Any]) -> None:
        confidence = signal.get("confidence", 0)
        score = signal.get("score", "N/A")
        direction = signal.get("signal") or signal.get("direction", "N/A")
        entry = signal.get("entry_price", "N/A")
        stop_loss = signal.get("stop_loss", "N/A")
        target = signal.get("target", "N/A")
        qty = position.get("quantity", "N/A")

        mode_text = "🟢 LIVE" if Config.ENABLE_LIVE_TRADING else "🛡️ SIM"

        msg = (
            f"📢 <b>Signal #{token}</b>\n"
            f"🎯 <b>Direction:</b> {direction}\n"
            f"📌 <b>Entry:</b> {entry} | <b>SL:</b> {stop_loss} | <b>TP:</b> {target}\n"
            f"📊 <b>Confidence:</b> {confidence:.2f} | <b>Score:</b> {score}\n"
            f"📦 <b>Quantity:</b> {qty}\n"
            f"⚡ <b>Mode:</b> {mode_text}"
        )
        self._send_message(msg)

    def _handle_command(self, command_text: str) -> None:
        text = command_text.strip().lower()
        logger.info(f"📩 Received command: '{text}'")

        try:
            if text == "/start":
                if self.control_callback:
                    success = self.control_callback("start", "")
                    reply = "✅ Trading started." if success else "❌ Failed to start."
                else:
                    reply = "⚠️ Start function not configured."
                self._send_message(reply)

            elif text == "/stop":
                if self.control_callback:
                    success = self.control_callback("stop", "")
                    reply = "🛑 Trading stopped." if success else "❌ Failed to stop."
                else:
                    reply = "⚠️ Stop function not configured."
                self._send_message(reply)

            elif text == "/status":
                if self.status_callback:
                    status = self.status_callback()
                    status_msg = (
                        "📊 <b>Bot Status</b>\n"
                        f"🔁 <b>Trading:</b> {'🟢 Running' if status.get('is_trading') else '🔴 Stopped'}\n"
                        f"📥 <b>Open Orders:</b> {status.get('open_orders', 0)}\n"
                        f"📈 <b>Trades Today:</b> {status.get('trades_today', 0)}\n"
                        f"🧠 <b>Mode:</b> {'🟢 LIVE' if status.get('live_mode') else '🛡️ SIM'}\n"
                        f"💰 <b>Equity:</b> {status.get('equity', 'N/A')}\n"
                        f"📉 <b>Daily Loss:</b> {status.get('daily_loss', 0):.2f}\n"
                        f"🔥 <b>Consecutive Losses:</b> {status.get('consecutive_losses', 0)}"
                    )
                    self._send_message(status_msg)
                else:
                    self._send_message("⚠️ Status callback not available.")

            elif text == "/summary":
                if self.summary_callback:
                    summary = self.summary_callback()
                    self._send_message(f"📋 <b>Daily Summary</b>\n\n{summary}")
                else:
                    self._send_message("⚠️ Summary not available.")

            elif text.startswith("/mode "):
                parts = text.split(" ", 1)
                if len(parts) < 2:
                    self._send_message("❌ Usage: /mode live or /mode shadow")
                    return
                mode = parts[1].strip().lower()
                if mode not in ["live", "shadow"]:
                    self._send_message("❌ Invalid mode. Use <code>/mode live</code> or <code>/mode shadow</code>")
                    return
                if self.control_callback:
                    success = self.control_callback("mode", mode)
                    reply = f"✅ Mode switched to <b>{mode.upper()}</b>." if success else "❌ Failed to switch mode."
                else:
                    reply = "⚠️ Mode control not configured."
                self._send_message(reply)

            elif text == "/help":
                help_text = (
                    "🤖 <b>Available Commands</b>:\n"
                    "<code>/start</code> – Start trading\n"
                    "<code>/stop</code> – Stop trading\n"
                    "<code>/status</code> – Show current status\n"
                    "<code>/summary</code> – Show trade summary\n"
                    "<code>/mode live|shadow</code> – Switch mode\n"
                    "<code>/help</code> – Show this help"
                )
                self._send_message(help_text)

            else:
                self._send_message("❓ Unknown command. Use <code>/help</code> for options.")

        except Exception as e:
            logger.error(f"Error handling command '{text}': {e}", exc_info=True)
            self._send_message("⚠️ Error processing your command.")

    def _poll_updates(self) -> None:
        # ✅ Fixed URL: no space in bot token
        url = f"https://api.telegram.org/bot{self.bot_token}/getUpdates"

        logger.info("📡 Telegram polling loop started...")
        while self.polling_active:
            try:
                response = requests.get(
                    url,
                    params={
                        "offset": self._update_offset + 1,
                        "limit": 100,
                        "timeout": 30,
                    },
                    timeout=45,
                )

                if response.status_code == 200:
                    result = response.json()
                    if not result.get("ok"):
                        logger.warning(f"Telegram API not OK: {result}")
                        time.sleep(5)
                        continue

                    updates = result.get("result", [])
                    for update in updates:
                        update_id = update["update_id"]
                        message = update.get("message")
                        if not message:
                            continue

                        text = message.get("text")
                        user_id = message.get("chat", {}).get("id")

                        # Validate user ID
                        if user_id != self.user_id:
                            logger.warning(f"Ignoring message from unauthorized user: {user_id}")
                            continue

                        if text:
                            # Update offset to prevent reprocessing
                            if update_id > self._update_offset:
                                self._update_offset = update_id
                            # Handle command
                            self._handle_command(text)
                else:
                    logger.warning(f"Telegram getUpdates failed: {response.status_code} - {response.text}")
                    time.sleep(5)

            except requests.exceptions.RequestException as e:
                logger.error(f"Network error during polling: {e}")
                time.sleep(5)
            except Exception as e:
                logger.error(f"Unexpected error in polling loop: {e}", exc_info=True)
                time.sleep(5)

        logger.info("🛑 Telegram polling loop stopped.")

    def start_polling(self) -> None:
        """Start the polling loop in a background thread."""
        if self.polling_active:
            logger.warning("Polling already active.")
            return

        self.polling_active = True
        self._polling_thread = threading.Thread(target=self._poll_updates, daemon=True)
        self._polling_thread.start()
        logger.info("✅ Telegram polling started.")

    def stop_polling(self) -> None:
        """Gracefully stop the polling loop."""
        if not self.polling_active:
            return
        self.polling_active = False
        logger.info("🛑 Stopping Telegram polling...")

        # Wait for polling thread to finish (but not from itself)
        if self._polling_thread and self._polling_thread.is_alive():
            if threading.current_thread() != self._polling_thread:
                self._polling_thread.join(timeout=5)
            # Don't force-kill; let it exit naturally
        logger.info("👋 Telegram polling stopped.")