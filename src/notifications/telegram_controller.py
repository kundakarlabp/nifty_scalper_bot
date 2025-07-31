"""
Telegram notification and command handling.

This module wraps the Telegram Bot API for sending rich messages and
processing user commands.  It uses long polling via ``getUpdates``
instead of webhooks to simplify deployment.  Commands supported:

* ``/start`` – begin trading (invokes the provided control callback)
* ``/stop`` – halt trading
* ``/status`` – get a snapshot of current bot status (via status callback)
* ``/summary`` – daily P&L summary (via summary callback)

The controller can be used independently or integrated into a
``RealTimeTrader`` class.  It is designed to operate even when no
Telegram credentials are configured; in that case all methods become
no‑ops.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, Optional

import requests

from src.config import Config

logger = logging.getLogger(__name__)


class TelegramController:
    """Wrapper around the Telegram bot API."""

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
        # Skip initial polling if no credentials
        if not self.bot_token or not self.user_id:
            logger.warning("Telegram credentials not set. Notifications disabled.")

    # --- Messaging ---
    def _api_url(self, method: str) -> str:
        return f"https://api.telegram.org/bot{self.bot_token}/{method}"

    def send_message(self, text: str, parse_mode: str = "Markdown") -> None:
        """Send a plain text or Markdown message to the configured user."""
        if not self.bot_token or not self.user_id:
            logger.debug("Skipping Telegram message because credentials are missing: %s", text)
            return
        try:
            payload = {
                "chat_id": self.user_id,
                "text": text,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True,
            }
            resp = requests.post(self._api_url("sendMessage"), json=payload, timeout=5)
            if not resp.ok:
                logger.error("Failed to send Telegram message: %s", resp.text)
        except Exception as exc:
            logger.error("Error sending Telegram message: %s", exc, exc_info=True)

    def send_startup_alert(self) -> None:
        self.send_message("🚀 Scalper bot initialised and ready.")

    def send_realtime_session_alert(self, state: str) -> None:
        if state.upper() == "START":
            self.send_message("▶️ Real‑time trading session started.")
        elif state.upper() == "STOP":
            self.send_message("⏹️ Real‑time trading session stopped.")

    def send_signal_alert(self, token: int, signal: Dict[str, Any], position: Dict[str, Any]) -> None:
        """Send a detailed alert when a new trading signal is generated."""
        direction = signal.get("signal")
        score = signal.get("score", 0)
        confidence = signal.get("confidence", 0)
        sl = signal.get("stop_loss")
        target = signal.get("target")
        qty = position.get("quantity") if position else None
        message = (
            f"📈 *New Signal*\n"
            f"Token: `{token}`\n"
            f"Direction: `{direction}`\n"
            f"Score: `{score:.2f}`\n"
            f"Confidence: `{confidence:.1f}/10`\n"
            f"Qty: `{qty}`\n"
            f"Entry: `{signal.get('entry_price'):.2f}`\n"
            f"SL: `{sl:.2f}` | Target: `{target:.2f}`"
        )
        self.send_message(message)

    # --- Polling and command handling ---
    def start_polling(self) -> None:
        """Begin long polling for incoming user messages."""
        if self.polling_active or not self.bot_token:
            return
        logger.info("Starting Telegram polling loop...")
        self.polling_active = True
        while self.polling_active:
            try:
                params = {
                    "timeout": 10,
                    "offset": self._update_offset,
                }
                resp = requests.get(self._api_url("getUpdates"), params=params, timeout=15)
                if not resp.ok:
                    logger.error("Telegram getUpdates failed: %s", resp.text)
                    time.sleep(5)
                    continue
                data = resp.json()
                for update in data.get("result", []):
                    self._update_offset = update["update_id"] + 1
                    message = update.get("message") or {}
                    chat_id = message.get("chat", {}).get("id")
                    if chat_id != self.user_id:
                        continue
                    text = (message.get("text") or "").strip().lower()
                    if text.startswith("/start"):
                        self._handle_start()
                    elif text.startswith("/stop"):
                        self._handle_stop()
                    elif text.startswith("/status"):
                        self._handle_status()
                    elif text.startswith("/summary"):
                        self._handle_summary()
                # Short pause to avoid spamming Telegram
                time.sleep(1)
            except Exception as exc:
                logger.error("Error in Telegram polling loop: %s", exc, exc_info=True)
                time.sleep(5)
        logger.info("Telegram polling stopped.")

    def stop_polling(self) -> None:
        """Stop the polling loop."""
        self.polling_active = False

    # --- Command handlers ---
    def _handle_start(self) -> None:
        self.send_message("▶️ Start command received.")
        if self.control_callback:
            result = self.control_callback("start")
            if result:
                self.send_message("✅ Trading started.")
            else:
                self.send_message("⚠️ Failed to start trading.")

    def _handle_stop(self) -> None:
        self.send_message("⏹️ Stop command received.")
        if self.control_callback:
            result = self.control_callback("stop")
            if result:
                self.send_message("✅ Trading stopped.")
            else:
                self.send_message("⚠️ Failed to stop trading.")

    def _handle_status(self) -> None:
        if self.status_callback:
            status = self.status_callback()
            status_lines = [f"*{k}*: `{v}`" for k, v in status.items()]
            message = "📊 *Status*\n" + "\n".join(status_lines)
            self.send_message(message)
        else:
            self.send_message("ℹ️ Status unavailable.")

    def _handle_summary(self) -> None:
        if self.summary_callback:
            summary = self.summary_callback()
            self.send_message(f"📈 *Daily Summary*\n{summary}")
        else:
            self.send_message("ℹ️ Summary unavailable.")
