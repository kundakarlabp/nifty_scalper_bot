# src/notifications/telegram_controller.py
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, Optional

import requests
from requests.adapters import HTTPAdapter
try:
    # available via requests' dependency
    from urllib3.util.retry import Retry  # type: ignore
except Exception:  # pragma: no cover
    Retry = None  # graceful degrade

from src.config import TelegramConfig

logger = logging.getLogger(__name__)


class TelegramController:
    """
    Telegram bot interface for receiving commands and sending alerts.

    Public callbacks (injected):
      - status_callback() -> dict
      - control_callback(command: str, arg: str) -> bool
      - summary_callback() -> str

    Commands:
      /start, /stop, /mode live|shadow, /status, /summary,
      /refresh, /health, /emergency, /help
    """

    def __init__(
        self,
        config: TelegramConfig,
        status_callback: Callable[[], Dict[str, Any]],
        control_callback: Callable[[str, str], bool],
        summary_callback: Callable[[], str],
    ) -> None:
        if not isinstance(config, TelegramConfig):
            raise TypeError("A valid TelegramConfig instance is required.")

        self.config = config
        self.base_url = f"https://api.telegram.org/bot{self.config.bot_token}"
        self.polling = False
        self.polling_thread: Optional[threading.Thread] = None

        # Callbacks from trader
        self.status_callback = status_callback
        self.control_callback = control_callback
        self.summary_callback = summary_callback

        # Single session with retries
        self.session = requests.Session()
        if Retry is not None:
            retry = Retry(
                total=5,
                read=5,
                connect=5,
                backoff_factor=0.5,
                status_forcelist=(429, 500, 502, 503, 504),
                allowed_methods=frozenset({"GET", "POST"}),
                raise_on_status=False,
            )
            adapter = HTTPAdapter(max_retries=retry)
            self.session.mount("https://", adapter)
            self.session.mount("http://", adapter)

        logger.info("TelegramController initialized.")

    # ---------- low-level HTTP helpers ----------
    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        timeout: int = 35,
    ) -> Optional[requests.Response]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        try:
            resp = self.session.request(method=method, url=url, params=params, json=json, timeout=timeout)
            # Handle rate limiting explicitly
            if resp.status_code == 429:
                try:
                    retry_after = resp.json().get("parameters", {}).get("retry_after", 1)
                except Exception:
                    retry_after = 1
                retry_after = max(1, int(retry_after))
                logger.warning("Telegram 429 rate limit; sleeping %ss then retrying once.", retry_after)
                time.sleep(retry_after)
                return self.session.request(method=method, url=url, params=params, json=json, timeout=timeout)
            return resp
        except requests.exceptions.RequestException as exc:
            logger.warning("Telegram HTTP error: %s %s (%s)", method, path, exc)
            return None

    # ---------- low-level send ----------
    def _send_message(self, text: str, parse_mode: Optional[str] = None, retries: int = 2, backoff_sec: float = 1.0) -> bool:
        payload: Dict[str, Any] = {
            "chat_id": self.config.chat_id,
            "text": text,
            "disable_notification": False,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode

        # Make sure we have a chat_id
        if not str(self.config.chat_id).strip():
            logger.error("Telegram chat_id is not configured. Cannot send message.")
            return False

        attempt = 0
        while attempt <= retries:
            attempt += 1
            resp = self._request("POST", "sendMessage", json=payload, timeout=15)
            if resp is None:
                # network issue; backoff
                if attempt <= retries:
                    time.sleep(backoff_sec)
                continue

            if resp.status_code == 200:
                logger.debug("Telegram message sent successfully.")
                return True

            # Log and retry on server errors
            try:
                body = resp.json()
            except Exception:
                body = {"text": resp.text}
            logger.warning("sendMessage failed (%s): %s", resp.status_code, body)

            if attempt <= retries and resp.status_code >= 500:
                time.sleep(backoff_sec)

            else:
                break

        logger.error("Failed to send Telegram message after %d attempts.", retries + 1)
        return False

    # ---------- public send helpers ----------
    def send_message(self, text: str, parse_mode: Optional[str] = None) -> bool:
        return self._send_message(text, parse_mode=parse_mode)

    def send_startup_alert(self) -> None:
        text = "🟢 Nifty Scalper Bot started.\nType /help for commands. Awaiting instructions…"
        self._send_message(text)

    def send_realtime_session_alert(self, action: str) -> None:
        if action.upper() == "START":
            text = "✅ Real-time trading session STARTED."
        elif action.upper() == "STOP":
            text = "🛑 Real-time trading session STOPPED."
        else:
            text = f"ℹ️ Session {action}."
        self._send_message(text)

    # Compatibility alias
    def send_alert(self, action: str) -> None:
        self.send_realtime_session_alert(action)

    def send_signal_alert(self, token: int, signal: Dict[str, Any], position: Dict[str, Any]) -> None:
        direction = signal.get("signal") or signal.get("direction", "ENTRY")
        entry = signal.get("entry_price", "N/A")
        sl = signal.get("stop_loss", "N/A")
        target = signal.get("target", "N/A")
        conf = float(signal.get("confidence", 0.0) or 0.0)
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
    def _send_status(self, status: Any) -> None:
        if isinstance(status, str):
            self._send_message(f"📊 Status:\n{status}")
            return

        is_trading = bool(status.get("is_trading", False))
        live_mode = bool(status.get("live_mode", False))

        open_positions = status.get("open_positions")
        if open_positions is None:
            open_positions = status.get("open_orders", 0)

        trades_today = status.get("closed_today")
        if trades_today is None:
            trades_today = status.get("trades_today", 0)

        daily_pnl = float(status.get("daily_pnl", 0.0) or 0.0)

        lines = [
            "📊 <b>Bot Status</b>",
            f"🔁 <b>Trading:</b> {'🟢 Running' if is_trading else '🔴 Stopped'}",
            f"🌐 <b>Mode:</b> {'🟢 LIVE' if live_mode else '🛡️ Shadow'}",
            f"📦 <b>Open Positions:</b> {open_positions}",
            f"📈 <b>Closed Today:</b> {trades_today}",
            f"💰 <b>Daily P&L:</b> {daily_pnl:.2f}",
        ]
        acct = status.get("account_size")
        sess = status.get("session_date")
        if acct is not None:
            lines.append(f"🏦 <b>Acct Size:</b> ₹{acct}")
        if sess is not None:
            lines.append(f"📅 <b>Session:</b> {sess}")

        self._send_message("\n".join(lines), parse_mode="HTML")

    def _send_summary(self, summary: str) -> None:
        self._send_message(summary, parse_mode="HTML")

    # ---------- command router ----------
    def _handle_command(self, command: str, arg: str = "", chat_id_hint: Optional[int] = None) -> None:
        logger.info("📩 Received command: '%s %s'", command, arg)

        # Learn chat_id if unset (first inbound message)
        if (not str(self.config.chat_id).strip()) and chat_id_hint is not None:
            self.config.chat_id = chat_id_hint  # type: ignore[attr-defined]
            logger.info("📌 Learned TELEGRAM_CHAT_ID=%s from inbound message.", chat_id_hint)

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
            self._send_status(self.status_callback())
            return

        if command == "summary":
            self._send_summary(self.summary_callback())
            return

        if command == "mode":
            v = (arg or "").strip().lower()
            if v not in ("live", "shadow"):
                self._send_message("Usage: /mode <live|shadow>")
                return
            ok = self.control_callback("mode", v)
            self._send_message("✅ Mode set to LIVE." if ok and v == "live" else
                               "✅ Mode set to SHADOW." if ok else
                               "⚠️ Failed to set mode.")
            return

        if command in ("start", "stop", "refresh", "health", "emergency"):
            ok = self.control_callback(command, arg)
            self._send_message("✅ Done." if ok else f"⚠️ Command '/{command} {arg}' failed.")
            return

        self._send_message(
            "❌ Unknown command.\n"
            "Try: /start, /stop, /mode live, /mode shadow, /status, /summary, /refresh, /health, /emergency, /help"
        )

    # ---------- polling loop ----------
    def _poll_updates(self) -> None:
        url_path = "getUpdates"
        offset: Optional[int] = None
        timeout = 30

        logger.info("📡 Telegram polling started. Awaiting commands...")
        while self.polling:
            try:
                params = {"timeout": timeout, "offset": offset, "allowed_updates": ["message", "edited_message"]}
                resp = self._request("GET", url_path, params=params, timeout=timeout + 5)
                if resp is None:
                    time.sleep(1)
                    continue

                if resp.status_code == 200:
                    data = resp.json()
                    results = data.get("result", []) if data.get("ok") else []
                    for result in results:
                        offset = int(result["update_id"]) + 1
                        message = result.get("message") or result.get("edited_message") or {}
                        text = (message.get("text") or "").strip()
                        if not text.startswith("/"):
                            continue
                        cmd_parts = text[1:].split(maxsplit=1)
                        command = cmd_parts[0].lower()
                        arg = cmd_parts[1] if len(cmd_parts) > 1 else ""
                        chat = message.get("chat") or {}
                        chat_id = chat.get("id")
                        self._handle_command(command, arg, chat_id_hint=chat_id)

                elif resp.status_code == 409:
                    logger.error("409 Conflict: another webhook/poller is active. Stopping polling.")
                    self.polling = False

                else:
                    try:
                        body = resp.json()
                    except Exception:
                        body = {"text": resp.text}
                    logger.error("Telegram getUpdates failed (%s): %s", resp.status_code, body)
                    time.sleep(2)

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
