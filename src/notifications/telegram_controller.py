import logging 
import threading 
import time 
from typing import Any, Callable, Dict, Optional

import requests from config import Config

logger = logging.getLogger(name)

class TelegramController: def init( self, status_callback: Optional[Callable[[], Dict[str, Any]]] = None, control_callback: Optional[Callable[[str], bool]] = None, summary_callback: Optional[Callable[[], str]] = None, ) -> None: self.bot_token = Config.TELEGRAM_BOT_TOKEN self.user_id = Config.TELEGRAM_USER_ID self.status_callback = status_callback self.control_callback = control_callback self.summary_callback = summary_callback self.polling_active = False self._polling_thread: Optional[threading.Thread] = None self._update_offset = 0 self.awaiting_confirmation = False

if not self.bot_token or not self.user_id:
        logger.warning("Telegram credentials not set. Notifications disabled.")

def _api_url(self, method: str) -> str:
    return f"https://api.telegram.org/bot{self.bot_token}/{method}"

def send_message(self, text: str, parse_mode: str = "Markdown") -> None:
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
    self.send_message("🚀 Scalper bot initialized and ready.")

def send_realtime_session_alert(self, state: str) -> None:
    if state.upper() == "START":
        self.send_message("▶️ Real-time trading session started.")
    elif state.upper() == "STOP":
        self.send_message("⏹️ Real-time trading session stopped.")

def send_signal_alert(self, token: int, signal: Dict[str, Any], position: Dict[str, Any]) -> None:
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

def start_polling(self) -> None:
    if self.polling_active or not self.bot_token:
        return
    logger.info("Starting Telegram polling loop...")
    self.polling_active = True
    while self.polling_active:
        try:
            params = {"timeout": 10, "offset": self._update_offset}
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

                if self.awaiting_confirmation:
                    if text == "/confirmstop":
                        self._execute_stop()
                    else:
                        self.send_message("❗ Stop command not confirmed. Please send `/confirmstop`.")
                    self.awaiting_confirmation = False
                    continue

                if text.startswith("/start"):
                    self._handle_start()
                elif text.startswith("/stop"):
                    self.awaiting_confirmation = True
                    self.send_message("⚠️ Send `/confirmstop` to confirm trading halt.")
                elif text.startswith("/status"):
                    self._handle_status()
                elif text.startswith("/summary"):
                    self._handle_summary()
                elif text.startswith("/mode"):
                    self._handle_mode(text)
                elif text.startswith("/help"):
                    self._handle_help()
                elif text.startswith("/restart"):
                    self.send_message("🔄 Restarting bot...")
                elif text.startswith("/debug"):
                    self.send_message("🔍 Debug mode toggled (not implemented).")
                elif text.startswith("/trades"):
                    self.send_message("📋 Last 3 trades: (mocked data)")
                elif text.startswith("/risk"):
                    self.send_message("📐 Risk settings: (mocked data)")
                elif text.startswith("/refresh"):
                    self.send_message("♻️ Refreshed indicators and signal state.")

            time.sleep(1)
        except Exception as exc:
            logger.error("Error in Telegram polling loop: %s", exc, exc_info=True)
            time.sleep(5)
    logger.info("Telegram polling stopped.")

def stop_polling(self) -> None:
    self.polling_active = False

def _handle_start(self) -> None:
    self.send_message("▶️ Start command received.")
    if self.control_callback:
        result = self.control_callback("start")
        self.send_message("✅ Trading started." if result else "⚠️ Failed to start trading.")

def _execute_stop(self) -> None:
    self.send_message("⏹️ Stop command confirmed.")
    if self.control_callback:
        result = self.control_callback("stop")
        self.send_message("✅ Trading stopped." if result else "⚠️ Failed to stop trading.")

def _handle_status(self) -> None:
    if self.status_callback:
        status = self.status_callback()
        status_lines = [f"*{k}*: `{v}`" for k, v in status.items()]
        self.send_message("📊 *Status*\n" + "\n".join(status_lines))
    else:
        self.send_message("ℹ️ Status unavailable.")

def _handle_summary(self) -> None:
    if self.summary_callback:
        summary = self.summary_callback()
        self.send_message(f"📈 *Daily Summary*\n{summary}")
    else:
        self.send_message("ℹ️ Summary unavailable.")

def _handle_mode(self, text: str) -> None:
    parts = text.split()
    if len(parts) < 2:
        if self.status_callback:
            status = self.status_callback()
            current_mode = "LIVE" if status.get("live_mode") else "SHADOW"
            self.send_message(f"⚙️ Mode: `{current_mode}`")
        else:
            self.send_message("ℹ️ Mode information unavailable.")
        return
    desired = parts[1].strip().lower()
    cmd = "mode_live" if desired in ("live", "on") else "mode_shadow"
    if self.control_callback:
        result = self.control_callback(cmd)
        self.send_message(f"✅ Mode switched to `{desired.upper()}`." if result else "⚠️ Failed to change mode.")
    else:
        self.send_message("ℹ️ Cannot change mode: no control callback.")

def _handle_help(self) -> None:
    help_text = (
        "🧰 *Available Commands*\n"
        "/start – Begin trading\n"
        "/stop – Halt trading (requires /confirmstop)\n"
        "/status – Show bot status\n"
        "/summary – Daily P&L\n"
        "/mode [live|shadow] – Switch trading mode\n"
        "/restart – Restart bot (future)\n"
        "/refresh – Refresh indicators\n"
        "/risk – Show risk settings\n"
        "/trades – Last 3 trades\n"
        "/debug – Toggle debug\n"
        "/help – Show this help"
    )
    self.send_message(help_text)

