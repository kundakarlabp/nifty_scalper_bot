from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, Optional

import requests

from src.config import Config

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
        self.send_message("\ud83d\ude80 Scalper bot initialised and ready.")

    def send_realtime_session_alert(self, state: str) -> None:
        if state.upper() == "START":
            self.send_message("\u25b6\ufe0f Real-time trading session started.")
        elif state.upper() == "STOP":
            self.send_message("\u23f9\ufe0f Real-time trading session stopped.")

    def send_signal_alert(self, token: int, signal: Dict[str, Any], position: Dict[str, Any]) -> None:
        direction = signal.get("signal")
        score = signal.get("score", 0)
        confidence = signal.get("confidence", 0)
        sl = signal.get("stop_loss")
        target = signal.get("target")
        qty = position.get("quantity") if position else None
        message = (
            f"\ud83d\udcc8 *New Signal*\n"
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
                    if text.startswith("/start"):
                        self._handle_start()
                    elif text.startswith("/stop"):
                        self._handle_stop()
                    elif text.startswith("/status"):
                        self._handle_status()
                    elif text.startswith("/summary"):
                        self._handle_summary()
                    elif text.startswith("/signal"):
                        self._handle_signal()
                    elif text.startswith("/lasttrade"):
                        self._handle_last_trade()
                    elif text.startswith("/open"):
                        self._handle_open_position()
                    elif text.startswith("/pnl"):
                        self._handle_pnl()
                    elif text.startswith("/resetday"):
                        self._handle_reset_day()
                    elif text.startswith("/log"):
                        self._handle_log()
                    elif text.startswith("/config"):
                        self._handle_config()
                time.sleep(1)
            except Exception as exc:
                logger.error("Error in Telegram polling loop: %s", exc, exc_info=True)
                time.sleep(5)
        logger.info("Telegram polling stopped.")

    def stop_polling(self) -> None:
        self.polling_active = False

    def _handle_start(self) -> None:
        self.send_message("\u25b6\ufe0f Start command received.")
        if self.control_callback and self.control_callback("start"):
            self.send_message("\u2705 Trading started.")
        else:
            self.send_message("\u26a0\ufe0f Failed to start trading.")

    def _handle_stop(self) -> None:
        self.send_message("\u23f9\ufe0f Stop command received.")
        if self.control_callback and self.control_callback("stop"):
            self.send_message("\u2705 Trading stopped.")
        else:
            self.send_message("\u26a0\ufe0f Failed to stop trading.")

    def _handle_status(self) -> None:
        if self.status_callback:
            status = self.status_callback()
            status_lines = [f"*{k}*: `{v}`" for k, v in status.items()]
            self.send_message("\ud83d\udcca *Status*\n" + "\n".join(status_lines))
        else:
            self.send_message("\u2139\ufe0f Status unavailable.")

    def _handle_summary(self) -> None:
        if self.summary_callback:
            summary = self.summary_callback()
            self.send_message(f"\ud83d\udcc8 *Daily Summary*\n{summary}")
        else:
            self.send_message("\u2139\ufe0f Summary unavailable.")

    def _handle_signal(self) -> None:
        if self.status_callback:
            signal = self.status_callback().get("last_signal", {})
            if signal:
                self.send_message(
                    f"\ud83d\udd39 *Latest Signal*\n"
                    f"Type: `{signal.get('signal')}`\n"
                    f"Score: `{signal.get('score')}`\n"
                    f"Confidence: `{signal.get('confidence')}`\n"
                    f"Entry: `{signal.get('entry_price')}`\n"
                    f"SL: `{signal.get('stop_loss')}` | Target: `{signal.get('target')}`"
                )
            else:
                self.send_message("\u2139\ufe0f No recent signal.")

    def _handle_last_trade(self) -> None:
        trades = self.status_callback().get("trade_history", [])
        if trades:
            last = trades[-1]
            self.send_message(
                f"\ud83d\udce6 *Last Trade*\n"
                f"Type: `{last['side']}`\n"
                f"Entry: `{last['entry_price']}`\n"
                f"Exit: `{last['exit_price']}`\n"
                f"P&L: `{last['pnl']}`"
            )
        else:
            self.send_message("\u2139\ufe0f No trades yet.")

    def _handle_open_position(self) -> None:
        position = self.status_callback().get("open_position", {})
        if position:
            self.send_message(
                f"\ud83d\udcc2 *Open Position*\n"
                f"Token: `{position['token']}`\n"
                f"Direction: `{position['direction']}`\n"
                f"Qty: `{position['quantity']}`\n"
                f"Entry: `{position['entry_price']}`"
            )
        else:
            self.send_message("\u2139\ufe0f No active positions.")

    def _handle_pnl(self) -> None:
        pnl = self.status_callback().get("live_pnl", None)
        if pnl is not None:
            self.send_message(f"\ud83d\udcb0 Live P&L: `{pnl}`")
        else:
            self.send_message("\u2139\ufe0f Live P&L unavailable.")

    def _handle_reset_day(self) -> None:
        if self.control_callback and self.control_callback("resetday"):
            self.send_message("\ud83d\udd04 Daily counters reset.")
        else:
            self.send_message("\u26a0\ufe0f Failed to reset day.")

    def _handle_log(self) -> None:
        try:
            with open("logs/trading_bot.log", "r") as f:
                lines = f.readlines()[-20:]
            self.send_message("\ud83d\udcc4 *Recent Logs*\n```\n" + "".join(lines) + "```", parse_mode="Markdown")
        except Exception as e:
            self.send_message(f"\u26a0\ufe0f Error reading log: {e}")

    def _handle_config(self) -> None:
        config_data = {
            "Capital": Config.ACCOUNT_SIZE,
            "Risk Per Trade": Config.RISK_PER_TRADE,
            "SL Points": Config.BASE_STOP_LOSS_POINTS,
            "TP Points": Config.BASE_TARGET_POINTS,
            "Max Drawdown": Config.MAX_DRAWDOWN,
            "Scoring Threshold": Config.CONFIDENCE_THRESHOLD,
        }
        msg = "\u2699\ufe0f *Bot Config*\n" + "\n".join(f"{k}: `{v}`" for k, v in config_data.items())
        self.send_message(msg)
