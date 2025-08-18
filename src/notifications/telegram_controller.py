# src/notifications/telegram_controller.py
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Callable, Dict, Optional

from src.config import TelegramConfig

logger = logging.getLogger(__name__)


class TelegramController:
    """
    Lightweight Telegram controller with:
    - /start, /stop, /mode live|shadow, /status, /summary, /refresh, /health, /emergency
    - NEW: /config get | /config set <key> <value>  (delegated to callbacks)
    """

    def __init__(
        self,
        config: TelegramConfig,
        status_callback: Optional[Callable[[], Dict[str, Any]]] = None,
        control_callback: Optional[Callable[[str, str], bool]] = None,
        summary_callback: Optional[Callable[[], str]] = None,
        config_getter: Optional[Callable[[], Dict[str, Any]]] = None,
        config_setter: Optional[Callable[[str, str], str]] = None,
    ):
        self.cfg = config
        self._status_cb = status_callback
        self._control_cb = control_callback
        self._summary_cb = summary_callback
        self._config_getter = config_getter
        self._config_setter = config_setter

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -------------- Telegram wire (placeholder/adapter) --------------

    def start_polling(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._poll_updates, daemon=True)
        self._thread.start()

    def stop_polling(self) -> None:
        self._stop_event.set()

    def _poll_updates(self) -> None:
        """
        Replace this with python-telegram-bot or your existing adapter.
        Here we simulate a loop that picks from a queue (left as an exercise).
        """
        logger.info("Telegram polling started.")
        while not self._stop_event.is_set():
            time.sleep(1.0)
        logger.info("Telegram polling stopped.")

    # -------------- Messaging helpers --------------

    def send_message(self, text: str) -> None:
        logger.info("TG: %s", text)
        # Plug in real send via bot token/chat id

    def _help_text(self) -> str:
        return (
            "Commands:\n"
            "/start, /stop, /mode <live|shadow>\n"
            "/status, /summary, /refresh, /health, /emergency\n"
            "/config get\n"
            "/config set <key> <value>\n"
            "Examples:\n"
            "  /config get\n"
            "  /config set adx_trend_strength 20\n"
            "  /config set require_vwap_alignment false\n"
        )

    # -------------- Command handler --------------

    def handle_command(self, raw: str) -> None:
        parts = (raw or "").strip().split()
        if not parts:
            return
        cmd = parts[0].lower()

        if cmd == "/help":
            self.send_message(self._help_text()); return

        if cmd == "/status":
            if self._status_cb:
                self.send_message(json.dumps(self._status_cb(), indent=2))
            return

        if cmd == "/summary":
            if self._summary_cb:
                self.send_message(self._summary_cb())
            return

        if cmd == "/mode":
            if self._control_cb and len(parts) > 1:
                ok = self._control_cb("mode", parts[1].lower())
                self.send_message("OK" if ok else "Failed")
            return

        if cmd == "/start":
            if self._control_cb:
                ok = self._control_cb("start", "")
                self.send_message("OK" if ok else "Failed")
            return

        if cmd == "/stop":
            if self._control_cb:
                ok = self._control_cb("stop", "")
                self.send_message("OK" if ok else "Failed")
            return

        if cmd == "/refresh":
            if self._control_cb:
                ok = self._control_cb("refresh", "")
                self.send_message("OK" if ok else "Failed")
            return

        if cmd == "/emergency":
            if self._control_cb:
                ok = self._control_cb("panic", "")
                self.send_message("OK" if ok else "Failed")
            return

        if cmd == "/health":
            self.send_message("healthy"); return

        if cmd == "/config":
            if len(parts) == 1 or parts[1].lower() == "get":
                if self._config_getter:
                    self.send_message(json.dumps(self._config_getter(), indent=2))
                else:
                    self.send_message("No config getter wired.")
                return
            if parts[1].lower() == "set":
                if self._config_setter and len(parts) >= 4:
                    key, val = parts[2], " ".join(parts[3:])
                    msg = self._config_setter(key, val)
                    self.send_message(msg)
                else:
                    self.send_message("Usage: /config set <key> <value>")
                return

        self.send_message("Unknown command. /help for options.")
