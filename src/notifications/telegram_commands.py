from __future__ import annotations

"""Telegram command listener using HTTP polling."""

import logging
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class TelegramCommands:
    """Poll a Telegram bot for commands and invoke a callback."""

    def __init__(
        self,
        bot_token: Optional[str],
        chat_id: Optional[str],
        on_cmd: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        self.token = bot_token or ""
        self.chat = str(chat_id or "")
        self.on_cmd = on_cmd
        self._offset: Optional[int] = None
        self._running = False
        self._th: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start polling for commands."""
        if not (self.token and self.chat):
            logger.warning("Telegram disabled (token/chat_id missing)")
            return
        if self._running:
            return
        self._running = True
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        logger.info("TelegramCommands started")

    def stop(self) -> None:
        """Stop polling and wait for the worker thread to exit."""
        self._running = False
        if self._th:
            self._th.join(timeout=1)
            self._th = None

    def _loop(self) -> None:
        import requests

        base = f"https://api.telegram.org/bot{self.token}"
        while self._running:
            try:
                params = {"timeout": 30}
                if self._offset is not None:
                    params["offset"] = self._offset
                r = requests.get(f"{base}/getUpdates", params=params, timeout=35)
                if r.status_code != 200:
                    time.sleep(1)
                    continue
                for upd in r.json().get("result", []):
                    self._offset = upd["update_id"] + 1
                    msg = upd.get("message") or {}
                    text = (msg.get("text") or "").strip()
                    chat_id = str((msg.get("chat") or {}).get("id") or "")
                    if not text or chat_id != self.chat:
                        continue
                    cmd, *rest = text.split(maxsplit=1)
                    arg = rest[0] if rest else ""
                    if self.on_cmd:
                        try:
                            self.on_cmd(cmd, arg)
                        except Exception:
                            logger.exception("cmd handler failed")
            except requests.exceptions.ReadTimeout:
                continue
            except Exception:
                logger.exception("telegram polling error")
                time.sleep(1)
