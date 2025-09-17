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
        backtest_runner: Optional[Callable[[Optional[str]], str]] = None,
    ) -> None:
        self.token = bot_token or ""
        self.chat = str(chat_id or "")
        self.on_cmd = on_cmd
        self._backtest_runner = backtest_runner
        self._offset: Optional[int] = None
        self._running = False
        self._th: Optional[threading.Thread] = None
        # runtime state for UX commands
        self.basis = "premium"
        self.unit_notional = 0.0
        self.lots = 0
        self.caps = 0.0
        self.risk_pct = 0.0
        self.exposure_mode = "premium"
        self.paused_until = 0.0

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
                    if self._handle_cmd(cmd, arg):
                        continue
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

    # ------------ command helpers ------------
    def _send(self, text: str) -> None:
        if not (self.token and self.chat):
            return
        try:
            import requests

            requests.post(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                json={"chat_id": self.chat, "text": text},
                timeout=10,
            )
        except Exception:
            logger.exception("telegram send error")

    def _handle_cmd(self, cmd: str, arg: str) -> bool:
        if cmd == "/pause":
            mins = 0
            if arg.endswith("m"):
                try:
                    mins = int(arg[:-1])
                except ValueError:
                    mins = 0
            self.paused_until = time.time() + mins * 60
            self._send(f"Paused for {mins}m")
            return True
        if cmd == "/risk":
            try:
                self.risk_pct = float(arg.strip("%"))
            except ValueError:
                self.risk_pct = 0.0
            self._send(f"risk_pct set to {self.risk_pct}%")
            return True
        if cmd == "/exposure":
            if arg in ("premium", "underlying"):
                self.exposure_mode = arg
            self._send(f"exposure mode = {self.exposure_mode}")
            return True
        if cmd == "/flatten":
            self.lots = 0
            self.unit_notional = 0.0
            self._send("positions flattened")
            return True
        if cmd == "/status" and arg == "brief":
            msg = (
                f"basis={self.basis} unit_notional={self.unit_notional} "
                f"lots={self.lots} caps={self.caps}"
            )
            self._send(msg)
            return True
        if cmd == "/backtest":
            if not self._backtest_runner:
                self._send("Backtest not available.")
                return True
            path = arg.strip() or None
            try:
                result = self._backtest_runner(path)
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("backtest runner failed")
                result = f"Backtest error: {exc}"
            if result:
                self._send(result)
            return True
        return False
