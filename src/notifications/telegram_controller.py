# src/notifications/telegram_controller.py
from __future__ import annotations

"""
Minimal Telegram controller compatible with existing main.py wiring.

- Matches constructor used by main.py:
    TelegramController(
        status_callback=...,     # () -> dict
        control_callback=...,    # (text:str) -> str
        summary_callback=...,    # () -> str
    )
- Single long-poll thread (no webhook)
- Basic commands delegated to control_callback
- Event helpers: notify_entry(), notify_fills(), notify_text()
"""

import hashlib
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import requests

from src.config import settings

log = logging.getLogger(__name__)


class TelegramController:
    def __init__(
        self,
        *,
        status_callback: Callable[[], Dict[str, Any]],
        control_callback: Callable[[str], str],
        summary_callback: Optional[Callable[[], str]] = None,
        http_timeout: float = 20.0,
    ) -> None:
        tg = getattr(settings, "telegram", object())
        self._token: Optional[str] = getattr(tg, "bot_token", None)
        self._chat_id: Optional[int] = getattr(tg, "chat_id", None)
        if not self._token or not self._chat_id:
            raise RuntimeError("TelegramController: bot_token or chat_id missing in settings.telegram")

        self._base = f"https://api.telegram.org/bot{self._token}"
        self._timeout = http_timeout

        # hooks
        self._status = status_callback
        self._control = control_callback
        self._summary = summary_callback or (lambda: "No summary.")

        # poll state
        self._poll_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._started = False
        self._last_update_id: Optional[int] = None

        # simple rate-limit + dedupe
        self._send_min_interval = 0.9
        self._last_sends: List[tuple[float, str]] = []

        # allowlist (just the configured chat)
        self._allowlist = {int(self._chat_id)}

    # ----------- outbound -----------
    def _rate_ok(self, text: str) -> bool:
        now = time.time()
        h = hashlib.md5(text.encode("utf-8")).hexdigest()
        # de-dupe same msg within 10s and enforce min interval
        self._last_sends[:] = [(t, hh) for t, hh in self._last_sends if now - t < 10]
        if self._last_sends and now - self._last_sends[-1][0] < self._send_min_interval:
            return False
        if any(hh == h for _, hh in self._last_sends):
            return False
        self._last_sends.append((now, h))
        return True

    def _send(self, text: str, parse_mode: Optional[str] = None) -> None:
        if not self._rate_ok(text):
            return
        payload = {"chat_id": self._chat_id, "text": text}
        if parse_mode:
            payload["parse_mode"] = parse_mode
        try:
            requests.post(f"{self._base}/sendMessage", json=payload, timeout=self._timeout)
        except Exception as e:
            log.debug("telegram send failed: %s", e)

    def notify_text(self, text: str) -> None:
        self._send(text)

    def send_startup_alert(self) -> None:
        try:
            s = self._status() if self._status else {}
        except Exception:
            s = {}
        self._send(
            "🚀 Bot started\n"
            f"🔁 Trading: {'🟢 LIVE' if s.get('live_trading') else '🟡 DRY'}\n"
            f"🧠 Broker: {s.get('broker')}\n"
            f"📦 Active: {s.get('active_orders', 0)}"
        )

    def notify_entry(self, *, symbol: str, side: str, qty: int, price: float, record_id: str) -> None:
        self._send(
            f"🟢 Entry placed\n{symbol} | {side}\nQty: {qty} @ {price:.2f}\nID: `{record_id}`",
            parse_mode="Markdown",
        )

    def notify_fills(self, fills: List[tuple[str, float]]) -> None:
        if not fills:
            return
        lines = ["✅ Fills"]
        for rid, px in fills:
            lines.append(f"• {rid} @ {px:.2f}")
        self._send("\n".join(lines))

    # ----------- inbound -----------
    def start_polling(self) -> None:
        if self._started:
            log.info("Telegram polling already running; skipping start.")
            return
        self._stop.clear()
        self._poll_thread = threading.Thread(target=self._poll_loop, name="tg-poll", daemon=True)
        self._poll_thread.start()
        self._started = True

    def stop_polling(self) -> None:
        if not self._started:
            return
        self._stop.set()
        if self._poll_thread:
            self._poll_thread.join(timeout=5)
        self._started = False

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                params = {"timeout": 25}
                if self._last_update_id is not None:
                    params["offset"] = self._last_update_id + 1
                r = requests.get(f"{self._base}/getUpdates", params=params, timeout=self._timeout + 10)
                data = r.json()
                if not data.get("ok"):
                    time.sleep(1.0)
                    continue
                for upd in data.get("result", []):
                    self._last_update_id = int(upd.get("update_id", 0))
                    self._handle_update(upd)
            except Exception as e:
                log.debug("Telegram poll error: %s", e)
                time.sleep(1.0)

    def _authorized(self, chat_id: int) -> bool:
        return int(chat_id) in self._allowlist

    def _handle_update(self, upd: Dict[str, Any]) -> None:
        msg = upd.get("message") or upd.get("edited_message")
        if not msg:
            return
        chat_id = msg.get("chat", {}).get("id")
        text = (msg.get("text") or "").strip()
        if not text:
            return
        if not self._authorized(int(chat_id)):
            self._send("Unauthorized.")
            return

        # Strip leading slash and route to control_callback
        cmd = text.lstrip("/")
        try:
            reply = self._control(cmd) if self._control else "No controller wired."
        except Exception as e:
            reply = f"Error: {e}"
        if reply:
            self._send(str(reply))