# src/notifications/telegram_controller.py
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import requests
from requests.adapters import HTTPAdapter

try:
    from urllib3.util.retry import Retry  # type: ignore
except Exception:  # pragma: no cover
    Retry = None

from src.config import settings

logger = logging.getLogger(__name__)


@dataclass
class _TGConfig:
    enabled: bool
    bot_token: Optional[str]
    chat_id: Optional[int]


def _build_cfg() -> _TGConfig:
    tg = getattr(settings, "telegram", None)
    enabled = bool(getattr(tg, "enabled", True))
    bot_token = getattr(tg, "bot_token", None)
    # Can be learned on first inbound message if not set
    chat_id = getattr(tg, "chat_id", getattr(settings, "TELEGRAM_CHAT_ID", None)) if tg is not None else None
    try:
        chat_id = int(chat_id) if chat_id is not None else None
    except Exception:
        chat_id = None
    return _TGConfig(enabled=enabled, bot_token=bot_token, chat_id=chat_id)


class TelegramController:
    """
    Telegram bot interface (long polling).
    - Ensures webhook is disabled before polling (prevents 409 conflicts)
    - Singleton polling guard within a process (no double polling)
    - Throttled 'no chat_id' logs
    - Status format matches requested style
    """

    # ---- process-wide singleton guard (no double polling) ----
    _global_lock = threading.Lock()
    _global_polling_active = False

    def __init__(
        self,
        status_callback: Callable[[], Dict[str, Any]],
        control_callback: Callable[[str, str], bool],
        summary_callback: Callable[[], str],
        *,
        cfg: Optional[_TGConfig] = None,
    ) -> None:
        self.cfg = cfg or _build_cfg()
        self.enabled = bool(self.cfg.enabled and self.cfg.bot_token)
        self.base_url = f"https://api.telegram.org/bot{self.cfg.bot_token}" if self.cfg.bot_token else None

        self.status_callback = status_callback
        self.control_callback = control_callback
        self.summary_callback = summary_callback

        self._poll_lock = threading.Lock()
        self._polling = False
        self._thread: Optional[threading.Thread] = None
        self._chat_missing_warned = False  # throttle noisy logs

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

        if not self.enabled:
            logger.warning("TelegramController disabled (no token or not enabled).")
        else:
            logger.info("TelegramController initialized.")

    # ---------- HTTP helpers ----------
    def _request(self, method: str, path: str, *, params=None, json=None, timeout: int = 35):
        if not self.enabled or not self.base_url:
            return None
        url = f"{self.base_url}/{path.lstrip('/')}"
        try:
            resp = self.session.request(method=method, url=url, params=params, json=json, timeout=timeout)
            if resp.status_code == 429:
                try:
                    retry_after = int(resp.json().get("parameters", {}).get("retry_after", 1))
                except Exception:
                    retry_after = 1
                time.sleep(max(1, retry_after))
                return self.session.request(method=method, url=url, params=params, json=json, timeout=timeout)
            return resp
        except requests.exceptions.RequestException as exc:
            logger.warning("Telegram HTTP error: %s %s (%s)", method, path, exc)
            return None

    # ---------- webhook controls ----------
    def _delete_webhook(self) -> bool:
        """Disable webhook; return True if disabled or not set."""
        if not self.enabled:
            return True
        try:
            self._request("POST", "deleteWebhook", json={"drop_pending_updates": True}, timeout=15)
            # verify
            info = self._request("GET", "getWebhookInfo", timeout=10)
            if info is None or info.status_code != 200:
                return True  # best effort
            data = info.json() if info.headers.get("content-type", "").startswith("application/json") else {}
            url = (data.get("result") or {}).get("url", "") if data.get("ok") else ""
            return not url
        except Exception:
            return True  # don't block polling on verification

    def _ensure_polling_ready(self) -> None:
        """Loop until webhook is cleared to avoid 409 conflicts."""
        for attempt in range(5):
            if self._delete_webhook():
                return
            sleep = 1 + attempt  # tiny backoff
            logger.info("Waiting for Telegram webhook to clear (%ss)...", sleep)
            time.sleep(sleep)

    # ---------- send ----------
    def _send_message(self, text: str, parse_mode: Optional[str] = None, retries: int = 2, backoff_sec: float = 1.0) -> bool:
        if not self.enabled:
            return False
        if self.cfg.chat_id is None:
            if not self._chat_missing_warned:
                logger.error("Telegram chat_id is not configured. Cannot send message.")
                self._chat_missing_warned = True
            return False

        payload: Dict[str, Any] = {"chat_id": self.cfg.chat_id, "text": text, "disable_notification": False}
        if parse_mode:
            payload["parse_mode"] = parse_mode

        attempt = 0
        while attempt <= retries:
            attempt += 1
            resp = self._request("POST", "sendMessage", json=payload, timeout=15)
            if resp is None:
                if attempt <= retries:
                    time.sleep(backoff_sec)
                continue
            if resp.status_code == 200:
                return True
            if attempt <= retries and resp.status_code >= 500:
                time.sleep(backoff_sec)
        return False

    # ---------- public helpers ----------
    def send_message(self, text: str, parse_mode: Optional[str] = None) -> bool:
        return self._send_message(text, parse_mode=parse_mode)

    def send_startup_alert(self) -> None:
        self._send_message("🟢 Nifty Scalper Bot started.\nType /help for commands. Awaiting instructions…")

    def send_realtime_session_alert(self, action: str) -> None:
        text = "✅ Real-time trading session STARTED." if action.upper() == "START" else \
               "🛑 Real-time trading session STOPPED." if action.upper() == "STOP" else f"ℹ️ Session {action}."
        self._send_message(text)

    def send_alert(self, action: str) -> None:
        self.send_realtime_session_alert(action)

    def send_signal_alert(self, token: int, signal: Dict[str, Any], position: Dict[str, Any]) -> None:
        direction = signal.get("side") or signal.get("signal") or signal.get("direction", "ENTRY")
        entry = signal.get("entry_price", signal.get("price", "N/A"))
        sl = signal.get("stop_loss", signal.get("sl_points", "N/A"))
        target = signal.get("target", signal.get("tp_points", "N/A"))
        conf = float(signal.get("confidence", 0.0) or 0.0)
        qty = position.get("quantity", position.get("lots", "N/A"))
        text = (
            f"🔥 <b>NEW SIGNAL #{token}</b>\n"
            f"🎯 Direction: <b>{direction}</b>\n"
            f"💰 Entry: <code>{entry}</code>\n"
            f"📉 Stop Loss: <code>{sl}</code>\n"
            f"🎯 Target: <code>{target}</code>\n"
            f"📊 Confidence: <code>{conf:.2f}</code>\n"
            f"🧮 Quantity/Lots: <code>{qty}</code>"
        )
        self._send_message(text, parse_mode="HTML")

    # ---------- status ----------
    @staticmethod
    def _fmt_uptime(sec: float) -> str:
        sec = int(max(0, sec))
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h}h {m}m"
        if m:
            return f"{m}m"
        return f"{s}s"

    def _send_status(self, status: Dict[str, Any]) -> None:
        is_trading = bool(status.get("is_trading", False))
        live_mode = bool(status.get("live_mode", False))
        quality = (status.get("quality") or "AUTO").upper()
        regime = (status.get("regime") or "auto")
        open_positions = status.get("open_positions", 0)
        closed_today = status.get("closed_today", status.get("trades_today", 0))
        daily_pnl = float(status.get("daily_pnl", 0.0) or 0.0)
        session_date = status.get("session_date") or status.get("session")
        uptime_sec = float(status.get("uptime_seconds", 0.0) or 0.0)

        lines = [
            "📊 <b>Bot Status</b>",
            f"🔁 <b>Trading:</b> {'🟢 Running' if is_trading else '🔴 Stopped'}",
            f"🌐 <b>Mode:</b> {'🟢 LIVE' if live_mode else '🛡️ SHADOW'}",
            f"✨ <b>Quality:</b> {quality}",
            f"🧭 <b>Regime:</b> {regime}",
            f"📦 <b>Open Positions:</b> {open_positions}",
            f"📈 <b>Closed Today:</b> {closed_today}",
            f"💰 <b>Daily P&L:</b> {daily_pnl:.2f}",
        ]
        if session_date:
            lines.append(f"📅 <b>Session:</b> {session_date}")
        lines.append(f"⏱️ <b>Uptime:</b> {self._fmt_uptime(uptime_sec)}")
        self._send_message("\n".join(lines), parse_mode="HTML")

    def _send_summary(self, summary: str) -> None:
        self._send_message(summary, parse_mode="HTML")

    # ---------- command routing ----------
    def _handle_command(self, command: str, arg: str = "", chat_id_hint: Optional[int] = None) -> None:
        logger.info("📩 Telegram command: '%s %s'", command, arg)

        # Learn chat_id on first inbound message
        if self.cfg.chat_id is None and chat_id_hint is not None:
            try:
                self.cfg.chat_id = int(chat_id_hint)
                logger.info("📌 Learned TELEGRAM_CHAT_ID=%s from inbound message.", chat_id_hint)
            except Exception:
                pass

        if command == "help":
            self._send_message(
                "🤖 <b>Commands</b>\n"
                "/start – start trading\n"
                "/stop – stop trading\n"
                "/mode live|shadow – switch mode\n"
                "/quality auto|conservative|aggressive – risk/latency profile\n"
                "/status – bot status\n"
                "/summary – daily summary\n"
                "/refresh – refresh instruments cache\n"
                "/health – system health\n"
                "/emergency – stop & cancel orders\n"
                "/ping – latency check",
                parse_mode="HTML",
            )
            return

        if command == "ping":
            self._send_message("pong");  return

        if command == "status":
            self._send_status(self.status_callback());  return

        if command == "summary":
            self._send_summary(self.summary_callback());  return

        if command == "mode":
            v = (arg or "").strip().lower()
            if v not in ("live", "shadow"):
                self._send_message("Usage: /mode <live|shadow>");  return
            ok = self.control_callback("mode", v)
            self._send_message("✅ Mode set to LIVE." if ok and v == "live"
                               else "✅ Mode set to SHADOW." if ok
                               else "⚠️ Failed to set mode.")
            return

        if command == "quality":
            v = (arg or "").strip().lower()
            if v not in ("auto", "conservative", "aggressive"):
                self._send_message("Usage: /quality <auto|conservative|aggressive>");  return
            ok = self.control_callback("quality", v)
            self._send_message(f"✅ Quality set to {v.upper()}." if ok else "⚠️ Failed to set quality.")
            return

        if command in ("start", "stop", "refresh", "health", "emergency"):
            ok = self.control_callback(command, arg)
            self._send_message("✅ Done." if ok else f"⚠️ Command '/{command} {arg}' failed.")
            return

        self._send_message(
            "❌ Unknown command.\n"
            "Try: /start, /stop, /mode live, /mode shadow, /quality auto, /status, /summary, /refresh, /health, /emergency, /help, /ping"
        )

    # ---------- polling loop ----------
    def _poll_updates(self) -> None:
        if not self.enabled:
            logger.info("Telegram disabled; polling thread exiting.")
            return

        self._ensure_polling_ready()  # avoid 409 conflicts

        url_path = "getUpdates"
        offset: Optional[int] = None
        timeout = 30

        logger.info("📡 Telegram polling started. Awaiting commands...")
        while self._polling:
            try:
                params = {"timeout": timeout, "offset": offset, "allowed_updates": ["message", "edited_message"]}
                resp = self._request("GET", url_path, params=params, timeout=timeout + 5)
                if resp is None:
                    time.sleep(1);  continue

                if resp.status_code == 200:
                    data = resp.json()
                    results = data.get("result", []) if data.get("ok") else []
                    for result in results:
                        offset = int(result["update_id"]) + 1
                        message = result.get("message") or result.get("edited_message") or {}
                        text = (message.get("text") or "").strip()
                        if not text.startswith("/"):
                            continue
                        parts = text[1:].split(maxsplit=1)
                        cmd = parts[0].lower()
                        arg = parts[1] if len(parts) > 1 else ""
                        chat = message.get("chat") or {}
                        chat_id = chat.get("id")
                        self._handle_command(cmd, arg, chat_id_hint=chat_id)

                elif resp.status_code == 409:
                    logger.error("409 Conflict: webhook active; disabling and retrying.")
                    self._ensure_polling_ready()
                    time.sleep(2)

                else:
                    try:
                        body = resp.json()
                    except Exception:
                        body = {"text": resp.text}
                    logger.error("getUpdates failed (%s): %s", resp.status_code, body)
                    time.sleep(2)

            except requests.exceptions.ReadTimeout:
                logger.debug("Telegram polling timeout — continuing...")
            except Exception as exc:
                logger.error("Error in Telegram polling: %s", exc, exc_info=True)
                time.sleep(5)

        logger.info("🛑 Telegram polling stopped.")

    # ---------- lifecycle (singleton) ----------
    def start_polling(self) -> None:
        if not self.enabled:
            logger.warning("Telegram disabled; not starting polling.")
            return
        with TelegramController._global_lock:
            if TelegramController._global_polling_active:
                logger.info("Telegram polling already active in this process; ignoring duplicate start.")
                return
            TelegramController._global_polling_active = True
        with self._poll_lock:
            if self._polling:
                logger.info("Telegram polling already running; ignoring duplicate start.")
                return
            self._polling = True
            self._thread = threading.Thread(target=self._poll_updates, name="TelegramPolling", daemon=True)
            self._thread.start()

    def stop_polling(self) -> None:
        with self._poll_lock:
            if not self._polling:
                return
            logger.info("🛑 Stopping Telegram polling...")
            self._polling = False
            th = self._thread
            self._thread = None
        if th and th.is_alive():
            th.join(timeout=5)
        with TelegramController._global_lock:
            TelegramController._global_polling_active = False