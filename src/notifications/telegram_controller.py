# src/notifications/telegram_controller.py
"""
TelegramController
------------------
Purpose:
- Poll Telegram long-poll API and route commands to the trader.
- Send formatted alerts (startup, status, signals, etc.).

Key points / changes:
- ✅ Singleton poller: lock-file at /tmp/tgpoll_<bot>_<instance>.lock prevents double polling.
- ✅ 409 auto-stop kept as a backstop, but we *proactively* avoid starting a 2nd poller.
- ✅ Clean stop() that releases the lock every time (incl. during exceptions).
- ✅ Small QoL: /mode live|shadow|quality on|off, /risk, /regime, /pause, /resume, /status…

Config knobs (env via Config):
- TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
- TELEGRAM_INSTANCE_ID (optional; default: hostname or 'default')
- TELEGRAM_POLL_TIMEOUT (default 30s), TELEGRAM_MIN_SEND_GAP_SEC (default 0.4s)
"""

from __future__ import annotations

import html
import logging
import os
import socket
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


class _FileLock:
    """Lightweight file lock to ensure only one poller instance per bot+instance id."""
    def __init__(self, path: str) -> None:
        self.path = path
        self._fd = None
        self._lock = threading.RLock()

    def acquire(self) -> bool:
        with self._lock:
            if self._fd is not None:
                return True
            try:
                # Use os.O_EXCL to fail if the file already exists.
                self._fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                os.write(self._fd, str(os.getpid()).encode("utf-8"))
                os.fsync(self._fd)
                logger.debug("Telegram lock acquired: %s", self.path)
                return True
            except FileExistsError:
                logger.warning("Telegram polling lock already present: %s", self.path)
                return False
            except Exception as e:
                logger.error("Failed to acquire Telegram lock: %s", e)
                return False

    def release(self) -> None:
        with self._lock:
            try:
                if self._fd is not None:
                    os.close(self._fd)
                    self._fd = None
                if os.path.exists(self.path):
                    os.unlink(self.path)
                logger.debug("Telegram lock released: %s", self.path)
            except Exception:
                # Do not raise in release path
                pass


class TelegramController:
    """
    Telegram bot interface for receiving commands and sending alerts.
    Communicates with RealTimeTrader via callback functions.

    Commands:
      /start
      /stop
      /mode live|shadow|quality on|off
      /risk <pct>                  (e.g. /risk 0.5 or /risk 0.5%)
      /regime auto|trend|range|off
      /pause <minutes>
      /resume
      /status
      /summary
      /refresh
      /health
      /emergency
      /help
    """

    def __init__(
        self,
        status_callback: Callable[[], Dict[str, Any]],
        control_callback: Callable[[str, str], bool],
        summary_callback: Callable[[], str],
    ) -> None:
        from src.config import Config  # lazy import so tests can monkeypatch

        self.bot_token: str = getattr(Config, "TELEGRAM_BOT_TOKEN", "")
        self.chat_id: str | int = getattr(Config, "TELEGRAM_CHAT_ID", "")

        if not self.bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN is required in Config")
        if not self.chat_id:
            raise ValueError("TELEGRAM_CHAT_ID is required in Config")

        # Instance id (for multi-deploy environments)
        instance_id = os.getenv("TELEGRAM_INSTANCE_ID") or socket.gethostname() or "default"
        self._lock_path = f"/tmp/tgpoll_{self.bot_token.split(':',1)[0]}_{instance_id}.lock"
        self._poll_lock = _FileLock(self._lock_path)

        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.polling = False
        self.polling_thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()

        # Callbacks to the trader
        self.status_callback = status_callback
        self.control_callback = control_callback
        self.summary_callback = summary_callback

        # poll state
        self._offset: Optional[int] = None
        self._timeout_s: int = int(getattr(Config, "TELEGRAM_POLL_TIMEOUT", 30))
        self._request_timeout_s: int = self._timeout_s + 5

        # light rate-limit on sends
        self._last_send_ts: float = 0.0
        self._min_send_gap_s: float = float(getattr(Config, "TELEGRAM_MIN_SEND_GAP_SEC", 0.4))

        logger.info("TelegramController initialized.")

    # ---------------- HTTP helpers ---------------- #

    def _post(self, method: str, payload: Dict[str, Any], *, json_mode: bool = True) -> Optional[Dict[str, Any]]:
        url = f"{self.base_url}/{method}"
        tries = 3
        backoff = 0.9
        last_exc = None
        for i in range(tries):
            try:
                resp = requests.post(url, json=payload if json_mode else None, data=None if json_mode else payload,
                                     timeout=self._request_timeout_s)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("ok"):
                        return data
                    logger.error("Telegram API error: %s", data)
                    return None
                if resp.status_code == 409:
                    # Another poller/webhook exists
                    logger.error("Telegram conflict (409). Another instance is active. Stopping polling.")
                    self.stop_polling()
                    return None
                logger.warning("Telegram POST %s failed (%s): %s", method, resp.status_code, resp.text)
            except Exception as exc:
                last_exc = exc
                logger.debug("Telegram POST %s error (attempt %d): %s", method, i + 1, exc)
            time.sleep(backoff * (2 ** i))
        if last_exc:
            logger.error("Telegram POST %s failed after retries: %s", method, last_exc)
        return None

    def _get(self, method: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        url = f"{self.base_url}/{method}"
        tries = 3
        backoff = 0.9
        last_exc = None
        for i in range(tries):
            try:
                resp = requests.get(url, params=params, timeout=self._request_timeout_s)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("ok"):
                        return data
                    logger.error("Telegram API error: %s", data)
                    return None
                if resp.status_code == 409:
                    logger.error("Telegram conflict (409). Another instance is active. Stopping polling.")
                    self.stop_polling()
                    return None
                logger.warning("Telegram GET %s failed (%s): %s", method, resp.status_code, resp.text)
            except Exception as exc:
                last_exc = exc
                logger.debug("Telegram GET %s error (attempt %d): %s", method, i + 1, exc)
            time.sleep(backoff * (2 ** i))
        if last_exc:
            logger.error("Telegram GET %s failed after retries: %s", method, last_exc)
        return None

    # ---------------- sending ---------------- #

    def _throttle(self) -> None:
        now = time.time()
        gap = now - self._last_send_ts
        if gap < self._min_send_gap_s:
            time.sleep(self._min_send_gap_s - gap)
        self._last_send_ts = time.time()

    def _send_message(self, text: str, parse_mode: Optional[str] = None, disable_web_page_preview: bool = True) -> bool:
        if not text:
            return False
        self._throttle()
        payload: Dict[str, Any] = {
            "chat_id": self.chat_id,
            "text": text,
            "disable_notification": False,
            "disable_web_page_preview": disable_web_page_preview,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode
        data = self._post("sendMessage", payload)
        ok = bool(data and data.get("ok"))
        if not ok:
            logger.error("Failed to send Telegram message.")
        return ok

    def send_message(self, text: str, parse_mode: Optional[str] = None) -> bool:
        return self._send_message(text, parse_mode=parse_mode or "HTML")

    def send_startup_alert(self) -> None:
        self._send_message("🟢 <b>Nifty Scalper Bot</b> started.\nType <code>/help</code> for commands.", parse_mode="HTML")

    def send_realtime_session_alert(self, action: str) -> None:
        msg = {"START": "✅ Real-time trading session <b>STARTED</b>.",
               "STOP": "🛑 Real-time trading session <b>STOPPED</b>."}.get(action.upper(), f"ℹ️ Session {html.escape(action)}.")
        self._send_message(msg, parse_mode="HTML")

    def send_alert(self, action: str) -> None:
        self.send_realtime_session_alert(action)

    def send_signal_alert(self, token: int, signal: Dict[str, Any], position: Dict[str, Any]) -> None:
        direction = html.escape(str(signal.get("signal") or signal.get("direction") or "ENTRY"))
        entry = html.escape(f"{signal.get('entry_price', 'N/A')}")
        sl = html.escape(f"{signal.get('stop_loss', 'N/A')}")
        target = html.escape(f"{signal.get('target', 'N/A')}")
        conf = float(signal.get("confidence", 0.0) or 0.0)
        qty = html.escape(f"{position.get('quantity', 'N/A')}")
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

    # ---------------- formatting helpers ---------------- #

    def _format_status(self, status: Any) -> Tuple[str, str]:
        if isinstance(status, str):
            return f"📊 Status:\n{html.escape(status)}", "HTML"

        is_trading = bool(status.get("is_trading", False))
        live_mode = bool(status.get("live_mode", False))
        quality_mode = bool(status.get("quality_mode", False))
        open_positions = int(status.get("open_positions", status.get("open_orders", 0)) or 0)
        trades_today = int(status.get("trades_today", status.get("closed_today", 0)) or 0)
        daily_pnl = float(status.get("daily_pnl", 0.0) or 0.0)
        acct = status.get("account_size")
        sess = status.get("session_date")

        lines = [
            "📊 <b>Bot Status</b>",
            f"🔁 <b>Trading:</b> {'🟢 Running' if is_trading else '🔴 Stopped'}",
            f"🌐 <b>Mode:</b> {'🟢 LIVE' if live_mode else '🛡️ Shadow'}",
            f"✨ <b>Quality:</b> {'ON' if quality_mode else 'OFF'}",
            f"📦 <b>Open Positions:</b> {open_positions}",
            f"📈 <b>Closed Today:</b> {trades_today}",
            f"💰 <b>Daily P&L:</b> {daily_pnl:.2f}",
        ]
        if acct is not None:
            lines.append(f"🏦 <b>Acct Size:</b> ₹{html.escape(str(acct))}")
        if sess is not None:
            lines.append(f"📅 <b>Session:</b> {html.escape(str(sess))}")

        return "\n".join(lines), "HTML"

    def _send_status(self, status: Any) -> None:
        txt, mode = self._format_status(status)
        self._send_message(txt, parse_mode=mode)

    def _send_summary(self, summary: str) -> None:
        self._send_message(summary, parse_mode="HTML")

    # ---------------- command router ---------------- #

    @staticmethod
    def _parse_command_and_arg(text: str) -> Tuple[str, str]:
        t = (text or "").strip()
        if not t.startswith("/"):
            return "", ""
        body = t[1:].strip()
        if not body:
            return "", ""
        parts = body.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        return cmd, arg

    @staticmethod
    def _normalize_risk_arg(arg: str) -> str:
        a = (arg or "").strip().replace("%", "")
        try:
            val = float(a)
            if val > 1.5:
                val = val / 100.0
            return f"{val:.4f}"
        except Exception:
            return ""

    def _handle_command(self, command: str, arg: str = "") -> None:
        logger.info("📩 Received command: '/%s %s'", command, arg)
        cmd = command.lower()

        if cmd == "help":
            self._send_message(
                "🤖 <b>Commands</b>\n"
                "/start – start trading\n"
                "/stop – stop trading\n"
                "/mode live|shadow|quality on|off – switch mode\n"
                "/risk &lt;pct&gt; – e.g. <code>/risk 0.5</code> (means 0.5%)\n"
                "/regime auto|trend|range|off – set regime gate\n"
                "/pause &lt;min&gt; – pause entries\n"
                "/resume – resume entries\n"
                "/status – bot status\n"
                "/summary – daily summary\n"
                "/refresh – refresh balance/instruments\n"
                "/health – system health\n"
                "/emergency – stop & cancel orders",
                parse_mode="HTML",
            )
            return

        if cmd == "status":
            try:
                status = self.status_callback()
            except Exception as e:
                logger.error("status_callback error: %s", e, exc_info=True)
                self._send_message("⚠️ Failed to fetch status.")
                return
            self._send_status(status)
            return

        if cmd == "summary":
            try:
                summary = self.summary_callback()
            except Exception as e:
                logger.error("summary_callback error: %s", e, exc_info=True)
                self._send_message("⚠️ Failed to fetch summary.")
                return
            self._send_summary(summary)
            return

        if cmd == "mode":
            a = (arg or "").strip().lower()
            ok = self.control_callback("mode", a)
            if not ok:
                self._send_message("⚠️ Failed to change mode.")
            return

        if cmd == "risk":
            norm = self._normalize_risk_arg(arg)
            ok = bool(norm) and self.control_callback("risk", norm)
            if not ok:
                self._send_message("⚠️ Usage: /risk 0.5  (meaning 0.5%)")
            else:
                self._send_message(f"✅ Risk per trade set to <b>{norm}%</b>.", parse_mode="HTML")
            return

        if cmd in {"start", "stop", "regime", "pause", "resume", "refresh", "health", "emergency"}:
            ok = self.control_callback(cmd, arg or "")
            if not ok:
                logger.warning("Command '/%s %s' failed.", cmd, arg)
                self._send_message(f"⚠️ Command '/{cmd} {html.escape(arg)}' failed.", parse_mode="HTML")
            return

        self._send_message("❌ Unknown command. Try <code>/help</code>.", parse_mode="HTML")

    # ---------------- polling loop ---------------- #

    def _poll_updates(self) -> None:
        logger.info("📡 Telegram polling started. Awaiting commands...")
        self._stop_evt.clear()
        while self.polling and not self._stop_evt.is_set():
            try:
                params = {"timeout": self._timeout_s}
                if self._offset is not None:
                    params["offset"] = self._offset
                data = self._get("getUpdates", params)
                if not data:
                    continue
                results = data.get("result", [])
                for item in results:
                    self._offset = int(item["update_id"]) + 1
                    message = item.get("message") or item.get("edited_message") or {}
                    text = (message.get("text") or "").strip()
                    if not text.startswith("/"):
                        continue
                    cmd, arg = self._parse_command_and_arg(text)
                    if cmd:
                        self._handle_command(cmd, arg)
            except requests.exceptions.ReadTimeout:
                logger.debug("Telegram polling timeout — continuing…")
            except Exception as exc:
                logger.error("Error in Telegram polling: %s", exc, exc_info=True)
                time.sleep(3.0)
        logger.info("🛑 Telegram polling stopped.")

    # ---------------- public lifecycle ---------------- #

    def start_polling(self) -> None:
        if self.polling:
            logger.warning("Polling already active.")
            return
        # SINGLETON GUARD
        if not self._poll_lock.acquire():
            logger.error("Another poller holds the lock. Not starting polling.")
            return

        self.polling = True
        self.polling_thread = threading.Thread(
            target=self._poll_updates, name="TelegramPolling", daemon=True
        )
        self.polling_thread.start()

    def stop_polling(self) -> None:
        logger.info("🛑 Stopping Telegram polling…")
        self.polling = False
        self._stop_evt.set()
        if self.polling_thread and self.polling_thread.is_alive():
            try:
                self.polling_thread.join(timeout=5)
            except Exception:
                pass
        self.polling_thread = None
        # Always release lock on stop
        self._poll_lock.release()