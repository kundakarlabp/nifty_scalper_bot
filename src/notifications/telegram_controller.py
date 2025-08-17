# src/notifications/telegram_controller.py
from __future__ import annotations

import html
import json
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


def _readable_exc(e: Exception) -> str:
    t = type(e).__name__
    return f"{t}: {e}"


class TelegramController:
    """
    Telegram bot interface for receiving commands and sending alerts.

    Commands:
      /start
      /stop
      /mode live|shadow|quality on|off|auto
      /regime auto|trend|range|off
      /risk <pct>                  (e.g., /risk 0.5 or /risk 0.5%)
      /pause <minutes>
      /resume
      /status
      /summary
      /refresh
      /health
      /emergency
      /help
    """

    # ----------------------------- lifecycle ----------------------------- #

    def __init__(
        self,
        status_callback: Callable[[], Dict[str, Any]],
        control_callback: Callable[[str, str], bool],
        summary_callback: Callable[[], str],
    ) -> None:
        # Lazy import so Config overrides in tests don’t explode imports
        from src.config import Config

        self.bot_token: str = getattr(Config, "TELEGRAM_BOT_TOKEN", "")
        self.chat_id: str | int = getattr(Config, "TELEGRAM_CHAT_ID", "")
        if not self.bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN is required in Config")
        if not self.chat_id:
            raise ValueError("TELEGRAM_CHAT_ID is required in Config")

        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self._timeout_longpoll: int = int(getattr(Config, "TELEGRAM_POLL_TIMEOUT", 30))
        self._http_timeout: int = self._timeout_longpoll + 5
        self._min_send_gap_s: float = float(getattr(Config, "TELEGRAM_MIN_SEND_GAP_SEC", 0.4))

        # poll state
        self.polling = False
        self._offset: Optional[int] = None
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # single-instance lock (prevents 409 if multiple containers run)
        self._lock_path = "/tmp/nifty_scalper_telegram.lock"
        self._lock_fh = None  # type: ignore

        # callbacks
        self.status_callback = status_callback
        self.control_callback = control_callback
        self.summary_callback = summary_callback

        # rate limit on sends
        self._last_send_ts: float = 0.0

        logger.info("TelegramController initialized.")

    # ------------------------------ locking ------------------------------ #

    def _acquire_lock(self) -> bool:
        """
        Acquire an exclusive lock so only one poller runs per host.
        Works on Linux (Railway). If locking fails, we return False.
        """
        try:
            self._lock_fh = open(self._lock_path, "w")
            try:
                # Prefer fcntl on POSIX
                import fcntl  # type: ignore

                fcntl.flock(self._lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._lock_fh.write(str(os.getpid()))
                self._lock_fh.flush()
                logger.info("📌 Telegram poll lock acquired (%s).", self._lock_path)
                return True
            except Exception as e:
                logger.warning("Locking with fcntl failed: %s", _readable_exc(e))
                # Try a poor-man check: if file empty or stale, take it
                self._lock_fh.seek(0)
                content = (self._lock_fh.read() or "").strip()
                if not content:
                    self._lock_fh.seek(0)
                    self._lock_fh.write(str(os.getpid()))
                    self._lock_fh.flush()
                    logger.info("📌 Telegram poll lock (best-effort) acquired.")
                    return True
                logger.info("🔒 Another poller appears active (pid: %s).", content)
                return False
        except Exception as e:
            logger.error("Failed to open lock file: %s", _readable_exc(e))
            return False

    def _release_lock(self) -> None:
        try:
            if self._lock_fh:
                try:
                    import fcntl  # type: ignore

                    fcntl.flock(self._lock_fh.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
                try:
                    self._lock_fh.close()
                except Exception:
                    pass
                # Best-effort cleanup
                try:
                    os.remove(self._lock_path)
                except Exception:
                    pass
                self._lock_fh = None
                logger.info("🔓 Telegram poll lock released.")
        except Exception:
            pass

    # ---------------------------- HTTP helpers --------------------------- #

    def _post(self, method: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        url = f"{self.base_url}/{method}"
        last_exc = None
        for i in range(3):
            try:
                resp = requests.post(url, json=payload, timeout=self._http_timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("ok"):
                        return data
                    logger.error("Telegram API error on %s: %s", method, data)
                    return None
                if resp.status_code == 409:
                    # Another getUpdates/webhook is active
                    logger.warning("Telegram 409 on %s. Will attempt webhook reset.", method)
                    self._reset_webhook()
                    time.sleep(1.2)
                    continue
                logger.warning("Telegram POST %s failed (%s): %s", method, resp.status_code, resp.text)
            except Exception as exc:
                last_exc = exc
                logger.debug("Telegram POST %s attempt %d failed: %s", method, i + 1, _readable_exc(exc))
            time.sleep(0.9 * (2**i))
        if last_exc:
            logger.error("Telegram POST %s failed after retries: %s", method, _readable_exc(last_exc))
        return None

    def _get(self, method: str, params: Dict[str, Any] | None = None) -> Optional[Dict[str, Any]]:
        url = f"{self.base_url}/{method}"
        last_exc = None
        for i in range(3):
            try:
                resp = requests.get(url, params=params or {}, timeout=self._http_timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("ok"):
                        return data
                    logger.error("Telegram API error on %s: %s", method, data)
                    return None
                if resp.status_code == 409:
                    logger.warning("Telegram 409 on %s. Will attempt webhook reset.", method)
                    self._reset_webhook()
                    time.sleep(1.2)
                    continue
                logger.warning("Telegram GET %s failed (%s): %s", method, resp.status_code, resp.text)
            except Exception as exc:
                last_exc = exc
                logger.debug("Telegram GET %s attempt %d failed: %s", method, i + 1, _readable_exc(exc))
            time.sleep(0.9 * (2**i))
        if last_exc:
            logger.error("Telegram GET %s failed after retries: %s", method, _readable_exc(last_exc))
        return None

    # ------------------------- Webhook compatibility ---------------------- #

    def _reset_webhook(self) -> None:
        """
        Ensure no webhook is set so getUpdates long-polling works.
        Safe to call multiple times.
        """
        try:
            info = self._get("getWebhookInfo") or {}
            res = info.get("result") or {}
            url = res.get("url") or ""
            if url:
                logger.info("Found existing webhook (%s). Deleting…", url)
                self._post("deleteWebhook", {"drop_pending_updates": False})
                # After delete, Telegram may still return 409 once; caller retries
        except Exception as e:
            logger.debug("getWebhookInfo/deleteWebhook error: %s", _readable_exc(e))

    # ------------------------------ sending ------------------------------- #

    def _throttle(self) -> None:
        now = time.time()
        gap = now - self._last_send_ts
        if gap < self._min_send_gap_s:
            time.sleep(self._min_send_gap_s - gap)
        self._last_send_ts = time.time()

    def _send_message(
        self,
        text: str,
        *,
        parse_mode: Optional[str] = "HTML",
        disable_web_page_preview: bool = True,
    ) -> bool:
        if not text:
            return False
        self._throttle()
        payload: Dict[str, Any] = {
            "chat_id": self.chat_id,
            "text": text,
            "disable_web_page_preview": disable_web_page_preview,
            "disable_notification": False,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode
        data = self._post("sendMessage", payload)
        ok = bool(data and data.get("ok"))
        if not ok:
            logger.error("Failed to send Telegram message.")
        return ok

    # public wrapper
    def send_message(self, text: str, parse_mode: Optional[str] = "HTML") -> bool:
        return self._send_message(text, parse_mode=parse_mode)

    def send_startup_alert(self) -> None:
        self._send_message("🟢 <b>Nifty Scalper Bot</b> started.\nType <code>/help</code> for commands.")

    def send_realtime_session_alert(self, action: str) -> None:
        msg = {
            "START": "✅ Real-time trading session <b>STARTED</b>.",
            "STOP": "🛑 Real-time trading session <b>STOPPED</b>.",
        }.get(action.upper(), f"ℹ️ Session {html.escape(action)}.")
        self._send_message(msg)

    # compatibility alias (older trader code)
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
        self._send_message(text)

    # -------------------------- formatting helpers ------------------------ #

    def _format_status(self, status: Any) -> Tuple[str, str]:
        # Allow trader to return a plain string
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
        regime = status.get("regime", "").upper() if status.get("regime") else None
        qreason = status.get("quality_reason")  # optional string from trader

        lines = [
            "📊 <b>Bot Status</b>",
            f"🔁 <b>Trading:</b> {'🟢 Running' if is_trading else '🔴 Stopped'}",
            f"🌐 <b>Mode:</b> {'🟢 LIVE' if live_mode else '🛡️ Shadow'}",
            f"✨ <b>Quality:</b> {'ON' if quality_mode else 'OFF'}",
        ]
        if regime:
            lines.append(f"🧭 <b>Regime:</b> {html.escape(regime)}")
        if qreason:
            lines.append(f"ℹ️ <i>{html.escape(str(qreason))}</i>")
        lines.extend(
            [
                f"📦 <b>Open Positions:</b> {open_positions}",
                f"📈 <b>Closed Today:</b> {trades_today}",
                f"💰 <b>Daily P&L:</b> {daily_pnl:.2f}",
            ]
        )
        if acct is not None:
            lines.append(f"🏦 <b>Acct Size:</b> ₹{html.escape(str(acct))}")
        if sess is not None:
            lines.append(f"📅 <b>Session:</b> {html.escape(str(sess))}")

        return "\n".join(lines), "HTML"

    def _send_status(self, status: Any) -> None:
        txt, mode = self._format_status(status)
        self._send_message(txt, parse_mode=mode)

    def _send_summary(self, summary: str) -> None:
        # summary string expected to be HTML from trader
        self._send_message(summary, parse_mode="HTML")

    # ------------------------- command parsing/router --------------------- #

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
            if val > 1.5:  # user meant percent
                val = val / 100.0
            # return normalized PERCENT string (for user echo); trader can parse either
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
                "/mode live|shadow|quality on|off|auto – switch mode\n"
                "/regime auto|trend|range|off – set regime gate\n"
                "/risk &lt;pct&gt; – e.g. <code>/risk 0.5</code> (means 0.5%)\n"
                "/pause &lt;min&gt; – pause entries\n"
                "/resume – resume entries\n"
                "/status – bot status\n"
                "/summary – daily summary\n"
                "/refresh – refresh balance/instruments\n"
                "/health – system health\n"
                "/emergency – cancel orders & clear (best-effort)",
            )
            return

        if cmd == "status":
            try:
                status = self.status_callback()
            except Exception as e:
                logger.error("status_callback error: %s", _readable_exc(e), exc_info=True)
                self._send_message("⚠️ Failed to fetch status.")
                return
            self._send_status(status)
            return

        if cmd == "summary":
            try:
                summary = self.summary_callback()
            except Exception as e:
                logger.error("summary_callback error: %s", _readable_exc(e), exc_info=True)
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

        if cmd == "regime":
            a = (arg or "").strip().lower()
            if a not in {"auto", "trend", "range", "off"}:
                self._send_message("⚠️ Usage: /regime auto|trend|range|off")
                return
            ok = self.control_callback("regime", a)
            if not ok:
                self._send_message("⚠️ Failed to change regime.")
            return

        if cmd == "risk":
            norm = self._normalize_risk_arg(arg)
            ok = bool(norm) and self.control_callback("risk", norm)
            if not ok:
                self._send_message("⚠️ Usage: /risk 0.5  (meaning 0.5%)")
            else:
                self._send_message(f"✅ Risk per trade set to <b>{norm}%</b>.")
            return

        if cmd in {"start", "stop", "pause", "resume", "refresh", "health", "emergency"}:
            ok = self.control_callback(cmd, arg or "")
            if not ok:
                self._send_message(f"⚠️ Command '/{cmd} {html.escape(arg)}' failed.")
            return

        self._send_message("❌ Unknown command. Try <code>/help</code>.")

    # ------------------------------ polling ------------------------------ #

    def _poll_updates(self) -> None:
        logger.info("📡 Telegram polling started. Awaiting commands...")
        self._stop_evt.clear()

        # Ensure no webhook conflicts
        self._reset_webhook()

        while self.polling and not self._stop_evt.is_set():
            try:
                params: Dict[str, Any] = {"timeout": self._timeout_longpoll}
                if self._offset is not None:
                    params["offset"] = self._offset

                data = self._get("getUpdates", params)
                if not data:
                    continue

                results = data.get("result", [])
                for item in results:
                    try:
                        self._offset = int(item["update_id"]) + 1
                    except Exception:
                        pass
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
                logger.error("Error in Telegram polling: %s", _readable_exc(exc), exc_info=True)
                time.sleep(3.0)

        logger.info("🛑 Telegram polling stopped.")

    # --------------------------- public methods --------------------------- #

    def start_polling(self) -> None:
        if self.polling:
            logger.warning("Polling already active.")
            return

        # Attempt singleton lock first
        if not self._acquire_lock():
            logger.warning(
                "Another process holds the Telegram poller lock. "
                "This instance will NOT poll (avoids 409)."
            )
            return

        self.polling = True
        self._thread = threading.Thread(target=self._poll_updates, name="TelegramPolling", daemon=True)
        self._thread.start()

    def stop_polling(self) -> None:
        logger.info("🛑 Stopping Telegram polling…")
        self.polling = False
        self._stop_evt.set()
        if self._thread and self._thread.is_alive():
            try:
                self._thread.join(timeout=5)
            except Exception:
                pass
        self._thread = None
        self._release_lock()