# src/notifications/telegram_controller.py
from __future__ import annotations

import html
import json
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


def _bool_env(v: Optional[str], default: bool = False) -> bool:
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


class TelegramController:
    """
    Lightweight, dependency-free Telegram long-polling controller (requests-based).

    Core contract (same as your previous version):
        status_callback() -> dict
        control_callback(cmd: str, arg: str) -> bool
        summary_callback() -> str

    Optional config bridges (if you want /config get|set):
        set_config_bridges(getter: () -> dict, setter: (key: str, val: str) -> str)

    Commands:
      /help
      /start /stop /refresh /health /emergency
      /status /summary /id /ping
      /mode [live|shadow]              (default: live)
      /quality [auto|on|off]           (default: auto)
      /regime [auto|trend|range|off]   (default: auto)
      /risk <pct>                      (e.g. 0.5 = 0.5%)
      /pause [minutes]                 (default: 1)
      /resume
      /config get
      /config set <key> <value>

    Startup behavior:
      - Deletes webhook (prevents 409 conflicts)
      - Optionally sets bot command menu
      - Can auto-latch first chat ID if not provided (toggle via env/arg)
    """

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        status_callback: Callable[[], Dict[str, Any]],
        control_callback: Callable[[str, str], bool],
        summary_callback: Callable[[], str],
        *,
        bot_token: Optional[str] = None,
        chat_id: Optional[int | str] = None,
        latch_first_chat: Optional[bool] = None,
        poll_timeout_sec: Optional[int] = None,
        min_send_gap_sec: Optional[float] = None,
    ) -> None:
        self.status_callback = status_callback
        self.control_callback = control_callback
        self.summary_callback = summary_callback

        # Allow wiring either directly or via settings/env
        if bot_token is None or chat_id is None:
            # Try pydantic settings first
            try:
                from src.config import settings  # pydantic v2 settings
                bot_token = bot_token or getattr(getattr(settings, "telegram", object()), "bot_token", None)
                chat_id = chat_id or getattr(getattr(settings, "telegram", object()), "chat_id", None)
            except Exception:
                pass
        # Finally, env fallback
        bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "").strip()

        if not bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN not provided (arg/settings/env).")
        self.bot_token: str = bot_token

        # Auto-latch toggle: default False unless explicitly enabled via env/arg
        if latch_first_chat is None:
            latch_first_chat = _bool_env(os.getenv("TELEGRAM_LATCH_FIRST_CHAT"), False)
        self._latch_first_chat: bool = bool(latch_first_chat)

        self.chat_id: str | int | None = None
        if str(chat_id).strip():
            try:
                self.chat_id = int(str(chat_id).strip())
            except Exception:
                self.chat_id = str(chat_id).strip()

        # Networking/polling config
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self._timeout_s: int = int(poll_timeout_sec or int(os.getenv("TELEGRAM_POLL_TIMEOUT", "30")))
        self._request_timeout_s: int = self._timeout_s + 5
        self._min_send_gap_s: float = float(min_send_gap_sec or float(os.getenv("TELEGRAM_MIN_SEND_GAP_SEC", "0.4")))

        # Poll state
        self.polling = False
        self._offset: Optional[int] = None
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Rate-limit on sends
        self._last_send_ts: float = 0.0

        # Optional config bridges
        self._config_getter: Optional[Callable[[], Dict[str, Any]]] = None
        self._config_setter: Optional[Callable[[str, str], str]] = None

        logger.info("TelegramController initialized (latch_first_chat=%s).", self._latch_first_chat)

    def set_config_bridges(
        self,
        getter: Optional[Callable[[], Dict[str, Any]]] = None,
        setter: Optional[Callable[[str, str], str]] = None,
    ) -> None:
        self._config_getter = getter
        self._config_setter = setter

    # ------------------------------------------------------------------ #
    # HTTP helpers
    # ------------------------------------------------------------------ #

    def _post(self, method: str, payload: Dict[str, Any], *, json_mode: bool = True) -> Optional[Dict[str, Any]]:
        url = f"{self.base_url}/{method}"
        tries = 3
        backoff = 0.9
        last_exc = None
        for i in range(tries):
            try:
                resp = requests.post(url, json=payload, timeout=self._request_timeout_s) if json_mode \
                    else requests.post(url, data=payload, timeout=self._request_timeout_s)

                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("ok"):
                        return data
                    logger.error("Telegram API error (%s): %s", method, data)
                    return None

                if resp.status_code == 409:
                    # Another poller/webhook active: stop gracefully
                    logger.error("Telegram 409 conflict on %s. Stopping polling.", method)
                    self.stop_polling()
                    return None

                logger.warning("Telegram POST %s failed (%s): %s", method, resp.status_code, resp.text[:180])
            except Exception as exc:
                last_exc = exc
                logger.debug("Telegram POST %s error (attempt %d): %s", method, i + 1, exc)
            time.sleep(backoff * (2**i))
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
                    logger.error("Telegram API error (%s): %s", method, data)
                    return None

                if resp.status_code == 409:
                    logger.error("Telegram 409 conflict on %s. Stopping polling.", method)
                    self.stop_polling()
                    return None

                logger.warning("Telegram GET %s failed (%s): %s", method, resp.status_code, resp.text[:180])
            except Exception as exc:
                last_exc = exc
                logger.debug("Telegram GET %s error (attempt %d): %s", method, i + 1, exc)
            time.sleep(backoff * (2**i))
        if last_exc:
            logger.error("Telegram GET %s failed after retries: %s", method, last_exc)
        return None

    # ------------------------------------------------------------------ #
    # Sending helpers
    # ------------------------------------------------------------------ #

    def _throttle(self) -> None:
        now = time.time()
        gap = now - self._last_send_ts
        if gap < self._min_send_gap_s:
            time.sleep(self._min_send_gap_s - gap)
        self._last_send_ts = time.time()

    def _send_message(
        self,
        text: str,
        parse_mode: Optional[str] = None,
        disable_web_page_preview: bool = True,
        chat_id: Optional[int | str] = None,
    ) -> bool:
        if not text:
            return False
        target = chat_id or self.chat_id
        if not target:
            return False
        self._throttle()
        payload: Dict[str, Any] = {
            "chat_id": target,
            "text": text,
            "disable_notification": False,
            "disable_web_page_preview": disable_web_page_preview,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode
        data = self._post("sendMessage", payload)
        return bool(data and data.get("ok"))

    def send_message(self, text: str, parse_mode: Optional[str] = None) -> bool:
        # default to HTML for formatted strings
        return self._send_message(text, parse_mode=parse_mode or "HTML")

    # ------------------------------------------------------------------ #
    # Formatting helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _fmt_uptime(seconds: Optional[float]) -> str:
        if not seconds or seconds <= 0:
            return "—"
        s = int(seconds)
        h, r = divmod(s, 3600)
        m, _ = divmod(r, 60)
        return f"{h}h {m}m" if h else f"{m}m"

    def _format_status(self, status: Any) -> Tuple[str, str]:
        """Returns (text, parse_mode)."""
        if isinstance(status, str):
            return f"📊 Status:\n{html.escape(status)}", "HTML"

        is_trading = bool(status.get("is_trading", False))
        live_mode = bool(status.get("live_mode", False))

        q_mode_raw = status.get("quality_mode", "AUTO")
        q_auto = bool(status.get("quality_auto", False))
        q_reason = status.get("quality_reason")
        if isinstance(q_mode_raw, str):
            q_mode_label = q_mode_raw.upper()
        elif isinstance(q_mode_raw, bool):
            q_mode_label = "ON" if q_mode_raw else "OFF"
        else:
            q_mode_label = "AUTO" if q_auto else "OFF"

        regime_mode = (status.get("regime_mode") or "AUTO")
        regime_reason = status.get("regime_reason")

        open_positions = int(status.get("open_positions", status.get("open_orders", 0)) or 0)
        trades_today = int(status.get("trades_today", status.get("closed_today", 0)) or 0)
        daily_pnl = float(status.get("daily_pnl", 0.0) or 0.0)
        acct = status.get("account_size")
        sess = status.get("session_date")
        uptime = self._fmt_uptime(float(status.get("uptime_sec", 0.0) or 0.0))

        lines = [
            "📊 <b>Bot Status</b>",
            f"🔁 <b>Trading:</b> {'🟢 Running' if is_trading else '🔴 Stopped'}",
            f"🌐 <b>Mode:</b> {'🟢 LIVE' if live_mode else '🛡️ Shadow'}",
            f"✨ <b>Quality:</b> {html.escape(q_mode_label)}" + (f" <i>({html.escape(q_reason)})</i>" if q_reason else ""),
            f"🧭 <b>Regime:</b> {html.escape(str(regime_mode))}" + (f" <i>({html.escape(str(regime_reason))})</i>" if regime_reason else ""),
            f"📦 <b>Open Positions:</b> {open_positions}",
            f"📈 <b>Closed Today:</b> {trades_today}",
            f"💰 <b>Daily P&L:</b> {daily_pnl:.2f}",
        ]
        if acct is not None:
            lines.append(f"🏦 <b>Acct Size:</b> ₹{html.escape(str(acct))}")
        if sess is not None:
            lines.append(f"📅 <b>Session:</b> {html.escape(str(sess))}")
        if uptime != "—":
            lines.append(f"⏱️ <b>Uptime:</b> {uptime}")
        return "\n".join(lines), "HTML"

    def _send_status(self, status: Any) -> None:
        txt, mode = self._format_status(status)
        self._send_message(txt, parse_mode=mode)

    def _send_summary(self, summary: str) -> None:
        self._send_message(summary, parse_mode="HTML")

    # ------------------------------------------------------------------ #
    # Command parsing/router
    # ------------------------------------------------------------------ #

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
        """
        Accepts '0.5' or '0.5%' or '1' (=> 1%). Returns normalized percent string.
        """
        a = (arg or "").strip().replace("%", "")
        try:
            val = float(a)
            if val > 1.5:  # user probably meant percent, not fraction
                val = val / 100.0
            return f"{val:.4f}"
        except Exception:
            return ""

    def _handle_command(self, message: Dict[str, Any]) -> None:
        text = (message.get("text") or "").strip()
        chat = message.get("chat") or {}
        cid = chat.get("id")

        # latch first chat if enabled and not set
        if not self.chat_id and self._latch_first_chat and cid:
            self.chat_id = cid
            logger.info("Telegram latched chat_id=%s", cid)

        # Enforce chat_id if configured
        if self.chat_id and cid and str(cid) != str(self.chat_id):
            logger.info("Telegram unauthorized chat: got %s expected %s", cid, self.chat_id)
            return

        cmd, arg = self._parse_command_and_arg(text)
        if not cmd:
            return

        logger.info("📩 Telegram cmd: /%s %s", cmd, arg)

        if cmd == "help":
            self._send_message(
                "🤖 <b>Commands</b>\n"
                "/start – start trading\n"
                "/stop – stop trading\n"
                "/mode [live|shadow] – default <code>live</code>\n"
                "/quality [auto|on|off] – default <code>auto</code>\n"
                "/regime [auto|trend|range|off] – default <code>auto</code>\n"
                "/risk &lt;pct&gt; – e.g. <code>/risk 0.5</code> (0.5%)\n"
                "/pause [min] – default <code>1</code>\n"
                "/resume – resume entries\n"
                "/status – bot status\n"
                "/summary – daily summary\n"
                "/refresh – refresh state\n"
                "/health – system health\n"
                "/id – show chat id\n"
                "/ping – latency check\n"
                "/config get | /config set &lt;k&gt; &lt;v&gt;\n"
                "/emergency – cancel orders & flatten (best-effort).",
                parse_mode="HTML",
            )
            return

        if cmd == "ping":
            self._send_message("pong")
            return

        if cmd == "id":
            self._send_message(f"chat_id: <code>{cid}</code>", parse_mode="HTML")
            return

        if cmd == "status":
            try:
                status = self.status_callback()
                self._send_status(status)
            except Exception as e:
                logger.error("/status error: %s", e, exc_info=True)
                self._send_message("⚠️ Failed to fetch status.")
            return

        if cmd == "summary":
            try:
                summary = self.summary_callback()
                self._send_summary(summary)
            except Exception as e:
                logger.error("/summary error: %s", e, exc_info=True)
                self._send_message("⚠️ Failed to fetch summary.")
            return

        if cmd == "mode":
            mode_arg = (arg.lower() if arg else "live")
            if mode_arg not in {"live", "shadow"}:
                self._send_message("⚠️ Usage: /mode [live|shadow]. Default is <code>live</code>.", parse_mode="HTML")
                return
            ok = self.control_callback("mode", mode_arg)
            self._send_message("✅ Mode set to <b>%s</b>." % mode_arg.upper() if ok else "⚠️ Failed to change mode.", parse_mode="HTML")
            return

        if cmd == "quality":
            q_arg = (arg.lower() if arg else "auto")
            if q_arg not in {"auto", "on", "off"}:
                self._send_message("⚠️ Usage: /quality [auto|on|off]. Default is <code>auto</code>.", parse_mode="HTML")
                return
            ok = self.control_callback("quality", q_arg)
            self._send_message("✅ Quality set to <b>%s</b>." % q_arg.upper() if ok else "⚠️ Failed to set quality.", parse_mode="HTML")
            return

        if cmd == "regime":
            r_arg = (arg.lower() if arg else "auto")
            if r_arg not in {"auto", "trend", "range", "off"}:
                self._send_message("⚠️ Usage: /regime [auto|trend|range|off]. Default is <code>auto</code>.", parse_mode="HTML")
                return
            ok = self.control_callback("regime", r_arg)
            self._send_message("✅ Regime set to <b>%s</b>." % r_arg.upper() if ok else "⚠️ Failed to set regime.", parse_mode="HTML")
            return

        if cmd == "risk":
            norm = self._normalize_risk_arg(arg)
            ok = bool(norm) and self.control_callback("risk", norm)
            self._send_message(f"✅ Risk per trade set to <b>{norm}%</b>." if ok else "⚠️ Usage: /risk 0.5", parse_mode="HTML")
            return

        if cmd == "pause":
            mins = 1
            if arg:
                try:
                    mins = max(1, int(float(arg)))
                except Exception:
                    mins = 1
            ok = self.control_callback("pause", str(mins))
            self._send_message(f"⏸️ Paused entries for <b>{mins} min</b>." if ok else "⚠️ Failed to pause.", parse_mode="HTML")
            return

        if cmd == "resume":
            ok = self.control_callback("resume", "")
            self._send_message("▶️ Resumed entries." if ok else "⚠️ Failed to resume.", parse_mode="HTML")
            return

        if cmd == "config":
            parts = (arg or "").split()
            if len(parts) == 0:
                self._send_message("Usage:\n/config get\n/config set <key> <value>")
                return
            sub = parts[0].lower()
            if sub == "get":
                if not self._config_getter:
                    self._send_message("Config getter not wired.")
                    return
                try:
                    cfg = self._config_getter()
                except Exception as e:
                    self._send_message(f"error: {e}")
                    return
                text = json.dumps(cfg, indent=2, default=str)
                self._send_message(f"<pre>{html.escape(text)}</pre>", parse_mode="HTML")
                return
            if sub == "set":
                if not self._config_setter:
                    self._send_message("Config setter not wired.")
                    return
                if len(parts) < 3:
                    self._send_message("Usage: /config set <key> <value>")
                    return
                key, value = parts[1], " ".join(parts[2:])
                try:
                    msg = self._config_setter(key, value)
                except Exception as e:
                    msg = f"error: {e}"
                self._send_message(f"<code>{html.escape(str(msg))}</code>", parse_mode="HTML")
                return
            self._send_message("Usage:\n/config get\n/config set <key> <value>")
            return

        if cmd in {"start", "stop", "refresh", "health", "emergency"}:
            ok = self.control_callback(cmd, "")
            if not ok:
                self._send_message(f"⚠️ Command '/{cmd}' failed.")
            return

        self._send_message("❌ Unknown command. Try <code>/help</code>.", parse_mode="HTML")

    # ------------------------------------------------------------------ #
    # Polling lifecycle
    # ------------------------------------------------------------------ #

    def _delete_webhook(self) -> None:
        self._post("deleteWebhook", {})

    def _set_my_commands(self) -> None:
        commands: List[Dict[str, str]] = [
            {"command": "status", "description": "Bot status"},
            {"command": "summary", "description": "Recent trades summary"},
            {"command": "mode", "description": "Mode: live | shadow"},
            {"command": "quality", "description": "Quality: auto | on | off"},
            {"command": "regime", "description": "Regime: auto | trend | range | off"},
            {"command": "risk", "description": "Risk per trade (e.g. 0.5)"},
            {"command": "pause", "description": "Pause entries (minutes)"},
            {"command": "resume", "description": "Resume entries"},
            {"command": "refresh", "description": "Refresh state"},
            {"command": "health", "description": "System health"},
            {"command": "emergency", "description": "Flatten & cancel"},
            {"command": "id", "description": "Show chat id"},
            {"command": "ping", "description": "Ping"},
            {"command": "help", "description": "Help"},
            {"command": "config", "description": "Get/Set runtime config"},
        ]
        self._post("setMyCommands", {"commands": commands})

    def _poll_updates(self) -> None:
        logger.info("📡 Telegram polling started. Awaiting commands...")
        self._stop_evt.clear()
        while self.polling and not self._stop_evt.is_set():
            try:
                params: Dict[str, Any] = {"timeout": self._timeout_s}
                if self._offset is not None:
                    params["offset"] = self._offset
                data = self._get("getUpdates", params)
                if not data:
                    continue
                for item in data.get("result", []):
                    self._offset = int(item["update_id"]) + 1
                    message = item.get("message") or item.get("edited_message") or {}
                    if not message or not (message.get("text") or "").startswith("/"):
                        # Allow auto-latch even if text isn't a command
                        if self._latch_first_chat and not self.chat_id and message.get("chat", {}).get("id"):
                            self.chat_id = message["chat"]["id"]
                            logger.info("Telegram latched chat_id=%s (non-command message).", self.chat_id)
                        continue
                    self._handle_command(message)
            except requests.exceptions.ReadTimeout:
                logger.debug("Telegram polling timeout — continuing…")
            except Exception as exc:
                logger.error("Error in Telegram polling: %s", exc, exc_info=True)
                time.sleep(3.0)
        logger.info("🛑 Telegram polling stopped.")

    def start_polling(self) -> None:
        if self.polling:
            logger.warning("Polling already active.")
            return
        # Clean webhook to avoid 409 conflicts
        self._delete_webhook()
        # Expose commands in Telegram UI
        self._set_my_commands()

        self.polling = True
        self._thread = threading.Thread(target=self._poll_updates, name="TelegramPolling", daemon=True)
        self._thread.start()
        logger.info("Telegram polling thread started.")

        # Optional startup ping if chat is known
        if self.chat_id:
            self._send_message(
                "🟢 <b>Nifty Scalper Bot</b> online.\nUse /status, /summary, /health, /config get, /config set <k> <v>, /emergency",
                parse_mode="HTML",
            )

    def stop_polling(self) -> None:
        if not self.polling:
            return
        logger.info("🛑 Stopping Telegram polling…")
        self.polling = False
        self._stop_evt.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._thread = None
