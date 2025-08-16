# src/notifications/telegram_controller.py
from __future__ import annotations

import html
import logging
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


class TelegramController:
    """
    Telegram bot interface for receiving commands and sending alerts.
    Communicates with RealTimeTrader via callback functions.

    Commands:
      /start
      /stop
      /mode live|shadow|quality on|off|auto
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

    # --- lifecycle ---------------------------------------------------------

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

    # --- HTTP helpers ------------------------------------------------------

    def _post(self, method: str, payload: Dict[str, Any], *, json_mode: bool = True) -> Optional[Dict[str, Any]]:
        url = f"{self.base_url}/{method}"
        tries = 3
        backoff = 0.9
        last_exc = None
        for i in range(tries):
            try:
                if json_mode:
                    resp = requests.post(url, json=payload, timeout=self._request_timeout_s)
                else:
                    resp = requests.post(url, data=payload, timeout=self._request_timeout_s)

                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("ok"):
                        return data
                    logger.error("Telegram API error: %s", data)
                    return None

                # 409 = another poller/webhook active, stop polling
                if resp.status_code == 409:
                    logger.error("Telegram conflict (409). Another instance is active. Stopping polling.")
                    self.stop_polling()
                    return None

                logger.warning("Telegram POST %s failed (%s): %s", method, resp.status_code, resp.text)
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
            time.sleep(backoff * (2**i))
        if last_exc:
            logger.error("Telegram GET %s failed after retries: %s", method, last_exc)
        return None

    # --- sending -----------------------------------------------------------

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

    # public simple wrapper
    def send_message(self, text: str, parse_mode: Optional[str] = None) -> bool:
        # default to HTML for your formatted strings; call with None for plain text
        return self._send_message(text, parse_mode=parse_mode or "HTML")

    def send_startup_alert(self) -> None:
        self._send_message("🟢 <b>Nifty Scalper Bot</b> started.\nType <code>/help</code> for commands.", parse_mode="HTML")

    def send_realtime_session_alert(self, action: str) -> None:
        msg = {"START": "✅ Real-time trading session <b>STARTED</b>.",
               "STOP": "🛑 Real-time trading session <b>STOPPED</b>."}.get(action.upper(), f"ℹ️ Session {html.escape(action)}.")
        self._send_message(msg, parse_mode="HTML")

    # compatibility with RealTimeTrader._safe_send_alert()
    def send_alert(self, action: str) -> None:
        self.send_realtime_session_alert(action)

    def send_signal_alert(self, token: int, signal: Dict[str, Any], position: Dict[str, Any]) -> None:
        # HTML-escape dynamic bits
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

    # --- formatting helpers -----------------------------------------------

    @staticmethod
    def _fmt_quality_line(status: Dict[str, Any]) -> str:
        # Backward compatible: if policy not provided, infer from boolean flag
        policy = (status.get("quality_policy") or "").strip().lower()
        qm = status.get("quality_mode")  # boolean, legacy
        if not policy:
            policy = "on" if qm else "off"

        label = {"on": "ON", "off": "OFF", "auto": "AUTO"}.get(policy, policy.upper() or "OFF")

        # Optional: reason text and regime metrics (if trader provides them)
        reason = status.get("quality_reason")
        parts = [f"✨ <b>Quality:</b> {label}"]
        if policy == "auto":
            if reason:
                parts.append(f"(auto: {html.escape(str(reason))})")
            # If regime supplied, display succinctly
            regime = (status.get("regime") or status.get("regime_mode") or "").upper()
            if regime:
                parts.append(f"[{regime}]")
        return " ".join(parts)

    def _format_status(self, status: Any) -> Tuple[str, str]:
        """Returns (text, parse_mode)."""
        if isinstance(status, str):
            return f"📊 Status:\n{html.escape(status)}", "HTML"

        is_trading = bool(status.get("is_trading", False))
        live_mode = bool(status.get("live_mode", False))
        open_positions = int(status.get("open_positions", status.get("open_orders", 0)) or 0)
        trades_today = int(status.get("trades_today", status.get("closed_today", 0)) or 0)
        daily_pnl = float(status.get("daily_pnl", 0.0) or 0.0)
        acct = status.get("account_size")
        sess = status.get("session_date")

        # Optional extras that may be present:
        regime = (status.get("regime") or status.get("regime_mode") or "")
        adx = status.get("adx")
        bbw = status.get("bb_width")
        slope = status.get("ema_slope")

        lines = [
            "📊 <b>Bot Status</b>",
            f"🔁 <b>Trading:</b> {'🟢 Running' if is_trading else '🔴 Stopped'}",
            f"🌐 <b>Mode:</b> {'🟢 LIVE' if live_mode else '🛡️ Shadow'}",
            self._fmt_quality_line(status),
            f"📦 <b>Open Positions:</b> {open_positions}",
            f"📈 <b>Closed Today:</b> {trades_today}",
            f"💰 <b>Daily P&L:</b> {daily_pnl:.2f}",
        ]
        if acct is not None:
            lines.append(f"🏦 <b>Acct Size:</b> ₹{html.escape(str(acct))}")
        if sess is not None:
            lines.append(f"📅 <b>Session:</b> {html.escape(str(sess))}")

        # Append a compact regime diagnostics line when available
        diag_bits = []
        if regime:
            diag_bits.append(f"Regime={html.escape(str(regime)).upper()}")
        if isinstance(adx, (int, float)):
            diag_bits.append(f"ADX={float(adx):.1f}")
        if isinstance(bbw, (int, float)):
            diag_bits.append(f"BBWidth={float(bbw):.3%}")
        if isinstance(slope, (int, float)):
            diag_bits.append(f"EMAΔ={float(slope):.2f}")
        if diag_bits:
            lines.append("🧭 " + " | ".join(diag_bits))

        return "\n".join(lines), "HTML"

    def _send_status(self, status: Any) -> None:
        txt, mode = self._format_status(status)
        self._send_message(txt, parse_mode=mode)

    def _send_summary(self, summary: str) -> None:
        # summary is already formatted HTML upstream
        self._send_message(summary, parse_mode="HTML")

    # --- command parsing/router -------------------------------------------

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
        Accepts '0.5' or '0.5%' or '1' (=> 1%). Returns a normalized string the trader can parse.
        """
        a = (arg or "").strip().replace("%", "")
        try:
            val = float(a)
            if val > 1.5:  # user probably meant percent, not fraction
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
                "/mode live|shadow|quality on|off|auto – switch mode or quality policy\n"
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
            # Accepts:
            #   /mode live
            #   /mode shadow
            #   /mode quality on
            #   /mode quality off
            #   /mode quality auto
            a = (arg or "").strip().lower()
            ok = self.control_callback("mode", a)
            if not ok:
                self._send_message("⚠️ Failed to change mode.")
            else:
                self._send_message(f"✅ Mode updated: <code>{html.escape(a)}</code>", parse_mode="HTML")
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

    # --- polling loop ------------------------------------------------------

    def _poll_updates(self) -> None:
        logger.info("📡 Telegram polling started. Awaiting commands...")
        self._stop_evt.clear()
        while self.polling and not self._stop_evt.is_set():
            try:
                params = {"timeout": self._timeout_s}
                if self._offset is not None:
                    params["offset"] = self._offset
                data = self._get("getUpdates", params)  # handles retries & 409
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

    # --- public lifecycle --------------------------------------------------

    def start_polling(self) -> None:
        if self.polling:
            logger.warning("Polling already active.")
            return
        self.polling = True
        self.polling_thread = threading.Thread(
            target=self._poll_updates,
            name="TelegramPolling",
            daemon=True,
        )
        self.polling_thread.start()

    def stop_polling(self) -> None:
        logger.info("🛑 Stopping Telegram polling…")
        self.polling = False
        self._stop_evt.set()
        if self.polling_thread and self.polling_thread.is_alive():
            self.polling_thread.join(timeout=5)
        self.polling_thread = None