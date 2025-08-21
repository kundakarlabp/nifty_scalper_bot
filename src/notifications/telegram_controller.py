from __future__ import annotations

"""
Telegram controller (long-poll).

- Single poll thread, resilient backoff
- Allowlist by chat id (settings.telegram.chat_id + optional extras)
- De-duplicated sends / throttled
- Rich command set: /help /status /mode /pause /resume /risk /config /flow(/diag) /logs /positions /active
"""

import hashlib
import json
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import requests

from src.config import settings

log = logging.getLogger(__name__)


def _truthy(s: Any) -> bool:
    return str(s).strip().lower() in ("1", "true", "on", "yes", "y", "live")


class TelegramController:
    def __init__(
        self,
        *,
        # read-only providers
        status_provider: Callable[[], Dict[str, Any]],
        positions_provider: Optional[Callable[[], Dict[str, Any]]] = None,
        actives_provider: Optional[Callable[[], List[Any]]] = None,
        # control hooks
        runner_pause: Optional[Callable[[], None]] = None,
        runner_resume: Optional[Callable[[], None]] = None,
        cancel_all: Optional[Callable[[], None]] = None,
        # runtime config hooks
        set_risk_pct: Optional[Callable[[float], None]] = None,
        toggle_trailing: Optional[Callable[[bool], None]] = None,
        set_trailing_mult: Optional[Callable[[float], None]] = None,
        toggle_partial: Optional[Callable[[bool], None]] = None,
        set_tp1_ratio: Optional[Callable[[float], None]] = None,
        set_breakeven_ticks: Optional[Callable[[int], None]] = None,
        set_live_mode: Optional[Callable[[bool], None]] = None,
        # diag/logs
        diag_provider: Optional[Callable[[], Dict[str, Any]]] = None,
        log_provider: Optional[Callable[[int], List[str]]] = None,
        http_timeout: float = 20.0,
    ) -> None:
        tg = settings.telegram
        self._token: Optional[str] = tg.bot_token
        self._chat_id: Optional[int] = tg.chat_id
        if not self._token or not self._chat_id:
            raise RuntimeError("TelegramController: bot_token or chat_id missing in settings.telegram")

        self._base = f"https://api.telegram.org/bot{self._token}"
        self._timeout = http_timeout

        # hooks/providers
        self._status_provider = status_provider
        self._positions_provider = positions_provider
        self._actives_provider = actives_provider
        self._runner_pause = runner_pause
        self._runner_resume = runner_resume
        self._cancel_all = cancel_all

        self._set_risk_pct = set_risk_pct
        self._toggle_trailing = toggle_trailing
        self._set_trailing_mult = set_trailing_mult
        self._toggle_partial = toggle_partial
        self._set_tp1_ratio = set_tp1_ratio
        self._set_breakeven_ticks = set_breakeven_ticks
        self._set_live_mode = set_live_mode

        self._diag_provider = diag_provider
        self._log_provider = log_provider

        # polling state
        self._poll_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._started = False
        self._last_update_id: Optional[int] = None

        # security: allowlist
        extra = list(getattr(tg, "extra_admin_ids", []) or [])
        self._allowlist = {int(self._chat_id), *[int(x) for x in extra]}

        # send rate & backoff
        self._send_min_interval = 0.9
        self._last_sends: List[tuple[float, str]] = []
        self._backoff = 1.0
        self._backoff_max = 20.0

        # prepared help text
        self._help = (
            "🤖 *Nifty Scalper Bot*\n"
            "/status — bot status\n"
            "/summary — recent trades summary\n"
            "/mode live|shadow — toggle live trading\n"
            "/pause [minutes] — pause entries\n"
            "/resume — resume entries\n"
            "/risk <pct> — set risk per trade\n"
            "/config [get <k>|set <k> <v>] — runtime config\n"
            "/positions — broker day positions\n"
            "/active [page] — active orders\n"
            "/flow — end-to-end diagnostic\n"
            "/logs [N] — recent logs (default 60)\n"
            "/health — system health (same as status)\n"
            "/ping — quick ping\n"
            "/id — show your chat id\n"
            "/emergency — flatten & cancel (confirm)"
        )

        # /config whitelist mapping (getter, setter, caster)
        self._cfg_map = {
            "risk.pct": (
                lambda: getattr(settings.risk, "risk_per_trade", 0.01) * 100.0,
                (self._set_risk_pct or (lambda v: setattr(settings.risk, "risk_per_trade", float(v) / 100.0))),
                float,
            ),
            "exec.trailing": (
                lambda: getattr(settings.executor, "enable_trailing", True),
                (self._toggle_trailing or (lambda v: setattr(settings.executor, "enable_trailing", bool(v)))),
                _truthy,
            ),
            "exec.trail_mult": (
                lambda: getattr(settings.executor, "trailing_atr_multiplier", 1.5),
                (self._set_trailing_mult or (lambda v: setattr(settings.executor, "trailing_atr_multiplier", float(v)))),
                float,
            ),
            "exec.partial": (
                lambda: getattr(settings.executor, "partial_tp_enable", False),
                (self._toggle_partial or (lambda v: setattr(settings.executor, "partial_tp_enable", bool(v)))),
                _truthy,
            ),
            "exec.tp1_ratio": (
                lambda: getattr(settings.executor, "tp1_qty_ratio", 0.5) * 100.0,
                (self._set_tp1_ratio or (lambda v: setattr(settings.executor, "tp1_qty_ratio", float(v) / 100.0))),
                float,
            ),
            "exec.be_ticks": (
                lambda: getattr(settings.executor, "breakeven_ticks", 2),
                (self._set_breakeven_ticks or (lambda v: setattr(settings.executor, "breakeven_ticks", int(v)))),
                int,
            ),
            "mode.live": (
                lambda: getattr(settings, "enable_live_trading", False),
                (self._set_live_mode or (lambda v: setattr(settings, "enable_live_trading", bool(v)))),
                _truthy,
            ),
        }

    # -------------------- outbound --------------------

    def _rate_ok(self, text: str) -> bool:
        now = time.time()
        h = hashlib.md5(text.encode("utf-8")).hexdigest()
        self._last_sends[:] = [(t, hh) for t, hh in self._last_sends if now - t < 10]
        if self._last_sends and now - self._last_sends[-1][0] < self._send_min_interval:
            return False
        if any(hh == h for _, hh in self._last_sends):
            return False
        self._last_sends.append((now, h))
        return True

    def _send(self, text: str, parse_mode: Optional[str] = None, disable_notification: bool = False) -> None:
        if not self._rate_ok(text):
            return
        delay = self._backoff
        while True:
            try:
                payload = {"chat_id": self._chat_id, "text": text, "disable_notification": disable_notification}
                if parse_mode:
                    payload["parse_mode"] = parse_mode
                requests.post(f"{self._base}/sendMessage", json=payload, timeout=self._timeout)
                self._backoff = 1.0
                return
            except Exception:
                time.sleep(delay)
                delay = min(self._backoff_max, delay * 2)
                self._backoff = delay

    def _send_inline(self, text: str, buttons: list[list[dict]]) -> None:
        payload = {
            "chat_id": self._chat_id,
            "text": text,
            "reply_markup": {"inline_keyboard": buttons},
        }
        try:
            requests.post(f"{self._base}/sendMessage", json=payload, timeout=self._timeout)
        except Exception as e:
            log.debug("Inline send failed: %s", e)

    def notify_text(self, text: str) -> None:
        self._send(text)

    def send_startup_alert(self) -> None:
        s = {}
        try:
            s = self._status_provider() if self._status_provider else {}
        except Exception:
            pass
        live = s.get("live_trading") or s.get("live") or False
        brk = (s.get("broker") or s.get("data_source") or "NA")
        self._send(
            "🚀 Nifty Scalper Bot online\n"
            f"🔁 Trading: {'🟢 LIVE' if live else '🟡 SHADOW'}\n"
            f"🧠 Source/Broker: {brk}"
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

    # -------------------- polling --------------------

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

    # -------------------- commands --------------------

    def _handle_update(self, upd: Dict[str, Any]) -> None:
        # inline callback
        if "callback_query" in upd:
            cq = upd["callback_query"]
            chat_id = cq.get("message", {}).get("chat", {}).get("id")
            if not self._authorized(int(chat_id)):
                return
            data = cq.get("data", "")
            try:
                if data == "confirm_cancel_all" and self._cancel_all:
                    self._cancel_all()
                    self._send("🧹 Cancelled all open orders.")
            finally:
                try:
                    requests.post(
                        f"{self._base}/answerCallbackQuery",
                        json={"callback_query_id": cq.get("id")},
                        timeout=self._timeout,
                    )
                except Exception:
                    pass
            return

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

        parts = text.split()
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd in ("/start", "/help"):
            return self._send(self._help, parse_mode="Markdown")

        if cmd in ("/status", "/health"):
            try:
                s = self._status_provider() if self._status_provider else {}
            except Exception:
                s = {}
            live = s.get("live_trading") or s.get("live") or False
            broker = s.get("broker") or s.get("data_source")
            active = s.get("active_orders", 0)
            when = s.get("time_ist")
            return self._send(
                f"📊 {when}\n"
                f"🔁 {'🟢 LIVE' if live else '🟡 SHADOW'} | {broker}\n"
                f"📦 Active: {active}"
            )

        if cmd == "/summary":
            return self._send("No summary yet.")  # your strategy can wire one later

        if cmd == "/mode":
            if not args:
                state = "LIVE" if getattr(settings, "enable_live_trading", False) else "SHADOW"
                return self._send(f"Mode is {state}. Usage: /mode live|shadow")
            want = args[0].lower()
            if want not in ("live", "shadow"):
                return self._send("Usage: /mode live|shadow")
            if self._set_live_mode:
                self._set_live_mode(want == "live")
            return self._send(f"Mode set to {'LIVE' if want == 'live' else 'SHADOW'}.")

        if cmd == "/pause":
            mins = 0
            if args:
                try: mins = max(0, int(args[0]))
                except Exception: mins = 0
            if self._runner_pause:
                self._runner_pause()
            if mins > 0:
                # optional timed resume could be added here if needed
                pass
            return self._send("⏸️ Entries paused.")

        if cmd == "/resume":
            if self._runner_resume:
                self._runner_resume()
            return self._send("▶️ Entries resumed.")

        if cmd == "/risk":
            if not args:
                pct = getattr(settings.risk, "risk_per_trade", 0.01) * 100.0
                return self._send(f"Current risk per trade: {pct:.2f}%. Usage: /risk 0.5")
            try:
                pct = float(args[0])
                if self._set_risk_pct:
                    self._set_risk_pct(pct)
                else:
                    setattr(settings.risk, "risk_per_trade", pct / 100.0)
                return self._send(f"Risk per trade set to {pct:.2f}%.")
            except Exception:
                return self._send("Invalid number. Example: /risk 0.5")

        if cmd == "/config":
            if not args:
                keys = "\n".join([f"- {k} = {g()}" for k, (g, _, _) in self._cfg_map.items()])
                return self._send("⚙️ Config (runtime)\n" + keys)
            if args[0] == "get" and len(args) == 2:
                k = args[1]
                if k in self._cfg_map:
                    g, _, _ = self._cfg_map[k]
                    return self._send(f"{k} = {g()}")
                return self._send("Unknown key.")
            if args[0] == "set" and len(args) == 3:
                k, val = args[1], args[2]
                if k in self._cfg_map:
                    g, s, cast = self._cfg_map[k]
                    try:
                        s(cast(val))
                        return self._send(f"Updated {k} → {g()}")
                    except Exception as e:
                        return self._send(f"Failed to set {k}: {e}")
                return self._send("Unknown key.")
            return self._send("Usage: /config [get <key>|set <key> <value>]")

        if cmd == "/positions":
            if not self._positions_provider:
                return self._send("No positions provider wired.")
            pos = self._positions_provider() or {}
            if not pos:
                return self._send("No positions (day).")
            lines = ["📒 Positions (day)"]
            for sym, p in pos.items():
                qty = p.get("quantity") if isinstance(p, dict) else getattr(p, "quantity", "?")
                avg = p.get("average_price") if isinstance(p, dict) else getattr(p, "average_price", "?")
                lines.append(f"• {sym}: qty={qty} avg={avg}")
            return self._send("\n".join(lines))

        if cmd == "/active":
            if not self._actives_provider:
                return self._send("No active-orders provider wired.")
            try:
                page = int(args[0]) if args else 1
            except Exception:
                page = 1
            acts = self._actives_provider() or []
            n = len(acts)
            page_size = 6
            pages = max(1, (n + page_size - 1) // page_size)
            page = max(1, min(page, pages))
            i0, i1 = (page - 1) * page_size, min(n, page * page_size)
            lines = [f"📦 Active Orders (p{page}/{pages})"]
            for rec in acts[i0:i1]:
                sym = getattr(rec, "symbol", "?")
                side = getattr(rec, "side", "?")
                qty = getattr(rec, "quantity", "?")
                rid = getattr(rec, "order_id", getattr(rec, "record_id", "?"))
                lines.append(f"• {sym} {side} qty={qty} id={rid}")
            return self._send("\n".join(lines))

        if cmd in ("/flow", "/diag"):
            if not self._diag_provider:
                return self._send("No diagnostic provider wired.")
            try:
                data = self._diag_provider() or {}
            except Exception as e:
                return self._send(f"diag failed: {e}")
            # Compact LED line
            checks = data.get("checks", [])
            leds = []
            ok_all = True
            for c in checks:
                ok = bool(c.get("ok"))
                ok_all = ok_all and ok
                leds.append(("🟢" if ok else "🔴") + " " + c.get("name"))
            title = "✅ Flow OK" if ok_all else "❗ Flow has issues"
            body = " · ".join(leds)
            return self._send(f"{title}\n{body}")

        if cmd == "/logs":
            n = 60
            if args:
                try: n = max(1, min(200, int(args[0])))
                except Exception: pass
            if not self._log_provider:
                return self._send("No logs provider wired.")
            lines = self._log_provider(n)
            if not lines:
                return self._send("No logs.")
            txt = "\n".join(lines)
            if len(txt) > 3500:  # Telegram payload limit safety
                txt = "\n".join(lines[-80:])
            return self._send("```text\n" + txt + "\n```", parse_mode="Markdown")

        if cmd == "/emergency":
            return self._send_inline(
                "Confirm flatten & cancel?",
                [[{"text": "✅ Confirm", "callback_data": "confirm_cancel_all"},
                  {"text": "❌ Abort", "callback_data": "abort"}]],
            )

        if cmd == "/id":
            return self._send(f"Chat id: `{self._chat_id}`", parse_mode="Markdown")

        if cmd == "/ping":
            return self._send("pong")

        # fallback
        return self._send(f"Unknown command: {cmd}")