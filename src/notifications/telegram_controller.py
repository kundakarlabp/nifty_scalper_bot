from __future__ import annotations

"""
Telegram controller with on-demand diagnostics.

Commands:
  /status [verbose]           – bot status
  /summary                    – last signal (if runner exposes it via status)
  /active [page]              – active orders
  /positions                  – broker day positions
  /cancel_all                 – cancel all (with confirm)
  /pause [minutes]            – pause entries (minutes optional)
  /resume                     – resume entries
  /risk <pct>                 – set risk per trade (e.g., /risk 0.5)
  /trail on|off               – toggle trailing exits
  /trailmult <x>              – set trailing ATR multiplier
  /partial on|off             – toggle partial TP
  /tp1 <pct>                  – TP1 qty percent (e.g., 40)
  /breakeven <ticks>          – BE ticks after TP1
  /mode live|dry              – live trading toggle
  /config [get <k>|set <k> <v>] – whitelist runtime keys
  /logs [N]                   – last N log lines (default 150)
  /health                     – system health snapshot (JSON)
  /tick                       – run one strategy cycle now (returns brief result)
  /ping                       – pong + uptime
  /id                         – show chat id
  /help                       – this help
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
        # diagnostics
        logs_provider: Optional[Callable[[int], str]] = None,
        health_provider: Optional[Callable[[], Dict[str, Any]]] = None,
        tick_provider: Optional[Callable[[], Dict[str, Any]]] = None,
        # runtime config hooks (optional; if absent, /config mutates settings via whitelist)
        set_risk_pct: Optional[Callable[[float], None]] = None,           # 0.5 -> 0.5%
        toggle_trailing: Optional[Callable[[bool], None]] = None,
        set_trailing_mult: Optional[Callable[[float], None]] = None,
        toggle_partial: Optional[Callable[[bool], None]] = None,
        set_tp1_ratio: Optional[Callable[[float], None]] = None,          # 40 -> 40%
        set_breakeven_ticks: Optional[Callable[[int], None]] = None,
        set_live_mode: Optional[Callable[[bool], None]] = None,           # True => LIVE
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
        self._status_provider = status_provider
        self._positions_provider = positions_provider
        self._actives_provider = actives_provider
        self._runner_pause = runner_pause
        self._runner_resume = runner_resume
        self._cancel_all = cancel_all

        self._logs_provider = logs_provider
        self._health_provider = health_provider
        self._tick_provider = tick_provider

        self._set_risk_pct = set_risk_pct
        self._toggle_trailing = toggle_trailing
        self._set_trailing_mult = set_trailing_mult
        self._toggle_partial = toggle_partial
        self._set_tp1_ratio = set_tp1_ratio
        self._set_breakeven_ticks = set_breakeven_ticks
        self._set_live_mode = set_live_mode

        # polling state
        self._poll_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._started = False
        self._last_update_id: Optional[int] = None

        # security: allowlist
        extra = getattr(tg, "extra_admin_ids", []) or []
        self._allowlist = {int(self._chat_id), *[int(x) for x in extra]}

        # send rate & backoff
        self._send_min_interval = 0.8  # seconds
        self._last_sends: List[tuple[float, str]] = []  # (timestamp, md5(text))
        self._backoff = 1.0
        self._backoff_max = 20.0

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
                lambda s: str(s).lower() in ("on", "true", "1", "yes"),
            ),
            "exec.trail_mult": (
                lambda: getattr(settings.executor, "trailing_atr_multiplier", 1.5),
                (self._set_trailing_mult or (lambda v: setattr(settings.executor, "trailing_atr_multiplier", float(v)))),
                float,
            ),
            "exec.partial": (
                lambda: getattr(settings.executor, "partial_tp_enable", False),
                (self._toggle_partial or (lambda v: setattr(settings.executor, "partial_tp_enable", bool(v)))),
                lambda s: str(s).lower() in ("on", "true", "1", "yes"),
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
                lambda s: str(s).lower() in ("on", "true", "1", "yes", "live"),
            ),
        }

    # -------------------- outbound --------------------

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
                time.sleep(delay + (0.3 * (hash(time.time()) % 1)))
                delay = min(self._backoff_max, delay * 2)
                self._backoff = delay

    def _send_inline(self, text: str, buttons: list[list[dict]]) -> None:
        payload = {"chat_id": self._chat_id, "text": text, "reply_markup": {"inline_keyboard": buttons}}
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
        self._send(
            "🚀 Nifty Scalper Bot started\n"
            f"🔁 Trading: {'🟢 LIVE' if s.get('live_trading') else '🟡 DRY'}\n"
            f"🧠 Broker: {s.get('broker')}\n"
            f"📦 Active: {s.get('active_orders', 0)}"
        )

    # -------------------- inbound (poll) --------------------

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

    # -------------------- command handling --------------------

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
            return self._send(
                "🤖 Nifty Scalper Bot\n"
                "/status [verbose] – bot status\n"
                "/summary – recent signal summary\n"
                "/active [page] – active orders\n"
                "/positions – broker day positions\n"
                "/cancel_all – cancel all (with confirm)\n"
                "/pause [minutes] – pause entries\n"
                "/resume – resume entries\n"
                "/risk <pct> – risk per trade (e.g., /risk 0.5)\n"
                "/trail on|off – toggle trailing\n"
                "/trailmult <x> – trailing ATR multiplier\n"
                "/partial on|off – partial TP\n"
                "/tp1 <pct> – TP1 percent of qty\n"
                "/breakeven <ticks> – BE ticks after TP1\n"
                "/mode live|dry – live trading toggle\n"
                "/config [get <k>|set <k> <v>] – runtime config\n"
                "/logs [N] – last N log lines\n"
                "/health – system health snapshot\n"
                "/tick – run one strategy cycle now\n"
                "/ping – ping"
            )

        if cmd == "/status":
            try:
                s = self._status_provider() if self._status_provider else {}
            except Exception:
                s = {}
            verbose = (args and args[0].lower().startswith("v"))
            if verbose:
                return self._send("```json\n" + json.dumps(s, indent=2) + "\n```", parse_mode="Markdown")
            return self._send(
                f"📊 {s.get('time_ist')}\n"
                f"🔁 {'🟢 LIVE' if s.get('live_trading') else '🟡 DRY'} | {s.get('broker')}\n"
                f"📦 Active: {s.get('active_orders', 0)}"
            )

        if cmd == "/summary":
            try:
                s = self._status_provider() if self._status_provider else {}
            except Exception:
                s = {}
            last = s.get("last_signal")
            if not last:
                return self._send("No recent signal.")
            return self._send("```json\n" + json.dumps(last, indent=2) + "\n```", parse_mode="Markdown")

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

        if cmd == "/cancel_all":
            return self._send_inline(
                "Confirm cancel all?",
                [[{"text": "✅ Confirm", "callback_data": "confirm_cancel_all"},
                  {"text": "❌ Abort", "callback_data": "abort"}]],
            )

        if cmd == "/pause":
            if self._runner_pause:
                self._runner_pause()
            # optional minutes arg is accepted, but we don't schedule resume here
            return self._send("⏸️ Entries paused.")

        if cmd == "/resume":
            if self._runner_resume:
                self._runner_resume()
            return self._send("▶️ Entries resumed.")

        if cmd == "/risk":
            if not args:
                return self._send("Usage: /risk 0.5  (for 0.5%)")
            try:
                pct = float(args[0])
                if self._set_risk_pct:
                    self._set_risk_pct(pct)
                else:
                    setattr(settings.risk, "risk_per_trade", pct / 100.0)
                return self._send(f"Risk per trade set to {pct:.2f}%.")
            except Exception:
                return self._send("Invalid number. Example: /risk 0.5")

        if cmd == "/trail":
            if not args:
                return self._send("Usage: /trail on|off")
            val = str(args[0]).lower() in ("on", "true", "1", "yes")
            if self._toggle_trailing:
                self._toggle_trailing(val)
            else:
                setattr(settings.executor, "enable_trailing", val)
            return self._send(f"Trailing {'enabled' if val else 'disabled'}.")

        if cmd == "/trailmult":
            if not args:
                return self._send("Usage: /trailmult 1.8")
            try:
                v = float(args[0])
                if self._set_trailing_mult:
                    self._set_trailing_mult(v)
                else:
                    setattr(settings.executor, "trailing_atr_multiplier", v)
                return self._send(f"Trailing ATR multiplier set to {v:.2f}.")
            except Exception:
                return self._send("Invalid number. Example: /trailmult 1.8")

        if cmd == "/partial":
            if not args:
                return self._send("Usage: /partial on|off")
            val = str(args[0]).lower() in ("on", "true", "1", "yes")
            if self._toggle_partial:
                self._toggle_partial(val)
            else:
                setattr(settings.executor, "partial_tp_enable", val)
            return self._send(f"Partial TP {'enabled' if val else 'disabled'}.")

        if cmd == "/tp1":
            if not args:
                return self._send("Usage: /tp1 40  (for 40% of qty)")
            try:
                pct = float(args[0])
                if self._set_tp1_ratio:
                    self._set_tp1_ratio(pct)
                else:
                    setattr(settings.executor, "tp1_qty_ratio", pct / 100.0)
                return self._send(f"TP1 set to {pct:.1f}%% of qty.")
            except Exception:
                return self._send("Invalid number. Example: /tp1 40")

        if cmd == "/breakeven":
            if not args:
                return self._send("Usage: /breakeven 2")
            try:
                ticks = int(args[0])
                if self._set_breakeven_ticks:
                    self._set_breakeven_ticks(ticks)
                else:
                    setattr(settings.executor, "breakeven_ticks", ticks)
                return self._send(f"Breakeven hop set to {ticks} ticks.")
            except Exception:
                return self._send("Invalid number. Example: /breakeven 2")

        if cmd == "/mode":
            if not args:
                return self._send("Usage: /mode live|dry")
            state = str(args[0]).lower()
            if state not in ("live", "dry"):
                return self._send("Usage: /mode live|dry")
            val = (state == "live")
            if self._set_live_mode:
                self._set_live_mode(val)
            else:
                setattr(settings, "enable_live_trading", val)
            return self._send(f"Mode set to {'LIVE' if val else 'DRY'}.")

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

        if cmd == "/logs":
            n = 150
            if args:
                try:
                    n = int(args[0])
                except Exception:
                    pass
            if not self._logs_provider:
                return self._send("Logs provider not wired.")
            tail = self._logs_provider(n)
            # Telegram messages have size limits; chunk if needed
            if len(tail) <= 3500:
                return self._send("```log\n" + tail + "\n```", parse_mode="Markdown")
            # chunk in ~3000 char messages
            i = 0
            while i < len(tail):
                self._send("```log\n" + tail[i:i+3000] + "\n```", parse_mode="Markdown", disable_notification=True)
                i += 3000
            return

        if cmd == "/health":
            if not self._health_provider:
                return self._send("Health provider not wired.")
            try:
                h = self._health_provider()
            except Exception as e:
                return self._send(f"Health error: {e}")
            return self._send("```json\n" + json.dumps(h, indent=2) + "\n```", parse_mode="Markdown")

        if cmd == "/tick":
            if not self._tick_provider:
                return self._send("Tick provider not wired.")
            res = self._tick_provider()
            return self._send("```json\n" + json.dumps(res, indent=2) + "\n```", parse_mode="Markdown")

        if cmd == "/ping":
            try:
                s = self._status_provider() if self._status_provider else {}
            except Exception:
                s = {}
            return self._send(
                f"Pong {s.get('time_ist')} • "
                f"{'LIVE' if s.get('live_trading') else 'DRY'} • active={s.get('active_orders', 0)}"
            )

        if cmd == "/id":
            return self._send(f"Chat ID: {self._chat_id}")

        # fallback
        return self._send(f"Unknown command: {cmd}")