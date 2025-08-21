from __future__ import annotations
"""
Telegram controller:
- Single long poll (no webhook)
- Allowlist (main chat_id + optional extras)
- /help, /status, /tick, /diag, /logs, /active, /positions, /pause, /resume,
  /mode, /risk, /trail, /trailmult, /partial, /tp1, /breakeven, /config
- Entry/fill notifications via notify_entry()/notify_fills()
"""
import hashlib
import json
import logging
import threading
import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional

import requests

from src.config import settings

log = logging.getLogger(__name__)


class TelegramController:
    def __init__(
        self,
        *,
        status_provider: Callable[[], Dict[str, Any]],
        positions_provider: Optional[Callable[[], Dict[str, Any]]] = None,
        actives_provider: Optional[Callable[[], List[Any]]] = None,
        diag_provider: Optional[Callable[[], Dict[str, Any]]] = None,
        # runner controls
        runner_pause: Optional[Callable[[], None]] = None,
        runner_resume: Optional[Callable[[], None]] = None,
        set_live_mode: Optional[Callable[[bool], None]] = None,
        # runtime knobs (if not provided, will mutate settings directly)
        set_risk_pct: Optional[Callable[[float], None]] = None,
        toggle_trailing: Optional[Callable[[bool], None]] = None,
        set_trailing_mult: Optional[Callable[[float], None]] = None,
        toggle_partial: Optional[Callable[[bool], None]] = None,
        set_tp1_ratio: Optional[Callable[[float], None]] = None,
        set_breakeven_ticks: Optional[Callable[[int], None]] = None,
        # logs provider (bound in main via ring buffer)
        logs_provider: Optional[Callable[[int], List[str]]] = None,
        http_timeout: float = 20.0,
    ) -> None:
        tg = settings.telegram
        self._token = tg.bot_token
        self._chat_id = tg.chat_id
        if not self._token or not self._chat_id:
            raise RuntimeError("TelegramController: bot_token or chat_id missing in settings.telegram")
        self._base = f"https://api.telegram.org/bot{self._token}"
        self._timeout = http_timeout

        self._status = status_provider
        self._positions = positions_provider
        self._actives = actives_provider
        self._diag = diag_provider
        self._runner_pause = runner_pause
        self._runner_resume = runner_resume
        self._set_live = set_live_mode

        self._set_risk = set_risk_pct
        self._toggle_trailing = toggle_trailing
        self._set_trailing_mult = set_trailing_mult
        self._toggle_partial = toggle_partial
        self._set_tp1_ratio = set_tp1_ratio
        self._set_be_ticks = set_breakeven_ticks
        self._logs_provider = logs_provider

        extra = tg.extra_admin_ids or []
        self._allow = {int(self._chat_id), *[int(x) for x in extra]}

        self._poll_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._started = False
        self._last_update_id: Optional[int] = None

        # send dedupe/rate
        self._send_min_interval = 0.9
        self._recent_hashes: deque[tuple[float, str]] = deque(maxlen=25)
        self._backoff = 1.0

        # config whitelist
        self._cfg_map = {
            "risk.pct": (
                lambda: settings.risk.risk_per_trade * 100.0,
                (self._set_risk or (lambda v: setattr(settings.risk, "risk_per_trade", float(v) / 100.0))),
                float,
            ),
            "exec.trailing": (
                lambda: settings.executor.enable_trailing,
                (self._toggle_trailing or (lambda v: setattr(settings.executor, "enable_trailing", bool(v)))),
                lambda s: str(s).lower() in ("on", "true", "1", "yes"),
            ),
            "exec.trail_mult": (
                lambda: settings.executor.trailing_atr_multiplier,
                (self._set_trailing_mult or (lambda v: setattr(settings.executor, "trailing_atr_multiplier", float(v)))),
                float,
            ),
            "exec.partial": (
                lambda: settings.executor.partial_tp_enable,
                (self._toggle_partial or (lambda v: setattr(settings.executor, "partial_tp_enable", bool(v)))),
                lambda s: str(s).lower() in ("on", "true", "1", "yes"),
            ),
            "exec.tp1_ratio": (
                lambda: settings.executor.tp1_qty_ratio * 100.0,
                (self._set_tp1_ratio or (lambda v: setattr(settings.executor, "tp1_qty_ratio", float(v) / 100.0))),
                float,
            ),
            "exec.be_ticks": (
                lambda: settings.executor.breakeven_ticks,
                (self._set_be_ticks or (lambda v: setattr(settings.executor, "breakeven_ticks", int(v)))),
                int,
            ),
            "mode.live": (
                lambda: settings.enable_live_trading,
                (self._set_live or (lambda v: setattr(settings, "enable_live_trading", bool(v)))),
                lambda s: str(s).lower() in ("on", "true", "1", "yes", "live"),
            ),
        }

    # ---------- outbound ----------
    def _rate_ok(self, text: str) -> bool:
        now = time.time()
        h = hashlib.md5(text.encode("utf-8")).hexdigest()
        # dedupe 10 seconds & min interval
        while self._recent_hashes and now - self._recent_hashes[0][0] > 10:
            self._recent_hashes.popleft()
        if self._recent_hashes and now - self._recent_hashes[-1][0] < self._send_min_interval:
            return False
        if any(x[1] == h for x in self._recent_hashes):
            return False
        self._recent_hashes.append((now, h))
        return True

    def _send(self, text: str, parse_mode: Optional[str] = None, disable_notification: bool = False) -> None:
        if not self._rate_ok(text):
            return
        payload = {"chat_id": self._chat_id, "text": text, "disable_notification": disable_notification}
        if parse_mode:
            payload["parse_mode"] = parse_mode
        delay = self._backoff
        while not self._stop.is_set():
            try:
                r = requests.post(f"{self._base}/sendMessage", json=payload, timeout=self._timeout)
                if r.ok:
                    self._backoff = 1.0
                    return
            except Exception:
                pass
            time.sleep(delay)
            delay = min(20.0, delay * 2.0)
            self._backoff = delay

    def _send_inline(self, text: str, buttons: list[list[dict]]) -> None:
        payload = {"chat_id": self._chat_id, "text": text, "reply_markup": {"inline_keyboard": buttons}}
        try:
            requests.post(f"{self._base}/sendMessage", json=payload, timeout=self._timeout)
        except Exception:
            pass

    def send_startup_alert(self) -> None:
        s = {}
        try: s = self._status() or {}
        except Exception: pass
        self._send(
            "🚀 Bot started\n"
            f"🔁 Trading: {'🟢 LIVE' if s.get('live_trading') else '🟡 DRY'}\n"
            f"🧠 Broker: {s.get('broker')}\n"
            f"📦 Active: {s.get('active_orders', 0)}"
        )

    def notify_entry(self, *, symbol: str, side: str, qty: int, price: float, record_id: str) -> None:
        self._send(f"🟢 Entry placed\n{symbol} | {side}\nQty: {qty} @ {price:.2f}\nID: `{record_id}`", parse_mode="Markdown")

    def notify_fills(self, fills: List[tuple[str, float]]) -> None:
        if not fills: return
        lines = ["✅ Fills"]
        for rid, px in fills:
            lines.append(f"• {rid} @ {px:.2f}")
        self._send("\n".join(lines))

    # ---------- polling ----------
    def start_polling(self) -> None:
        if self._started:
            log.info("Telegram already polling; skip.")
            return
        self._stop.clear()
        self._poll_thread = threading.Thread(target=self._poll_loop, name="tg-poll", daemon=True)
        self._poll_thread.start()
        self._started = True
        log.info("Telegram polling thread started.")

    def stop_polling(self) -> None:
        if not self._started:
            return
        self._stop.set()
        if self._poll_thread:
            self._poll_thread.join(timeout=5)
        self._started = False
        log.info("Telegram polling stopped.")

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                params = {"timeout": 25}
                if self._last_update_id is not None:
                    params["offset"] = self._last_update_id + 1
                r = requests.get(f"{self._base}/getUpdates", params=params, timeout=self._timeout + 10)
                data = r.json()
                if not data.get("ok"):
                    time.sleep(1.0); continue
                for upd in data.get("result", []):
                    self._last_update_id = int(upd.get("update_id", 0))
                    self._handle_update(upd)
            except Exception as e:
                log.debug("Telegram poll error: %s", e)
                time.sleep(1.0)

    # ---------- auth & routing ----------
    def _authorized(self, chat_id: int) -> bool:
        return int(chat_id) in self._allow

    def _handle_update(self, upd: Dict[str, Any]) -> None:
        # callback buttons
        if "callback_query" in upd:
            cq = upd["callback_query"]
            chat_id = cq.get("message", {}).get("chat", {}).get("id")
            if not self._authorized(int(chat_id)): return
            data = cq.get("data", "")
            if data == "confirm_cancel_all":
                self._send("Not wired: cancel_all (left to runner/executor if needed).")
            try:
                requests.post(f"{self._base}/answerCallbackQuery", json={"callback_query_id": cq.get("id")}, timeout=self._timeout)
            except Exception:
                pass
            return

        msg = upd.get("message") or upd.get("edited_message")
        if not msg: return
        chat_id = msg.get("chat", {}).get("id")
        text = (msg.get("text") or "").strip()
        if not text: return
        if not self._authorized(int(chat_id)):
            self._send("Unauthorized."); return

        parts = text.split()
        cmd = parts[0].lower()
        args = parts[1:]

        # commands
        if cmd in ("/start", "/help"):
            return self._send(
                "🤖 Nifty Scalper Bot\n"
                "/help – this help\n"
                "/status [v] – status (v=json)\n"
                "/tick – single cycle tick now\n"
                "/diag – deep health/flow check\n"
                "/logs [n] – last n logs (default 50)\n"
                "/active – active orders\n"
                "/positions – broker day positions\n"
                "/pause | /resume – toggle entries\n"
                "/mode live|dry – live toggle\n"
                "/risk <pct> – set risk per trade\n"
                "/trail on|off – toggle trailing\n"
                "/trailmult <x> – set trailing ATR multiplier\n"
                "/partial on|off – toggle partial TP\n"
                "/tp1 <pct> – TP1 split percent\n"
                "/breakeven <ticks> – BE ticks after TP1\n"
                "/config [get <k>|set <k> <v>] – runtime config"
            )

        if cmd == "/status":
            try: s = self._status() or {}
            except Exception: s = {}
            if args and args[0].lower().startswith("v"):
                return self._send("```json\n" + json.dumps(s, indent=2) + "\n```", parse_mode="Markdown")
            return self._send(
                f"📊 {s.get('time_ist')} | {'🟢 LIVE' if s.get('live_trading') else '🟡 DRY'} | {s.get('broker')}\n"
                f"📦 Active: {s.get('active_orders', 0)}"
            )

        if cmd == "/tick":
            # This is informational — runner’s loop is continuous; using /diag for details
            s = self._status() or {}
            return self._send(f"⏱ tick requested @ {s.get('time_ist')} (loop is continuous).")

        if cmd == "/diag":
            if not self._diag:
                return self._send("Diag not wired.")
            d = self._diag() or {}
            ok = "🟢 All good" if d.get("ok") else "❗ Flow has issues"
            checks = []
            for c in d.get("checks", []):
                checks.append(("🟢" if c.get("ok") else "🔴") + " " + c.get("name", ""))
            text = ok + "\n" + " · ".join(checks)
            return self._send(text)

        if cmd == "/logs":
            n = 50
            if args:
                try: n = max(1, min(300, int(args[0])))
                except Exception: pass
            if not self._logs_provider:
                return self._send("Logs not wired.")
            lines = self._logs_provider(n)
            if not lines:
                return self._send("No logs captured yet.")
            blob = "\n".join(lines[-n:])
            # Telegram message limit ~4k chars; split if needed
            chunks: List[str] = []
            while blob:
                chunks.append(blob[:3500])
                blob = blob[3500:]
            for ch in chunks:
                self._send("```\n" + ch + "\n```", parse_mode="Markdown", disable_notification=True)
            return

        if cmd == "/active":
            if not self._actives:
                return self._send("No active provider.")
            acts = self._actives() or []
            if not acts:
                return self._send("No active orders.")
            lines = ["📦 Active Orders"]
            for r in acts:
                sym = getattr(r, "symbol", "?")
                side = getattr(r, "side", "?")
                qty = getattr(r, "quantity", "?")
                rid = getattr(r, "order_id", getattr(r, "record_id", "?"))
                lines.append(f"• {sym} {side} qty={qty} id={rid}")
            return self._send("\n".join(lines))

            # /positions
        if cmd == "/positions":
            if not self._positions:
                return self._send("No positions provider.")
            pos = self._positions() or {}
            if not pos: return self._send("No positions.")
            lines = ["📒 Positions (day)"]
            for sym, p in pos.items():
                qty = p.get("quantity") if isinstance(p, dict) else getattr(p, "quantity", "?")
                avg = p.get("average_price") if isinstance(p, dict) else getattr(p, "average_price", "?")
                lines.append(f"• {sym}: qty={qty} avg={avg}")
            return self._send("\n".join(lines))

        if cmd == "/pause":
            if self._runner_pause: self._runner_pause(); return self._send("⏸️ Paused entries.")
            return self._send("Pause not wired.")

        if cmd == "/resume":
            if self._runner_resume: self._runner_resume(); return self._send("▶️ Resumed entries.")
            return self._send("Resume not wired.")

        if cmd == "/mode":
            if not args: return self._send("Usage: /mode live|dry")
            val = args[0].lower() == "live"
            if self._set_live: self._set_live(val)
            settings.enable_live_trading = val
            return self._send(f"Mode set to {'LIVE' if val else 'DRY'}.")

        if cmd == "/risk":
            if not args: return self._send("Usage: /risk 0.5 (pct)")
            try:
                pct = float(args[0])
                if self._set_risk: self._set_risk(pct)
                else: settings.risk.risk_per_trade = pct / 100.0
                return self._send(f"Risk per trade set to {pct:.2f}%.")
            except Exception:
                return self._send("Invalid number. Example: /risk 0.5")

        if cmd == "/trail":
            if not args: return self._send("Usage: /trail on|off")
            val = args[0].lower() in ("on", "true", "1", "yes")
            if self._toggle_trailing: self._toggle_trailing(val)
            else: settings.executor.enable_trailing = val
            return self._send(f"Trailing {'enabled' if val else 'disabled'}.")

        if cmd == "/trailmult":
            if not args: return self._send("Usage: /trailmult 1.5")
            try:
                v = float(args[0])
                if self._set_trailing_mult: self._set_trailing_mult(v)
                else: settings.executor.trailing_atr_multiplier = v
                return self._send(f"Trailing ATR multiplier set to {v:.2f}.")
            except Exception:
                return self._send("Invalid number. Example: /trailmult 1.8")

        if cmd == "/partial":
            if not args: return self._send("Usage: /partial on|off")
            val = args[0].lower() in ("on","true","1","yes")
            if self._toggle_partial: self._toggle_partial(val)
            else: settings.executor.partial_tp_enable = val
            return self._send(f"Partial TP {'enabled' if val else 'disabled'}.")

        if cmd == "/tp1":
            if not args: return self._send("Usage: /tp1 40 (percent)")
            try:
                pct = float(args[0])
                if self._set_tp1_ratio: self._set_tp1_ratio(pct)
                else: settings.executor.tp1_qty_ratio = pct / 100.0
                return self._send(f"TP1 set to {pct:.1f}% of qty.")
            except Exception:
                return self._send("Invalid number. Example: /tp1 40")

        if cmd == "/breakeven":
            if not args: return self._send("Usage: /breakeven 2 (ticks)")
            try:
                k = int(args[0])
                if self._set_be_ticks: self._set_be_ticks(k)
                else: settings.executor.breakeven_ticks = k
                return self._send(f"Breakeven hop set to {k} ticks.")
            except Exception:
                return self._send("Invalid integer. Example: /breakeven 2")

        if cmd == "/config":
            if not args:
                keys = "\n".join([f"- {k} = {g()}" for k,(g,_,_) in self._cfg_map.items()])
                return self._send("⚙️ Config (runtime)\n" + keys)
            if args[0] == "get" and len(args) == 2:
                k = args[1]
                if k in self._cfg_map:
                    g, _, _ = self._cfg_map[k]
                    return self._send(f"{k} = {g()}")
                return self._send("Unknown key.")
            if args[0] == "set" and len(args) == 3:
                k, val = args[1], args[2]
                if k not in self._cfg_map:
                    return self._send("Unknown key.")
                g, s, cast = self._cfg_map[k]
                try:
                    s(cast(val))
                    return self._send(f"Updated {k} → {g()}")
                except Exception as e:
                    return self._send(f"Failed to set {k}: {e}")
            return self._send("Usage: /config [get <k>|set <k> <v>]")

        return self._send(f"Unknown command: {cmd}")