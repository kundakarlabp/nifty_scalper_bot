from __future__ import annotations
"""
Telegram long-poll controller (webhook-safe) for the Nifty Scalper Bot.

Key features
- Forces webhook OFF before starting long-poll (avoids "dead" bot after deploy)
- Single polling thread, deduped sends, gentle backoff
- Admin allowlist (primary chat_id + optional extra_admin_ids)
- Rich commands: /help, /status, /active, /positions, /logs, /diag, /tick,
  /cancel_all (inline confirm), /pause, /resume, /mode, /risk, /trail, /trailmult,
  /partial, /tp1, /breakeven, /config get|set (whitelist)
- Event hooks: notify_entry(), notify_fills(), notify_text()
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


def _booly(s: str) -> bool:
    return str(s).strip().lower() in ("on", "true", "1", "yes", "y", "live")


class TelegramController:
    def __init__(
        self,
        *,
        # providers (must be pure/read-only)
        status_provider: Callable[[], Dict[str, Any]],
        positions_provider: Optional[Callable[[], Dict[str, Any]]] = None,
        actives_provider: Optional[Callable[[], List[Any]]] = None,
        logs_provider: Optional[Callable[[int], List[str]]] = None,
        diag_provider: Optional[Callable[[], Dict[str, Any]]] = None,
        tick_fn: Optional[Callable[[], Dict[str, Any]]] = None,
        # control hooks
        runner_pause: Optional[Callable[[], None]] = None,
        runner_resume: Optional[Callable[[], None]] = None,
        cancel_all: Optional[Callable[[], None]] = None,
        # optional runtime config hooks (else fall back to mutating `settings`)
        set_risk_pct: Optional[Callable[[float], None]] = None,           # 0.5 -> 0.5%
        toggle_trailing: Optional[Callable[[bool], None]] = None,
        set_trailing_mult: Optional[Callable[[float], None]] = None,
        toggle_partial: Optional[Callable[[bool], None]] = None,
        set_tp1_ratio: Optional[Callable[[float], None]] = None,          # 40 -> 40%
        set_breakeven_ticks: Optional[Callable[[int], None]] = None,
        set_live_mode: Optional[Callable[[bool], None]] = None,           # True => LIVE
        http_timeout: float = 25.0,
    ) -> None:
        tg = getattr(settings, "telegram", object())
        self._token: Optional[str] = getattr(tg, "bot_token", None)
        self._chat_id: Optional[int] = getattr(tg, "chat_id", None)
        if not self._token or not self._chat_id:
            raise RuntimeError("TelegramController: bot_token or chat_id missing in settings.telegram")

        self._base = f"https://api.telegram.org/bot{self._token}"
        self._timeout = float(http_timeout)

        # allowlist: primary chat + extras
        extra = getattr(tg, "extra_admin_ids", []) or []
        self._allowlist = {int(self._chat_id), *[int(x) for x in extra]}

        # providers
        self._status_provider = status_provider
        self._positions_provider = positions_provider
        self._actives_provider = actives_provider
        self._logs_provider = logs_provider
        self._diag_provider = diag_provider
        self._tick_fn = tick_fn

        # controls
        self._runner_pause = runner_pause
        self._runner_resume = runner_resume
        self._cancel_all = cancel_all

        # runtime config (optional)
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

        # send dedupe/backoff
        self._send_min_interval = 0.9  # seconds
        self._recent: List[tuple[float, str]] = []  # (ts, md5)
        self._backoff = 1.0
        self._backoff_max = 20.0

        # whitelist for /config
        self._cfg_map = {
            "risk.pct": (
                lambda: getattr(settings.risk, "risk_per_trade", 0.01) * 100.0,
                (self._set_risk_pct or (lambda v: setattr(settings.risk, "risk_per_trade", float(v) / 100.0))),
                float,
            ),
            "exec.trailing": (
                lambda: getattr(settings.executor, "enable_trailing", True),
                (self._toggle_trailing or (lambda v: setattr(settings.executor, "enable_trailing", bool(v)))),
                _booly,
            ),
            "exec.trail_mult": (
                lambda: getattr(settings.executor, "trailing_atr_multiplier", 1.5),
                (self._set_trailing_mult or (lambda v: setattr(settings.executor, "trailing_atr_multiplier", float(v)))),
                float,
            ),
            "exec.partial": (
                lambda: getattr(settings.executor, "partial_tp_enable", False),
                (self._toggle_partial or (lambda v: setattr(settings.executor, "partial_tp_enable", bool(v)))),
                _booly,
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
                _booly,
            ),
        }

    # -------------------- webhook safety / startup --------------------

    def _ensure_poll_mode(self) -> None:
        """
        Ensure webhooks are OFF so long polling works immediately after redeploys.
        """
        try:
            r = requests.get(f"{self._base}/getWebhookInfo", timeout=self._timeout)
            info = r.json() if r.ok else {}
            url = (info.get("result", {}) or {}).get("url") or info.get("url")
            if url:
                requests.post(
                    f"{self._base}/deleteWebhook",
                    json={"drop_pending_updates": True},
                    timeout=self._timeout,
                )
                log.info("Telegram webhook cleared to enable long-polling.")
        except Exception as e:
            log.debug("getWebhookInfo/deleteWebhook failed: %s", e)

        # Basic token sanity
        try:
            g = requests.get(f"{self._base}/getMe", timeout=self._timeout)
            if not g.ok:
                log.error("Telegram getMe failed: %s", g.text)
        except Exception as e:
            log.error("Telegram getMe error: %s", e)

    # -------------------- outbound --------------------

    def _rate_ok(self, text: str) -> bool:
        now = time.time()
        h = hashlib.md5(text.encode("utf-8")).hexdigest()
        # keep last 10s; enforce min interval
        self._recent[:] = [(t, hh) for t, hh in self._recent if now - t < 10]
        if self._recent and now - self._recent[-1][0] < self._send_min_interval:
            return False
        if any(hh == h for _, hh in self._recent):
            return False
        self._recent.append((now, h))
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
                r = requests.post(f"{self._base}/sendMessage", json=payload, timeout=self._timeout)
                if not r.ok:
                    raise RuntimeError(r.text)
                self._backoff = 1.0
                return
            except Exception as e:
                log.debug("sendMessage error: %s", e)
                time.sleep(delay)
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
        mode = "🟢 LIVE" if s.get("live_trading") else "🟡 DRY"
        self._send(
            "🚀 Nifty Scalper Bot online\n"
            f"🔁 {mode} · Broker: {s.get('broker')} · Active: {s.get('active_orders', 0)}"
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

    # -------------------- inbound (poll) --------------------

    def start_polling(self) -> None:
        if self._started:
            log.info("Telegram polling already running; skip.")
            return
        self._ensure_poll_mode()
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
                params = {"timeout": 30}
                if self._last_update_id is not None:
                    params["offset"] = self._last_update_id + 1
                r = requests.get(f"{self._base}/getUpdates", params=params, timeout=self._timeout + 10)
                data = r.json() if r.ok else {}
                if not data or not data.get("ok"):
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
            return self._send(
                "🤖 *Nifty Scalper Bot*\n"
                "*Info*\n"
                "• /status [verbose] – status summary\n"
                "• /active [page] – active orders\n"
                "• /positions – broker day positions\n"
                "• /logs [n] – last n log lines (default 30)\n"
                "• /diag – full pipeline diagnostics\n"
                "• /tick – force a single runner tick\n"
                "• /ping – quick heartbeat\n\n"
                "*Control*\n"
                "• /pause · /resume\n"
                "• /cancel_all (asks to confirm)\n"
                "• /mode live|dry – toggle live trading\n\n"
                "*Tuning*\n"
                "• /risk <pct> – e.g., /risk 0.5\n"
                "• /trail on|off · /trailmult <x>\n"
                "• /partial on|off · /tp1 <pct>\n"
                "• /breakeven <ticks>\n"
                "• /config [get <k>|set <k> <v>]  (keys: risk.pct, exec.trailing, exec.trail_mult, exec.partial, exec.tp1_ratio, exec.be_ticks, mode.live)\n",
                parse_mode="Markdown"
            )

        if cmd == "/ping":
            s = {}
            try:
                s = self._status_provider() if self._status_provider else {}
            except Exception:
                pass
            return self._send(f"🏓 {_bool_time()} | {'LIVE' if s.get('live_trading') else 'DRY'}")

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

        if cmd == "/logs":
            n = 30
            if args:
                try:
                    n = max(1, min(500, int(args[0])))
                except Exception:
                    pass
            if not self._logs_provider:
                return self._send("No logs provider.")
            lines = self._logs_provider(n) or ["<empty>"]
            text = "\n".join(lines[-n:])
            # split if too long
            chunks = _split_long(text, 3800)
            for c in chunks:
                self._send("```\n" + c + "\n```", parse_mode="Markdown")
            return

        if cmd == "/diag":
            if not self._diag_provider:
                return self._send("No diagnostics provider.")
            try:
                d = self._diag_provider() or {}
            except Exception as e:
                return self._send(f"Diag failed: {e}")
            # compact status line
            icons = []
            for c in d.get("checks", []):
                icons.append(("🟢" if c.get("ok") else "🔴") + " " + c.get("name", "?"))
            header = "❗ Flow has issues" if not d.get("ok", False) else "✅ Flow looks good"
            self._send(header + "\n" + " · ".join(icons))
            # full JSON
            self._send("```json\n" + json.dumps(d, indent=2) + "\n```", parse_mode="Markdown")
            return

        if cmd == "/tick":
            if not self._tick_fn:
                return self._send("Tick not wired.")
            try:
                res = self._tick_fn() or {}
                self._send("```json\n" + json.dumps(res, indent=2) + "\n```", parse_mode="Markdown")
            except Exception as e:
                self._send(f"Tick failed: {e}")
            return

        if cmd == "/cancel_all":
            return self._send_inline(
                "Confirm cancel all?",
                [[{"text": "✅ Confirm", "callback_data": "confirm_cancel_all"},
                  {"text": "❌ Abort", "callback_data": "abort"}]],
            )

        if cmd == "/pause":
            if self._runner_pause:
                self._runner_pause()
                return self._send("⏸️ Entries paused.")
            return self._send("Pause not wired.")

        if cmd == "/resume":
            if self._runner_resume:
                self._runner_resume()
                return self._send("▶️ Entries resumed.")
            return self._send("Resume not wired.")

        if cmd == "/mode":
            if not args:
                return self._send("Usage: /mode live|dry")
            val = _booly(args[0])
            if self._set_live_mode:
                self._set_live_mode(val)
            else:
                setattr(settings, "enable_live_trading", val)
            return self._send(f"Mode set to {'LIVE' if val else 'DRY'}.")

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
            val = _booly(args[0])
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
            val = _booly(args[0])
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

        # fallback
        return self._send(f"Unknown command: {cmd}")


def _bool_time() -> str:
    return time.strftime("%H:%M:%S")


def _split_long(s: str, n: int) -> List[str]:
    out, cur = [], []
    cur_len = 0
    for line in s.splitlines():
        if cur_len + len(line) + 1 > n and cur:
            out.append("\n".join(cur))
            cur = []
            cur_len = 0
        cur.append(line)
        cur_len += len(line) + 1
    if cur:
        out.append("\n".join(cur))
    return out