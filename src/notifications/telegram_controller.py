from __future__ import annotations

"""
Telegram command & control for the Nifty Scalper Bot.

Highlights
- Long-poll worker (single thread), de-duped sends, backoff.
- Robust command set with runtime toggles and diagnostics.
- Pluggable providers/hooks so it’s easy to wire from main.py:
    - status_provider() -> dict
    - summary_provider(n:int) -> List[dict]
    - diag_provider() -> dict
    - logs_provider(n:int) -> List[str]
    - positions_provider() -> dict
    - actives_provider() -> List[Any]
    - runner_pause(minutes:int|None), runner_resume()
    - cancel_all(), flatten_all()
    - tick_once() -> dict|None
    - set_risk_pct(float), toggle_trailing(bool), set_trailing_mult(float),
      toggle_partial(bool), set_tp1_ratio(float), set_breakeven_ticks(int),
      set_live_mode(bool), set_quality_mode(str), set_regime_mode(str)
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
        summary_provider: Optional[Callable[[int], List[Dict[str, Any]]]] = None,
        diag_provider: Optional[Callable[[], Dict[str, Any]]] = None,
        logs_provider: Optional[Callable[[int], List[str]]] = None,
        positions_provider: Optional[Callable[[], Dict[str, Any]]] = None,
        actives_provider: Optional[Callable[[], List[Any]]] = None,
        # control hooks
        runner_pause: Optional[Callable[[Optional[int]], None]] = None,
        runner_resume: Optional[Callable[[], None]] = None,
        cancel_all: Optional[Callable[[], None]] = None,
        flatten_all: Optional[Callable[[], None]] = None,
        tick_once: Optional[Callable[[], Optional[Dict[str, Any]]]] = None,
        # runtime config hooks
        set_risk_pct: Optional[Callable[[float], None]] = None,
        toggle_trailing: Optional[Callable[[bool], None]] = None,
        set_trailing_mult: Optional[Callable[[float], None]] = None,
        toggle_partial: Optional[Callable[[bool], None]] = None,
        set_tp1_ratio: Optional[Callable[[float], None]] = None,
        set_breakeven_ticks: Optional[Callable[[int], None]] = None,
        set_live_mode: Optional[Callable[[bool], None]] = None,
        set_quality_mode: Optional[Callable[[str], None]] = None,
        set_regime_mode: Optional[Callable[[str], None]] = None,
        http_timeout: float = 20.0,
    ) -> None:
        tg = getattr(settings, "telegram", object())
        self._enabled: bool = bool(getattr(tg, "enabled", True))
        self._token: Optional[str] = getattr(tg, "bot_token", None)
        self._chat_id: Optional[int] = getattr(tg, "chat_id", None)
        if not self._enabled or not self._token or not self._chat_id:
            raise RuntimeError("TelegramController: disabled or bot_token/chat_id missing")

        self._base = f"https://api.telegram.org/bot{self._token}"
        self._timeout = http_timeout

        # hooks
        self._status = status_provider
        self._summary = summary_provider
        self._diag = diag_provider
        self._logs = logs_provider
        self._positions = positions_provider
        self._actives = actives_provider

        self._pause = runner_pause
        self._resume = runner_resume
        self._cancel_all = cancel_all
        self._flatten_all = flatten_all
        self._tick_once = tick_once

        self._set_risk_pct = set_risk_pct
        self._toggle_trailing = toggle_trailing
        self._set_trailing_mult = set_trailing_mult
        self._toggle_partial = toggle_partial
        self._set_tp1_ratio = set_tp1_ratio
        self._set_breakeven_ticks = set_breakeven_ticks
        self._set_live_mode = set_live_mode
        self._set_quality_mode = set_quality_mode
        self._set_regime_mode = set_regime_mode

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
                time.sleep(delay)
                delay = min(self._backoff_max, delay * 2)
                self._backoff = delay

    def _send_inline(self, text: str, buttons: list[list[dict]]) -> None:
        payload = {"chat_id": self._chat_id, "text": text, "reply_markup": {"inline_keyboard": buttons}}
        try:
            requests.post(f"{self._base}/sendMessage", json=payload, timeout=self._timeout)
        except Exception as e:
            log.debug("Inline send failed: %s", e)

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

    def send_menu(self) -> None:
        self._send(
            "📋 Commands\n"
            "/status – bot status\n"
            "/summary [n] – recent signals\n"
            "/flow – green/red flow\n"
            "/diag – detailed diagnostics\n"
            "/logs [n] – last N logs\n"
            "/mode live|shadow – toggle mode\n"
            "/quality auto|on|off – execution quality\n"
            "/regime auto|trend|range|off – regime bias\n"
            "/risk <pct> – risk per trade\n"
            "/trail on|off – trailing\n"
            "/trailmult <x> – trailing ATR multiplier\n"
            "/partial on|off – partial TP\n"
            "/tp1 <pct> – TP1 quantity percent\n"
            "/be <ticks> – breakeven ticks after TP1\n"
            "/pause [min] – pause entries\n"
            "/resume – resume entries\n"
            "/orders – open orders\n"
            "/positions – broker positions\n"
            "/flatten – market exit all (confirm)\n"
            "/tick – run one decision tick now\n"
            "/id – show chat id\n"
            "/help – this menu"
        )

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
                if data == "confirm_flatten" and self._flatten_all:
                    self._flatten_all()
                    self._send("🏁 Flattened all positions and cancelled orders.")
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
        chat = msg.get("chat", {})
        chat_id = chat.get("id")
        text = (msg.get("text") or "").strip()
        if not text:
            return
        if not self._authorized(int(chat_id)):
            self._send("Unauthorized.")
            return

        parts = text.split()
        cmd = parts[0].lower()
        args = parts[1:]

        # ---- menu / help
        if cmd in ("/start", "/help", "/menu"):
            return self.send_menu()

        # ---- basic status
        if cmd == "/status":
            s = {}
            try:
                s = self._status() or {}
            except Exception:
                pass
            return self._send(
                f"📊 {s.get('time_ist','?')}\n"
                f"{'🟢 LIVE' if s.get('live_trading') else '🟡 SHADOW'} | {s.get('broker')}\n"
                f"📦 Active: {s.get('active_orders',0)} | ⏸ {s.get('paused')}\n"
                f"🎚 Quality: {s.get('quality_mode')} | Regime: {s.get('regime_mode')}"
            )

        if cmd == "/summary":
            n = 10
            if args:
                try:
                    n = max(1, min(30, int(args[0])))
                except Exception:
                    pass
            if not self._summary:
                return self._send("No summary provider wired.")
            data = self._summary(n) or []
            if not data:
                return self._send("No recent signals.")
            lines = ["🧾 Recent signals:"]
            for d in data:
                lines.append(
                    f"{d.get('time','?')} · {d.get('side','?')} "
                    f"s={d.get('score')} c={d.get('confidence')} "
                    f"sl={d.get('sl_points')} tp={d.get('tp_points')}"
                )
            return self._send("\n".join(lines))

        if cmd == "/flow":
            if not self._diag:
                return self._send("No diagnostics wired.")
            d = self._diag() or {}
            def dot(ok: bool) -> str:
                return "🟢" if ok else "🔴"
            ch = {c["name"]: c for c in d.get("checks", [])}
            order = ["market_open","spot_ltp","spot_ohlc","strike_selection","option_ohlc","indicators","signal","sizing","execution_ready","open_orders"]
            bullets = " · ".join([f"{dot(bool(ch.get(k,{}).get('ok')))} {k}" for k in order])
            header = "✅ Flow OK" if d.get("ok") else "❗ Flow has issues"
            return self._send(f"{header}\n{bullets}")

        if cmd == "/diag":
            if not self._diag:
                return self._send("No diagnostics wired.")
            d = self._diag() or {}
            return self._send("```json\n" + json.dumps(d, indent=2) + "\n```", parse_mode="Markdown")

        if cmd == "/logs":
            n = 60
            if args:
                try:
                    n = max(5, min(500, int(args[0])))
                except Exception:
                    pass
            if not self._logs:
                return self._send("No logs provider wired.")
            lines = self._logs(n) or []
            if not lines:
                return self._send("No logs available.")
            # Telegram message limit—chunk
            chunk = []
            size = 0
            for ln in lines:
                ln = ln.rstrip()
                if size + len(ln) + 1 > 3500:
                    self._send("```\n" + "\n".join(chunk) + "\n```", parse_mode="Markdown", disable_notification=True)
                    chunk, size = [], 0
                chunk.append(ln)
                size += len(ln) + 1
            if chunk:
                self._send("```\n" + "\n".join(chunk) + "\n```", parse_mode="Markdown", disable_notification=True)
            return

        if cmd == "/mode":
            if not args:
                return self._send("Usage: /mode live|shadow")
            state = args[0].lower()
            live = state in ("live", "on", "true")
            if self._set_live_mode:
                self._set_live_mode(live)
            return self._send(f"Mode set to {'LIVE' if live else 'SHADOW'}.")

        if cmd == "/quality":
            if not args:
                return self._send("Usage: /quality auto|on|off")
            val = args[0].lower()
            if val not in ("auto","on","off"):
                return self._send("Usage: /quality auto|on|off")
            if self._set_quality_mode:
                self._set_quality_mode(val)
            return self._send(f"Quality mode: {val}")

        if cmd == "/regime":
            if not args:
                return self._send("Usage: /regime auto|trend|range|off")
            val = args[0].lower()
            if val not in ("auto","trend","range","off"):
                return self._send("Usage: /regime auto|trend|range|off")
            if self._set_regime_mode:
                self._set_regime_mode(val)
            return self._send(f"Regime mode: {val}")

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
            val = args[0].lower() in ("on","true","1","yes")
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
            val = args[0].lower() in ("on","true","1","yes")
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

        if cmd in ("/be", "/breakeven"):
            if not args:
                return self._send("Usage: /be 2")
            try:
                ticks = int(args[0])
                if self._set_breakeven_ticks:
                    self._set_breakeven_ticks(ticks)
                else:
                    setattr(settings.executor, "breakeven_ticks", ticks)
                return self._send(f"Breakeven ticks set to {ticks}.")
            except Exception:
                return self._send("Invalid number. Example: /be 2")

        if cmd == "/pause":
            minutes = None
            if args:
                try:
                    minutes = int(args[0])
                except Exception:
                    pass
            if self._pause:
                self._pause(minutes)
            return self._send(f"⏸️ Entries paused{f' for {minutes}m' if minutes else ''}.")

        if cmd == "/resume":
            if self._resume:
                self._resume()
            return self._send("▶️ Entries resumed.")

        if cmd == "/orders":
            if not self._actives:
                return self._send("No active-orders provider wired.")
            acts = self._actives() or []
            if not acts:
                return self._send("No open orders.")
            lines = ["📦 Active Orders"]
            for rec in acts:
                sym = getattr(rec, "symbol", "?")
                side = getattr(rec, "side", "?")
                qty = getattr(rec, "quantity", "?")
                rid = getattr(rec, "order_id", getattr(rec, "record_id", "?"))
                lines.append(f"• {sym} {side} qty={qty} id={rid}")
            return self._send("\n".join(lines))

        if cmd == "/positions":
            if not self._positions:
                return self._send("No positions provider wired.")
            pos = self._positions() or {}
            if not pos:
                return self._send("No positions (day).")
            lines = ["📒 Positions (day)"]
            for sym, p in pos.items():
                qty = p.get("quantity") if isinstance(p, dict) else getattr(p, "quantity", "?")
                avg = p.get("average_price") if isinstance(p, dict) else getattr(p, "average_price", "?")
                lines.append(f"• {sym}: qty={qty} avg={avg}")
            return self._send("\n".join(lines))

        if cmd == "/flatten":
            return self._send_inline(
                "Confirm FLATTEN (exit market positions and cancel orders)?",
                [[{"text": "✅ Confirm", "callback_data": "confirm_flatten"},
                  {"text": "❌ Abort", "callback_data": "abort"}]],
            )

        if cmd == "/cancel_all":
            return self._send_inline(
                "Confirm cancel all orders?",
                [[{"text": "✅ Confirm", "callback_data": "confirm_cancel_all"},
                  {"text": "❌ Abort", "callback_data": "abort"}]],
            )

        if cmd == "/tick":
            if not self._tick_once:
                return self._send("No tick hook wired.")
            res = self._tick_once()
            if not res:
                return self._send("Tick completed. No actionable signal.")
            return self._send("Tick: " + json.dumps(res, default=str))

        if cmd == "/id":
            return self._send(f"Chat id: `{self._chat_id}`", parse_mode="Markdown")

        # fallback
        return self._send(f"Unknown command: {cmd}")