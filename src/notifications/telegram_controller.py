from __future__ import annotations

import hashlib
import inspect
import json
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import requests

from src.config import settings

log = logging.getLogger(__name__)


class TelegramController:
    """
    Mandatory Telegram controller

    Features:
      • Startup alert + rate-limited outbound sending
      • Long‑polling updates with allowlist (owner + extra admins)
      • Operational commands:
          /help, /status [v], /equity, /active [p], /positions, /logs [n]
          /pause, /resume, /mode live|dry, /cancel_all (inline confirm)
          /tick, /tickdry, /why (last flow), /signal (last signal)
          /check (deep diag), /diag (compact)
      • Live tuning hooks:
          /risk <pct>, /trail on|off, /trailmult <x>, /partial on|off,
          /tp1 <pct>, /breakeven <ticks>,
          /minscore <n>, /conf <x>, /atrp <n>, /slmult <x>, /tpmult <x>,
          /trend <tp_boost> <sl_relax>, /range <tp_tighten> <sl_tighten>
    """

    def __init__(
        self,
        *,
        # Providers (all optional but recommended)
        status_provider: Optional[Callable[[], Dict[str, Any]]] = None,
        positions_provider: Optional[Callable[[], Dict[str, Any]]] = None,
        actives_provider: Optional[Callable[[], List[Any]]] = None,
        diag_provider: Optional[Callable[[], Dict[str, Any]]] = None,
        logs_provider: Optional[Callable[[int], List[str]]] = None,
        last_signal_provider: Optional[Callable[[], Optional[Dict[str, Any]]]] = None,
        last_flow_provider: Optional[Callable[[], Optional[Dict[str, Any]]]] = None,
        equity_provider: Optional[Callable[[], Dict[str, Any]]] = None,
        # Controls
        runner_pause: Optional[Callable[[], None]] = None,
        runner_resume: Optional[Callable[[], None]] = None,
        runner_tick: Optional[Callable[..., Optional[Dict[str, Any]]]] = None,   # may accept dry=bool
        runner_tick_mock: Optional[Callable[[], Optional[Dict[str, Any]]]] = None,
        cancel_all: Optional[Callable[[], None]] = None,
        set_live_mode: Optional[Callable[[bool], None]] = None,
        # Execution mutators
        set_risk_pct: Optional[Callable[[float], None]] = None,
        toggle_trailing: Optional[Callable[[bool], None]] = None,
        set_trailing_mult: Optional[Callable[[float], None]] = None,
        toggle_partial: Optional[Callable[[bool], None]] = None,
        set_tp1_ratio: Optional[Callable[[float], None]] = None,
        set_breakeven_ticks: Optional[Callable[[int], None]] = None,
        # Strategy mutators
        set_min_score: Optional[Callable[[int], None]] = None,
        set_conf_threshold: Optional[Callable[[float], None]] = None,
        set_atr_period: Optional[Callable[[int], None]] = None,
        set_sl_mult: Optional[Callable[[float], None]] = None,
        set_tp_mult: Optional[Callable[[float], None]] = None,
        set_trend_boosts: Optional[Callable[[float, float], None]] = None,
        set_range_tighten: Optional[Callable[[float, float], None]] = None,
        http_timeout: float = 20.0,
    ) -> None:
        tg = getattr(settings, "telegram", object())
        self._token: Optional[str] = getattr(tg, "bot_token", None)
        self._chat_id: Optional[int] = int(getattr(tg, "chat_id", 0) or 0)

        # Telegram is mandatory for this bot
        if not self._token or not self._chat_id:
            raise RuntimeError(
                "TelegramController: TELEGRAM__BOT_TOKEN and TELEGRAM__CHAT_ID are required."
            )

        self._base = f"https://api.telegram.org/bot{self._token}"
        self._timeout = float(http_timeout)

        # Hooks
        self._status = status_provider
        self._positions = positions_provider
        self._actives = actives_provider
        self._diag = diag_provider
        self._logs = logs_provider
        self._last_signal = last_signal_provider
        self._last_flow = last_flow_provider
        self._equity = equity_provider

        self._pause = runner_pause
        self._resume = runner_resume
        self._tick = runner_tick
        self._tick_mock = runner_tick_mock
        self._cancel_all = cancel_all
        self._set_live_mode = set_live_mode

        self._set_risk_pct = set_risk_pct
        self._toggle_trailing = toggle_trailing
        self._set_trailing_mult = set_trailing_mult
        self._toggle_partial = toggle_partial
        self._set_tp1_ratio = set_tp1_ratio
        self._set_breakeven_ticks = set_breakeven_ticks

        self._set_min_score = set_min_score
        self._set_conf_threshold = set_conf_threshold
        self._set_atr_period = set_atr_period
        self._set_sl_mult = set_sl_mult
        self._set_tp_mult = set_tp_mult
        self._set_trend_boosts = set_trend_boosts
        self._set_range_tighten = set_range_tighten

        # Polling state
        self._poll_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._started = False
        self._last_update_id: Optional[int] = None

        # Allowlist (owner + extra admins)
        extra = getattr(tg, "extra_admin_ids", []) or []
        self._allowlist = {int(self._chat_id), *[int(x) for x in extra if str(x).strip()]}

        # Rate control / backoff
        self._send_min_interval = 0.9
        self._last_sends: List[tuple[float, str]] = []
        self._backoff = 1.0
        self._backoff_max = 20.0

    # ---------------- Outbound ----------------
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

    def _send(self, text: str, *, parse_mode: Optional[str] = None, silent: bool = False) -> None:
        if not text:
            return
        if not self._rate_ok(text):
            return
        delay = self._backoff
        while True:
            try:
                payload = {"chat_id": self._chat_id, "text": text, "disable_notification": bool(silent)}
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
        try:
            requests.post(
                f"{self._base}/sendMessage",
                json={"chat_id": self._chat_id, "text": text, "reply_markup": {"inline_keyboard": buttons}},
                timeout=self._timeout,
            )
        except Exception as e:
            log.debug("Inline send failed: %s", e)

    def send_startup_alert(self) -> None:
        status = {}
        try:
            status = self._status() if self._status else {}
        except Exception:
            pass
        self._send(
            "🚀 Bot started\n"
            f"🔁 Trading: {'🟢 LIVE' if status.get('live_trading') else '🟡 DRY'}\n"
            f"🧠 Broker: {status.get('broker')}\n"
            f"📦 Active: {status.get('active_orders', 0)}"
        )

    # ---------------- Polling ----------------
    def start_polling(self) -> None:
        if self._started:
            log.info("Telegram polling already running; skip start.")
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

    # ---------------- Helpers ----------------
    def _authorized(self, chat_id: int) -> bool:
        return int(chat_id) in self._allowlist

    def _do_tick(self, *, dry: bool) -> str:
        """Run one decision cycle. If provider doesn't accept 'dry', emulate."""
        if not self._tick and not self._tick_mock:
            return "Tick not wired."
        # Prefer real runner_tick if present
        if self._tick:
            accepts_dry = False
            try:
                sig = inspect.signature(self._tick)  # type: ignore[arg-type]
                accepts_dry = "dry" in sig.parameters
            except Exception:
                pass
            try:
                if accepts_dry:
                    res = self._tick(dry=dry)  # type: ignore[misc]
                else:
                    # emulate "dry" by toggling offhours flag temporarily
                    allow_off_before = bool(settings.allow_offhours_testing)
                    try:
                        if dry:
                            setattr(settings, "allow_offhours_testing", True)
                        res = self._tick()
                    finally:
                        setattr(settings, "allow_offhours_testing", allow_off_before)
                return "✅ Tick executed." if res else "Tick executed (no action)."
            except Exception as e:
                return f"Tick error: {e}"
        # Fallback mock (useful after-hours)
        try:
            res = self._tick_mock() if self._tick_mock else None
            return "✅ Mock tick executed." if res else "Mock tick executed (no signal)."
        except Exception as e:
            return f"Mock tick error: {e}"

    # ---------------- Command handling ----------------
    def _handle_update(self, upd: Dict[str, Any]) -> None:
        # Inline callback buttons (confirmations)
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

        # Normal message
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

        # Help
        if cmd in ("/start", "/help"):
            return self._send(
                "🤖 Nifty Scalper — commands\n"
                "*Core*\n"
                "/status [v] • /equity • /active [p] • /positions\n"
                "/cancel_all • /pause • /resume • /mode live|dry\n"
                "/tick • /tickdry • /why • /signal • /logs [n]\n"
                "/diag • /check\n"
                "*Execution*\n"
                "/risk <pct> • /trail on|off • /trailmult <x>\n"
                "/partial on|off • /tp1 <pct> • /breakeven <ticks>\n"
                "*Strategy*\n"
                "/minscore <n> • /conf <x> • /atrp <n>\n"
                "/slmult <x> • /tpmult <x> • /trend a b • /range a b\n",
                parse_mode="Markdown",
            )

        # Status
        if cmd == "/status":
            s = {}
            try:
                s = self._status() if self._status else {}
            except Exception:
                pass
            verbose = (args and args[0].lower().startswith("v"))
            if verbose:
                return self._send("```json\n" + json.dumps(s, indent=2) + "\n```", parse_mode="Markdown")
            return self._send(
                f"📊 {s.get('time_ist')}\n"
                f"🔁 {'🟢 LIVE' if s.get('live_trading') else '🟡 DRY'} | {s.get('broker')}\n"
                f"📦 Active: {s.get('active_orders', 0)}"
            )

        # Equity snapshot
        if cmd == "/equity":
            e = {}
            try:
                e = self._equity() if self._equity else {}
            except Exception:
                pass
            return self._send("```json\n" + json.dumps(e, indent=2) + "\n```", parse_mode="Markdown")

        # Actives
        if cmd == "/active":
            if not self._actives:
                return self._send("No active-orders provider wired.")
            try:
                page = int(args[0]) if args else 1
            except Exception:
                page = 1
            acts = self._actives() or []
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

        # Positions
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

        # Cancel all
        if cmd == "/cancel_all":
            return self._send_inline(
                "Confirm cancel all?",
                [[{"text": "✅ Confirm", "callback_data": "confirm_cancel_all"},
                  {"text": "❌ Abort", "callback_data": "abort"}]],
            )

        # Pause / Resume
        if cmd == "/pause":
            if self._pause:
                self._pause()
                return self._send("⏸️ Entries paused.")
            return self._send("Pause not wired.")

        if cmd == "/resume":
            if self._resume:
                self._resume()
                return self._send("▶️ Entries resumed.")
            return self._send("Resume not wired.")

        # Mode
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

        # Tick / Tickdry
        if cmd in ("/tick", "/tickdry"):
            dry = (cmd == "/tickdry") or (args and args[0].lower().startswith("dry"))
            out = self._do_tick(dry=dry)
            return self._send(out)

        # Logs
        if cmd == "/logs":
            if not self._logs:
                return self._send("Logs provider not wired.")
            try:
                n = int(args[0]) if args else 30
                lines = self._logs(max(5, min(200, n))) or []
                if not lines:
                    return self._send("No logs available.")
                block = "\n".join(lines[-n:])
                if len(block) > 3500:
                    block = block[-3500:]
                return self._send("```text\n" + block + "\n```", parse_mode="Markdown")
            except Exception as e:
                return self._send(f"Logs error: {e}")

        # Flow/why
        if cmd == "/why":
            d = {}
            try:
                d = self._last_flow() or {}
            except Exception:
                pass
            if not d:
                return self._send("No recent decision.")
            return self._send("```json\n" + json.dumps(d, indent=2) + "\n```", parse_mode="Markdown")

        # Last signal snapshot
        if cmd == "/signal":
            d = {}
            try:
                d = self._last_signal() or {}
            except Exception:
                pass
            if not d:
                return self._send("No last signal available.")
            return self._send("```json\n" + json.dumps(d, indent=2) + "\n```", parse_mode="Markdown")

        # Diag compact
        if cmd == "/diag":
            if not self._diag:
                return self._send("Diag provider not wired.")
            try:
                d = self._diag() or {}
                ok = d.get("ok", False)
                checks = d.get("checks", [])
                summary = []
                for c in checks:
                    mark = "🟢" if c.get("ok") else "🔴"
                    summary.append(f"{mark} {c.get('name')}")
                head = "✅ Flow looks good" if ok else "❗ Flow has issues"
                return self._send(f"{head}\n" + " · ".join(summary))
            except Exception as e:
                return self._send(f"Diag error: {e}")

        # Diag deep
        if cmd == "/check":
            if not self._diag:
                return self._send("Diag provider not wired.")
            try:
                d = self._diag() or {}
                lines = ["🔍 Full system check"]
                for c in d.get("checks", []):
                    mark = "🟢" if c.get("ok") else "🔴"
                    extra = c.get("hint") or c.get("detail") or ""
                    if extra:
                        lines.append(f"{mark} {c.get('name')} — {extra}")
                    else:
                        lines.append(f"{mark} {c.get('name')}")
                lines.append("📈 last_signal: " + ("present" if d.get("last_signal") else "none"))
                return self._send("\n".join(lines))
            except Exception as e:
                return self._send(f"Check error: {e}")

        # ---------- Execution tuning ----------
        if cmd == "/risk":
            if not args:
                return self._send("Usage: /risk 0.5  (for 0.5%)")
            try:
                pct = float(args[0])
                if self._set_risk_pct:
                    self._set_risk_pct(pct)
                else:
                    settings.risk.risk_per_trade = pct / 100.0
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
                settings.executor.enable_trailing = val
            return self._send(f"Trailing {'ON' if val else 'OFF'}.")

        if cmd == "/trailmult":
            if not args:
                return self._send("Usage: /trailmult 1.5")
            try:
                x = float(args[0])
                if self._set_trailing_mult:
                    self._set_trailing_mult(x)
                else:
                    settings.executor.trailing_atr_multiplier = x
                return self._send(f"Trailing ATR multiplier set to {x}.")
            except Exception:
                return self._send("Invalid number. Example: /trailmult 1.5")

        if cmd == "/partial":
            if not args:
                return self._send("Usage: /partial on|off")
            val = str(args[0]).lower() in ("on", "true", "1", "yes")
            if self._toggle_partial:
                self._toggle_partial(val)
            else:
                settings.executor.partial_tp_enable = val
            return self._send(f"Partial TP {'ON' if val else 'OFF'}.")

        if cmd == "/tp1":
            if not args:
                return self._send("Usage: /tp1 40   (percentage of qty for TP1)")
            try:
                pct = float(args[0])
                if self._set_tp1_ratio:
                    self._set_tp1_ratio(pct)
                else:
                    settings.executor.tp1_qty_ratio = pct / 100.0
                return self._send(f"TP1 ratio set to {pct:.1f}%.")
            except Exception:
                return self._send("Invalid number. Example: /tp1 40")

        if cmd == "/breakeven":
            if not args:
                return self._send("Usage: /breakeven <ticks>")
            try:
                ticks = int(args[0])
                if self._set_breakeven_ticks:
                    self._set_breakeven_ticks(ticks)
                else:
                    settings.executor.breakeven_ticks = ticks
                return self._send(f"Breakeven ticks set to {ticks}.")
            except Exception:
                return self._send("Invalid integer. Example: /breakeven 2")

        # ---------- Strategy tuning ----------
        if cmd == "/minscore":
            try:
                n = int(args[0])
                if self._set_min_score:
                    self._set_min_score(n)
                else:
                    settings.strategy.min_signal_score = n
                return self._send(f"Min signal score → {n}.")
            except Exception:
                return self._send("Usage: /minscore <int>")

        if cmd == "/conf":
            try:
                x = float(args[0])
                if self._set_conf_threshold:
                    self._set_conf_threshold(x)
                else:
                    settings.strategy.confidence_threshold = x
                return self._send(f"Confidence threshold → {x}.")
            except Exception:
                return self._send("Usage: /conf <float>")

        if cmd == "/atrp":
            try:
                n = int(args[0])
                if self._set_atr_period:
                    self._set_atr_period(n)
                else:
                    settings.strategy.atr_period = n
                return self._send(f"ATR period → {n}.")
            except Exception:
                return self._send("Usage: /atrp <int>")

        if cmd == "/slmult":
            try:
                x = float(args[0])
                if self._set_sl_mult:
                    self._set_sl_mult(x)
                else:
                    settings.strategy.atr_sl_multiplier = x
                return self._send(f"SL multiplier → {x}.")
            except Exception:
                return self._send("Usage: /slmult <float>")

        if cmd == "/tpmult":
            try:
                x = float(args[0])
                if self._set_tp_mult:
                    self._set_tp_mult(x)
                else:
                    settings.strategy.atr_tp_multiplier = x
                return self._send(f"TP multiplier → {x}.")
            except Exception:
                return self._send("Usage: /tpmult <float>")

        if cmd == "/trend":
            if len(args) != 2:
                return self._send("Usage: /trend <tp_boost> <sl_relax>")
            try:
                a, b = float(args[0]), float(args[1])
                if self._set_trend_boosts:
                    self._set_trend_boosts(a, b)
                else:
                    settings.strategy.trend_tp_boost = a
                    settings.strategy.trend_sl_relax = b
                return self._send(f"Trend shaping → tp+{a}, sl+{b}.")
            except Exception:
                return self._send("Usage: /trend <float> <float>")

        if cmd == "/range":
            if len(args) != 2:
                return self._send("Usage: /range <tp_tighten> <sl_tighten>")
            try:
                a, b = float(args[0]), float(args[1])
                if self._set_range_tighten:
                    self._set_range_tighten(a, b)
                else:
                    settings.strategy.range_tp_tighten = a
                    settings.strategy.range_sl_tighten = b
                return self._send(f"Range shaping → tp{a:+}, sl{b:+}.")
            except Exception:
                return self._send("Usage: /range <float> <float>")

        # Unknown command
        return self._send("Unknown command. Try /help.")

    # Public helper
    def send_message(self, text: str) -> None:
        self._send(text)