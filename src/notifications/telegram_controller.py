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
    Production-safe Telegram controller:

    - Pulls token/chat_id from settings.telegram
    - Provides /status /diag /check /logs /tick /tickdry + tuning commands
    - Health Cards message for /health (and used inside /check when verbose)
    - Dedup + rate limiting + backoff on send
    """

    # ---------------- init ----------------
    def __init__(
        self,
        *,
        # providers
        status_provider: Callable[[], Dict[str, Any]],
        positions_provider: Optional[Callable[[], Dict[str, Any]]] = None,
        actives_provider: Optional[Callable[[], List[Any]]] = None,
        diag_provider: Optional[Callable[[], Dict[str, Any]]] = None,
        logs_provider: Optional[Callable[[int], List[str]]] = None,
        last_signal_provider: Optional[Callable[[], Optional[Dict[str, Any]]]] = None,
        # controls
        runner_pause: Optional[Callable[[], None]] = None,
        runner_resume: Optional[Callable[[], None]] = None,
        runner_tick: Optional[Callable[..., Optional[Dict[str, Any]]]] = None,
        cancel_all: Optional[Callable[[], None]] = None,
        # bot/strategy mutators
        set_live_mode: Optional[Callable[[bool], None]] = None,
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
        if not self._token or not self._chat_id:
            raise RuntimeError("TelegramController: TELEGRAM__BOT_TOKEN or TELEGRAM__CHAT_ID missing")

        self._base = f"https://api.telegram.org/bot{self._token}"
        self._timeout = http_timeout
        self._session = requests.Session()

        # hooks
        self._status_provider = status_provider
        self._positions_provider = positions_provider
        self._actives_provider = actives_provider
        self._diag_provider = diag_provider
        self._logs_provider = logs_provider
        self._last_signal_provider = last_signal_provider

        self._runner_pause = runner_pause
        self._runner_resume = runner_resume
        self._runner_tick = runner_tick
        self._cancel_all = cancel_all

        self._set_live_mode = set_live_mode
        self._set_min_score = set_min_score
        self._set_conf_threshold = set_conf_threshold
        self._set_atr_period = set_atr_period
        self._set_sl_mult = set_sl_mult
        self._set_tp_mult = set_tp_mult
        self._set_trend_boosts = set_trend_boosts
        self._set_range_tighten = set_range_tighten

        # polling state
        self._poll_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._started = False
        self._last_update_id: Optional[int] = None

        # allowlist
        extra = getattr(tg, "extra_admin_ids", []) or []
        self._allowlist = {int(self._chat_id), *[int(x) for x in extra]}

        # rate-limit / backoff
        self._send_min_interval = 0.9
        self._last_sends: List[tuple[float, str]] = []
        self._backoff = 1.0
        self._backoff_max = 20.0

    # --------------- outbound helpers ---------------
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
                self._session.post(f"{self._base}/sendMessage", json=payload, timeout=self._timeout)
                self._backoff = 1.0
                return
            except Exception:
                time.sleep(delay)
                delay = min(self._backoff_max, delay * 2)
                self._backoff = delay

    def _send_inline(self, text: str, buttons: list[list[dict]]) -> None:
        payload = {"chat_id": self._chat_id, "text": text, "reply_markup": {"inline_keyboard": buttons}}
        try:
            self._session.post(f"{self._base}/sendMessage", json=payload, timeout=self._timeout)
        except Exception as e:
            log.debug("Inline send failed: %s", e)

    # public
    def send_message(self, text: str, *, parse_mode: Optional[str] = None) -> None:
        self._send(text, parse_mode=parse_mode)

    # --------------- polling loop ---------------
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
                r = self._session.get(f"{self._base}/getUpdates", params=params, timeout=self._timeout + 10)
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

    # --------------- helpers ---------------
    def _authorized(self, chat_id: int) -> bool:
        return int(chat_id) in self._allowlist

    def _do_tick(self, *, dry: bool) -> str:
        if not self._runner_tick:
            return "Tick not wired."
        accepts_dry = False
        try:
            sig = inspect.signature(self._runner_tick)  # type: ignore[arg-type]
            accepts_dry = "dry" in sig.parameters
        except Exception:
            accepts_dry = False

        live_before = getattr(settings, "enable_live_trading", False)
        allow_off_before = getattr(settings, "allow_offhours_testing", False)
        try:
            if dry:
                try:
                    setattr(settings, "allow_offhours_testing", True)
                except Exception:
                    pass
                if accepts_dry:
                    res = self._runner_tick(dry=True)  # type: ignore[misc]
                else:
                    if self._set_live_mode:
                        self._set_live_mode(False)
                    res = self._runner_tick()
            else:
                res = self._runner_tick()
        except Exception as e:
            return f"Tick error: {e}"
        finally:
            try:
                if self._set_live_mode:
                    self._set_live_mode(bool(live_before))
            except Exception:
                pass
            try:
                setattr(settings, "allow_offhours_testing", bool(allow_off_before))
            except Exception:
                pass

        return "✅ Tick executed." if res else ("Dry tick executed (no action)." if dry else "Tick executed (no action).")

    # --------------- health cards ---------------
    def _health_cards(self) -> str:
        """Multiline ‘cards’ view summarising health; used by /health and /check verbose."""
        try:
            s = self._status_provider() if self._status_provider else {}
        except Exception:
            s = {}
        try:
            diag = self._diag_provider() if self._diag_provider else {}
        except Exception:
            diag = {}

        # Sections
        def mark(ok: bool) -> str:
            return "🟢" if ok else "🔴"

        # Build
        lines: List[str] = []
        lines.append("🔍 *Health Report*")
        # Top row
        if diag and isinstance(diag.get("checks"), list):
            # group a few primary checks
            prim = []
            for name in [
                "Data source (Kite)",
                "Trading window",
                "OHLC fetch",
                "Equity snapshot",
                "Risk gates",
            ]:
                found = next((c for c in diag["checks"] if c["name"] == name), None)
                if found:
                    prim.append(f"{mark(found.get('ok', False))} {name.split(' ')[0]}")
            if prim:
                lines.append(" ".join(prim))

        # Data card
        lines.append(f"{mark(True)}🪣 *Data*")
        lines.append("rows=?  age=?s  tf=?  src=?")

        # Strategy card
        lines.append(f"{mark(True)}🧠 *Strategy*")
        lines.append("score=—  conf=—  rr=—  regime=—  last_age=—s")

        # Orders card
        lines.append(f"{mark(True)}🛠 *Orders*")
        lines.append("active=—  last_exec_age=—s  err=—")

        # Risk card
        lines.append(f"{mark(True)}💰 *Risk*")
        lines.append("pnl=0.00  loss=0.00/0.00  trades=0  streak=0  equity=0.00  floor=0.00")

        # System card
        lines.append(f"{mark(True)}⚙️ *System*")
        lines.append("ticks=—  errs=—  rate=—%  cpu=—%  mem=—%  last_err=—")

        # Broker card
        lines.append(f"{mark(bool(s.get('broker') and s.get('broker') != 'None'))}🧩 *Broker*")
        lines.append(f"live={s.get('live_trading')}  broker={s.get('broker')}")

        return "\n".join(lines)

    # --------------- commands ---------------
    def _handle_update(self, upd: Dict[str, Any]) -> None:
        # Inline callbacks
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
                    self._session.post(
                        f"{self._base}/answerCallbackQuery",
                        json={"callback_query_id": cq.get("id")},
                        timeout=self._timeout,
                    )
                except Exception:
                    pass
            return

        # Text messages
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

        # HELP
        if cmd in ("/start", "/help"):
            return self._send(
                "🤖 Nifty Scalper Bot — commands\n"
                "*Core*\n"
                "/status [verbose]  ·  /health  ·  /diag  ·  /check\n"
                "/tick  ·  /tickdry  ·  /logs [n]\n"
                "/pause  ·  /resume  ·  /mode live|dry\n"
                "*Strategy*\n"
                "/minscore n  ·  /conf x  ·  /atrp n  ·  /slmult x  ·  /tpmult x\n"
                "/trend tp_boost sl_relax  ·  /range tp_tighten sl_tighten\n",
                parse_mode="Markdown",
            )

        # STATUS
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

        # HEALTH (cards)
        if cmd == "/health":
            return self._send(self._health_cards(), parse_mode="Markdown")

        # DIAG (compact line)
        if cmd == "/diag":
            if not self._diag_provider:
                return self._send("Diag provider not wired.")
            try:
                d = self._diag_provider() or {}
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

        # CHECK (multiline, with hints)
        if cmd == "/check":
            if not self._diag_provider:
                return self._send("Diag provider not wired.")
            try:
                d = self._diag_provider() or {}
                lines = ["🔍 Full system check"]
                for c in d.get("checks", []):
                    mark = "🟢" if c.get("ok") else "🔴"
                    extra = c.get("hint") or c.get("detail") or ""
                    if extra:
                        lines.append(f"{mark} {c.get('name')} — {extra}")
                    else:
                        lines.append(f"{mark} {c.get('name')}")
                if d.get("last_signal"):
                    lines.append("📈 last_signal: present")
                else:
                    lines.append("📈 last_signal: none")
                return self._send("\n".join(lines))
            except Exception as e:
                return self._send(f"Check error: {e}")

        # MODE
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
            return self._send(f"Mode set to {'LIVE' if val else 'DRY'} and rewired.")

        # TICKs
        if cmd in ("/tick", "/tickdry"):
            dry = (cmd == "/tickdry") or (args and args[0].lower().startswith("dry"))
            out = self._do_tick(dry=dry)
            return self._send(out)

        # LOGS
        if cmd == "/logs":
            if not self._logs_provider:
                return self._send("Logs provider not wired.")
            try:
                n = int(args[0]) if args else 30
                lines = self._logs_provider(max(5, min(200, n))) or []
                if not lines:
                    return self._send("No logs available.")
                block = "\n".join(lines[-n:])
                if len(block) > 3500:
                    block = block[-3500:]
                return self._send("```text\n" + block + "\n```", parse_mode="Markdown")
            except Exception as e:
                return self._send(f"Logs error: {e}")

        # PAUSE/RESUME
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

        # CANCEL ALL
        if cmd == "/cancel_all":
            return self._send_inline(
                "Confirm cancel all?",
                [[{"text": "✅ Confirm", "callback_data": "confirm_cancel_all"},
                  {"text": "❌ Abort", "callback_data": "abort"}]],
            )

        # --- Strategy tuning ---
        if cmd == "/minscore":
            if not args:
                return self._send("Usage: /minscore 3")
            try:
                n = int(args[0])
                if self._set_min_score:
                    self._set_min_score(n)
                return self._send(f"Min signal score set to {n}.")
            except Exception:
                return self._send("Invalid integer.")

        if cmd == "/conf":
            if not args:
                return self._send("Usage: /conf 0.6")
            try:
                x = float(args[0])
                if self._set_conf_threshold:
                    self._set_conf_threshold(x)
                return self._send(f"Confidence threshold set to {x:.2f}.")
            except Exception:
                return self._send("Invalid number.")

        if cmd == "/atrp":
            if not args:
                return self._send("Usage: /atrp 14")
            try:
                n = int(args[0])
                if self._set_atr_period:
                    self._set_atr_period(n)
                return self._send(f"ATR period set to {n}.")
            except Exception:
                return self._send("Invalid integer.")

        if cmd == "/slmult":
            if not args:
                return self._send("Usage: /slmult 1.3")
            try:
                x = float(args[0])
                if self._set_sl_mult:
                    self._set_sl_mult(x)
                return self._send(f"SL ATR multiplier set to {x:.2f}.")
            except Exception:
                return self._send("Invalid number.")

        if cmd == "/tpmult":
            if not args:
                return self._send("Usage: /tpmult 2.2")
            try:
                x = float(args[0])
                if self._set_tp_mult:
                    self._set_tp_mult(x)
                return self._send(f"TP ATR multiplier set to {x:.2f}.")
            except Exception:
                return self._send("Invalid number.")

        if cmd == "/trend":
            if len(args) < 2:
                return self._send("Usage: /trend <tp_boost> <sl_relax>")
            try:
                a, b = float(args[0]), float(args[1])
                if self._set_trend_boosts:
                    self._set_trend_boosts(a, b)
                return self._send(f"Trend boosts: tp+{a}, sl+{b}")
            except Exception:
                return self._send("Invalid numbers.")

        if cmd == "/range":
            if len(args) < 2:
                return self._send("Usage: /range <tp_tighten> <sl_tighten>")
            try:
                a, b = float(args[0]), float(args[1])
                if self._set_range_tighten:
                    self._set_range_tighten(a, b)
                return self._send(f"Range tighten: tp{a:+}, sl{b:+}")
            except Exception:
                return self._send("Invalid numbers.")

        # Unknown
        return self._send("Unknown command. Try /help.")