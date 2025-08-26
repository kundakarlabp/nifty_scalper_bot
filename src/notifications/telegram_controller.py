# src/notifications/telegram_controller.py
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
    Telegram control plane with:
      * compact /diag
      * multiline /check "health cards"
      * safe rate limiting & retries
      * same command set you already use
    """

    # ------------------------------------------------------------------ #
    # ctor
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        *,
        # providers (all optional except status)
        status_provider: Callable[[], Dict[str, Any]],
        positions_provider: Optional[Callable[[], Dict[str, Any]]] = None,
        actives_provider: Optional[Callable[[], List[Any]]] = None,
        diag_provider: Optional[Callable[[], Dict[str, Any]]] = None,
        logs_provider: Optional[Callable[[int], List[str]]] = None,
        last_signal_provider: Optional[Callable[[], Optional[Dict[str, Any]]]] = None,
        # controls
        runner_pause: Optional[Callable[[], None]] = None,
        runner_resume: Optional[Callable[[], None]] = None,
        runner_tick: Optional[Callable[..., Optional[Dict[str, Any]]]] = None,  # may accept dry=bool
        cancel_all: Optional[Callable[[], None]] = None,
        # execution mutators
        set_risk_pct: Optional[Callable[[float], None]] = None,
        toggle_trailing: Optional[Callable[[bool], None]] = None,
        set_trailing_mult: Optional[Callable[[float], None]] = None,
        toggle_partial: Optional[Callable[[bool], None]] = None,
        set_tp1_ratio: Optional[Callable[[float], None]] = None,
        set_breakeven_ticks: Optional[Callable[[int], None]] = None,
        set_live_mode: Optional[Callable[[bool], None]] = None,
        # strategy mutators
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
        self._set_risk_pct = set_risk_pct
        self._toggle_trailing = toggle_trailing
        self._set_trailing_mult = set_trailing_mult
        self._toggle_partial = toggle_partial
        self._set_tp1_ratio = set_tp1_ratio
        self._set_breakeven_ticks = set_breakeven_ticks
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

    # ------------------------------------------------------------------ #
    # outbound helpers
    # ------------------------------------------------------------------ #
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
        try:
            self._session.post(
                f"{self._base}/sendMessage",
                json={"chat_id": self._chat_id, "text": text, "reply_markup": {"inline_keyboard": buttons}},
                timeout=self._timeout,
            )
        except Exception as e:
            log.debug("Inline send failed: %s", e)

    # public API
    def send_message(self, text: str, *, parse_mode: Optional[str] = None) -> None:
        self._send(text, parse_mode=parse_mode)

    # ------------------------------------------------------------------ #
    # polling
    # ------------------------------------------------------------------ #
    def start_polling(self) -> None:
        if self._started:
            log.info("Telegram polling already running; skipping start.")
            return
        self._stop.clear()
        self._poll_thread = threading.Thread(target=self._poll_loop, name="tg-poll", daemon=True)
        self._poll_thread.start()
        self._started = True
        log.info("Telegram polling started (chat_id=%s).", self._chat_id)

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

    # ------------------------------------------------------------------ #
    # health card rendering
    # ------------------------------------------------------------------ #
    @staticmethod
    def _mark(ok: Optional[bool]) -> str:
        return "🟢" if ok else "🔴"

    def _render_health_cards(self, diag: Dict[str, Any], status: Dict[str, Any]) -> str:
        """
        Build the multiline 'Health Report' like the screenshot.
        Works even if diag/status are missing fields.
        """
        # convenience getters
        def g(d, *keys, default=None):
            cur = d or {}
            for k in keys:
                cur = cur.get(k, {})
            return cur if cur not in (None, {}) else default

        # flat checks into dict for quick lookup
        checks = {str(c.get("name")).lower(): bool(c.get("ok")) for c in (diag.get("checks") or [])}

        # Data
        data_rows = g(diag, "data", default={})
        rows = data_rows.get("rows")
        age = data_rows.get("age")
        tf = data_rows.get("tf")
        src = data_rows.get("src")

        # Strategy
        strat = g(diag, "strategy", default={})
        score = strat.get("score")
        conf = strat.get("confidence")
        rr = strat.get("rr")
        regime = (strat.get("regime") or "").replace("_", " ")
        last_age = strat.get("last_age")

        # Orders
        orders = g(diag, "orders", default={})
        actives = orders.get("active")
        last_exec_age = orders.get("last_exec_age")
        last_err = orders.get("err")

        # Risk
        risk = g(diag, "risk", default={})
        pnl = risk.get("pnl", 0.0)
        loss = risk.get("loss", "0/0")
        trades = risk.get("trades", 0)
        streak = risk.get("streak", 0)
        equity = risk.get("equity", 0.0)
        floor = risk.get("floor", 0.0)

        # System
        sysd = g(diag, "system", default={})
        ticks = sysd.get("ticks")
        errs = sysd.get("errs")
        rate = sysd.get("rate")
        cpu = sysd.get("cpu")
        mem = sysd.get("mem")

        # Broker
        live = status.get("live_trading")
        broker = status.get("broker")

        # Compose
        lines: List[str] = []
        lines.append("🔎 *Health Report*")
        lines.append("")
        # headline check line (like compact /diag)
        headline = []
        for key in (
            "flow",
            "telegram controller",
            "broker session",
            "zerodha credentials",
            "data source",
            "instrument token",
            "trading window",
            "ohlc fetch",
            "equity snapshot",
            "risk gates",
            "executor",
        ):
            # allow variety of names from the runner
            ok = None
            for k in checks:
                if key in k:
                    ok = checks[k]
                    break
            if ok is not None:
                headline.append(f"{self._mark(ok)} {key.replace('session', 'session (Kite)')}")
        if headline:
            lines.append(" · ".join(headline))
            lines.append("")

        # Sections
        def sec(title: str, body: List[str], ok: Optional[bool] = None):
            mark = "" if ok is None else f"{self._mark(ok)} "
            lines.append(f"{mark}*{title}*")
            if body:
                for b in body:
                    lines.append(b)
            else:
                lines.append("—")
            lines.append("")

        # Data
        sec("🧰 Data", [f"rows={rows or 0}  age={age or '—'}  tf={tf or '—'}  src={src or '—'}"],
            checks.get("ohlc fetch"))

        # Strategy
        sec("🧠 Strategy", [
            f"score={score or '—'}  conf={conf or '—'}  rr={rr or '—'}  regime={regime or '—'}  last_age={last_age or '—'}"
        ], checks.get("strategy"))

        # Orders
        sec("🛠 Orders", [
            f"active={actives or '—'}  last_exec_age={last_exec_age or '—'}  err={last_err or '—'}"
        ], checks.get("executor"))

        # Risk
        sec("💰 Risk", [
            f"pnl={pnl:.2f}  loss={loss}  trades={trades}  streak={streak}  equity={equity:.2f}  floor={floor:.2f}"
        ], checks.get("risk gates"))

        # System
        sec("⚙️ System", [
            f"ticks={ticks or '—'}  errs={errs or '—'}  rate={rate or '—'}  cpu={cpu or '—'}  mem={mem or '—'}  last_err={last_err or '—'}"
        ], checks.get("flow"))

        # Broker
        sec("🧩 Broker", [f"live={'Yes' if live else 'None'}  broker={broker or 'None'}"],
            checks.get("broker session"))

        # last signal
        try:
            ls = self._last_signal_provider() if self._last_signal_provider else None
        except Exception:
            ls = None
        lines.append(f"📈 last_signal: {'present' if ls else 'none'}")

        text = "\n".join(lines)
        # Telegram Markdown-safe block (avoid backticks inside)
        return text.replace("_", "\\_")

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
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
            pass

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

    # ------------------------------------------------------------------ #
    # command handling
    # ------------------------------------------------------------------ #
    def _handle_update(self, upd: Dict[str, Any]) -> None:
        # callbacks
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

        # text
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
                "/status [verbose] — basic or JSON status\n"
                "/active [page] — list active orders\n"
                "/positions — broker day positions\n"
                "/cancel_all — cancel all (with confirm)\n"
                "/pause | /resume — control entries\n"
                "/mode live|dry — toggle live trading\n"
                "/tick — one tick | /tickdry — one dry tick (after-hours ok)\n"
                "/logs [n] — recent log lines\n"
                "/diag — compact dot list  •  /check — Health Report\n"
                "*Execution Tuning*\n"
                "/risk <pct> • /trail on|off • /trailmult <x>\n"
                "/partial on|off • /tp1 <pct> • /breakeven <ticks>\n"
                "*Strategy Tuning*\n"
                "/minscore <n> • /conf <x> • /atrp <n>\n"
                "/slmult <x> • /tpmult <x> • /trend a b • /range a b\n",
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

        # ACTIVES
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

        # POSITIONS
        if cmd == "/positions":
            if not self._positions_provider:
                return self._send("No positions provider wired.")
            pos = self._positions_provider() or {}
            if not pos:
                return self._send("No positions (day).")
            lines = ["📒 Positions (day)"]
            for sym, p in pos.items():
                if isinstance(p, dict):
                    qty = p.get("quantity")
                    avg = p.get("average_price")
                else:
                    qty = getattr(p, "quantity", "?")
                    avg = getattr(p, "average_price", "?")
                lines.append(f"• {sym}: qty={qty} avg={avg}")
            return self._send("\n".join(lines))

        # CANCEL ALL
        if cmd == "/cancel_all":
            return self._send_inline(
                "Confirm cancel all?",
                [[{"text": "✅ Confirm", "callback_data": "confirm_cancel_all"},
                  {"text": "❌ Abort", "callback_data": "abort"}]],
            )

        # PAUSE / RESUME
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
            return self._send(f"Mode set to {'LIVE' if val else 'DRY'}.")

        # TICKS
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

        # DIAG (compact)
        if cmd == "/diag":
            if not self._diag_provider:
                return self._send("No checks available.")
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

        # CHECK (multiline cards)
        if cmd == "/check":
            if not self._diag_provider:
                return self._send("No checks available.")
            try:
                d = self._diag_provider() or {}
                s = self._status_provider() if self._status_provider else {}
                card = self._render_health_cards(d, s)
                # Markdown: we avoid triple backticks to keep line wrapping consistent on mobile
                return self._send(card, parse_mode="Markdown")
            except Exception as e:
                return self._send(f"Check error: {e}")

        # --- execution tuning ---
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
            return self._send(f"Trailing {'enabled' if val else 'disabled'}.")

        if cmd == "/trailmult":
            if not args:
                return self._send("Usage: /trailmult 1.4")
            try:
                x = float(args[0])
                if self._set_trailing_mult:
                    self._set_trailing_mult(x)
                return self._send(f"Trailing ATR multiplier set to {x:.2f}.")
            except Exception:
                return self._send("Invalid number. Example: /trailmult 1.4")

        if cmd == "/partial":
            if not args:
                return self._send("Usage: /partial on|off")
            val = str(args[0]).lower() in ("on", "true", "1", "yes")
            if self._toggle_partial:
                self._toggle_partial(val)
            return self._send(f"Partial TP {'enabled' if val else 'disabled'}.")

        if cmd == "/tp1":
            if not args:
                return self._send("Usage: /tp1 50   (for 50%)")
            try:
                pct = float(args[0])
                if self._set_tp1_ratio:
                    self._set_tp1_ratio(pct)
                return self._send(f"TP1 ratio set to {pct:.1f}%.")
            except Exception:
                return self._send("Invalid number. Example: /tp1 50")

        if cmd == "/breakeven":
            if not args:
                return self._send("Usage: /breakeven 2   (ticks)")
            try:
                ticks = int(args[0])
                if self._set_breakeven_ticks:
                    self._set_breakeven_ticks(ticks)
                return self._send(f"Breakeven ticks set to {ticks}.")
            except Exception:
                return self._send("Invalid integer. Example: /breakeven 2")

        # strategy tuning
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

        # unknown
        return self._send("Unknown command. Try /help.")