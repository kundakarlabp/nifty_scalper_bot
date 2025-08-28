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
    - Provides /status /health /diag /check /logs /tick /tickdry /mode /pause /resume
      plus /positions, /active and strategy tuning commands
    - Health Cards message for /health (also used by /check-like detail)
    - Dedup + rate limiting + backoff on send
    """

    @classmethod
    def create(cls, *args: Any, **kwargs: Any) -> Optional["TelegramController"]:
        """Factory that returns a controller or ``None`` if credentials are missing."""
        try:
            return cls(*args, **kwargs)
        except RuntimeError:
            log.warning("Telegram disabled: missing bot token or chat ID")
            return None

    # ---------------- init ----------------
    def __init__(
        self,
        *,
        # providers
        status_provider: Callable[[], Dict[str, Any]],
        positions_provider: Optional[Callable[[], Dict[str, Any]]] = None,
        actives_provider: Optional[Callable[[], List[Any]]] = None,
        diag_provider: Optional[Callable[[], Dict[str, Any]]] = None,          # detailed (/check)
        logs_provider: Optional[Callable[[int], List[str]]] = None,
        last_signal_provider: Optional[Callable[[], Optional[Dict[str, Any]]]] = None,
        compact_diag_provider: Optional[Callable[[], Dict[str, Any]]] = None,  # NEW: compact (/diag)
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
        self._compact_diag_provider = compact_diag_provider  # NEW
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

    @staticmethod
    def _escape_markdown(text: str) -> str:
        """Escape Markdown special characters for safe Telegram messages."""
        escape_chars = "_*[]()~`>#+-=|{}.!\\"
        return "".join(
            "\\" + ch if ch in escape_chars else ch
            for ch in text
        )

    def _send(self, text: str, parse_mode: Optional[str] = None, disable_notification: bool = False) -> None:
        if not text or not text.strip():
            return
        if not self._rate_ok(text):
            return
        delay = self._backoff
        if parse_mode:
            text = self._escape_markdown(text)
        while True:
            try:
                payload = {"chat_id": self._chat_id, "text": text, "disable_notification": disable_notification}
                if parse_mode:
                    payload["parse_mode"] = parse_mode
                response = self._session.post(
                    f"{self._base}/sendMessage", json=payload, timeout=self._timeout
                )
                try:
                    data = response.json()
                except Exception as e:
                    log.error("Telegram send JSON decode failed: %s", e)
                    raise
                if not response.ok or not data.get("ok"):
                    log.error(
                        "Telegram send failed: status=%s, data=%s",
                        response.status_code,
                        data,
                    )
                    raise RuntimeError("telegram send failed")
                self._backoff = 1.0
                return
            except Exception:
                time.sleep(delay)
                delay = min(self._backoff_max, delay * 2)
                self._backoff = delay

    def _send_inline(self, text: str, buttons: list[list[dict]]) -> None:
        payload = {"chat_id": self._chat_id, "text": text, "reply_markup": {"inline_keyboard": buttons}}
        try:
            response = self._session.post(
                f"{self._base}/sendMessage", json=payload, timeout=self._timeout
            )
            try:
                data = response.json()
            except Exception as e:
                log.error("Inline send JSON decode failed: %s", e)
                return
            if not response.ok or not data.get("ok"):
                log.error(
                    "Inline send failed: status=%s, data=%s",
                    response.status_code,
                    data,
                )
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
                setattr(settings, "allow_offhours_testing", True)
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
                log.debug("Failed to restore live trading mode", exc_info=True)
            try:
                setattr(settings, "allow_offhours_testing", bool(allow_off_before))
            except Exception:
                log.debug("Failed to restore allow_offhours_testing flag", exc_info=True)

        return "✅ Tick executed." if res else ("Dry tick executed (no action)." if dry else "Tick executed (no action).")

    # --------------- health cards ---------------
    def _health_cards(self) -> str:
        """Multiline health cards view; based on runner's diag bundle."""
        try:
            diag = self._diag_provider() if self._diag_provider else {}
        except Exception:
            diag = {}

        def mark(ok: bool) -> str:
            return "🟢" if ok else "🔴"

        lines: List[str] = []
        lines.append("🔍 *Health Report*")

        for c in diag.get("checks", []):
            name = c.get("name")
            detail = c.get("detail") or c.get("hint") or ""
            lines.append(f"{mark(c.get('ok', False))} {name} — {detail}")

        lines.append(f"📈 Last signal: {'present' if diag.get('last_signal') else 'none'}")
        lines.append(f"\nOverall: {'✅ ALL GOOD' if diag.get('ok') else '❗ Issues present'}")
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
            if data == "confirm_cancel_all" and self._cancel_all:
                self._cancel_all()
                self._send("🧹 Cancelled all open orders.")
            try:
                self._session.post(
                    f"{self._base}/answerCallbackQuery",
                    json={"callback_query_id": cq.get("id")},
                    timeout=self._timeout,
                )
            except Exception:
                log.debug("Failed to answer callback query", exc_info=True)
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
                "/status [verbose] · /health · /diag · /check\n"
                "/positions · /active [page]\n"
                "/tick · /tickdry · /logs [n]\n"
                "/pause · /resume · /mode live|dry · /cancel_all\n"
                "*Strategy*\n"
                "/minscore n · /conf x · /atrp n · /slmult x · /tpmult x\n"
                "/trend tp_boost sl_relax · /range tp_tighten sl_tighten\n",
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

        # DIAG (compact one-line summary)
        if cmd == "/diag":
            try:
                if self._compact_diag_provider:
                    diag_data = self._compact_diag_provider() or {}
                elif self._diag_provider:
                    # Fallback: derive compact view from detailed bundle
                    full = self._diag_provider() or {}
                    msgs = {}
                    for c in full.get("checks", []):
                        name = (c.get("name") or "").lower().replace(" ", "_")
                        msgs[name] = "ok" if c.get("ok") else "blocked"
                    diag_data = {
                        "ok": bool(full.get("ok", False)),
                        "status_messages": msgs,
                    }
                else:
                    return self._send("Diag provider not wired.")

                is_ok = bool(diag_data.get("ok", False))
                summary_items = diag_data.get("status_messages", {})

                parts_line: List[str] = []
                ok_statuses = {"ok", "skipped", "no-eval", "dry mode"}
                for name, status in summary_items.items():
                    icon = "🟢" if status in ok_statuses else "🔴"
                    parts_line.append(f"{icon} {name.replace('_', ' ').capitalize()}")

                head = "✅ Flow looks good" if is_ok else "❗ Flow has issues"
                return self._send(f"{head}\n" + " · ".join(parts_line))
            except Exception as e:
                return self._send(f"Diag error: {e}")

        # CHECK (multiline, with hints)
        if cmd == "/check":
            if not self._diag_provider:
                return self._send("Check provider not wired.")
            try:
                d = self._diag_provider() or {}
                if d is None:
                    return self._send("Failed to get health data.")

                lines = ["🔍 Full system check"]
                for c in d.get("checks", []):
                    mark = "🟢" if c.get("ok") else "🔴"
                    extra = c.get("hint") or c.get("detail") or ""
                    lines.append(f"{mark} {c.get('name')} — {extra}")

                lines.append(f"📈 last_signal: {'present' if d.get('last_signal') else 'none'}")

                return self._send("\n".join(lines))
            except Exception as e:
                return self._send(f"Check error: {e}")

        # POSITIONS
        if cmd == "/positions":
            if not self._positions_provider:
                return self._send("Positions provider not wired.")
            pos = self._positions_provider() or {}
            err = getattr(getattr(self._positions_provider, "__self__", None), "last_error", None)
            if err:
                return self._send(f"Positions error: {err}")
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

        # ACTIVE ORDERS
        if cmd == "/active":
            if not self._actives_provider:
                return self._send("Active-orders provider not wired.")
            try:
                page = int(args[0]) if args else 1
            except Exception:
                page = 1
            acts = self._actives_provider() or []
            err = getattr(getattr(self._actives_provider, "__self__", None), "last_error", None)
            if err:
                return self._send(f"Active-orders error: {err}")
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

        # MODE
        if cmd == "/mode":
            if not args:
                return self._send("Usage: /mode live|dry")
            state = str(args[0]).lower()
            if state not in ("live", "dry"):
                return self._send("Usage: /mode live|dry")
            val = (state == "live")
            runner = getattr(self._set_live_mode, "__self__", None) if self._set_live_mode else None
            if self._set_live_mode:
                try:
                    self._set_live_mode(val)
                except Exception as e:
                    return self._send(f"Mode error: {e}")
            else:
                setattr(settings, "enable_live_trading", val)
            msg = f"Mode set to {'LIVE' if val else 'DRY'} and rewired."
            self._send(msg)
            if runner is not None and getattr(runner, "kite", None) is None:
                self._send("⚠️ Broker session missing.")
            return

        # TICK / TICKDRY
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

        # CANCEL ALL
        if cmd == "/cancel_all":
            return self._send_inline(
                "Confirm cancel all?",
                [[{"text": "✅ Confirm", "callback_data": "confirm_cancel_all"},
                  {"text": "❌ Abort", "callback_data": "abort"}]],
            )

        # --- Strategy tuning ---
        if cmd == "/minscore":
            if self._set_min_score:
                try:
                    n = int(args[0])
                    self._set_min_score(n)
                    return self._send(f"Min signal score set to {n}.")
                except Exception:
                    return self._send("Invalid integer.")
            return self._send("Min score not wired.")

        if cmd == "/conf":
            if self._set_conf_threshold:
                try:
                    x = float(args[0])
                    self._set_conf_threshold(x)
                    return self._send(f"Confidence threshold set to {x:.2f}.")
                except Exception:
                    return self._send("Invalid number.")
            return self._send("Conf threshold not wired.")

        if cmd == "/atrp":
            if self._set_atr_period:
                try:
                    n = int(args[0])
                    self._set_atr_period(n)
                    return self._send(f"ATR period set to {n}.")
                except Exception:
                    return self._send("Invalid integer.")
            return self._send("ATR period not wired.")

        if cmd == "/slmult":
            if self._set_sl_mult:
                try:
                    x = float(args[0])
                    self._set_sl_mult(x)
                    return self._send(f"SL ATR multiplier set to {x:.2f}.")
                except Exception:
                    return self._send("Invalid number.")
            return self._send("SL multiplier not wired.")

        if cmd == "/tpmult":
            if self._set_tp_mult:
                try:
                    x = float(args[0])
                    self._set_tp_mult(x)
                    return self._send(f"TP ATR multiplier set to {x:.2f}.")
                except Exception:
                    return self._send("Invalid number.")
            return self._send("TP multiplier not wired.")

        if cmd == "/trend":
            if self._set_trend_boosts:
                if len(args) < 2:
                    return self._send("Usage: /trend <tp_boost> <sl_relax>")
                try:
                    a, b = float(args[0]), float(args[1])
                    self._set_trend_boosts(a, b)
                    return self._send(f"Trend boosts: tp+{a}, sl+{b}")
                except Exception:
                    return self._send("Invalid numbers.")
            return self._send("Trend boosts not wired.")

        if cmd == "/range":
            if self._set_range_tighten:
                if len(args) < 2:
                    return self._send("Usage: /range <tp_tighten> <sl_tighten>")
                try:
                    a, b = float(args[0]), float(args[1])
                    self._set_range_tighten(a, b)
                    return self._send(f"Range tighten: tp{a:+}, sl{b:+}")
                except Exception:
                    return self._send("Invalid numbers.")
            return self._send("Range tighten not wired.")

        # Unknown
        return self._send("Unknown command. Try /help.")
