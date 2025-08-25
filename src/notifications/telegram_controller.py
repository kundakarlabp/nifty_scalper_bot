# Path: src/notifications/telegram_controller.py
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import requests

log = logging.getLogger(__name__)


class TelegramController:
    """
    Lightweight Telegram Bot controller using the HTTP Bot API.
    - Supports start/stop polling in a background thread
    - Minimal command set for status, logs, manual tick, mock tick, pause/resume, etc.
    - Safe to use in production containers (no webhook required)

    Public methods used by the app:
      - start_polling(), stop_polling()
      - send_message(text), send_startup_alert()
      - notify_entry(...), notify_fills([...])
    """

    def __init__(
        self,
        *,
        bot_token: str,
        chat_id: str,
        # Providers (all optional but recommended)
        status_provider: Optional[Callable[[], Dict[str, Any]]] = None,
        positions_provider: Optional[Callable[[], Dict[str, Any]]] = None,
        actives_provider: Optional[Callable[[], List[Any]]] = None,
        diag_provider: Optional[Callable[[], Dict[str, Any]]] = None,
        logs_provider: Optional[Callable[[int], List[str]]] = None,
        last_signal_provider: Optional[Callable[[], Optional[Dict[str, Any]]]] = None,
        # Controls / actions (optional)
        runner_pause: Optional[Callable[[], None]] = None,
        runner_resume: Optional[Callable[[], None]] = None,
        runner_tick: Optional[Callable[[], Optional[Dict[str, Any]]]] = None,
        runner_tick_mock: Optional[Callable[[], Optional[Dict[str, Any]]]] = None,
        cancel_all: Optional[Callable[[], None]] = None,
        # Risk/executor tunables (optional)
        set_risk_pct: Optional[Callable[[float], None]] = None,
        toggle_trailing: Optional[Callable[[bool], None]] = None,
        set_trailing_mult: Optional[Callable[[float], None]] = None,
        toggle_partial: Optional[Callable[[bool], None]] = None,
        set_tp1_ratio: Optional[Callable[[float], None]] = None,
        set_breakeven_ticks: Optional[Callable[[int], None]] = None,
        set_live_mode: Optional[Callable[[bool], None]] = None,
        # Strategy tunables (optional)
        set_min_score: Optional[Callable[[int], None]] = None,
        set_conf_threshold: Optional[Callable[[float], None]] = None,
        set_atr_period: Optional[Callable[[int], None]] = None,
        set_sl_mult: Optional[Callable[[float], None]] = None,
        set_tp_mult: Optional[Callable[[float], None]] = None,
        set_trend_boosts: Optional[Callable[[float, float], None]] = None,
        set_range_tighten: Optional[Callable[[float, float], None]] = None,
    ) -> None:
        if not bot_token or not chat_id:
            raise ValueError("TelegramController requires bot_token and chat_id")

        self.bot_token = bot_token.strip()
        self.chat_id = str(chat_id).strip()

        # Providers & actions
        self._status = status_provider
        self._positions = positions_provider
        self._actives = actives_provider
        self._diag = diag_provider
        self._logs = logs_provider
        self._last_signal = last_signal_provider

        self._pause = runner_pause
        self._resume = runner_resume
        self._tick = runner_tick
        self._tick_mock = runner_tick_mock
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

        # Polling state
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_update_id: Optional[int] = None

    # ----------------------- public API -----------------------

    def start_polling(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll_loop, name="tg-poll", daemon=True)
        self._thread.start()
        log.info("Telegram polling started.")

    def stop_polling(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        log.info("Telegram polling stopped.")

    def send_message(self, text: str, *, disable_web_page_preview: bool = True) -> None:
        try:
            self._api_call(
                "sendMessage",
                {"chat_id": self.chat_id, "text": str(text), "disable_web_page_preview": disable_web_page_preview},
            )
        except Exception as e:
            log.warning("Telegram send_message failed: %s", e)

    def send_startup_alert(self) -> None:
        self.send_message("✅ Nifty Scalper Bot online. Use /help for commands.")

    def notify_entry(self, *, symbol: str, side: str, qty: int, price: float, record_id: str) -> None:
        msg = f"🟢 ENTRY {side} {qty} {symbol} @ {price:.2f}\n#id {record_id}"
        self.send_message(msg)

    def notify_fills(self, fills: List[Dict[str, Any]] | List[tuple]) -> None:
        if not fills:
            return
        lines = ["📗 Fills:"]
        for f in fills:
            if isinstance(f, dict):
                s = f.get("symbol") or f.get("tradingsymbol") or "-"
                q = f.get("qty") or f.get("quantity") or f.get("filled_quantity") or "-"
                p = f.get("price") or f.get("avg_price") or f.get("fill_price") or "-"
                lines.append(f"• {s} qty={q} @ {p}")
            else:
                try:
                    rid, px = f
                    lines.append(f"• {rid} @ {px}")
                except Exception:
                    lines.append(f"• {f}")
        self.send_message("\n".join(lines))

    # ----------------------- polling loop -----------------------

    def _poll_loop(self) -> None:
        base = f"https://api.telegram.org/bot{self.bot_token}/getUpdates"
        timeout = 20
        while not self._stop.is_set():
            try:
                params = {"timeout": timeout}
                if self._last_update_id is not None:
                    params["offset"] = self._last_update_id + 1
                r = requests.get(base, params=params, timeout=timeout + 5)
                r.raise_for_status()
                data = r.json()
                if not data.get("ok"):
                    time.sleep(2)
                    continue

                for upd in data.get("result", []):
                    self._last_update_id = upd.get("update_id", self._last_update_id)
                    msg = (upd.get("message") or upd.get("edited_message")) or {}
                    text = (msg.get("text") or "").strip()
                    chat_id = str(((msg.get("chat") or {}).get("id")) or "")
                    if not text or chat_id != self.chat_id:
                        continue
                    self._handle_command(text)
            except Exception as e:
                log.warning("Telegram polling error: %s", e)
                time.sleep(2)

    # ----------------------- commands -----------------------

    def _handle_command(self, text: str) -> None:
        t = text.strip()
        if t.lower() in ("/help", "help", "/start"):
            self._reply_help(); return

        if t.lower() == "/status":
            self._reply_status(); return
        if t.lower() == "/positions":
            self._reply_positions(); return
        if t.lower() == "/actives":
            self._reply_actives(); return
        if t.lower() == "/diag":
            self._reply_diag(); return
        if t.lower().startswith("/logs"):
            try:
                n = int(t.split(maxsplit=1)[1])
            except Exception:
                n = 30
            self._reply_logs(n); return
        if t.lower() == "/last":
            self._reply_last_signal(); return

        if t.lower() == "/pause" and self._pause:
            self._pause(); self.send_message("⏸ Paused."); return
        if t.lower() == "/resume" and self._resume:
            self._resume(); self.send_message("▶️ Resumed."); return
        if t.lower() == "/tick" and self._tick:
            res = self._safe_call(self._tick); self._reply_json("Manual tick", res); return
        if t.lower() == "/mock" and self._tick_mock:
            res = self._safe_call(self._tick_mock); self._reply_json("Mock tick", res); return
        if t.lower() == "/cancelall" and self._cancel_all:
            self._cancel_all(); self.send_message("🧹 Cancelled all open orders."); return

        # Tunables: simple space‑separated args
        try:
            if t.lower().startswith("/risk "):
                pct = float(t.split()[1])
                if self._set_risk_pct: self._set_risk_pct(pct)
                self.send_message(f"Risk per trade set to {pct:.2f}%."); return
            if t.lower().startswith("/trailing "):
                on = t.split()[1].lower() in ("1", "on", "true", "yes")
                if self._toggle_trailing: self._toggle_trailing(on)
                self.send_message(f"Trailing {'enabled' if on else 'disabled'}."); return
            if t.lower().startswith("/trailmult "):
                m = float(t.split()[1])
                if self._set_trailing_mult: self._set_trailing_mult(m)
                self.send_message(f"Trailing ATR multiplier set to {m}."); return
            if t.lower().startswith("/partial "):
                on = t.split()[1].lower() in ("1", "on", "true", "yes")
                if self._toggle_partial: self._toggle_partial(on)
                self.send_message(f"Partial TP {'enabled' if on else 'disabled'}."); return
            if t.lower().startswith("/tp1 "):
                pct = float(t.split()[1])
                if self._set_tp1_ratio: self._set_tp1_ratio(pct)
                self.send_message(f"TP1 qty ratio set to {pct:.2f}%"); return
            if t.lower().startswith("/be "):
                ticks = int(t.split()[1])
                if self._set_breakeven_ticks: self._set_breakeven_ticks(ticks)
                self.send_message(f"Breakeven ticks set to {ticks}."); return
            if t.lower().startswith("/live "):
                on = t.split()[1].lower() in ("1", "on", "true", "yes")
                if self._set_live_mode: self._set_live_mode(on)
                self.send_message(f"Live trading set to {on}."); return

            if t.lower().startswith("/minscore "):
                n = int(t.split()[1]);  self._try(self._set_min_score, n); return
            if t.lower().startswith("/conf "):
                x = float(t.split()[1]); self._try(self._set_conf_threshold, x); return
            if t.lower().startswith("/atrp "):
                n = int(t.split()[1]);  self._try(self._set_atr_period, n); return
            if t.lower().startswith("/slmult "):
                x = float(t.split()[1]); self._try(self._set_sl_mult, x); return
            if t.lower().startswith("/tpmult "):
                x = float(t.split()[1]); self._try(self._set_tp_mult, x); return
            if t.lower().startswith("/trend "):
                tp_boost, sl_relax = map(float, t.split()[1:3]); self._try(self._set_trend_boosts, tp_boost, sl_relax); return
            if t.lower().startswith("/range "):
                tp_t, sl_t = map(float, t.split()[1:3]); self._try(self._set_range_tighten, tp_t, sl_t); return
        except Exception as e:
            self.send_message(f"⚠️ Bad command: {t}\n{e}")
            return

        self.send_message("Unknown command. Use /help")

    # ----------------------- replies -----------------------

    def _reply_help(self) -> None:
        self.send_message(
            "🤖 Commands:\n"
            "/status | /positions | /actives | /diag | /last\n"
            "/logs [N]\n"
            "/pause | /resume | /tick | /mock | /cancelall\n"
            "/risk <pct> | /trailing <on|off> | /trailmult <x>\n"
            "/partial <on|off> | /tp1 <pct> | /be <ticks> | /live <on|off>\n"
            "/minscore <n> | /conf <x> | /atrp <n> | /slmult <x> | /tpmult <x>\n"
            "/trend <tp_boost> <sl_relax> | /range <tp_tighten> <sl_tighten>"
        )

    def _reply_status(self) -> None:
        payload = self._safe_call(self._status)
        self._reply_json("Status", payload)

    def _reply_positions(self) -> None:
        payload = self._safe_call(self._positions)
        self._reply_json("Positions", payload)

    def _reply_actives(self) -> None:
        payload = self._safe_call(self._actives)
        self._reply_json("Active orders", payload)

    def _reply_diag(self) -> None:
        payload = self._safe_call(self._diag)
        self._reply_json("Diagnostics", payload)

    def _reply_logs(self, n: int) -> None:
        if not self._logs:
            self.send_message("No logs provider wired.")
            return
        try:
            lines = self._logs(int(n)) or []
            if not lines:
                self.send_message("No logs.")
                return
            text = "📝 Logs (latest):\n" + "\n".join(lines[-n:])
            # Telegram message limit guard
            if len(text) > 3900:
                text = text[-3900:]
            self.send_message(text, disable_web_page_preview=True)
        except Exception as e:
            self.send_message(f"Log fetch failed: {e}")

    def _reply_last_signal(self) -> None:
        payload = self._safe_call(self._last_signal)
        self._reply_json("Last signal", payload)

    # ----------------------- helpers -----------------------

    def _api_call(self, method: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"https://api.telegram.org/bot{self.bot_token}/{method}"
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
        return r.json()

    def _reply_json(self, title: str, payload: Any) -> None:
        try:
            pretty = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        except Exception:
            pretty = str(payload)
        self.send_message(f"📦 {title}:\n<pre>{pretty}</pre>", disable_web_page_preview=True)

    def _safe_call(self, fn: Optional[Callable[..., Any]], *args, **kwargs) -> Any:
        try:
            if fn:
                return fn(*args, **kwargs)
            return {"ok": False, "error": "not_wired"}
        except Exception as e:
            log.exception("Provider/action error: %s", e)
            return {"ok": False, "error": str(e)}

    def _try(self, fn: Optional[Callable[..., Any]], *args) -> None:
        if not fn:
            self.send_message("Not wired.")
            return
        try:
            fn(*args)
            self.send_message("OK.")
        except Exception as e:
            self.send_message(f"Failed: {e}")