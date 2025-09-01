from __future__ import annotations

import hashlib
import inspect
import json
import logging
import threading
import time
import os
from datetime import datetime, timedelta, time as dt_time
from typing import Any, Callable, Dict, List, Optional
from zoneinfo import ZoneInfo
import statistics as stats

from src.logs.journal import read_trades_between

import requests

from src.config import settings

log = logging.getLogger(__name__)


def _kpis(trades: List[Dict[str, float]]) -> Dict[str, float]:
    """Compute basic KPIs from a list of trade dicts."""
    if not trades:
        return {}
    rs = [t.get("pnl_R", 0.0) for t in trades]
    pos = [r for r in rs if r > 0]
    neg = [-r for r in rs if r < 0]
    pf = (sum(pos) / sum(neg)) if neg else float("inf")
    win = (sum(1 for r in rs if r > 0) / len(rs)) * 100.0
    avg_r = sum(rs) / len(rs)
    med_r = stats.median(rs)
    eq = 0.0
    peak = 0.0
    mdd = 0.0
    for r in rs:
        eq += r
        peak = max(peak, eq)
        mdd = min(mdd, eq - peak)
    losses = [r for r in rs if r < 0]
    p95_loss = 0.0
    if losses:
        try:
            p95_loss = stats.quantiles(losses, n=20, method="inclusive")[0]
        except Exception:
            p95_loss = min(losses)
    return {
        "trades": len(rs),
        "PF": round(pf, 2),
        "Win%": round(win, 1),
        "AvgR": round(avg_r, 2),
        "MedianR": round(med_r, 2),
        "MaxDD_R": round(-mdd, 2),
        "p95_loss_R": round(abs(p95_loss), 2),
    }


def _fmt_micro(
    sym: str,
    micro: dict | None,
    last_bar_ts: str | None,
    lag_s: int | None,
) -> str:
    """Format microstructure diagnostics for Telegram output."""

    m = micro or {}
    spread = m.get("spread_pct")
    depth_ok = m.get("depth_ok")
    return (
        f"quote: {sym} src={m.get('source', '-')} "
        f"ltp={m.get('ltp')} bid={m.get('bid')} ask={m.get('ask')} "
        f"spread%={spread if spread is not None else 'N/A'} "
        f"bid5={m.get('bid5')} ask5={m.get('ask5')} "
        f"depth={depth_ok if depth_ok is not None else 'N/A'} "
        f"last_bar_ts={last_bar_ts} lag_s={lag_s}"
    )

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
        risk_provider: Optional[Callable[[], Dict[str, Any]]] = None,
        limits_provider: Optional[Callable[[], Dict[str, Any]]] = None,
        risk_reset_today: Optional[Callable[[], None]] = None,
        bars_provider: Optional[Callable[[int], str]] = None,
        quotes_provider: Optional[Callable[[str], str]] = None,
        probe_provider: Optional[Callable[[], Dict[str, Any]]] = None,
        trace_provider: Optional[Callable[[int], None]] = None,
        selftest_provider: Optional[Callable[[str], str]] = None,
        backtest_provider: Optional[Callable[[Optional[str]], str]] = None,
        atm_provider: Optional[Callable[[], Dict[str, Any]]] = None,
        l1_provider: Optional[Callable[[], Optional[Dict[str, Any]]]] = None,
        # controls
        runner_pause: Optional[Callable[[], None]] = None,
        runner_resume: Optional[Callable[[], None]] = None,
        runner_tick: Optional[Callable[..., Optional[Dict[str, Any]]]] = None,
        cancel_all: Optional[Callable[[], None]] = None,
        open_trades_provider: Optional[Callable[[], List[Dict[str, Any]]]] = None,
        cancel_trade: Optional[Callable[[str], None]] = None,
        reconcile_once: Optional[Callable[[], int]] = None,
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
        self._risk_provider = risk_provider
        self._limits_provider = limits_provider
        self._risk_reset = risk_reset_today
        self._logs_provider = logs_provider
        self._last_signal_provider = last_signal_provider
        self._bars_provider = bars_provider
        self._quotes_provider = quotes_provider
        self._probe_provider = probe_provider
        self._trace_provider = trace_provider
        self._selftest_provider = selftest_provider
        self._backtest_provider = backtest_provider
        self._atm_provider = atm_provider
        self._l1_provider = l1_provider

        self._runner_pause = runner_pause
        self._runner_resume = runner_resume
        self._runner_tick = runner_tick
        self._cancel_all = cancel_all
        self._open_trades_provider = open_trades_provider
        self._cancel_trade = cancel_trade
        self._reconcile_once = reconcile_once

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
                "/status [verbose] · /health · /diag · /check · /components\n"
                "/positions · /active [page] · /risk · /limits\n"
                "/tick · /tickdry · /backtest [csv] · /logs [n]\n"
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

        # COMPONENTS
        if cmd == "/components":
            status = self._status_provider() if self._status_provider else {}
            names = status.get("components", {})
            dp_health = status.get("data_provider_health", {})
            oc_health = status.get("order_connector_health", {})
            text = (
                "⚙️ *Components*\n"
                f"Strategy: `{names.get('strategy')}`\n"
                f"Data: `{names.get('data_provider')}`  health={dp_health.get('status')}\n"
                f"Connector: `{names.get('order_connector')}`  health={oc_health.get('status')}\n"
            )
            return self._send(text, parse_mode="Markdown")

        # API HEALTH
        if cmd == "/apihealth":
            status = self._status_provider() if self._status_provider else {}
            api = status.get("api_health", {})

            def fmt(name: str, d: Dict[str, Any]) -> str:
                return (
                    f"{name}: {d.get('state')} err={d.get('err_rate', 0):.2%} "
                    f"p95={d.get('p95_ms', 0)}ms n={d.get('n', 0)} open_until={d.get('open_until')}"
                )

            lines = [
                fmt("Orders", api.get("orders", {})),
                fmt("Modify", api.get("modify", {})),
                fmt("Hist", api.get("hist", {})),
                fmt("Quote", api.get("quote", {})),
            ]
            return self._send("\n".join(lines))

        if cmd == "/reload":
            runner = getattr(getattr(self, "_runner_tick", None), "__self__", None)
            if runner and hasattr(runner, "_maybe_hot_reload_cfg"):
                before = runner.strategy_cfg.version
                runner._maybe_hot_reload_cfg()
                after = runner.strategy_cfg.version
                return self._send(
                    f"🔁 Config reloaded: {runner.strategy_cfg.name} v{after} (was v{before})"
                )
            return self._send("⚠️ Reload failed: runner missing")

        if cmd == "/config":
            runner = getattr(getattr(self, "_runner_tick", None), "__self__", None)
            if not runner:
                return self._send("Config unavailable")
            c = runner.strategy_cfg
            text = (
                "🧭 *Strategy Config*\n"
                f"name: `{c.name}` v{c.version}\n"
                f"tz: {c.tz}\n"
                f"ATR% band: {c.atr_min}–{c.atr_max}\n"
                f"score gates: trend {c.score_trend_min}, range {c.score_range_min}\n"
                f"micro: spread open {c.max_spread_pct_open}, reg {c.max_spread_pct_regular}, close {c.max_spread_pct_last20m}, depth×{c.depth_multiplier}\n"
                f"options: OI≥{c.min_oi}, Δ∈[{c.delta_min},{c.delta_max}], reATM>{c.re_atm_drift_pct}%\n"
                f"lifecycle: tp1 {c.tp1_R_min}–{c.tp1_R_max}R, tp2(T/R) {c.tp2_R_trend}/{c.tp2_R_range}, trail {c.trail_atr_mult}, time {c.time_stop_min}m\n"
                f"gamma: {c.gamma_enabled} after {c.gamma_after}\n"
                f"warmup: bars {c.min_bars_required}/{c.indicator_min_bars}\n"
            )
            return self._send(text, parse_mode="Markdown")

        # Circuit breaker admin
        if cmd == "/cb" and args:
            runner = getattr(getattr(self, "_runner_tick", None), "__self__", None)
            if args[0].lower() == "reset" and runner:
                for cb in [
                    getattr(getattr(runner, "order_executor", None), "cb_orders", None),
                    getattr(getattr(runner, "order_executor", None), "cb_modify", None),
                    getattr(getattr(runner, "data_source", None), "cb_hist", None),
                    getattr(getattr(runner, "data_source", None), "cb_quote", None),
                ]:
                    if cb:
                        cb.reset()
                return self._send("Breakers reset.")
            if args[0].lower() == "open" and runner and len(args) >= 3:
                name = args[1].lower()
                try:
                    secs = int(args[2])
                except Exception:
                    secs = 30
                cb_map = {
                    "orders": getattr(getattr(runner, "order_executor", None), "cb_orders", None),
                    "modify": getattr(getattr(runner, "order_executor", None), "cb_modify", None),
                    "hist": getattr(getattr(runner, "data_source", None), "cb_hist", None),
                    "quote": getattr(getattr(runner, "data_source", None), "cb_quote", None),
                }
                cb = cb_map.get(name)
                if cb:
                    cb.force_open(secs)
                    return self._send(f"Breaker {name} forced OPEN {secs}s")
                return self._send("Unknown breaker name")

        # DIAG – detailed status + last signal
        if cmd == "/diag":
            try:
                status = self._status_provider() if self._status_provider else {}
                plan = self._last_signal_provider() if self._last_signal_provider else {}
                verbose = os.getenv("DIAG_VERBOSE", "true").lower() != "false"

                def be(val: Any) -> str:
                    return "✅" if val else "❌"

                lines: List[str] = ["📊 Status"]
                lines.append(f"• Market Open: {be(status.get('market_open'))}")
                lines.append(f"• Within Window: {be(status.get('within_window'))}")
                lines.append(f"• Daily DD Hit: {be(status.get('daily_dd_hit'))}")
                lines.append(f"• Cooloff Until: {status.get('cooloff_until', '-')}")
                lines.append(f"• Trades Today: {status.get('trades_today')}")
                lines.append(f"• Consecutive Losses: {status.get('consecutive_losses')}")
                lines.append("")
                lines.append("📈 Signal")
                reason_block = plan.get("reason_block") or "-"
                if verbose:
                    micro = plan.get("micro", {})
                    lines.append(
                        "• Action: {a} | Option: {o} | Strike: {s} | Qty: {q}".format(
                            a=plan.get("action"),
                            o=plan.get("option_type"),
                            s=plan.get("strike"),
                            q=plan.get("qty_lots"),
                        )
                    )
                    lines.append(
                        "• Regime: {r} | Score: {sc} | RR: {rr}".format(
                            r=plan.get("regime"),
                            sc=plan.get("score"),
                            rr=plan.get("rr"),
                        )
                    )
                    lines.append(
                        "• ATR%: {a} | Spread%: {sp} | Depth OK: {d}".format(
                            a=plan.get("atr_pct"),
                            sp=micro.get("spread_pct"),
                            d=be(micro.get("depth_ok")),
                        )
                    )
                    lines.append(
                        "• Entry: {e} | SL: {sl} | TP1: {tp1} | TP2: {tp2}".format(
                            e=plan.get("entry"),
                            sl=plan.get("sl"),
                            tp1=plan.get("tp1"),
                            tp2=plan.get("tp2"),
                        )
                    )
                    lines.append(f"• Block: {reason_block}")
                    reasons = plan.get("reasons") or []
                    if reasons:
                        lines.append("• Reasons:")
                        for r in reasons[:4]:
                            lines.append(f"  - {r}")
                    else:
                        lines.append("• Reasons: -")
                    lines.append(f"• TS: {plan.get('ts')}")
                    lines.append(f"• Eval Count: {plan.get('eval_count')}")
                    lines.append(f"• Last Eval: {plan.get('last_eval_ts')}")
                else:
                    lines.append(
                        "• Action: {a} | Option: {o} | Strike: {s} | Qty: {q}".format(
                            a=plan.get("action"),
                            o=plan.get("option_type"),
                            s=plan.get("strike"),
                            q=plan.get("qty_lots"),
                        )
                    )
                    lines.append(
                        "• Regime: {r} | Score: {sc} | RR: {rr}".format(
                            r=plan.get("regime"),
                            sc=plan.get("score"),
                            rr=plan.get("rr"),
                        )
                    )
                    lines.append(f"• Block: {reason_block}")
                return self._send("\n".join(lines), parse_mode="Markdown")
            except Exception as e:
                return self._send(f"Diag error: {e}")

        if cmd == "/greeks":
            runner = getattr(getattr(self, "_runner_tick", None), "__self__", None)
            if not runner:
                return self._send("Runner unavailable.")
            delta_units = round(runner._portfolio_delta_units(), 1)
            gmode = runner.now_ist.weekday() == 3 and runner.now_ist.time() >= dt_time(14, 45)
            text = (
                "📐 *Portfolio Greeks*\n"
                f"Δ(units): {delta_units} | gamma_mode: {gmode}"
            )
            return self._send(text, parse_mode="Markdown")

        if cmd == "/plan":
            runner = getattr(getattr(self, "_runner_tick", None), "__self__", None)
            if not runner:
                return self._send("Runner unavailable.")
            try:
                snap = runner.telemetry_snapshot()
                plan = snap.get("signal", {})
                text = json.dumps(plan, ensure_ascii=False, indent=2)
                if len(text) > 2000:
                    text = text[:2000] + "\n... (truncated)"
                return self._send("🧩 *Plan*\n```\n" + text + "\n```", parse_mode="Markdown")
            except Exception as e:
                return self._send(f"Plan error: {e}")

        if cmd == "/events":
            runner = getattr(getattr(self, "_runner_tick", None), "__self__", None)
            if not runner or not runner.event_cal:
                return self._send("No events calendar loaded.")
            now = runner.now_ist
            active = runner.event_cal.active(now)
            lines = [f"📅 Events v{runner.event_cal.version} tz={runner.event_cal.tz.key}"]
            for ev in active[:5]:
                lines.append(
                    f"ACTIVE: {ev.name}  guard: {ev.guard_start().time()}→{ev.guard_end().time()}  block={ev.block_trading} widen+{ev.post_widen_spread_pct:.2f}%"
                )
            if not active:
                lines.append("No active guard window.")
            return self._send("\n".join(lines))

        if cmd == "/nextevent":
            runner = getattr(getattr(self, "_runner_tick", None), "__self__", None)
            if not runner or not runner.event_cal:
                return self._send("No events calendar loaded.")
            ev = runner.event_cal.next_event(runner.now_ist)
            if not ev:
                return self._send("No upcoming events within horizon.")
            return self._send(
                f"Next: {ev.name}\nGuard: {ev.guard_start().isoformat()} → {ev.guard_end().isoformat()}  block={ev.block_trading} widen+{ev.post_widen_spread_pct:.2f}%"
            )

        if cmd == "/eventguard":
            runner = getattr(getattr(self, "_runner_tick", None), "__self__", None)
            if not runner:
                return self._send("Runner unavailable.")
            state = args[0].lower() if args else "on"
            runner.event_guard_enabled = state != "off"
            return self._send(
                f"EventGuard: {'ON' if runner.event_guard_enabled else 'OFF'}"
            )

        if cmd == "/audit":
            runner = getattr(getattr(self, "_runner_tick", None), "__self__", None)
            if not runner:
                return self._send("Runner unavailable.")
            try:
                s = runner.telemetry_snapshot()
                sig, bars = s["signal"], s["bars"]
                def pf(ok: bool, val: Any) -> tuple[str, Any]:
                    return ("✅ PASS ", val) if ok else ("❌ FAIL ", val)
                window_ok = getattr(runner, "within_window", True)
                re = getattr(runner, "risk_engine", None)
                cool_ok = (re.state.cooloff_until is None) if re else True
                dd_ok = (re.state.cum_R_today > -re.cfg.max_daily_dd_R) if re else True
                bc_ok = (bars.get("bar_count") or 0) >= runner.strategy_cfg.min_bars_required
                stale_ok = True
                try:
                    last = bars.get("last_bar_ts")
                    if last:
                        last_dt = datetime.fromisoformat(str(last))
                        if last_dt.tzinfo is None:
                            last_dt = last_dt.replace(tzinfo=ZoneInfo(runner.strategy_cfg.tz))
                        age = (runner.now_ist - last_dt).total_seconds()
                        stale_ok = 0 <= age <= 90
                except Exception:
                    stale_ok = False
                regime_ok = sig.get("regime") in ("TREND", "RANGE")
                atr = sig.get("atr_pct") or 0.0
                atr_ok = runner.strategy_cfg.atr_min <= atr <= runner.strategy_cfg.atr_max
                need = (
                    runner.strategy_cfg.score_trend_min
                    if sig.get("regime") == "TREND"
                    else runner.strategy_cfg.score_range_min
                )
                if runner.strategy_cfg.lower_score_temp:
                    need = min(need, 6)
                score_ok = (sig.get("score") or 0) >= need
                micro = sig.get("micro") or {}
                msp = micro.get("spread_pct")
                mdp = micro.get("depth_ok")
                ob_reason = sig.get("reason_block")
                lines = []
                for name, ok, val in [
                    ("window", window_ok, getattr(runner, "window_tuple", "-")),
                    ("cooloff", cool_ok, getattr(re.state, "cooloff_until", None) if re else None),
                    ("daily_dd", dd_ok, getattr(re.state, "cum_R_today", 0.0) if re else 0.0),
                    ("bar_count", bc_ok, bars.get("bar_count")),
                    ("data_stale", stale_ok, bars.get("last_bar_ts")),
                    ("regime", regime_ok, sig.get("regime")),
                    ("atr_pct", atr_ok, atr),
                    ("score", score_ok, f"{sig.get('score')}/{need}"),
                ]:
                    mark, val = pf(ok, val)
                    lines.append(f"{name}: {mark}{val}")
                if msp is None or mdp is None:
                    sp_line = "N/A (no_quote)"
                    dp_line = "N/A (no_quote)"
                else:
                    sp_line = msp
                    dp_line = "✅" if mdp else "❌"
                lines.append(
                    f"micro: spread%={sp_line} depth={dp_line} src={sig.get('quote_src','-')}"
                )
                api = s.get("api_health", {})
                rh = s.get("router", {})
                lines.append(
                    f"orders_cb: {api.get('orders', {}).get('state', '?')} p95={api.get('orders', {}).get('p95_ms')}"
                )
                lines.append(
                    f"quote_cb: {api.get('quote', {}).get('state', '?')} p95={api.get('quote', {}).get('p95_ms')}"
                )
                lines.append(
                    f"router ack_p95: {rh.get('ack_p95_ms')} queues={rh.get('queues')}"
                )
                lines.append(f"reason_block: *{ob_reason}*")
                return self._send("🧪 *Audit*\n" + "\n".join(lines), parse_mode="Markdown")
            except Exception as e:
                return self._send(f"Audit error: {e}")

        if cmd == "/why":
            try:
                status = self._status_provider() if self._status_provider else {}
                plan = self._last_signal_provider() if self._last_signal_provider else {}
                now = time.time()
                def mark(ok: bool) -> str:
                    return "PASS" if ok else "FAIL"
                def val(x: Any) -> str:
                    return str(x)
                last_ts = plan.get("last_bar_ts")
                last_ts_sec = 0.0
                if last_ts:
                    try:
                        last_dt = datetime.fromisoformat(str(last_ts))
                        last_ts_sec = last_dt.timestamp()
                    except Exception:
                        last_ts_sec = 0.0
                gates = []
                within = bool(status.get("within_window"))
                gates.append(("window", within, val(status.get("within_window"))))
                cooloff = bool(status.get("cooloff_until") and status.get("cooloff_until") != "-")
                gates.append(("cooloff", not cooloff, status.get("cooloff_until")))
                dd = bool(status.get("daily_dd_hit"))
                gates.append(("daily_dd", not dd, status.get("day_realized_loss")))
                bar_count = int(plan.get("bar_count") or 0)
                bc = bar_count >= 20
                gates.append(("bar_count", bc, bar_count))
                lag = int(now - last_ts_sec) if last_ts_sec else None
                data_stale = (lag or 0) <= 150
                gates.append(("data_stale", data_stale, f"lag_s={lag}" if lag is not None else "-"))
                regime = plan.get("regime") in ("TREND", "RANGE")
                gates.append(("regime", regime, plan.get("regime")))
                atr = plan.get("atr_pct")
                atr_min = plan.get("atr_min")
                gates.append(("atr_pct", atr is not None and atr_min is not None and atr >= atr_min, atr))
                score = float(plan.get("score") or 0.0)
                reg = str(plan.get("regime"))
                need = 9 if reg == "TREND" else 8
                gates.append(("score", score >= need, score))
                sp = plan.get("spread_pct")
                dp = plan.get("depth_ok")
                reason_block = plan.get("reason_block") or "-"
                reasons = plan.get("reasons") or []
                lines = ["/why gates"]
                for name, ok, value in gates:
                    lines.append(f"{name}: {mark(ok)} {value}")
                sp_line = "N/A (no_quote)" if sp is None else round(sp, 3)
                dp_line = "N/A (no_quote)" if dp is None else ("✅" if dp else "❌")
                lines.append(
                    f"micro: spread%={sp_line} depth={dp_line} src={plan.get('quote_src','-')}"
                )
                if plan.get("option"):
                    o = plan["option"]
                    lines.append(
                        f"option: {o['kind']} {o['strike']} {o['expiry']} token={o['token']} src={plan.get('quote_src','-')}"
                    )
                lines.append(
                    f"features: {'PASS' if plan.get('feature_ok') else 'FAIL'} "
                    f"reasons={','.join(plan.get('reasons', [])) or '-'}"
                )
                if atr is not None and atr_min is not None:
                    lines.append(
                        f"ATR%: {round(atr,4)} (min {atr_min}) → {'PASS' if 'atr_low' not in plan.get('reasons', []) else 'FAIL'}"
                    )
                else:
                    lines.append("ATR%: N/A")
                if plan.get("probe_window_from") and plan.get("probe_window_to"):
                    lines.append(
                        f"\u2022 Probe window: {plan['probe_window_from']} \u2192 {plan['probe_window_to']} (IST)"
                    )
                lines.append(f"reason_block: {reason_block}")
                if reasons:
                    lines.append("reasons: " + ", ".join(str(r) for r in reasons))
                return self._send("\n".join(lines), parse_mode="Markdown")
            except Exception as e:
                return self._send(f"Why error: {e}")

        if cmd == "/probe":
            if not self._probe_provider:
                return self._send("Probe provider unavailable.")
            try:
                info = self._probe_provider()
                return self._send(f"Probe: {info}")
            except Exception as e:
                return self._send(f"Probe error: {e}")

        if cmd == "/atm":
            if not self._atm_provider:
                return self._send("ATM provider unavailable.")
            try:
                info = self._atm_provider()
                return self._send(f"ATM: {info}")
            except Exception as e:
                return self._send(f"ATM error: {e}")

        if cmd == "/tick":
            if not self._l1_provider:
                return self._send("L1 provider unavailable.")
            try:
                t = self._l1_provider()
                return self._send(f"L1: {t if t else 'n/a'}")
            except Exception as e:
                return self._send(f"L1 error: {e}")

        if cmd == "/bars":
            runner = getattr(getattr(self, "_runner_tick", None), "__self__", None)
            if not runner:
                return self._send("Runner unavailable.")
            try:
                n = int(args[0]) if args else 5
            except Exception:
                n = 5
            try:
                df = runner.data_source.get_last_bars(n)
                if df is None:
                    return self._send("No bars available.")
                rows = []
                for ts, r in df.iterrows():
                    rows.append(
                        f"{str(ts)[11:16]}  O:{r.open:.1f} H:{r.high:.1f} L:{r.low:.1f} C:{r.close:.1f}  "
                        f"VWAP:{getattr(r, 'vwap', float('nan')):.1f}  ATR%:{getattr(r, 'atr_pct', float('nan')):.2f}"
                    )
                src = getattr(runner.last_plan or {}, "data_source", None) or (runner.last_plan or {}).get("data_source", "broker")
                text = "📊 *Bars*  src=" + str(src) + "\n" + "\n".join(rows[-n:])
                return self._send(text, parse_mode="Markdown")
            except Exception as e:
                return self._send(f"Bars error: {e}")

        if cmd == "/quotes":
            try:
                plan = self._last_signal_provider() if self._last_signal_provider else {}
                sym = plan.get("strike") or plan.get("symbol") or "-"
                text = "📈 *Quotes*\n" + _fmt_micro(
                    sym,
                    plan.get("micro"),
                    plan.get("last_bar_ts"),
                    plan.get("last_bar_lag_s"),
                )
                return self._send(text, parse_mode="Markdown")
            except Exception as e:
                return self._send(f"Quotes error: {e}")

        if cmd == "/trace":
            runner = getattr(getattr(self, "_runner_tick", None), "__self__", None)
            if not runner:
                return self._send("Runner unavailable.")
            try:
                n = int(args[0]) if args else 5
            except Exception:
                n = 5
            runner.trace_ticks_remaining = max(1, min(50, n))
            return self._send(f"Tracing next {runner.trace_ticks_remaining} evals.")

        if cmd == "/traceoff":
            runner = getattr(getattr(self, "_runner_tick", None), "__self__", None)
            if not runner:
                return self._send("Runner unavailable.")
            runner.trace_ticks_remaining = 0
            return self._send("Trace off.")

        if cmd == "/summary":
            runner = getattr(getattr(self, "_runner_tick", None), "__self__", None)
            if not runner:
                return self._send("Runner unavailable.")
            arg = (args[0].lower() if args else "week")
            tz = ZoneInfo(getattr(runner.settings, "TZ", "Asia/Kolkata"))
            now = datetime.now(tz)
            if arg == "month":
                start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            else:
                start = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
            trades = read_trades_between(start, now)
            k = _kpis(trades)
            rh = getattr(runner.order_executor, "router_health", lambda: {})()
            cb = getattr(runner.order_executor, "api_health", lambda: {})()
            text = (
                f"📈 *Summary* ({arg}) {start.date()} → {now.date()}\n"
                f"Trades: {k.get('trades',0)} | PF: {k.get('PF')} | Win%: {k.get('Win%')} | "
                f"AvgR: {k.get('AvgR')} | MedR: {k.get('MedianR')} | MaxDD_R: {k.get('MaxDD_R')} | "
                f"p95_loss_R: {k.get('p95_loss_R')}\n"
                f"Router ack_p95: {rh.get('ack_p95_ms')} | Orders CB: {cb.get('orders',{}).get('state')}"
            )
            return self._send(text, parse_mode="Markdown")

        if cmd == "/lasttrades":
            runner = getattr(getattr(self, "_runner_tick", None), "__self__", None)
            jrnl = getattr(runner, "journal", None) if runner else None
            if not jrnl:
                return self._send("Journal unavailable.")
            rows = jrnl.last_trades(10)
            if not rows:
                return self._send("No closed trades yet.")
            lines = []
            for t in rows:
                ts = str(t.get("ts_exit", ""))[:16]
                lines.append(
                    f"{ts}  {t.get('side')} {t.get('symbol')}  R={t.get('R')}  pnl_R={t.get('pnl_R')}"
                )
            return self._send("\U0001F5DE *Last Trades*\n" + "\n".join(lines), parse_mode="Markdown")

        if cmd == "/hb":
            runner = getattr(getattr(self, "_runner_tick", None), "__self__", None)
            if not runner:
                return self._send("Runner unavailable.")
            arg = (args[0].lower() if args else "on")
            runner.hb_enabled = (arg != "off")
            return self._send(f"Heartbeat: {'ON' if runner.hb_enabled else 'OFF'}")

        if cmd == "/selftest":
            try:
                opt = args[0].lower() if args else "ce"
                if not self._selftest_provider:
                    return self._send("selftest not wired")
                text = self._selftest_provider(opt)
                return self._send(text)
            except Exception as e:
                return self._send(f"Selftest error: {e}")

        if cmd == "/smoketest":
            try:
                opt = args[0].lower() if args else "ce"
                if not self._selftest_provider:
                    return self._send("smoketest not wired")
                text = self._selftest_provider(opt)
                return self._send(f"smoketest: {text}")
            except Exception as e:
                return self._send(f"Smoketest error: {e}")

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

        if cmd == "/risk":
            if not self._risk_provider:
                return self._send("Risk provider not wired.")
            snap = self._risk_provider() or {}
            lines = [
                "Risk",
                (
                    f"date: {snap.get('session_date')}  cum_R_today: {snap.get('cum_R_today')}  "
                    f"trades_today: {snap.get('trades_today')}  consec_losses: {snap.get('consecutive_losses')}"
                ),
                (
                    f"cooloff_until: {snap.get('cooloff_until') or '-'}  roll10_avgR: {snap.get('roll10_avgR')}"
                ),
                f"daily_caps: {snap.get('daily_caps_hit_recent', [])}",
                f"skip_next_open_date: {snap.get('skip_next_open_date')}",
            ]
            expo = snap.get("exposure") or {}
            if expo:
                notional = expo.get("notional_rupees", 0.0)
                lots = expo.get("lots_by_symbol", {})
                exp_line = f"exposure: notional ₹{notional:,.0f}"
                for sym, lot_count in lots.items():
                    exp_line += f"  {sym}: {lot_count} lots"
                lines.append(exp_line)
            return self._send("\n".join(lines))

        if cmd == "/limits":
            if not self._limits_provider:
                return self._send("Limits provider not wired.")
            cfg = self._limits_provider() or {}
            lines = ["Limits"]
            for k, v in cfg.items():
                lines.append(f"{k}: {v}")
            return self._send("\n".join(lines))

        if cmd == "/riskresettoday":
            if not self._risk_reset:
                return self._send("Risk reset not wired.")
            try:
                self._risk_reset()
                return self._send("Risk counters reset.")
            except Exception as e:
                return self._send(f"Risk reset error: {e}")

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

        if cmd == "/backtest":
            if not self._backtest_provider:
                return self._send("Backtest not wired.")
            try:
                path = args[0] if args else None
                res = self._backtest_provider(path)
                return self._send(res)
            except Exception as e:
                return self._send(f"Backtest error: {e}")

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

        if cmd == "/watch":
            if not args:
                return self._send("Usage: /watch on|off")
            val = args[0].lower() in ("on", "true", "1", "yes")
            settings.TELEGRAM__PRETRADE_ALERTS = val
            return self._send(f"Pre-trade alerts: {'ON' if val else 'OFF'}")

        if cmd == "/orders":
            if not self._open_trades_provider:
                return self._send("Orders provider not wired.")
            try:
                legs = self._open_trades_provider() or []
            except Exception as e:
                return self._send(f"Orders error: {e}")
            if not legs:
                return self._send("No open trades.")
            lines: List[str] = []
            current: Optional[str] = None
            for leg in legs:
                trade = str(leg.get("trade"))
                if trade != current:
                    lines.append(f"Trade {trade} [{leg.get('status', '')}]")
                    current = trade
                lines.append(
                    f"  {leg.get('leg')} {leg.get('sym')} {leg.get('state')} "
                    f"filled={leg.get('filled')}/{leg.get('qty')} avg={leg.get('avg', 0)} "
                    f"age={leg.get('age_s')}s"
                )
            return self._send("\n".join(lines))

        if cmd == "/router":
            if self._diag_provider:
                try:
                    rh = self._diag_provider().get("router", {})
                except Exception as e:
                    return self._send(f"Router error: {e}")
                text = (
                    "🚦 Router\n"
                    f"ack_p95_ms: {rh.get('ack_p95_ms')}\n"
                    f"queues: {rh.get('queues')}\n"
                    f"inflight: {rh.get('inflight')}\n"
                )
                return self._send(text)
            return self._send("Router diag not wired.")

        if cmd == "/cancel":
            if not args:
                return self._send("Usage: /cancel <trade_id|all>")
            target = args[0]
            if target == "all":
                if self._cancel_all:
                    self._cancel_all()
                    return self._send("🧹 Cancelled all open orders.")
                return self._send("Cancel-all not wired.")
            if self._cancel_trade:
                try:
                    self._cancel_trade(target)
                    return self._send(f"Cancel requested for {target}.")
                except Exception as e:
                    return self._send(f"Cancel error: {e}")
            return self._send("Cancel not wired.")

        if cmd == "/reconcile":
            if self._reconcile_once:
                try:
                    n = int(self._reconcile_once())
                    return self._send(f"Reconciled {n} legs.")
                except Exception as e:
                    return self._send(f"Reconcile error: {e}")
            return self._send("Reconciler not wired.")

        # PAUSE / RESUME
        if cmd == "/pause":
            if self._runner_pause:
                self._runner_pause()
                return self._send("⏸️ Entries paused.")
            return self._send("Pause not wired.")
        if cmd == "/resume":
            resumed = 0
            if self._runner_resume:
                self._runner_resume()
            runner = getattr(getattr(self, "_runner_tick", None), "__self__", None)
            if runner and getattr(runner, "journal", None):
                try:
                    rehydrated = runner.journal.rehydrate_open_legs()
                    for leg in rehydrated:
                        fsm = runner.order_executor.get_or_create_fsm(leg["trade_id"])
                        runner.order_executor.attach_leg_from_journal(fsm, leg)
                    runner.reconciler.step(runner.now_ist)
                    resumed = len(rehydrated)
                except Exception:
                    resumed = 0
            if resumed:
                return self._send(f"▶️ Entries resumed. Rehydrated {resumed} legs.")
            return self._send("▶️ Entries resumed.")

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
