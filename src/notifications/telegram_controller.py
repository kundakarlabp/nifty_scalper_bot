from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
import re
import statistics as stats
import threading
import time
from collections import deque
from collections.abc import Mapping
from datetime import date, datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from zoneinfo import ZoneInfo

import requests

from src.config import settings
from src.diagnostics import checks
from src.diagnostics.registry import run, run_all
from src.execution.micro_filters import micro_from_quote
from src.execution.order_executor import fetch_quote_with_depth
from src.logs.journal import read_trades_between
from src.risk.position_sizing import PositionSizer
from src.strategies.runner import StrategyRunner
from src.strategies.warmup import check as warmup_check
from src.utils import strike_selector
from src.utils.emit import emit_debug
from src.utils.expiry import last_tuesday_of_month, next_tuesday_expiry
from src.utils.freshness import compute as compute_freshness
from src.diagnostics.metrics import daily_summary

log = logging.getLogger(__name__)


def _format_number(value: Any, digits: int = 2) -> str:
    """Return ``value`` formatted as a decimal string."""

    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _format_float(value: Any, digits: int = 2) -> str:
    """Return ``value`` formatted with ``digits`` decimal places without grouping."""

    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _env_float(name: str, default: float) -> float:
    """Return ``float`` from environment variable ``name`` with ``default`` fallback."""

    raw = os.getenv(name)
    try:
        return float(raw) if raw is not None else default
    except (TypeError, ValueError):
        return default


class _CmdGate:
    """Simple per-chat command rate limiter."""

    def __init__(self) -> None:
        self._last: Dict[tuple[int, str], float] = {}

    def allow(self, chat_id: int, key: str, interval: float) -> bool:
        now = time.time()
        marker = (chat_id, key)
        last = self._last.get(marker, 0.0)
        if now - last < interval:
            return False
        self._last[marker] = now
        return True


_COMMAND_GATE = _CmdGate()


COMMAND_HELP_OVERRIDES: dict[str, str] = {
    "/score": "Latest strategy score breakdown (debug)",
    "/selftest": "Run component health checks",
    "/shadow": "List shadow-mode blockers (no orders)",
    "/sizer": "Show position sizer parameters",
    "/slmult": "Set stop-loss ATR multiple",
    "/smoketest": "Smoke-test status & controls",
    "/start": "Resume runner if paused/stopped",
    "/state": "Equity/trade counters & cooldowns",
    "/status": "High-level bot status summary",
    "/summary": "One-line performance summary",
    "/tick": "Latest L1 quote for ATM option",
    "/tpmult": "Get/Set take-profit ATR multiple",
    "/diag": "Recent diagnostic trace events",
    "/diagstatus": "Compact diagnostic status summary",
    "/diagtrace": "Structured diagnostic trace events",
    "/trace": "Show trace details or enable tracing",
    "/traceoff": "Disable trace logging",
    "/trend": "Adjust trend-mode TP/SL boosts",
    "/warmup": "Warm-up status (historical bootstrap)",
    "/watch": "Toggle pre-trade watch alerts",
    "/why": "Explain last decision and gates",
    "/deep": "Detailed status snapshot (debug)",
    "/whydetail": "Gate chain with reasons (debug)",
    "/errors": "Recent error ring entries",
    "/statusjson": "Raw status snapshot JSON",
    "/logs": "Latest structured debug log snapshot (default 20 lines)",
}


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
    try:
        spread_pct = f"{float(spread) * 100.0:.2f}"
    except (TypeError, ValueError):
        spread_pct = "N/A"
    return (
        f"quote: {sym} src={m.get('source', '-')} "
        f"ltp={m.get('ltp')} bid={m.get('bid')} ask={m.get('ask')} "
        f"spread%={spread_pct} "
        f"bid5={m.get('bid5')} ask5={m.get('ask5')} "
        f"depth={depth_ok if depth_ok is not None else 'N/A'} "
        f"last_bar_ts={last_bar_ts} lag_s={lag_s}"
    )


def _diag_trace_limit() -> int:
    """Return the maximum number of diagnostic trace events to display."""

    raw = getattr(settings, "DIAG_TRACE_EVENTS", None)
    if raw is None:
        raw = getattr(settings, "diag_trace_events", None)
    if isinstance(raw, bool):
        return 20 if raw else 0
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 0
    return max(value, 0)


def _sanitize_json(value: Any) -> Any:
    """Return a JSON-serializable representation of ``value``."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _sanitize_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_json(v) for v in value]
    return str(value)


def _normalize_trace_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return a sanitized, compact representation of a trace record."""

    item: dict[str, Any] = {}
    for key in ("ts", "level", "comp", "event", "trace_id", "symbol", "side"):
        val = record.get(key)
        if val not in (None, ""):
            item[key] = _sanitize_json(val)
    msg = record.get("msg") or record.get("message")
    if msg not in (None, ""):
        item["msg"] = _sanitize_json(msg)
    extras = {
        k: _sanitize_json(v)
        for k, v in record.items()
        if k not in item and k not in {"msg", "message"} and v not in (None, "")
    }
    if extras:
        item["data"] = extras
    return item


def _format_trace_summary_line(record: dict[str, Any]) -> str:
    """Format ``record`` for human-readable Telegram output."""

    normalized = dict(_normalize_trace_record(record))
    parts: list[str] = []
    ts = normalized.pop("ts", None)
    level = normalized.pop("level", None)
    comp = normalized.pop("comp", None)
    event = normalized.pop("event", None)
    trace_id = normalized.pop("trace_id", None)
    msg = normalized.pop("msg", None)
    data = normalized.pop("data", None)
    if ts is not None:
        parts.append(str(ts))
    if level is not None:
        parts.append(str(level))
    if comp and event:
        parts.append(f"{comp}.{event}")
    else:
        if comp:
            parts.append(str(comp))
        if event:
            parts.append(str(event))
    if trace_id is not None:
        parts.append(f"trace={trace_id}")
    if msg is not None:
        parts.append(str(msg))
    for key in sorted(normalized):
        parts.append(f"{key}={normalized[key]}")
    if data:
        parts.append(
            "data=" + json.dumps(data, ensure_ascii=False, sort_keys=True)
        )
    return " ".join(str(p) for p in parts if str(p))


class TelegramController:
    """
    Production-safe Telegram controller:

    - Pulls token/chat_id from settings.telegram
    - Provides /status /health /diag /check /logs /tick /force_eval /mode /pause /resume
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
        diag_provider: Optional[
            Callable[[], Dict[str, Any]]
        ] = None,  # detailed (/check)
        logs_provider: Optional[Callable[[int], List[str]]] = None,
        last_signal_provider: Optional[Callable[[], Optional[Dict[str, Any]]]] = None,
        compact_diag_provider: Optional[
            Callable[[], Dict[str, Any]]
        ] = None,  # NEW: compact (/diag)
        risk_provider: Optional[Callable[[], Dict[str, Any]]] = None,
        limits_provider: Optional[Callable[[], Dict[str, Any]]] = None,
        risk_reset_today: Optional[Callable[[], None]] = None,
        bars_provider: Optional[Callable[[int], str]] = None,
        quotes_provider: Optional[Callable[..., str]] = None,
        probe_provider: Optional[Callable[[], Dict[str, Any]]] = None,
        trace_provider: Optional[Callable[[int], None]] = None,
        selftest_provider: Optional[Callable[[str], str]] = None,
        backtest_provider: Optional[Callable[[Optional[str]], str]] = None,
        filecheck_provider: Optional[Callable[[str], str]] = None,
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
            raise RuntimeError(
                "TelegramController: TELEGRAM__BOT_TOKEN or TELEGRAM__CHAT_ID missing"
            )

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
        self._filecheck_provider = filecheck_provider
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
        self._last_summary_date: Optional[date] = None
        self._last_eod_summary_date: Optional[date] = None

        # allowlist
        extra = getattr(tg, "extra_admin_ids", []) or []
        self._allowlist = {int(self._chat_id), *[int(x) for x in extra]}

        # rate-limit / backoff
        self._send_min_interval = 0.9
        self._last_sends: List[tuple[float, str]] = []
        self._backoff = 1.0
        self._backoff_max = 20.0

        # heartbeat (auto-edit) state
        self._hb_lock = threading.Lock()
        self._hb_by_chat: Dict[int, Dict[str, Any]] = {}
        self._hb_thread: Optional[threading.Thread] = None
        self._hb_run = False

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
        return "".join("\\" + ch if ch in escape_chars else ch for ch in text)

    def _send(
        self,
        text: str,
        parse_mode: Optional[str] = None,
        disable_notification: bool = False,
    ) -> None:
        if not text or not text.strip():
            return
        if not self._rate_ok(text):
            return
        delay = self._backoff
        if parse_mode:
            text = self._escape_markdown(text)
        while True:
            try:
                payload = {
                    "chat_id": self._chat_id,
                    "text": text,
                    "disable_notification": disable_notification,
                }
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
        payload = {
            "chat_id": self._chat_id,
            "text": text,
            "reply_markup": {"inline_keyboard": buttons},
        }
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

    def send_eod_summary(self) -> None:
        """Send end-of-day flatten notice with daily stats."""
        tz = ZoneInfo(getattr(settings, "TZ", "Asia/Kolkata"))
        now = datetime.now(tz)
        if self._last_eod_summary_date == now.date():
            return
        stats = daily_summary(now)
        text = (
            f"\U0001F514 EOD flat R={stats['R']} "
            f"Hit={stats['hit_rate']}% AvgR={stats['avg_R']} "
            f"Slip={stats['slippage_bps']}bps"
        )
        self._send(text)
        self._last_eod_summary_date = now.date()

    def _maybe_send_close_summary(self) -> None:
        """Emit a daily P&L summary at market close."""
        if self._stop.is_set():
            return
        tz = ZoneInfo(getattr(settings, "TZ", "Asia/Kolkata"))
        now = datetime.now(tz)
        cutoff = dt_time(15, 29)
        if now.time() >= cutoff and self._last_summary_date != now.date():
            stats = daily_summary(now)
            text = (
                f"\U0001F4CA Close R={stats['R']} "
                f"Hit={stats['hit_rate']}% AvgR={stats['avg_R']} "
                f"Slip={stats['slippage_bps']}bps"
            )
            self._send(text)
            self._last_summary_date = now.date()

    # --------------- polling loop ---------------
    def start_polling(self) -> None:
        if self._started:
            log.info("Telegram polling already running; skipping start.")
            return
        self._stop.clear()
        self._poll_thread = threading.Thread(
            target=self._poll_loop, name="tg-poll", daemon=True
        )
        self._poll_thread.start()
        self._started = True
        log.info("Telegram polling started (chat_id=%s).", self._chat_id)

    def stop_polling(self) -> None:
        if not self._started:
            return
        self._stop.set()
        try:
            self._session.close()
        except Exception:
            pass
        if self._poll_thread:
            self._poll_thread.join(timeout=5)
        self._started = False

    def _poll_loop(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                params = {"timeout": 25}
                if self._last_update_id is not None:
                    params["offset"] = self._last_update_id + 1
                r = self._session.get(
                    f"{self._base}/getUpdates",
                    params=params,
                    timeout=self._timeout + 10,
                )
                data = r.json()
                if not data.get("ok"):
                    time.sleep(1.0)
                    continue
                for upd in data.get("result", []):
                    self._last_update_id = int(upd.get("update_id", 0))
                    self._handle_update(upd)
                backoff = 1.0
                self._maybe_send_close_summary()
            except (OSError, requests.exceptions.ConnectionError) as e:
                log.warning("Telegram poll network error: %s", e)
                try:
                    self._session.close()
                except Exception:
                    pass
                self._session = requests.Session()
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
                continue
            except Exception as e:
                log.debug("Telegram poll error: %s", e)
                time.sleep(1.0)

    # --------------- helpers ---------------
    def _authorized(self, chat_id: int) -> bool:
        return int(chat_id) in self._allowlist

    def _list_commands(self) -> List[str]:
        """Return all supported Telegram command strings."""
        source = inspect.getsource(self._handle_update)
        cmds = set(re.findall(r"cmd == \"(\/[\w_]+)\"", source))
        for group in re.findall(r"cmd in \(([^)]*)\)", source):
            for part in group.split(","):
                part = part.strip().strip("'\"")
                if part.startswith("/"):
                    cmds.add(part)
        return sorted(cmds)

    def _help_text(self) -> str:
        cmds = self._list_commands()
        lines = ["🤖 Nifty Scalper Bot — commands", ""]
        for cmd in sorted(COMMAND_HELP_OVERRIDES):
            if cmd in cmds:
                lines.append(f"{cmd:<10} — {COMMAND_HELP_OVERRIDES[cmd]}")
        other_cmds = [c for c in cmds if c not in COMMAND_HELP_OVERRIDES]
        if other_cmds:
            lines.append("")
            lines.append("Other commands:")
            for i in range(0, len(other_cmds), 6):
                lines.append(" · ".join(other_cmds[i : i + 6]))
        return "\n".join(lines)

    def _resume_entries(self) -> str:
        rehydrated = 0
        if self._runner_resume:
            try:
                self._runner_resume()
            except Exception:  # pragma: no cover - defensive logging
                log.exception("Runner resume callback failed")
                return "Resume error. Check logs."
        runner = getattr(getattr(self, "_runner_tick", None), "__self__", None)
        if runner and getattr(runner, "journal", None):
            try:
                rehydrated_legs = runner.journal.rehydrate_open_legs()
                for leg in rehydrated_legs:
                    fsm = runner.order_executor.get_or_create_fsm(leg["trade_id"])
                    runner.order_executor.attach_leg_from_journal(fsm, leg)
                runner.reconciler.step(runner.now_ist)
                rehydrated = len(rehydrated_legs)
            except Exception:  # pragma: no cover - defensive logging
                log.debug("Failed to rehydrate legs after resume", exc_info=True)
                rehydrated = 0
        if rehydrated:
            return f"▶️ Entries resumed. Rehydrated {rehydrated} legs."
        return "▶️ Entries resumed."

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
                log.debug(
                    "Failed to restore allow_offhours_testing flag", exc_info=True
                )

        return (
            "✅ Tick executed."
            if res
            else (
                "Dry tick executed (no action)."
                if dry
                else "Tick executed (no action)."
            )
        )

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

        lines.append(
            f"📈 Last signal: {'present' if diag.get('last_signal') else 'none'}"
        )
        lines.append(
            f"\nOverall: {'✅ ALL GOOD' if diag.get('ok') else '❗ Issues present'}"
        )
        return "\n".join(lines)

    def _resolve_runner(self) -> Optional[StrategyRunner]:
        """Return the wired runner instance if available."""

        candidate = getattr(getattr(self, "_runner_tick", None), "__self__", None)
        if isinstance(candidate, StrategyRunner):
            return candidate
        try:
            return StrategyRunner.get_singleton()
        except Exception:
            return None

    def _quick_snapshot(self) -> Dict[str, Any]:
        """Collect a lightweight snapshot for quick-look commands."""

        try:
            tz = ZoneInfo(getattr(settings, "TZ", "Asia/Kolkata"))
        except Exception:
            tz = timezone.utc

        snap: Dict[str, Any] = {
            "timestamp": datetime.now(tz),
            "runner_ready": False,
            "plan": {},
            "status": {},
        }

        runner = self._resolve_runner()
        if not runner:
            return snap

        snap["runner_ready"] = True

        try:
            telemetry = runner.telemetry_snapshot()
        except Exception:
            telemetry = {}
        if isinstance(telemetry, dict):
            snap["telemetry"] = telemetry
        else:
            snap["telemetry"] = {}

        signal = snap["telemetry"].get("signal", {})
        plan: Dict[str, Any] = {}
        if isinstance(signal, Mapping):
            plan.update(signal)

        raw_plan = getattr(runner, "last_plan", None)
        if isinstance(raw_plan, Mapping):
            for key in (
                "action",
                "option_type",
                "qty_lots",
                "strike",
                "entry",
                "sl",
                "tp1",
                "tp2",
                "rr",
                "score",
                "regime",
                "reason_block",
                "reasons",
                "opt_entry",
                "opt_tp1",
                "opt_tp2",
                "opt_sl",
                "opt_lot_cost",
            ):
                plan.setdefault(key, raw_plan.get(key))
        snap["plan"] = plan

        micro = plan.get("micro") if isinstance(plan, Mapping) else None
        snap["micro"] = micro if isinstance(micro, Mapping) else {}

        try:
            status = runner.get_status_snapshot()
        except Exception:
            status = {}
        if isinstance(status, dict):
            snap["status"] = status

        try:
            equity_snapshot = runner.get_equity_snapshot()
        except Exception:
            equity_snapshot = {}
        equity_value: Optional[float] = None
        if isinstance(equity_snapshot, dict):
            candidate = equity_snapshot.get("equity_cached")
            if isinstance(candidate, (int, float)):
                equity_value = float(candidate)
        if equity_value is None:
            cached = getattr(runner, "_equity_cached_value", None)
            if isinstance(cached, (int, float)):
                equity_value = float(cached)
        snap["equity"] = equity_value

        status_dict = snap["status"]
        snap["market_open"] = bool(status_dict.get("market_open"))
        snap["open_positions"] = status_dict.get("open_positions")

        return snap

    def _render_plan_summary(self, plan: Mapping[str, Any]) -> str:
        """Render a compact multi-line summary of the current plan."""

        if not plan:
            return "No active plan."

        action = plan.get("action") or plan.get("side") or "idle"
        option_type = plan.get("option_type") or ""
        qty = plan.get("qty_lots")
        header = f"Action: {action} {option_type}".strip()
        if qty not in (None, ""):
            header += f" | lots={qty}"

        entry_line = []
        for label in ("entry", "sl", "tp1", "tp2"):
            value = plan.get(label)
            if value not in (None, ""):
                entry_line.append(f"{label.upper()}={_format_number(value)}")

        meta_line = []
        for label, digits in (("rr", 2), ("score", 2), ("atr_pct", 2), ("regime", None)):
            value = plan.get(label)
            if value in (None, ""):
                continue
            if label == "regime":
                meta_line.append(f"regime={value}")
            else:
                meta_line.append(f"{label}={_format_number(value, digits)}")

        lines = [header]
        if entry_line:
            lines.append(" | ".join(entry_line))
        if meta_line:
            lines.append(" | ".join(meta_line))
        return "\n".join(lines)

    def _render_reason_details(self, plan: Mapping[str, Any]) -> str:
        """Describe the current gating outcome from the plan snapshot."""

        reason = plan.get("reason_block")
        reasons = plan.get("reasons")
        if not reason and not reasons:
            return "All gates passed."

        lines: List[str] = []
        if reason:
            lines.append(f"Blocked by: {reason}")

        if isinstance(reasons, Mapping):
            for key, value in reasons.items():
                if isinstance(value, Mapping):
                    detail = ", ".join(f"{k}={v}" for k, v in value.items())
                    lines.append(f"- {key}: {detail}")
                else:
                    lines.append(f"- {key}: {value}")
        elif isinstance(reasons, (list, tuple)):
            for item in reasons:
                lines.append(f"- {item}")

        return "\n".join(lines) if lines else "Blocked without detail."

    def _recent_errors(self, limit: int = 20) -> List[str]:
        """Return the most recent error log lines."""

        paths = [
            Path("logs/error.log"),
            Path("logs/errors.log"),
            Path("logs/bot-error.log"),
        ]
        for path in paths:
            if not path.exists():
                continue
            try:
                with path.open("r", encoding="utf-8", errors="ignore") as handle:
                    lines = list(deque(handle, maxlen=limit))
            except Exception:
                log.debug("Failed to read error log %s", path, exc_info=True)
                continue
            return [line.rstrip("\n") for line in lines]
        return []

    def _handle_quick_commands(
        self, cmd: str, args: List[str], chat_id: int
    ) -> bool:
        """Handle compact snapshot commands added for mobile quick-checks."""

        quick_commands = {
            "/hb": _env_float("TG_RATE_HB_SEC", 3.0),
            "/plan": _env_float("TG_RATE_PLAN_SEC", 3.0),
            "/whydetail": _env_float("TG_RATE_WHYDETAIL_SEC", 5.0),
            "/deep": _env_float("TG_RATE_DEEP_SEC", 5.0),
            "/errors": _env_float("TG_RATE_ERRORS_SEC", 10.0),
            "/statusjson": _env_float("TG_RATE_STATUSJSON_SEC", 5.0),
        }
        if cmd not in quick_commands:
            return False

        if cmd == "/hb" and args and args[0].lower() in {"on", "off"}:
            return self._handle_hb_toggle(chat_id, args)

        interval = quick_commands[cmd]
        if not _COMMAND_GATE.allow(chat_id, cmd, interval):
            self._send("⏳ Please wait before repeating this command.")
            return True

        snap = self._quick_snapshot()

        if cmd == "/statusjson":
            try:
                payload = json.dumps(snap, default=str, ensure_ascii=False)
            except TypeError:
                payload = json.dumps(_sanitize_json(snap), ensure_ascii=False)
            self._send(payload[:3500])
            return True
        if not snap.get("runner_ready"):
            self._send("Runner not ready.")
            return True

        status = snap.get("status", {})
        plan = snap.get("plan", {})
        runner = self._resolve_runner()

        if cmd == "/hb":
            self._send(self._build_heartbeat_text(snap))
            return True

        if cmd == "/plan":
            side = plan.get("action") or plan.get("side")
            status_flag = str(plan.get("status") or "").lower()
            has_signal = bool(side) and str(side).lower() not in {"idle", "none", "hold"}
            has_details = any(
                plan.get(field) not in (None, "")
                for field in (
                    "entry",
                    "tp1",
                    "tp2",
                    "sl",
                    "opt_entry",
                    "opt_tp1",
                    "opt_tp2",
                    "opt_sl",
                    "opt_lot_cost",
                    "strike",
                )
            )
            if not plan or (status_flag == "no_signal" and not has_signal and not has_details):
                self._send("📉 No active signal right now.")
                return True

            esc = self._escape_markdown
            option_type = plan.get("option_type") or plan.get("ot") or "—"
            strike = plan.get("strike") or plan.get("symbol") or "—"
            entry = plan.get("entry")
            sl = plan.get("sl")
            tp1 = plan.get("tp1")
            tp2 = plan.get("tp2")
            rr = plan.get("rr")
            qty = plan.get("qty_lots") or plan.get("qty")
            lot = plan.get("lot") or plan.get("lot_size")
            quote_age = (
                plan.get("age")
                or plan.get("quote_age_s")
                or plan.get("bar_age_s")
                or plan.get("last_bar_lag_s")
            )
            spread = plan.get("spr") or plan.get("spread_pct")
            micro = snap.get("micro")
            if spread is None and isinstance(micro, Mapping):
                spread = micro.get("spread_pct")
            if quote_age is None and isinstance(micro, Mapping):
                quote_age = micro.get("age")
            micro_detail = plan.get("micro") if isinstance(plan.get("micro"), Mapping) else {}
            if not micro_detail and isinstance(micro, Mapping):
                micro_detail = micro

            side_display = str(side).upper() if has_signal else "—"

            lines = ["*📋 Trading Plan*"]
            lines.append(
                f"• {esc(side_display)} {esc(str(option_type))} {esc(str(strike))}"
            )
            lines.append(
                "• Entry: {entry} | SL: {sl}\n"
                "• TP1: {tp1} | TP2: {tp2} | RR: {rr}"
                .format(
                    entry=esc(_format_number(entry) if entry not in (None, "") else "—"),
                    sl=esc(_format_number(sl) if sl not in (None, "") else "—"),
                    tp1=esc(_format_number(tp1) if tp1 not in (None, "") else "—"),
                    tp2=esc(_format_number(tp2) if tp2 not in (None, "") else "—"),
                    rr=esc(_format_float(rr) if rr not in (None, "") else "—"),
                )
            )
            lines.append(
                "• Size: {qty} lots × {lot}"
                .format(
                    qty=esc(str(qty) if qty not in (None, "") else "—"),
                    lot=esc(str(lot) if lot not in (None, "") else "—"),
                )
            )
            lines.append(
                "• Quote: age={age}s spread%={spr}%".format(
                    age=esc(
                        _format_float(quote_age, 1)
                        if isinstance(quote_age, (int, float))
                        else "—"
                    ),
                    spr=esc(
                        _format_float((float(spread) * 100.0), 2)
                        if isinstance(spread, (int, float))
                        else "—"
                    ),
                )
            )

            option_fields_present = any(
                plan.get(field) not in (None, "")
                for field in ("opt_entry", "opt_tp1", "opt_tp2", "opt_sl", "opt_lot_cost")
            )
            if option_fields_present:
                lines.append(
                    "Option → entry ₹{e} SL ₹{sl} TP1 ₹{tp1} TP2 ₹{tp2} lot ₹{lc}".format(
                        e=esc(
                            _format_number(plan.get("opt_entry"))
                            if plan.get("opt_entry") not in (None, "")
                            else "—"
                        ),
                        sl=esc(
                            _format_number(plan.get("opt_sl"))
                            if plan.get("opt_sl") not in (None, "")
                            else "—"
                        ),
                        tp1=esc(
                            _format_number(plan.get("opt_tp1"))
                            if plan.get("opt_tp1") not in (None, "")
                            else "—"
                        ),
                        tp2=esc(
                            _format_number(plan.get("opt_tp2"))
                            if plan.get("opt_tp2") not in (None, "")
                            else "—"
                        ),
                        lc=esc(
                            _format_number(plan.get("opt_lot_cost"), 2)
                            if plan.get("opt_lot_cost") not in (None, "")
                            else "—"
                        ),
                    )
                )

            if micro_detail:
                reason = micro_detail.get("reason")
                status = (
                    "OK"
                    if not reason or reason in {"ok", None}
                    else f"BLOCK → {reason}"
                )
                lines.append(
                    "• Micro: {status} (depth_ok={depth} spr%={spr})".format(
                        status=esc(str(status)),
                        depth=esc(
                            str(micro_detail.get("depth_ok"))
                            if micro_detail.get("depth_ok") not in (None, "")
                            else "—"
                        ),
                        spr=esc(
                            _format_float(
                                float(micro_detail.get("spread_pct", 0.0)) * 100.0,
                                2,
                            )
                            if isinstance(
                                micro_detail.get("spread_pct"),
                                (int, float),
                            )
                            else "—",
                        ),
                    )
                )

            self._send("\n".join(lines), parse_mode="Markdown")
            return True

        if cmd == "/whydetail":
            chain: list[str] = []
            warm = getattr(runner, "_warm", None) if runner else None
            if warm is not None:
                warm_ok = getattr(warm, "ok", False)
                have = getattr(warm, "have_bars", None)
                need = getattr(warm, "required_bars", None)
                chain.append(f"warmup:{'ok' if warm_ok else 'fail'}({have}/{need})")
            plan_snapshot = getattr(runner, "_last_plan", None) if runner else None
            if not isinstance(plan_snapshot, Mapping):
                plan_snapshot = plan
            micro_detail: Mapping[str, Any] = {}
            if isinstance(plan_snapshot, Mapping):
                micro_raw = plan_snapshot.get("micro")
                if isinstance(micro_raw, Mapping):
                    micro_detail = micro_raw
            if not micro_detail and isinstance(snap.get("micro"), Mapping):
                micro_detail = snap["micro"]  # type: ignore[assignment]
            if micro_detail:
                reason = micro_detail.get("reason") or micro_detail.get("block_reason")
                micro_ok = not reason or str(reason).lower() in {"ok", "pass"}
                spread_val = micro_detail.get("spread_pct")
                spread_txt = (
                    _format_float((float(spread_val) * 100.0), 2)
                    if isinstance(spread_val, (int, float))
                    else "-"
                )
                chain.append(
                    "micro:{status}(spr={spr} depth={depth})".format(
                        status="ok" if micro_ok else "fail",
                        spr=spread_txt,
                        depth=micro_detail.get("depth_ok"),
                    )
                )
            rr_val = plan_snapshot.get("rr") if isinstance(plan_snapshot, Mapping) else None
            if isinstance(rr_val, (int, float)):
                chain.append(f"score:ok(rr={_format_float(rr_val)})")
            lots = getattr(runner, "_last_computed_lots", None) if runner else None
            if lots is not None:
                chain.append(f"sizing:{'ok' if lots > 0 else 'fail'}(lots={lots})")
            last_label = getattr(runner, "_last_decision_label", None) if runner else None
            last_reason = getattr(runner, "_last_decision_reason", None) if runner else None
            label_txt = last_label or "idle"
            if last_reason:
                chain.append(f"decision:{label_txt}(reason={last_reason})")
            else:
                chain.append(f"decision:{label_txt}")
            text = " → ".join(chain) if chain else "No gate details available."
            self._send(text)
            return True

        if cmd == "/deep":
            plan_snapshot = getattr(runner, "_last_plan", None) if runner else None
            if not isinstance(plan_snapshot, Mapping):
                plan_snapshot = plan
            plan_data = plan_snapshot if isinstance(plan_snapshot, Mapping) else {}
            side = plan_data.get("action") or plan_data.get("side") or "—"
            option_type = plan_data.get("option_type") or plan_data.get("ot") or "—"
            strike = (
                plan_data.get("strike")
                or plan_data.get("symbol")
                or plan_data.get("option_symbol")
                or "—"
            )
            entry = (
                _format_number(plan_data.get("entry"))
                if plan_data.get("entry") not in (None, "")
                else "—"
            )
            sl = (
                _format_number(plan_data.get("sl"))
                if plan_data.get("sl") not in (None, "")
                else "—"
            )
            tp1 = (
                _format_number(plan_data.get("tp1"))
                if plan_data.get("tp1") not in (None, "")
                else "—"
            )
            tp2 = (
                _format_number(plan_data.get("tp2"))
                if plan_data.get("tp2") not in (None, "")
                else "—"
            )
            rr = (
                _format_float(plan_data.get("rr"))
                if plan_data.get("rr") not in (None, "")
                else "—"
            )
            qty = plan_data.get("qty_lots") or plan_data.get("qty") or "—"
            lot = plan_data.get("lot") or plan_data.get("lot_size") or "—"
            spread = plan_data.get("spr") or plan_data.get("spread_pct")
            if spread is None and isinstance(plan_data.get("micro"), Mapping):
                spread = plan_data.get("micro", {}).get("spread_pct")
            if spread is None and isinstance(snap.get("micro"), Mapping):
                spread = snap.get("micro", {}).get("spread_pct")
            spread_txt = (
                _format_float((float(spread) * 100.0), 2)
                if isinstance(spread, (int, float))
                else "—"
            )
            age = (
                plan_data.get("age")
                or plan_data.get("quote_age_s")
                or plan_data.get("bar_age_s")
                or plan_data.get("last_bar_lag_s")
            )
            micro_detail = plan_data.get("micro") if isinstance(plan_data.get("micro"), Mapping) else {}
            if not micro_detail and isinstance(snap.get("micro"), Mapping):
                micro_detail = snap["micro"]  # type: ignore[assignment]
            if age is None and isinstance(micro_detail, Mapping):
                age = micro_detail.get("age")
            age_txt = (
                _format_float(age, 1) if isinstance(age, (int, float)) else "—"
            )
            equity = _format_number(snap.get("equity"), 0)
            position = status.get("pos")
            if position in (None, ""):
                position = status.get("open_positions")
            if position in (None, ""):
                position = "—"
            micro_reason = "—"
            micro_mode = "—"
            depth_ok = "—"
            micro_spread = "—"
            if isinstance(micro_detail, Mapping) and micro_detail:
                micro_reason = micro_detail.get("reason") or micro_detail.get("block_reason") or "OK"
                micro_mode = micro_detail.get("mode") or "—"
                depth_val = micro_detail.get("depth_ok")
                depth_ok = depth_val if depth_val not in (None, "") else "—"
                spread_val = micro_detail.get("spread_pct")
                if isinstance(spread_val, (int, float)):
                    micro_spread = _format_float((float(spread_val) * 100.0), 2)
            last_label = getattr(runner, "_last_decision_label", None) if runner else None
            last_reason = getattr(runner, "_last_decision_reason", None) if runner else None
            lines = [
                f"• MARKET  {'open' if snap.get('market_open') else 'closed'}  tick_age={age_txt}s",
                f"• EQUITY  {equity}  POS={position}",
                (
                    "• PLAN    {side} {ot} {strike}  e={entry} sl={sl} tp1={tp1} tp2={tp2} rr={rr} "
                    "lots={qty}×{lot}".format(
                        side=side,
                        ot=option_type,
                        strike=strike,
                        entry=entry,
                        sl=sl,
                        tp1=tp1,
                        tp2=tp2,
                        rr=rr,
                        qty=qty,
                        lot=lot,
                    )
                ),
                f"• QUOTE   age={age_txt}s spr={spread_txt}%",
            ]
            if isinstance(micro_detail, Mapping) and micro_detail:
                lines.append(
                    "• MICRO   mode={mode} gate={gate} (depth_ok={depth} spr%={spr})".format(
                        mode=micro_mode,
                        gate="OK" if str(micro_reason).lower() in {"ok", "none", ""} else micro_reason,
                        depth=depth_ok,
                        spr=micro_spread,
                    )
                )
            decision_line = "• DECISION last={label} reason={reason}".format(
                label=last_label or "—",
                reason=last_reason or "—",
            )
            lines.append(decision_line)
            self._send("\n".join(lines))
            return True



        if cmd == "/errors":
            limit = 20
            if args and args[0].isdigit():
                limit = max(1, min(int(args[0]), 100))
            ring = getattr(self, "error_ring", None)
            if isinstance(ring, list) and ring:
                subset = ring[-limit:]
                lines = [
                    f"- {item.get('type', 'Error')}: {item.get('msg')} @ {item.get('where')}"
                    for item in subset
                ]
                self._send("\n".join(lines))
                return True
            lines = self._recent_errors(limit)
            if not lines:
                self._send("No recent errors found.")
                return True
            text = "Recent errors:\n" + "\n".join(lines)
            if len(text) > 3500:
                text = text[-3500:]
            self._send(text)
            return True

        return False

    def _build_heartbeat_text(self, snap: Mapping[str, Any]) -> str:
        """Render the compact heartbeat line from a quick snapshot."""

        status = snap.get("status") if isinstance(snap.get("status"), Mapping) else {}
        plan = snap.get("plan") if isinstance(snap.get("plan"), Mapping) else {}
        micro = snap.get("micro") if isinstance(snap.get("micro"), Mapping) else {}

        ts = status.get("time_ist")
        timestamp = snap.get("timestamp")
        if not ts and isinstance(timestamp, datetime):
            ts = timestamp.strftime("%H:%M:%S")
        broker = status.get("broker") or "ok"
        market = "open" if snap.get("market_open") else "closed"
        runner_state = "ready" if snap.get("runner_ready") else "down"

        position = status.get("pos")
        if position in (None, ""):
            position = status.get("open_positions")
        if position in (None, ""):
            position = "—"

        equity = _format_number(snap.get("equity"), 0)

        side = plan.get("action") or plan.get("side") or "—"
        option_type = plan.get("option_type") or plan.get("ot") or "—"
        strike = plan.get("strike") or plan.get("symbol") or "—"

        age = (
            plan.get("age")
            or plan.get("quote_age_s")
            or plan.get("bar_age_s")
            or plan.get("last_bar_lag_s")
        )
        spread = plan.get("spr") or plan.get("spread_pct")
        if spread is None and isinstance(micro, Mapping):
            spread = micro.get("spread_pct")
        if age is None and isinstance(micro, Mapping):
            age = micro.get("age")

        age_str = _format_float(age, 1) if isinstance(age, (int, float)) else "-"
        spread_pct = (
            _format_float((float(spread) * 100.0), 2)
            if isinstance(spread, (int, float))
            else "-"
        )

        plan_line = f"{side} {option_type} {strike} age={age_str}s spr={spread_pct}%"
        header = (
            f"hb {ts or '-'} broker={broker} market={market} "
            f"strategy={runner_state} pos={position} equity={equity}"
        )
        return f"{header} plan={plan_line}"

    def _handle_hb_toggle(self, chat_id: int, args: List[str]) -> bool:
        """Handle `/hb on|off [interval]` commands for auto-heartbeat."""

        if not _COMMAND_GATE.allow(chat_id, "/hb_cfg", 1.5):
            self._send("⏳ Please wait before repeating this command.")
            return True

        action = args[0].lower()

        if action == "off":
            with self._hb_lock:
                state = self._hb_by_chat.setdefault(
                    chat_id,
                    {"enabled": False, "interval": 30.0, "message_id": None, "last": 0.0},
                )
                state["enabled"] = False
            self._hb_maybe_stop()
            self._send("hb auto-update disabled.")
            return True

        interval = _env_float("HB_DEFAULT_SEC", 30.0)
        if len(args) > 1:
            try:
                interval = float(args[1])
            except (TypeError, ValueError):
                interval = _env_float("HB_DEFAULT_SEC", 30.0)
        interval = max(interval, 5.0)

        with self._hb_lock:
            state = self._hb_by_chat.setdefault(
                chat_id,
                {"enabled": False, "interval": interval, "message_id": None, "last": 0.0},
            )
            state["enabled"] = True
            state["interval"] = interval
            state["last"] = 0.0
            state["message_id"] = None
        self._hb_ensure_thread()
        self._send(f"hb auto-update enabled every {int(interval)}s.")
        return True

    def _hb_ensure_thread(self) -> None:
        with self._hb_lock:
            if self._hb_run and self._hb_thread and self._hb_thread.is_alive():
                return
            self._hb_run = True
            thread = threading.Thread(target=self._hb_loop, name="tg-hb", daemon=True)
            self._hb_thread = thread
        thread.start()

    def _hb_maybe_stop(self) -> None:
        with self._hb_lock:
            if any(state.get("enabled") for state in self._hb_by_chat.values()):
                return
            self._hb_run = False

    def _hb_loop(self) -> None:
        while True:
            with self._hb_lock:
                run = self._hb_run
                chat_ids = list(self._hb_by_chat.keys())
            if not run:
                break
            now = time.time()
            for cid in chat_ids:
                self._hb_process_chat(cid, now)
            time.sleep(1.0)
        with self._hb_lock:
            self._hb_thread = None

    def _hb_process_chat(self, chat_id: int, now: float) -> None:
        with self._hb_lock:
            state = self._hb_by_chat.get(chat_id)
            if not state or not state.get("enabled"):
                return
            try:
                interval = float(state.get("interval", 30.0))
            except (TypeError, ValueError):
                interval = 30.0
            interval = max(interval, 5.0)
            last = float(state.get("last", 0.0) or 0.0)
            message_id = state.get("message_id")
        if now - last < interval:
            return

        try:
            snap = self._quick_snapshot()
        except Exception:
            log.debug("heartbeat snapshot failed", exc_info=True)
            snap = {}

        text = self._build_heartbeat_text(snap)
        ok, new_id = self._hb_send_or_edit(chat_id, message_id, text)

        with self._hb_lock:
            state = self._hb_by_chat.setdefault(chat_id, {})
            state["last"] = now
            state["message_id"] = new_id if ok else None

    def _hb_send_or_edit(
        self, chat_id: int, message_id: Optional[int], text: str
    ) -> tuple[bool, Optional[int]]:
        payload: Dict[str, Any] = {
            "chat_id": chat_id,
            "text": text,
            "disable_web_page_preview": True,
        }
        endpoint = "editMessageText" if message_id else "sendMessage"
        if message_id:
            payload["message_id"] = message_id
        ok, result = self._hb_api_call(endpoint, payload)
        if not ok:
            return False, None
        new_id: Optional[int] = None
        if isinstance(result, Mapping):
            msg_id = result.get("message_id")
            if isinstance(msg_id, int):
                new_id = msg_id
        if message_id and new_id is None:
            new_id = message_id
        return True, new_id

    def _hb_api_call(
        self, endpoint: str, payload: Dict[str, Any]
    ) -> tuple[bool, Optional[Mapping[str, Any]]]:
        try:
            response = self._session.post(
                f"{self._base}/{endpoint}", json=payload, timeout=self._timeout
            )
            data = response.json()
        except Exception:
            log.debug("heartbeat %s request failed", endpoint, exc_info=True)
            return False, None
        if not response.ok or not data.get("ok"):
            log.debug(
                "heartbeat %s failed: status=%s data=%s",
                endpoint,
                response.status_code,
                data,
            )
            return False, None
        result = data.get("result")
        return True, result if isinstance(result, Mapping) else None

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

        if self._handle_quick_commands(cmd, args, int(chat_id)):
            return

        # START / HELP
        if cmd == "/start":
            runner = getattr(getattr(self, "_runner_tick", None), "__self__", None)
            runner = runner or StrategyRunner.get_singleton()
            if not self._runner_resume and runner is None:
                return self._send("Runner not ready.")
            msg = self._resume_entries()
            return self._send(msg + "\n\nUse /help to list commands.")

        if cmd == "/help":
            return self._send(self._help_text(), parse_mode="Markdown")

        # STATUS
        if cmd == "/status":
            verbose = args and args[0].lower().startswith("v")
            if verbose:
                try:
                    s = self._status_provider() if self._status_provider else {}
                except Exception:
                    s = {}
                return self._send(
                    "```json\n" + json.dumps(s, indent=2) + "\n```",
                    parse_mode="Markdown",
                )
            if not _COMMAND_GATE.allow(int(chat_id), cmd, _env_float("TG_RATE_STATUS_SEC", 3.0)):
                self._send("⏳ Please wait before repeating this command.")
                return

            snap = self._quick_snapshot()
            if not snap.get("runner_ready"):
                return self._send("Runner not ready.")

            status = snap.get("status", {})
            plan = snap.get("plan", {})
            esc = self._escape_markdown

            runner_state = "ready" if snap.get("runner_ready") else "down"
            market_state = "open" if snap.get("market_open") else "closed"
            broker = status.get("broker") or "ok"
            equity = _format_number(snap.get("equity"), 0)
            position = status.get("pos")
            if position in (None, ""):
                position = status.get("open_positions")
            if position in (None, ""):
                position = "—"
            plan_side = plan.get("action") or plan.get("side") or "—"
            plan_ot = plan.get("option_type") or plan.get("ot") or "—"
            plan_strike = plan.get("strike") or plan.get("symbol") or "—"
            entry = (
                _format_number(plan.get("entry"))
                if plan.get("entry") not in (None, "")
                else "—"
            )
            sl = (
                _format_number(plan.get("sl"))
                if plan.get("sl") not in (None, "")
                else "—"
            )
            tp1 = (
                _format_number(plan.get("tp1"))
                if plan.get("tp1") not in (None, "")
                else "—"
            )
            rr = (
                _format_float(plan.get("rr"))
                if plan.get("rr") not in (None, "")
                else "—"
            )

            spread = plan.get("spr") or plan.get("spread_pct")
            micro = snap.get("micro") if isinstance(snap.get("micro"), Mapping) else {}
            if spread is None and isinstance(micro, Mapping):
                spread = micro.get("spread_pct")
            age = (
                plan.get("age")
                or plan.get("quote_age_s")
                or plan.get("bar_age_s")
                or plan.get("last_bar_lag_s")
            )
            if age is None and isinstance(micro, Mapping):
                age = micro.get("age")
            spread_txt = (
                _format_float((float(spread) * 100.0), 2)
                if isinstance(spread, (int, float))
                else "—"
            )
            age_txt = (
                _format_float(age, 1) if isinstance(age, (int, float)) else "—"
            )

            lines = [
                f"*RUNNER*  {esc(runner_state)}",
                f"*MARKET*  {esc(market_state)}",
                f"*BROKER*  {esc(str(broker))}",
                f"*EQUITY*  {esc(equity)}  *POS* {esc(str(position))}",
                (
                    "*PLAN*    {side} {ot} {strike}  e={entry} sl={sl} tp1={tp1} rr={rr}".format(
                        side=esc(str(plan_side)),
                        ot=esc(str(plan_ot)),
                        strike=esc(str(plan_strike)),
                        entry=esc(entry),
                        sl=esc(sl),
                        tp1=esc(tp1),
                        rr=esc(rr),
                    )
                ),
                f"*QUOTE*   age={esc(age_txt)}s spr={esc(spread_txt)}%",
            ]

            return self._send("\n".join(lines), parse_mode="Markdown")

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

        if cmd == "/state":
            runner = StrategyRunner.get_singleton()
            if not runner:
                return self._send("runner not ready")
            status = runner.get_status_snapshot()
            plan = (
                runner.get_last_signal_debug()
                if hasattr(runner, "get_last_signal_debug")
                else {}
            )
            eq = getattr(runner, "_equity_cached_value", 0.0)
            trail_on = bool(getattr(getattr(runner, "order_executor", None), "enable_trailing", False))
            lines = [
                (
                    f"eq={round(float(eq),2)} trades={status.get('trades_today')} "
                    f"cooloff={status.get('cooloff_until', '-')} "
                    f"losses={status.get('consecutive_losses')} "
                    f"evals={getattr(runner, 'eval_count', 0)} "
                    f"tp_basis={getattr(settings, 'tp_basis', 'premium')} "
                    f"trail={'on' if trail_on else 'off'}"
                )
            ]
            lines.append(
                "spot: entry={e} sl={sl} tp1={tp1} tp2={tp2}".format(
                    e=plan.get("entry"),
                    sl=plan.get("sl"),
                    tp1=plan.get("tp1"),
                    tp2=plan.get("tp2"),
                )
            )
            basis = str(settings.EXPOSURE_BASIS).lower()
            if basis == "premium":
                lines.append(
                    "Option → entry ₹{e} SL ₹{sl} TP1 ₹{tp1} TP2 ₹{tp2} lot ₹{lc}".format(
                        e=plan.get("opt_entry"),
                        sl=plan.get("opt_sl"),
                        tp1=plan.get("opt_tp1"),
                        tp2=plan.get("opt_tp2"),
                        lc=plan.get("opt_lot_cost"),
                    )
                )
            return self._send("\n".join(lines))

        if cmd == "/selftest":
            if args:
                results = [run(args[0])]
            else:
                results = run_all()
            lines = [
                f"{'✅' if r.ok else '❌'} {r.name} — {r.msg} ({r.took_ms}ms)"
                for r in results
            ]
            for r in results:
                if not r.ok and r.fix:
                    lines.append(f"• fix: {r.fix}")
            return self._send("\n".join(lines)[:3500])

        if cmd == "/probe":
            if self._probe_provider:
                try:
                    info = self._probe_provider()
                    return self._send(f"Probe: {info}")
                except Exception as e:
                    return self._send(f"Probe error: {e}")
            runner = StrategyRunner.get_singleton()
            if not runner:
                return self._send("runner not ready")
            ds = runner.debug_snapshot()
            return self._send(
                f"bars={ds['bars']} last={ds['last_bar_ts']} lag_s={ds['lag_s']} rr={ds['rr_threshold']} risk%={ds['risk_pct']}"
            )

        if cmd == "/score":
            runner = StrategyRunner.get_singleton()
            if not runner:
                return self._send("runner not ready")
            items = getattr(runner, "_score_items", None)
            total = getattr(runner, "_score_total", None)
            thr = getattr(runner.strategy_cfg, "min_signal_score", 0.0)
            if items is None:
                return self._send("📊 no score breakdown yet")
            top = sorted(items.items(), key=lambda kv: abs(kv[1]), reverse=True)[:12]
            lines = [
                f"📊 Score breakdown (thr={thr})",
                f"• total={total}  {'✅ PASS' if total is not None and total >= thr else '❌ FAIL'}",
            ]
            for name, val in top:
                sign = "➕" if val >= 0 else "➖"
                lines.append(f"{sign} {name}: {val}")
            return self._send("\n".join(lines))

        if cmd == "/shadow":
            runner = StrategyRunner.get_singleton()
            p = runner.last_plan if runner else {}
            sb = p.get("shadow_blockers") or []
            if not sb:
                return self._send("no shadow blockers")
            return self._send("would_block: " + ", ".join(sb))

        if cmd == "/lastplan":
            runner = StrategyRunner.get_singleton()
            p = runner.last_plan if runner else {}
            txt = json.dumps(p, default=str)[:3500]
            return self._send(f"<pre>{txt}</pre>", parse_mode="HTML")

        if cmd == "/bars":
            if self._bars_provider:
                try:
                    n = int(args[0]) if args else 5
                except Exception:
                    n = 5
                try:
                    text = self._bars_provider(n)
                    return self._send(text)
                except Exception as e:
                    return self._send(f"Bars error: {e}")
            runner = StrategyRunner.get_singleton()
            if not runner:
                return self._send("runner not ready")
            ds = runner.debug_snapshot()
            return self._send(
                f"bars={ds['bars']} last={ds['last_bar_ts']} lag_s={ds['lag_s']} gates={ds['gates']}"
            )

        if cmd == "/expiry":
            return self._send(
                f"weekly={next_tuesday_expiry()} | monthly={last_tuesday_of_month()}"
            )

        if cmd == "/sizer":
            ps = PositionSizer.from_settings(
                risk_per_trade=settings.risk.risk_per_trade,
                min_lots=settings.risk.min_lots,
                max_lots=settings.risk.max_lots,
                max_position_size_pct=settings.risk.max_position_size_pct,
                exposure_basis=settings.EXPOSURE_BASIS,
            )
            qty, lots, diag = ps.size_from_signal(
                entry_price=200,
                stop_loss=190,
                lot_size=50,
                equity=100_000,
                spot_sl_points=10,
                delta=0.5,
            )
            return self._send(
                f"risk%={settings.risk.risk_per_trade} rupees={diag['risk_rupees']} lots={lots} qty={qty}"
            )

        if cmd == "/healthjson":
            js = json.dumps([r.__dict__ for r in run_all()], ensure_ascii=False)
            return self._send(js[:3500])

        if cmd == "/logtail":
            n = 200
            if args and args[0].isdigit():
                n = int(args[0])
            p = Path("logs/bot.log")
            if not p.exists():
                return self._send("no log")
            lines = p.read_text(errors="ignore").splitlines()[-n:]
            return self._send(
                "```" + "\n".join(lines)[-3500:] + "```", parse_mode="Markdown"
            )

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
                f"score gate: ≥{c.raw.get('strategy', {}).get('min_score', 0.0)}\n"
                f"micro: mode {c.raw.get('micro', {}).get('mode')}, cap {c.raw.get('micro', {}).get('max_spread_pct')}, depth_min_lots {c.depth_min_lots}\n"
                f"options: OI≥{c.min_oi}, Δ∈[{c.delta_min},{c.delta_max}]\n"
                f"lifecycle: tp1 {c.tp1_R_min}–{c.tp1_R_max}R, tp2(T/R) {c.tp2_R_trend}/{c.tp2_R_range}, trail {c.trail_atr_mult}, time {c.time_stop_min}m\n"
                f"gamma: {c.gamma_enabled} after {c.gamma_after}\n"
                f"warmup: bars {c.min_bars_required}\n"
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
                    "orders": getattr(
                        getattr(runner, "order_executor", None), "cb_orders", None
                    ),
                    "modify": getattr(
                        getattr(runner, "order_executor", None), "cb_modify", None
                    ),
                    "hist": getattr(
                        getattr(runner, "data_source", None), "cb_hist", None
                    ),
                    "quote": getattr(
                        getattr(runner, "data_source", None), "cb_quote", None
                    ),
                }
                cb = cb_map.get(name)
                if cb:
                    cb.force_open(secs)
                    return self._send(f"Breaker {name} forced OPEN {secs}s")
                return self._send("Unknown breaker name")

        # DIAG – recent trace events from the diagnostics ring buffer
        if cmd == "/diag":
            ring = getattr(checks, "TRACE_RING", None)
            limit_cfg = _diag_trace_limit()
            if ring is None or limit_cfg <= 0:
                return self._send("Diagnostic trace capture disabled.")
            enabled_fn = getattr(ring, "enabled", None)
            if not enabled_fn or not enabled_fn():
                return self._send("Diagnostic trace capture disabled.")
            try:
                records = ring.tail(None)
            except Exception as exc:
                return self._send(f"Diag error: {exc}")
            if not records:
                return self._send("No diagnostic events captured.")
            limit = limit_cfg
            if args:
                try:
                    requested = int(args[0])
                except (TypeError, ValueError):
                    return self._send("Usage: /diag [count]")
                limit = max(1, min(requested, limit_cfg))
            tail = records[-limit:]
            lines = [_format_trace_summary_line(rec) for rec in tail]
            text = "\n".join(lines)
            return self._send(f"```text\n{text}\n```", parse_mode="Markdown")

        if cmd == "/diagstatus":
            if not self._compact_diag_provider:
                return self._send("Diag status provider not wired.")
            try:
                summary = self._compact_diag_provider() or {}
            except Exception as exc:
                return self._send(f"Diag status error: {exc}")
            if not summary:
                return self._send("No diagnostic summary available.")
            status_messages = summary.get("status_messages") or {}
            overall = "ok" if summary.get("ok") else "issues"
            lines = [f"overall: {overall}"]
            for key in sorted(status_messages):
                lines.append(f"{key}: {status_messages[key]}")
            text = "\n".join(lines)
            return self._send(f"```text\n{text}\n```", parse_mode="Markdown")

        if cmd == "/diagtrace":
            limit = _diag_trace_limit()
            ring = getattr(checks, "TRACE_RING", None)
            if limit <= 0 or ring is None:
                return self._send("Diagnostic trace capture disabled.")
            enabled_fn = getattr(ring, "enabled", None)
            if not enabled_fn or not enabled_fn():
                return self._send("Diagnostic trace capture disabled.")
            try:
                records = ring.tail(None)
            except Exception as exc:
                return self._send(f"Diag trace error: {exc}")
            if not records:
                return self._send("No diagnostic events captured.")
            target_trace = args[0].strip() if args else ""
            filtered_records = (
                [r for r in records if str(r.get("trace_id")) == target_trace]
                if target_trace
                else records
            )
            if not filtered_records:
                return self._send(f"No events found for trace {target_trace}.")
            payload = [
                _normalize_trace_record(rec)
                for rec in filtered_records[-limit:]
            ]
            text = json.dumps(payload, ensure_ascii=False, indent=2)
            return self._send(f"```json\n{text}\n```", parse_mode="Markdown")

        if cmd == "/greeks":
            runner = getattr(getattr(self, "_runner_tick", None), "__self__", None)
            if not runner:
                return self._send("Runner unavailable.")
            delta_units = round(runner._portfolio_delta_units(), 1)
            gmode = runner.now_ist.weekday() == 1 and runner.now_ist.time() >= dt_time(
                14, 45
            )
            text = (
                "📐 *Portfolio Greeks*\n"
                f"Δ(units): {delta_units} | gamma_mode: {gmode}"
            )
            return self._send(text, parse_mode="Markdown")

        if cmd == "/plan":
            runner = StrategyRunner.get_singleton()
            if not runner:
                return self._send("runner not ready")
            plan = runner.last_plan if runner else {}
            lines = [
                "spot: entry={e} sl={sl} tp1={tp1} tp2={tp2}".format(
                    e=plan.get("entry"),
                    sl=plan.get("sl"),
                    tp1=plan.get("tp1"),
                    tp2=plan.get("tp2"),
                )
            ]
            basis = str(settings.EXPOSURE_BASIS).lower()
            if basis == "premium":
                lines.append(
                    "Option → entry ₹{e} SL ₹{sl} TP1 ₹{tp1} TP2 ₹{tp2} lot ₹{lc}".format(
                        e=plan.get("opt_entry"),
                        sl=plan.get("opt_sl"),
                        tp1=plan.get("opt_tp1"),
                        tp2=plan.get("opt_tp2"),
                        lc=plan.get("opt_lot_cost"),
                    )
                )
            return self._send("\n".join(lines))

        if cmd == "/events":
            runner = getattr(getattr(self, "_runner_tick", None), "__self__", None)
            if not runner or not runner.event_cal:
                return self._send("No events calendar loaded.")
            now = runner.now_ist
            active = runner.event_cal.active(now)
            lines = [
                f"📅 Events v{runner.event_cal.version} tz={runner.event_cal.tz.key}"
            ]
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
                bc_ok = (
                    bars.get("bar_count") or 0
                ) >= runner.strategy_cfg.min_bars_required
                stale_ok = True
                try:
                    last = bars.get("last_bar_ts")
                    if last:
                        last_dt = datetime.fromisoformat(str(last))
                        if last_dt.tzinfo is None:
                            last_dt = last_dt.replace(
                                tzinfo=ZoneInfo(runner.strategy_cfg.tz)
                            )
                        age = (runner.now_ist - last_dt).total_seconds()
                        stale_ok = 0 <= age <= 150
                except Exception:
                    stale_ok = False
                regime_ok = sig.get("regime") in ("TREND", "RANGE")
                atr = sig.get("atr_pct") or 0.0
                atr_ok = (
                    runner.strategy_cfg.atr_min <= atr <= runner.strategy_cfg.atr_max
                )
                need = float(
                    runner.strategy_cfg.raw.get("strategy", {}).get("min_score", 0.35)
                )
                score_ok = (sig.get("score") or 0) >= need
                micro = sig.get("micro") or {}
                msp = micro.get("spread_pct")
                mdp = micro.get("depth_ok")
                ob_reason = sig.get("reason_block")
                lines = []
                for name, ok, val in [
                    ("window", window_ok, getattr(runner, "window_tuple", "-")),
                    (
                        "cooloff",
                        cool_ok,
                        getattr(re.state, "cooloff_until", None) if re else None,
                    ),
                    (
                        "daily_dd",
                        dd_ok,
                        getattr(re.state, "cum_R_today", 0.0) if re else 0.0,
                    ),
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
                return self._send(
                    "🧪 *Audit*\n" + "\n".join(lines), parse_mode="Markdown"
                )
            except Exception as e:
                return self._send(f"Audit error: {e}")

        if cmd == "/why":
            if not _COMMAND_GATE.allow(
                int(chat_id), cmd, _env_float("TG_RATE_WHY_SEC", 3.0)
            ):
                self._send("⏳ Please wait before repeating this command.")
                return
            try:
                status = self._status_provider() if self._status_provider else {}
                plan = (
                    self._last_signal_provider() if self._last_signal_provider else {}
                )
                runner = getattr(getattr(self, "_runner_tick", None), "__self__", None)
                warm = getattr(runner, "_warm", None)
                fresh = getattr(runner, "_fresh", None)
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
                cooloff = bool(
                    status.get("cooloff_until") and status.get("cooloff_until") != "-"
                )
                gates.append(("cooloff", not cooloff, status.get("cooloff_until")))
                dd = bool(status.get("daily_dd_hit"))
                gates.append(("daily_dd", not dd, status.get("day_realized_loss")))
                bar_count = int(plan.get("bar_count") or 0)
                if warm:
                    gates.append(
                        (
                            "bar_count",
                            bool(getattr(warm, "ok", False)),
                            f"have={getattr(warm, 'have_bars', bar_count)} need={getattr(warm, 'required_bars', '-')}",
                        )
                    )
                else:
                    gates.append(("bar_count", bar_count >= 20, bar_count))
                if fresh:
                    gates.append(
                        (
                            "data_stale",
                            bool(getattr(fresh, "ok", False)),
                            f"tick_lag={getattr(fresh, 'tick_lag_s', None)} bar_lag={getattr(fresh, 'bar_lag_s', None)}",
                        )
                    )
                else:
                    lag = int(now - last_ts_sec) if last_ts_sec else None
                    data_stale = (lag or 0) <= 150
                    gates.append(
                        (
                            "data_stale",
                            data_stale,
                            f"lag_s={lag}" if lag is not None else "-",
                        )
                    )
                regime = plan.get("regime") in ("TREND", "RANGE")
                gates.append(("regime", regime, plan.get("regime")))
                atr = plan.get("atr_pct")
                atr_min = plan.get("atr_min")
                atr_ok = atr is not None and atr_min is not None and atr >= atr_min
                gates.append(("atr_pct", atr_ok, atr if atr is not None else "N/A"))
                score_val = plan.get("score")
                score = float(score_val) if score_val is not None else None
                reg = str(plan.get("regime"))
                need = 9 if reg == "TREND" else 8
                gates.append(
                    (
                        "score",
                        score is not None and score >= need,
                        score if score is not None else "N/A",
                    )
                )
                sp = plan.get("spread_pct")
                dp = plan.get("depth_ok")
                reason_block = plan.get("reason_block") or "-"
                reason_label = (
                    checks.REASON_MAP.get(reason_block, "")
                    if reason_block != "-"
                    else ""
                )
                reasons = plan.get("reasons") or []
                micro = plan.get("micro") if isinstance(plan.get("micro"), Mapping) else {}
                spread_val = plan.get("spr") or plan.get("spread_pct")
                if spread_val is None and isinstance(micro, Mapping):
                    spread_val = micro.get("spread_pct")
                depth_val = plan.get("depth_ok")
                if depth_val is None and isinstance(micro, Mapping):
                    depth_val = micro.get("depth_ok")
                age_val = (
                    plan.get("age")
                    or plan.get("quote_age_s")
                    or plan.get("bar_age_s")
                    or plan.get("last_bar_lag_s")
                )
                if age_val is None and isinstance(micro, Mapping):
                    age_val = micro.get("age")
                spread_txt = (
                    _format_float((float(spread_val) * 100.0), 2)
                    if isinstance(spread_val, (int, float))
                    else "—"
                )
                age_txt = (
                    _format_float(age_val, 1)
                    if isinstance(age_val, (int, float))
                    else "—"
                )
                depth_txt = str(depth_val) if depth_val is not None else "—"
                micro_reason = ""
                if isinstance(micro, Mapping):
                    micro_reason = micro.get("reason") or ""

                summary_reasons = plan.get("reasons")
                summary_extra = ""
                if isinstance(summary_reasons, (list, tuple)) and summary_reasons:
                    summary_extra = str(summary_reasons[0])
                elif isinstance(summary_reasons, Mapping) and summary_reasons:
                    first_key = next(iter(summary_reasons))
                    summary_extra = f"{first_key}:{summary_reasons[first_key]}"

                if reason_block and reason_block != "-":
                    summary = f"why: BLOCK {reason_block}"
                    if summary_extra and summary_extra != reason_block:
                        summary += f" ({summary_extra})"
                else:
                    runner_label = status.get("state") or status.get("runner_state")
                    if not runner_label:
                        runner_label = "ready" if within else "down"
                    summary = f"why: {runner_label}"

                summary += f" spr={spread_txt}% age={age_txt}s depth_ok={depth_txt}"
                if micro_reason and micro_reason not in {"ok", reason_block}:
                    summary += f" reason={micro_reason}"

                lines = ["/why gates", summary]
                banners_raw = status.get("banners")
                if isinstance(banners_raw, str):
                    banners_seq = [banners_raw]
                elif isinstance(banners_raw, (list, tuple, set)):
                    banners_seq = [str(item) for item in banners_raw]
                else:
                    banners_seq = []
                if banners_seq:
                    if any("auto_relax" in item for item in banners_seq):
                        lines.append("auto_relax: active")
                    lines.append("Banners: " + ", ".join(banners_seq))
                for name, ok, value in gates:
                    lines.append(f"{name}: {mark(ok)} {value}")
                if reason_block in {"no_option_quote", "no_option_token"}:
                    lines.append("micro: N/A (no_quote)")
                else:
                    sp_line = "N/A (no_quote)" if sp is None else round(sp, 3)
                    dp_line = "N/A (no_quote)" if dp is None else ("✅" if dp else "❌")
                    lines.append(
                        f"micro: spread%={sp_line} depth={dp_line} src={plan.get('quote_src','-')}"
                    )
                lines.append(
                    "spot: entry={e} sl={sl} tp1={tp1} tp2={tp2}".format(
                        e=plan.get("entry"),
                        sl=plan.get("sl"),
                        tp1=plan.get("tp1"),
                        tp2=plan.get("tp2"),
                    )
                )
                basis = str(settings.EXPOSURE_BASIS).lower()
                if basis == "premium":
                    lines.append(
                        "Option → entry ₹{e} SL ₹{sl} TP1 ₹{tp1} TP2 ₹{tp2} lot ₹{lc} atr {atr} atr_pct {atrp}".format(
                            e=plan.get("opt_entry"),
                            sl=plan.get("opt_sl"),
                            tp1=plan.get("opt_tp1"),
                            tp2=plan.get("opt_tp2"),
                            lc=plan.get("opt_lot_cost"),
                            atr=plan.get("opt_atr"),
                            atrp=plan.get("opt_atr_pct"),
                        )
                    )
                lines.append(
                    f"tp_basis: {getattr(settings, 'tp_basis', 'premium')}"
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
                    atr_reasons = plan.get("reasons", [])
                    atr_blocked = any(
                        isinstance(reason, str)
                        and reason.startswith("atr_out_of_band")
                        for reason in atr_reasons
                    )
                    atr_max = plan.get("atr_max")
                    lines.append(
                        f"ATR%: {round(atr,4)} (band {atr_min}..{atr_max}) → {'PASS' if not atr_blocked else 'FAIL'}"
                    )
                else:
                    lines.append("ATR%: N/A")
                if plan.get("probe_window_from") and plan.get("probe_window_to"):
                    lines.append(
                        f"\u2022 Probe window: {plan['probe_window_from']} \u2192 {plan['probe_window_to']} (IST)"
                    )
                if reason_label:
                    lines.append(
                        f"reason_block: {reason_block} ({reason_label})"
                    )
                else:
                    lines.append(f"reason_block: {reason_block}")
                if reasons:
                    lines.append("reasons: " + ", ".join(str(r) for r in reasons))
                try:
                    from src.diagnostics.registry import run as diag_run

                    diag_checks = [
                        "data_window",
                        "atr",
                        "regime",
                        "micro",
                        "risk_gates",
                    ]
                    for r in (diag_run(c) for c in diag_checks):
                        mark = "✅" if r.ok else "❌"
                        lines.append(f"{mark} {r.name}: {r.msg}")
                        if not r.ok and r.fix:
                            lines.append(f"• fix: {r.fix}")
                except Exception:
                    pass
                emit_debug(
                    "why",
                    {
                        "status": status,
                        "plan": plan,
                        "gates": [
                            {"name": name, "ok": ok, "value": value} for name, ok, value in gates
                        ],
                        "summary": summary,
                    },
                )
                return self._send("\n".join(lines)[:3500], parse_mode="Markdown")
            except Exception as e:
                return self._send(f"Why error: {e}")

        if cmd == "/atm":
            if not self._atm_provider:
                return self._send("ATM provider unavailable.")
            try:
                info = self._atm_provider()
                return self._send(f"ATM: {info}")
            except Exception as e:
                return self._send(f"ATM error: {e}")

        if cmd in ("/tick", "/l1"):
            if not self._l1_provider:
                return self._send("L1 provider unavailable.")
            try:
                t = self._l1_provider()
                return self._send(f"L1: {t if t else 'n/a'}")
            except Exception as e:
                return self._send(f"L1 error: {e}")

        if cmd == "/quotes":
            try:
                if self._quotes_provider:
                    opt = args[0] if args else "both"
                    runner = StrategyRunner.get_singleton()
                    if runner is None:
                        return self._send("runner not ready")
                    return self._send(self._quotes_provider(opt, runner=runner))
                plan = (
                    self._last_signal_provider() if self._last_signal_provider else {}
                )
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
            trace_setter: Optional[Callable[[int], None]] = None
            if self._trace_provider:
                trace_setter = self._trace_provider
            else:
                runner = getattr(getattr(self, "_runner_tick", None), "__self__", None)
                if runner is not None:
                    trace_setter = lambda n: setattr(
                        runner, "trace_ticks_remaining", max(1, min(50, n))
                    )
            if args:
                try:
                    requested = int(args[0])
                except (TypeError, ValueError):
                    requested = None
                if trace_setter and requested is not None:
                    count = max(1, min(50, requested))
                    trace_setter(count)
                    return self._send(f"Tracing next {count} evals.")
            if not args:
                return self._send("Usage: /trace <trace_id>")
            trace_id = args[0].strip()
            if not trace_id:
                return self._send("Usage: /trace <trace_id>")
            ring = getattr(checks, "TRACE_RING", None)
            if ring is None:
                return self._send("Diagnostic trace capture disabled.")
            enabled_fn = getattr(ring, "enabled", None)
            if not enabled_fn or not enabled_fn():
                return self._send("Diagnostic trace capture disabled.")
            try:
                records = ring.tail(None)
            except Exception as exc:
                return self._send(f"Trace error: {exc}")
            filtered = [r for r in records if str(r.get("trace_id")) == trace_id]
            if not filtered:
                return self._send(f"No events found for trace {trace_id}.")
            payload = [_normalize_trace_record(rec) for rec in filtered]
            text = json.dumps(payload, ensure_ascii=False, indent=2)
            return self._send(f"```json\n{text}\n```", parse_mode="Markdown")

        if cmd == "/traceoff":
            if self._trace_provider:
                try:
                    self._trace_provider(0)
                except Exception as exc:
                    return self._send(f"Trace off error: {exc}")
                return self._send("Trace off.")
            runner = getattr(getattr(self, "_runner_tick", None), "__self__", None)
            if not runner:
                return self._send("Runner unavailable.")
            runner.trace_ticks_remaining = 0
            return self._send("Trace off.")

        if cmd == "/summary":
            runner = getattr(getattr(self, "_runner_tick", None), "__self__", None)
            if not runner:
                return self._send("Runner unavailable.")
            arg = args[0].lower() if args else "week"
            tz = ZoneInfo(getattr(runner.settings, "TZ", "Asia/Kolkata"))
            now = datetime.now(tz)
            if arg == "month":
                start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            else:
                start = (now - timedelta(days=now.weekday())).replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
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
            return self._send(
                "\U0001f5de *Last Trades*\n" + "\n".join(lines), parse_mode="Markdown"
            )

        if cmd == "/hb":
            runner = getattr(getattr(self, "_runner_tick", None), "__self__", None)
            if not runner:
                return self._send("Runner unavailable.")
            arg = args[0].lower() if args else "on"
            runner.hb_enabled = arg != "off"
            return self._send(f"Heartbeat: {'ON' if runner.hb_enabled else 'OFF'}")

        if cmd == "/smoketest":
            try:
                opt = args[0].lower() if args else "ce"
                if not self._selftest_provider:
                    return self._send("smoketest not wired")
                text = self._selftest_provider(opt)
                return self._send(f"smoketest: {text}")
            except Exception as e:
                return self._send(f"Smoketest error: {e}")

        if cmd == "/filecheck":
            try:
                if not self._filecheck_provider:
                    return self._send("filecheck not wired")
                if not args:
                    return self._send("Usage: /filecheck <path>")
                text = self._filecheck_provider(args[0])
                return self._send(text)
            except Exception as e:
                return self._send(f"filecheck error: {e}")

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

                lines.append(
                    f"📈 last_signal: {'present' if d.get('last_signal') else 'none'}"
                )

                return self._send("\n".join(lines))
            except Exception as e:
                return self._send(f"Check error: {e}")

        # POSITIONS
        if cmd == "/positions":
            if not self._positions_provider:
                return self._send("Positions provider not wired.")
            pos = self._positions_provider() or {}
            err = getattr(
                getattr(self._positions_provider, "__self__", None), "last_error", None
            )
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
            err = getattr(
                getattr(self._actives_provider, "__self__", None), "last_error", None
            )
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
                basis = expo.get("basis", "premium")
                exp_line = f"exposure: notional ({basis}) ₹{notional:,.0f}"
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
            val = state == "live"
            runner = (
                getattr(self._set_live_mode, "__self__", None)
                if self._set_live_mode
                else None
            )
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

        # FORCE EVAL
        if cmd == "/force_eval":
            dry = bool(args and args[0].lower().startswith("dry"))
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
                n = int(args[0]) if args else 20
                span = max(5, min(200, n))
                lines = self._logs_provider(span) or []
                if not lines:
                    return self._send("No logs available.")
                block = "\n".join(lines[-min(len(lines), n):])
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
            return self._send(self._resume_entries())

        if cmd == "/emergency_stop":
            runner = StrategyRunner.get_singleton()
            if not runner:
                return self._send("runner not ready")
            try:
                if self._cancel_all:
                    self._cancel_all()
                runner.shutdown()
                kill_file = os.getenv("KILL_SWITCH_FILE", "")
                if kill_file:
                    try:
                        Path(kill_file).touch()
                    except Exception:
                        log.exception("failed to persist kill switch")
                return self._send("Emergency stop executed.")
            except Exception as e:
                return self._send(f"Emergency stop failed: {e}")

        # CANCEL ALL
        if cmd == "/cancel_all":
            return self._send_inline(
                "Confirm cancel all?",
                [
                    [
                        {"text": "✅ Confirm", "callback_data": "confirm_cancel_all"},
                        {"text": "❌ Abort", "callback_data": "abort"},
                    ]
                ],
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

        if cmd == "/atrmin":
            runner = StrategyRunner.get_singleton()
            if not runner:
                return self._send("runner not ready")
            try:
                v = float(args[0])
                runner.strategy_cfg.raw["atr_min"] = v
                runner.strategy_cfg.atr_min = v
                return self._send(f"atr_min set to {v}")
            except Exception:
                return self._send("usage: /atrmin <percent>")

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

        if cmd == "/warmup":
            runner = StrategyRunner.get_singleton()
            if not runner or not getattr(runner, "data_source", None):
                return self._send("runner not ready")
            w = getattr(runner, "_warm", None)
            have = getattr(runner, "_last_bar_count", None)
            if w is None:
                if have is None:
                    try:
                        bars_df = runner.data_source.get_last_bars(
                            runner.strategy_cfg.min_bars_required
                        )
                        have = int(len(bars_df)) if hasattr(bars_df, "__len__") else 0
                    except Exception:
                        have = 0
                w = warmup_check(runner.strategy_cfg, int(have))
            label = "PASS ✅" if w.ok else "FAIL ❌"
            reasons = ", ".join(w.reasons) if w.reasons else "-"
            return self._send(
                f"🧊 Warmup {label}\n• have={w.have_bars} need={w.required_bars}\n• reasons: {reasons}"
            )

        if cmd == "/fresh":
            runner = StrategyRunner.get_singleton()
            if not runner or not getattr(runner, "data_source", None):
                return self._send("runner not ready")
            now = datetime.now(timezone.utc)
            ds = runner.data_source
            tick_dt = None
            tick_attr = getattr(ds, "last_tick_ts", None)
            if callable(tick_attr):
                try:
                    tick_dt = tick_attr()
                except Exception:  # pragma: no cover - defensive log path
                    tick_dt = None
            if tick_dt is None:
                tick_dt_accessor = getattr(ds, "last_tick_dt", None)
                if callable(tick_dt_accessor):
                    try:
                        tick_dt = tick_dt_accessor()
                    except Exception:  # pragma: no cover - defensive log path
                        tick_dt = None
            if tick_dt is None:
                tick_dt = getattr(ds, "_last_tick_ts", None)

            fr = compute_freshness(
                now=now,
                last_tick_ts=tick_dt,
                last_bar_open_ts=ds.last_bar_open_ts(),
                tf_seconds=ds.timeframe_seconds,
                max_tick_lag_s=runner.strategy_cfg.max_tick_lag_s,
                max_bar_lag_s=runner.strategy_cfg.max_bar_lag_s,
            )
            label = "PASS ✅" if fr.ok else "FAIL ❌"
            return self._send(
                f"⏱ Freshness {label}\n• tick_lag={fr.tick_lag_s}s\n• bar_lag={fr.bar_lag_s}s"
            )

        if cmd == "/microcap":
            runner = StrategyRunner.get_singleton()
            if not runner:
                return self._send("runner not ready")
            try:
                v = float(args[0])
                cfg = runner.strategy_cfg.raw.setdefault("micro", {})
                cfg["max_spread_pct"] = v
                cfg["dynamic"] = False
                return self._send(f"micro max_spread_pct set to {v}% (dynamic=off)")
            except Exception:
                return self._send("usage: /microcap <percent>  e.g. /microcap 1.0")

        if cmd == "/depthmin":
            runner = StrategyRunner.get_singleton()
            if not runner:
                return self._send("runner not ready")
            try:
                v = int(args[0])
                cfg = runner.strategy_cfg.raw.setdefault("micro", {})
                cfg["depth_min_lots"] = v
                return self._send(f"micro depth_min_lots set to {v}")
            except Exception:
                return self._send("usage: /depthmin <lots>")

        if cmd == "/micromode":
            runner = StrategyRunner.get_singleton()
            if not runner:
                return self._send("runner not ready")
            try:
                v = str(args[0]).upper()
                assert v in ("HARD", "SOFT")
                cfg = runner.strategy_cfg.raw.setdefault("micro", {})
                cfg["mode"] = v
                return self._send(f"micro mode = {v}")
            except Exception:
                return self._send("usage: /micromode HARD|SOFT")

        if cmd == "/micro":
            runner = StrategyRunner.get_singleton()
            if not runner:
                return self._send("runner not ready")
            try:
                inst_dump = strike_selector._fetch_instruments_nfo(runner.kite) or []
                spot = strike_selector._get_spot_ltp(
                    runner.kite, getattr(settings.instruments, "spot_symbol", "")
                )
                atm = strike_selector.resolve_weekly_atm(spot or 0.0, inst_dump)
                info = atm.get("ce") if atm else None
                if not info:
                    return self._send("📉 Micro: N/A (no_option_token)")
                tsym, lot = info
                q = fetch_quote_with_depth(runner.kite, tsym)
                spread_pct, depth_ok = micro_from_quote(
                    q, lot_size=lot, depth_min_lots=runner.strategy_cfg.depth_min_lots
                )
                if spread_pct is None or depth_ok is None:
                    return self._send("📉 Micro: N/A (no_quote)")
                src = q.get("source", "-")
                try:
                    spread_pct_fmt = f"{float(spread_pct) * 100.0:.2f}"
                except (TypeError, ValueError):
                    spread_pct_fmt = "N/A"
                return self._send(
                    f"📉 Micro\n• spread%={spread_pct_fmt}\n• depth_ok={depth_ok}\n• src={src}"
                )
            except Exception:
                log.exception("/micro failed")
                return self._send("📉 Micro: N/A (error)")

        # Unknown
        return self._send("Unknown command. Try /help.")
