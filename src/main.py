from __future__ import annotations

import logging
import signal
import sys
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any, Deque, Dict, List, Optional

from src.config import settings
from src.server.health import start_health_server
from src.strategies.runner import StrategyRunner
from src.notifications.telegram_controller import TelegramController

# Optional broker SDK (works in paper mode if missing)
try:
    from kiteconnect import KiteConnect  # type: ignore
except Exception:  # pragma: no cover
    KiteConnect = None  # type: ignore


# ---------------- Logging ----------------
class RingBufferLogHandler(logging.Handler):
    """Keep a rolling buffer of recent logs for /logs."""
    def __init__(self, capacity: int = 2000) -> None:
        super().__init__()
        self.capacity = int(capacity)
        self.buf: Deque[str] = deque(maxlen=self.capacity)
        self._fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.buf.append(self._fmt.format(record))
        except Exception:
            pass

    def tail(self, n: int) -> List[str]:
        n = max(1, min(int(n), self.capacity))
        return list(self.buf)[-n:]


def _setup_logging() -> RingBufferLogHandler:
    root = logging.getLogger()
    if not root.handlers:
        root.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        root.addHandler(ch)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    rb = RingBufferLogHandler(capacity=2000)
    root.addHandler(rb)
    return rb


def _now_ist_str() -> str:
    now_ist = datetime.now(timezone(timedelta(hours=5, minutes=30)))
    return now_ist.strftime("%Y-%m-%d %H:%M:%S")


# ---------------- Application ----------------
class Application:
    def __init__(self) -> None:
        self._log_handler = _setup_logging()
        self.log = logging.getLogger("main")

        self._stop = threading.Event()
        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)

        self._start_ts = time.time()

        # Broker (live if enabled and KiteConnect available)
        self.kite = self._setup_broker()

        # Core runner
        self.runner = StrategyRunner(kite=self.kite)

        # Telegram (wired after runner so providers exist)
        self.telegram: Optional[TelegramController] = None
        self._wire_telegram()

        # Health server
        self._start_health()

    # ---------- broker ----------
    def _setup_broker(self):
        if settings.enable_live_trading and KiteConnect is not None:
            try:
                kc = KiteConnect(api_key=settings.zerodha.api_key)
                kc.set_access_token(settings.zerodha.access_token)
                logging.info("KiteConnect initialized (live mode)")
                return kc
            except Exception as e:
                logging.warning(f"Kite init failed; continuing in paper mode: {e}")
        else:
            logging.info("Live trading disabled or KiteConnect missing; paper mode")
        return None

    # ---------- health server ----------
    def _start_health(self) -> None:
        if settings.health.enable_server:
            start_health_server(status_provider=self.runner.get_status_snapshot)

    # ---------- status snapshot for /status fallback ----------
    def _status_payload(self) -> Dict[str, Any]:
        try:
            r = self.runner.get_status_snapshot()
        except Exception:
            r = {}
        return {
            "time_ist": _now_ist_str(),
            "live_trading": bool(settings.enable_live_trading),
            "paused": bool(r.get("paused", False)),
            "broker": "Kite" if self.kite is not None else "Paper",
            "active_orders": r.get("active_orders", 0),
            "uptime_sec": int(time.time() - self._start_ts),
        }

    # ---------- telegram wiring ----------
    def _wire_telegram(self) -> None:
        tg = settings.telegram
        if not (tg.enabled and tg.bot_token and tg.chat_id):
            self.log.info("Telegram not started (disabled or credentials missing).")
            return

        # Providers (what the controller can ask the runner)
        providers = dict(
            status_provider=self.runner.get_status_snapshot,
            positions_provider=getattr(self.runner.executor, "get_positions_kite", None),
            actives_provider=getattr(self.runner.executor, "get_active_orders", None),
            diag_provider=self._diag_provider_wrapper,
            logs_provider=self._log_handler.tail,
            last_signal_provider=self.runner.get_last_signal_debug,
            flow_provider=self.runner.get_last_flow_debug,
            equity_provider=self.runner.get_equity_snapshot,
            config_provider=self._config_subset,
            sizing_test_provider=self._sizing_test,
        )

        # Controls (what the controller can command)
        self.telegram = TelegramController(
            **providers,
            runner_pause=self.runner.pause,
            runner_resume=self.runner.resume,
            runner_tick=self.runner.runner_tick,     # supports dry=True
            cancel_all=getattr(self.runner.executor, "cancel_all_orders", None),
            set_live_mode=self.runner.set_live_mode,
            # Execution mutators (fall back to settings if runner hooks absent)
            set_risk_pct=self._set_risk_pct,
            toggle_trailing=self._toggle_trailing,
            set_trailing_mult=self._set_trailing_mult,
            toggle_partial=self._toggle_partial,
            set_tp1_ratio=self._set_tp1_ratio,
            set_breakeven_ticks=self._set_breakeven_ticks,
            # Strategy mutators
            set_min_score=self._set_min_score,
            set_conf_threshold=self._set_conf_threshold,
            set_atr_period=self._set_atr_period,
            set_sl_mult=self._set_sl_mult,
            set_tp_mult=self._set_tp_mult,
            set_trend_boosts=self._set_trend_boosts,
            set_range_tighten=self._set_range_tighten,
        )
        self.telegram.start_polling()
        self.telegram.send_startup_alert()

    # ---------- provider wrappers ----------
    def _diag_provider_wrapper(self) -> Dict[str, Any]:
        # Compact health/flow summary for /diag and detailed for /check
        flow = self.runner.get_last_flow_debug() or {}
        checks = []

        # basic gates
        checks.append({"name": "within_window", "ok": bool(flow.get("within_window", False))})
        checks.append({"name": "paused", "ok": not bool(flow.get("paused", False))})
        checks.append({"name": "data_ok", "ok": bool(flow.get("data_ok", False))})
        checks.append({"name": "signal_ok", "ok": bool(flow.get("signal_ok", False))})

        # risk gates
        rg = flow.get("risk_gates", {}) or {}
        for k in ("equity_floor", "daily_drawdown", "loss_streak", "trades_per_day", "sl_valid"):
            if k in rg:
                checks.append({"name": f"risk:{k}", "ok": bool(rg[k])})

        # sizing
        qty = int(flow.get("qty", 0))
        checks.append({"name": "sizing_qty>0", "ok": qty > 0})

        ok = all(c.get("ok", False) for c in checks if not c["name"].startswith("paused"))
        return {"ok": ok, "checks": checks, "last_signal": self.runner.get_last_signal_debug()}

    def _config_subset(self) -> Dict[str, Any]:
        return {
            "live": settings.enable_live_trading,
            "log_level": settings.log_level,
            "time_window": [settings.data.time_filter_start, settings.data.time_filter_end],
            "lookback_minutes": settings.data.lookback_minutes,
            "use_live_equity": settings.risk.use_live_equity,
            "equity_floor": settings.risk.min_equity_floor,
            "risk_per_trade": settings.risk.risk_per_trade,
            "dd_pct": settings.risk.max_daily_drawdown_pct,
            "max_trades": settings.risk.max_trades_per_day,
            "min_bars": settings.strategy.min_bars_for_signal,
            "min_score": settings.strategy.min_signal_score,
            "conf_threshold": settings.strategy.confidence_threshold,
            "rr_min": getattr(settings.strategy, "rr_min", None),
            "lot_size": settings.instruments.nifty_lot_size,
            "entry_type": settings.executor.entry_order_type,
            "tick_size": settings.executor.tick_size,
            "slippage_ticks": settings.executor.slippage_ticks,
        }

    def _sizing_test(self, entry: float, sl: float) -> Dict[str, Any]:
        qty_lots = self.runner.sizing_test(entry, sl)  # returns {"qty": int, "diag": {...}}
        return qty_lots

    # ---------- mutators passed to Telegram ----------
    def _set_risk_pct(self, pct_fraction: float) -> None:
        # pct_fraction is already a fraction (0.01 = 1%) from controller
        settings.risk.risk_per_trade = float(pct_fraction)

    def _toggle_trailing(self, v: bool) -> None:
        settings.executor.enable_trailing = bool(v)

    def _set_trailing_mult(self, x: float) -> None:
        settings.executor.trailing_atr_multiplier = float(x)

    def _toggle_partial(self, v: bool) -> None:
        settings.executor.partial_tp_enable = bool(v)

    def _set_tp1_ratio(self, r: float) -> None:
        # controller passes 0..1
        settings.executor.tp1_qty_ratio = float(r)

    def _set_breakeven_ticks(self, n: int) -> None:
        settings.executor.breakeven_ticks = int(n)

    def _set_min_score(self, n: int) -> None:
        settings.strategy.min_signal_score = int(n)

    def _set_conf_threshold(self, x: float) -> None:
        settings.strategy.confidence_threshold = float(x)

    def _set_atr_period(self, n: int) -> None:
        settings.strategy.atr_period = int(n)

    def _set_sl_mult(self, x: float) -> None:
        settings.strategy.atr_sl_multiplier = float(x)

    def _set_tp_mult(self, x: float) -> None:
        settings.strategy.atr_tp_multiplier = float(x)

    def _set_trend_boosts(self, tp_boost: float, sl_relax: float) -> None:
        settings.strategy.trend_tp_boost = float(tp_boost)
        settings.strategy.trend_sl_relax = float(sl_relax)

    def _set_range_tighten(self, tp_tight: float, sl_tight: float) -> None:
        settings.strategy.range_tp_tighten = float(tp_tight)
        settings.strategy.range_sl_tighten = float(sl_tight)

    # ---------- signal handling ----------
    def _on_signal(self, signum: int, frame: Any) -> None:  # noqa: ARG002
        self.log.info("Signal %s received, shutting down…", signum)
        self._stop.set()

    # ---------- main loop ----------
    def start(self) -> None:
        self.log.info(
            "Starting Nifty Scalper Bot | live_trading=%s",
            "true" if settings.enable_live_trading else "false",
        )

        heartbeat_every = 300.0
        last_hb = 0.0

        while not self._stop.is_set():
            try:
                # If you have a websocket, it should call runner.process_tick(...) from its own thread.
                # This loop ensures periodic processing still happens in paper mode.
                self.runner.process_tick(tick=None)
                self.runner.health_check()
            except Exception as e:
                logging.getLogger(__name__).warning(f"main loop warn: {e}")

            now = time.time()
            if now - last_hb > heartbeat_every:
                s = self._status_payload()
                self.log.info(
                    "⏱ heartbeat | live=%d paused=%s active=%d",
                    1 if s.get("live_trading") else 0,
                    s.get("paused"),
                    s.get("active_orders", 0),
                )
                last_hb = now

            if self._stop.wait(timeout=1.5):
                break

        # Shutdown
        try:
            if self.telegram:
                self.telegram.stop_polling()
        except Exception:
            pass
        try:
            self.runner.shutdown()
        except Exception:
            pass
        self.log.info("Nifty Scalper Bot stopped.")


if __name__ == "__main__":
    Application().start()