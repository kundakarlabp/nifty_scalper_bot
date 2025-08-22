# src/main.py
from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any, Deque, Dict, List, Optional

from src.config import settings
from src.server import health as health_server
from src.strategies.runner import StrategyRunner
from src.notifications.telegram_controller import TelegramController

# Optional broker SDK (kept safe)
try:
    from kiteconnect import KiteConnect  # type: ignore
    from kiteconnect.exceptions import NetworkException, TokenException, InputException  # type: ignore
except Exception:  # pragma: no cover
    KiteConnect = None  # type: ignore
    NetworkException = TokenException = InputException = Exception  # fallbacks


# ---------------- Logging ----------------
class RingBufferLogHandler(logging.Handler):
    """Lightweight in-memory log buffer for /logs on demand."""
    def __init__(self, capacity: int = 1000) -> None:
        super().__init__()
        self.capacity = int(capacity)
        self.buf: Deque[str] = deque(maxlen=self.capacity)
        self._fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self._fmt.format(record)
            self.buf.append(line)
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


def _now_ist_naive() -> datetime:
    ist = datetime.now(timezone(timedelta(hours=5, minutes=30)))
    return ist.replace(tzinfo=None)


# ---------------- Application ----------------
class Application:
    def __init__(self) -> None:
        self._log_handler = _setup_logging()
        self.log = logging.getLogger(__name__)

        # live toggle mirrors settings
        self.live_trading = bool(settings.enable_live_trading)

        # graceful shutdown
        self._stop_event = threading.Event()
        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)

        # start time
        self._start_ts = time.time()

        # runner with event sink -> telegram notifications
        self._last_signal: Optional[Dict[str, Any]] = None
        self.runner = StrategyRunner(event_sink=self._on_event)

        # telegram (wired after runner so hooks can be passed)
        self.tg: Optional[TelegramController] = None

    # ---------- status / payload ----------
    def _status_payload(self) -> Dict[str, Any]:
        r: Dict[str, Any] = {}
        try:
            r = self.runner.to_status_dict()
        except Exception:
            r = {}
        return {
            "time_ist": _now_ist_naive().isoformat(sep=" ", timespec="seconds"),
            "live_trading": bool(self.live_trading),
            "paused": bool(r.get("paused")),
            "broker": r.get("broker") or ("Kite" if KiteConnect else "none"),
            "data_source": r.get("data_source"),
            "active_orders": r.get("active_orders", 0),
            "uptime_sec": int(time.time() - self._start_ts),
        }

    # ---------- health server ----------
    def _start_health(self) -> None:
        # Safe defaults if server config is absent
        srv = getattr(settings, "server", None)
        host = getattr(srv, "host", "0.0.0.0")
        port = int(getattr(srv, "port", 8000))
        t = threading.Thread(
            target=health_server.run,
            kwargs={"callback": self._status_payload, "host": host, "port": port},
            daemon=True,
            name="health-http",
        )
        t.start()

    # ---------- telegram wiring ----------
    def _wire_telegram(self) -> None:
        tg = getattr(settings, "telegram", None)
        enabled = bool(getattr(tg, "enabled", True)) if tg is not None else False
        bot_token = getattr(tg, "bot_token", None) if tg is not None else None
        chat_id = getattr(tg, "chat_id", None) if tg is not None else None

        if not (enabled and bot_token and chat_id):
            self.log.info("Telegram not started (disabled or credentials missing).")
            return

        # Providers
        def positions_provider() -> Dict[str, Any]:
            ex = getattr(self.runner, "executor", None)
            return ex.get_positions_kite() if ex else {}

        def actives_provider() -> List[Any]:
            ex = getattr(self.runner, "executor", None)
            return ex.get_active_orders() if ex else []

        def diag_provider() -> Dict[str, Any]:
            try:
                return self.runner.diagnose()
            except Exception:
                return {"ok": False, "error": "diagnose() not available"}

        def logs_provider(n: int) -> List[str]:
            return self._log_handler.tail(n)

        def last_signal_provider() -> Optional[Dict[str, Any]]:
            return self._last_signal

        # Controls / mutators
        def runner_pause() -> None:
            try:
                self.runner.pause()
            except Exception:
                pass

        def runner_resume() -> None:
            try:
                self.runner.resume()
            except Exception:
                pass

        def runner_tick() -> Optional[Dict[str, Any]]:
            # one synchronous tick using a temporary event
            stop_evt = threading.Event()
            try:
                res = self.runner.run_once(stop_evt)
                if res:
                    self._last_signal = res
                return res
            except Exception as e:
                self.log.exception("Manual tick error: %s", e)
                return None

        def cancel_all() -> None:
            ex = getattr(self.runner, "executor", None)
            if ex:
                try:
                    ex.cancel_all_orders()
                except Exception:
                    pass

        # Execution mutators
        def set_risk_pct(pct: float) -> None:
            settings.risk.risk_per_trade = float(pct) / 100.0

        def toggle_trailing(v: bool) -> None:
            settings.executor.enable_trailing = bool(v)

        def set_trailing_mult(x: float) -> None:
            settings.executor.trailing_atr_multiplier = float(x)

        def toggle_partial(v: bool) -> None:
            settings.executor.partial_tp_enable = bool(v)

        def set_tp1_ratio(pct: float) -> None:
            settings.executor.tp1_qty_ratio = float(pct) / 100.0

        def set_breakeven_ticks(ticks: int) -> None:
            settings.executor.breakeven_ticks = int(ticks)

        def set_live_mode(v: bool) -> None:
            # update all three: app flag, settings, runner
            self.live_trading = bool(v)
            settings.enable_live_trading = bool(v)
            try:
                # keep runner in sync with /mode changes
                self.runner._live = bool(v)  # minimal, avoids full re-init
            except Exception:
                pass
            self.log.info("Live mode set to %s.", "True" if v else "False")

        # Strategy mutators
        def set_min_score(n: int) -> None:
            settings.strategy.min_signal_score = int(n)

        def set_conf_threshold(x: float) -> None:
            settings.strategy.confidence_threshold = float(x)

        def set_atr_period(n: int) -> None:
            settings.strategy.atr_period = int(n)

        def set_sl_mult(x: float) -> None:
            settings.strategy.atr_sl_multiplier = float(x)

        def set_tp_mult(x: float) -> None:
            settings.strategy.atr_tp_multiplier = float(x)

        def set_trend_boosts(tp_boost: float, sl_relax: float) -> None:
            settings.strategy.trend_tp_boost = float(tp_boost)
            settings.strategy.trend_sl_relax = float(sl_relax)

        def set_range_tighten(tp_t: float, sl_t: float) -> None:
            settings.strategy.range_tp_tighten = float(tp_t)
            settings.strategy.range_sl_tighten = float(sl_t)

        # Instantiate controller
        self.tg = TelegramController(
            status_provider=self._status_payload,
            positions_provider=positions_provider,
            actives_provider=actives_provider,
            diag_provider=diag_provider,
            logs_provider=logs_provider,
            last_signal_provider=last_signal_provider,
            runner_pause=runner_pause,
            runner_resume=runner_resume,
            runner_tick=runner_tick,
            cancel_all=cancel_all,
            set_risk_pct=set_risk_pct,
            toggle_trailing=toggle_trailing,
            set_trailing_mult=set_trailing_mult,
            toggle_partial=toggle_partial,
            set_tp1_ratio=set_tp1_ratio,
            set_breakeven_ticks=set_breakeven_ticks,
            set_live_mode=set_live_mode,
            set_min_score=set_min_score,
            set_conf_threshold=set_conf_threshold,
            set_atr_period=set_atr_period,
            set_sl_mult=set_sl_mult,
            set_tp_mult=set_tp_mult,
            set_trend_boosts=set_trend_boosts,
            set_range_tighten=set_range_tighten,
        )
        self.tg.start_polling()
        self.tg.send_startup_alert()

    # ---------- runner event sink -> telegram ----------
    def _on_event(self, evt: Dict[str, Any]) -> None:
        try:
            et = (evt or {}).get("type")
            if et == "ENTRY_PLACED" and self.tg:
                self.tg.notify_entry(
                    symbol=str(evt.get("symbol")),
                    side=str(evt.get("side")),
                    qty=int(evt.get("qty", 0)),
                    price=float(evt.get("price", 0.0)),
                    record_id=str(evt.get("record_id")),
                )
            elif et in ("FILL", "FILLS") and self.tg:  # accept both singular/plural
                fills = evt.get("fills") or []
                if fills:
                    self.tg.notify_fills(fills)

            # cache last signal-ish payloads
            if et in ("ENTRY_PLACED", "SIGNAL", "FILL", "FILLS"):
                self._last_signal = evt
        except Exception:
            pass

    # ---------- signals/shutdown ----------
    def _on_signal(self, signum: int, frame: Any) -> None:
        self.log.info("Signal %s received, shutting down…", signum)
        self._stop_event.set()

    # ---------- main loop ----------
    def run(self) -> None:
        self.log.info("Starting Nifty Scalper Bot | live_trading=%s", self.live_trading)
        self._start_health()
        self._wire_telegram()

        cadence = 0.5
        hb_every = 60.0  # heartbeat log every ~1 minute
        last_hb = 0.0

        while not self._stop_event.is_set():
            try:
                res = self.runner.run_once(self._stop_event)
                if res:
                    self._last_signal = res
            except (NetworkException, TokenException, InputException) as e:
                logging.getLogger(__name__).error("Transient broker error: %s", e)
            except Exception as e:
                logging.getLogger(__name__).exception("Main loop error: %s", e)

            now = time.time()
            if now - last_hb > hb_every:
                s = self._status_payload()
                self.log.info(
                    "⏱ heartbeat | live=%d paused=%s active=%d src=%s",
                    1 if s.get("live_trading") else 0,
                    s.get("paused"),
                    s.get("active_orders", 0),
                    s.get("data_source"),
                )
                last_hb = now

            if self._stop_event.wait(timeout=cadence):
                break

        if self.tg:
            try:
                self.tg.stop_polling()
            except Exception:
                pass
        self.log.info("Nifty Scalper Bot stopped.")


# ---------------- CLI ----------------
def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="nifty_scalper_bot")
    sub = p.add_subparsers(dest="cmd", required=False)
    sub.add_parser("start", help="Start trading loop (default)")
    sub.add_parser("backtest", help="Run backtest from CSV file")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    cmd = args.cmd or "start"

    if cmd == "start":
        app = Application()
        app.run()
    elif cmd == "backtest":
        logging.getLogger(__name__).info("Backtest command not yet implemented.")
    return 0


if __name__ == "__main__":
    sys.exit(main())