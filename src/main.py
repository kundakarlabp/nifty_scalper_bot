# Path: src/main.py
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

import numpy as np
import pandas as pd

from src.config import settings
from src.server import health as health_server
from src.strategies.runner import StrategyRunner
from src.notifications.telegram_controller import TelegramController

# strategy + sizing used by mock tick
from src.strategies.scalping_strategy import EnhancedScalpingStrategy
from src.risk.position_sizing import PositionSizer

# Optional broker SDK
try:
    from kiteconnect import KiteConnect  # type: ignore
    from kiteconnect.exceptions import NetworkException, TokenException, InputException  # type: ignore
except Exception:  # fallback if kiteconnect not installed
    KiteConnect = None  # type: ignore
    NetworkException = TokenException = InputException = Exception


# ---------------- Logging ----------------
class RingBufferLogHandler(logging.Handler):
    def __init__(self, capacity: int = 1000) -> None:
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
        level = getattr(logging, str(getattr(settings, "log_level", "INFO")).upper(), logging.INFO)
        root.setLevel(level)
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

        self.live_trading = bool(getattr(settings, "enable_live_trading", False))
        self._stop_event = threading.Event()
        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)

        self._start_ts = time.time()
        self._last_signal: Optional[Dict[str, Any]] = None

        # Runner (broker is optional; paper mode works with kite=None)
        self.runner = StrategyRunner(kite=None, telegram_controller=None)

        # Telegram controller instance (wired after settings check)
        self.tg: Optional[TelegramController] = None

    # ---------- status payload ----------
    def _status_payload(self) -> Dict[str, Any]:
        try:
            r = self.runner.to_status_dict()  # runner may expose this
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
        host = "0.0.0.0"
        port = 8000
        t = threading.Thread(
            target=health_server.run,
            kwargs={"callback": self._status_payload, "host": host, "port": port},
            daemon=True,
            name="health-http",
        )
        t.start()

    # ---------- telegram wiring ----------
    def _wire_telegram(self) -> None:
        # Nested telegram settings (env: TELEGRAM__ENABLED, TELEGRAM__BOT_TOKEN, TELEGRAM__CHAT_ID)
        tg_cfg = getattr(settings, "telegram", object())
        tg_enabled = bool(getattr(tg_cfg, "enabled", False))
        bot_token = getattr(tg_cfg, "bot_token", None)
        chat_id = getattr(tg_cfg, "chat_id", None)

        if not (tg_enabled and bot_token and chat_id):
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

        # Controls
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

        def runner_tick(dry: bool = False) -> Optional[Dict[str, Any]]:
            """
            When dry=True, we won't place orders; runner should honor settings.enable_live_trading False.
            """
            stop_evt = threading.Event()
            try:
                res = self.runner.run_once(stop_evt)
                if res:
                    self._last_signal = res
                return res
            except Exception as e:
                self.log.exception("Manual tick error: %s", e)
                return None

        # --- MOCK tick (off-hours validation) ---
        def runner_tick_mock() -> Optional[Dict[str, Any]]:
            try:
                return self._mock_tick_once()
            except Exception as e:
                self.log.exception("Mock tick error: %s", e)
                return {"error": str(e)}

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
            self.live_trading = bool(v)
            settings.enable_live_trading = bool(v)
            try:
                self.runner._live = bool(v)
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
            cancel_all=lambda: getattr(getattr(self.runner, "executor", None), "cancel_all_orders", lambda: None)(),
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

    # ---------- MOCK tick: generate synthetic candles & run strategy/sizing ----------
    def _mock_tick_once(self) -> Dict[str, Any]:
        # Generate last 60 minutes of synthetic 1‑min data
        n = 60
        idx = pd.date_range(end=_now_ist_naive(), periods=n, freq="1min")
        base = 500.0
        steps = np.random.normal(0, 0.8, size=n).cumsum()
        close = base + steps
        high = close + np.random.uniform(0.2, 1.0, size=n)
        low = close - np.random.uniform(0.2, 1.0, size=n)
        open_ = close + np.random.uniform(-0.5, 0.5, size=n)
        vol = np.random.randint(100, 1000, size=n)

        option_df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)
        spot_df = option_df.copy()  # simple proxy

        # Run the strategy just like the runner would
        strat = EnhancedScalpingStrategy()
        current_price = float(option_df["close"].iloc[-1])
        sig = strat.generate_signal(option_df, current_price, spot_df)
        if not sig:
            return {"signal_ok": False}

        # Sizing (use PositionSizer directly)
        entry_price = float(sig["entry_price"])
        stop_loss = float(sig["stop_loss"])
        lot_size = int(getattr(getattr(settings, "instruments", object()), "nifty_lot_size", 75))
        equity = float(getattr(settings, "risk_default_equity", 30000.0))
        risk_per_trade = float(getattr(getattr(settings, "risk", object()), "risk_per_trade", 0.01))
        min_lots = int(getattr(getattr(settings, "instruments", object()), "min_lots", 1))
        max_lots = int(getattr(getattr(settings, "instruments", object()), "max_lots", 10))
        max_pos_pct = float(getattr(getattr(settings, "risk", object()), "max_position_size_pct", 0.10))

        sizer = PositionSizer(
            risk_per_trade=risk_per_trade,
            min_lots=min_lots,
            max_lots=max_lots,
            max_position_size_pct=max_pos_pct,
        )
        qty, lots, _ = sizer.size_from_signal(
            entry_price=entry_price, stop_loss=stop_loss, lot_size=lot_size, equity=equity
        )
        return {
            "signal_ok": True,
            "lots": int(lots),
            "qty": int(qty),
            "side": sig["side"],
            "sl_points": float(sig["sl_points"]),
            "tp_points": float(sig["tp_points"]),
            "confidence": float(sig["confidence"]),
            "score": int(sig["score"]),
        }

    # ---------- runner events ----------
    def _on_event(self, evt: Dict[str, Any]) -> None:
        if not evt:
            return
        try:
            et = evt.get("type")
            if et == "ENTRY_PLACED" and self.tg:
                self.tg.notify_entry(
                    symbol=str(evt.get("symbol")),
                    side=str(evt.get("side")),
                    qty=int(evt.get("qty", 0)),
                    price=float(evt.get("price", 0.0)),
                    record_id=str(evt.get("record_id")),
                )
            elif et in ("FILL", "FILLS") and self.tg:
                fills = evt.get("fills") or []
                if fills:
                    self.tg.notify_fills(fills)
            if et in ("ENTRY_PLACED", "SIGNAL", "FILL", "FILLS"):
                self._last_signal = evt
        except Exception:
            pass

    def _on_signal(self, signum: int, frame: Any) -> None:
        self.log.info("Signal %s received, shutting down…", signum)
        self._stop_event.set()

    # ---------- main loop ----------
    def run(self) -> None:
        self.log.info("Starting Nifty Scalper Bot | live_trading=%s", self.live_trading)
        self._start_health()
        self._wire_telegram()

        cadence = 0.5
        hb_every = 300.0
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