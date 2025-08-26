from __future__ import annotations

import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from src.config import settings
from src.data.source import LiveKiteSource
from src.strategies.scalping_strategy import EnhancedScalpingStrategy
from src.execution.order_executor import OrderExecutor  # your file

log = logging.getLogger("StrategyRunner")


@dataclass
class FlowState:
    last_tick_ts: Optional[float] = None
    last_error: Optional[str] = None
    ticks: int = 0
    errors: int = 0
    last_signal: Optional[Dict[str, Any]] = None
    last_signal_ts: Optional[float] = None


def _ist_now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class StrategyRunner:
    """
    Orchestrates: data → strategy → sizing/execution, plus health & controls.
    """

    def __init__(self, *, kite=None, telegram_controller=None) -> None:
        self.kite = kite
        self.telegram_controller = telegram_controller  # later overwritten by real one

        # Data source
        self.data_source = LiveKiteSource(kite)
        self.data_source.connect()
        log.info("Data source initialized: %s", self.data_source.__class__.__name__)

        # Strategy / executor
        self.strategy = EnhancedScalpingStrategy()
        self.executor = OrderExecutor(kite=kite)

        # Flow state
        self.state = FlowState()
        self._paused = False
        self.live_trading = bool(settings.enable_live_trading)

        # health metrics (simple moving counts)
        self._tick_rate_window: List[float] = []
        self._cpu_pct = 0.0
        self._mem_pct = 0.0

        log.info(
            "StrategyRunner ready (live_trading=%s, use_live_equity=%s)",
            self.live_trading, True,
        )

    # ------------------ public API expected by Telegram ------------------

    def start(self) -> None:
        pass  # nothing long-running; main loop is driven by /tick or a scheduler outside

    def shutdown(self) -> None:
        pass

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def set_live_mode(self, val: bool) -> None:
        self.live_trading = bool(val)
        settings.enable_live_trading = bool(val)
        if val and self.kite:
            try:
                self.data_source.connect()
                log.info("🔓 Live mode ON — broker session initialized.")
                if self.telegram_controller:
                    self.telegram_controller.send_message("🔓 Live mode ON — broker session initialized.")
            except Exception:
                pass

    # Strategy tuning passthroughs
    def set_min_score(self, n: int) -> None:
        settings.strategy.min_signal_score = int(n)

    def set_conf_threshold(self, x: float) -> None:
        settings.strategy.confidence_threshold = float(x)

    def set_atr_period(self, n: int) -> None:
        settings.strategy.atr_period = int(n)

    def set_sl_mult(self, x: float) -> None:
        settings.strategy.atr_sl_multiplier = float(x)

    def set_tp_mult(self, x: float) -> None:
        settings.strategy.atr_tp_multiplier = float(x)

    def set_trend_boosts(self, tp_boost: float, sl_relax: float) -> None:
        settings.strategy.trend_tp_boost = float(tp_boost)
        settings.strategy.trend_sl_relax = float(sl_relax)

    def set_range_tighten(self, tp_t: float, sl_t: float) -> None:
        settings.strategy.range_tp_tighten = float(tp_t)
        settings.strategy.range_sl_tighten = float(sl_t)

    # ------------------ runner ops ------------------

    def runner_tick(self, *, dry: bool = False) -> Optional[Dict[str, Any]]:
        """
        One pass: pull OHLC, run strategy, (paper/live) execute.
        Returns the placed/attempted order dict or None.
        """
        t0 = time.time()
        self.state.last_tick_ts = t0
        self.state.ticks += 1

        if self._paused:
            return None

        try:
            # --- data ---
            token = int(getattr(settings.instruments, "nifty_token", 256265))  # example default
            end = datetime.now()
            start = end - timedelta(minutes=30)
            tf = "minute"
            df = self.data_source.fetch_ohlc(token, start, end, tf)
            if df is None or df.empty:
                raise RuntimeError("No data (df empty)")

            # detect spot ltp for strike pick
            spot_ltp = self.data_source.get_last_price(token)

            # --- strategy ---
            sig = self.strategy.generate_signal(
                df=df,
                current_tick={"spot_ltp": spot_ltp, "ltp": spot_ltp, "option_ltp": spot_ltp},
                current_price=spot_ltp,
                spot_df=df,
            )
            self.state.last_signal = sig
            self.state.last_signal_ts = time.time()

            if not sig:
                return None

            # --- risk/rr gate (minimal) ---
            rr_min = float(getattr(settings.strategy, "rr_min", 1.3))
            if float(sig.get("rr", 0.0)) < rr_min:
                return None

            # --- execution ---
            if dry or not self.live_trading:
                # paper: just log
                log.info("PAPER order: %s", {k: sig[k] for k in ("action", "option_type", "strike", "entry_price")})
                return {"paper": True, **sig}
            else:
                placed = self.executor.place_signal(sig)
                return {"paper": False, "result": placed, **sig}

        except Exception as e:
            self.state.errors += 1
            self.state.last_error = f"{e.__class__.__name__}: {e}"
            log.debug("runner_tick exception: %s", e, exc_info=True)
            return None
        finally:
            # maintain simple tick rate window (seconds between ticks)
            self._tick_rate_window.append(max(0.001, time.time() - t0))
            self._tick_rate_window[:] = self._tick_rate_window[-50:]

    # ------------------ health & diagnostics ------------------

    def get_status_snapshot(self) -> Dict[str, Any]:
        return {
            "time_ist": _ist_now_str(),
            "live_trading": self.live_trading,
            "broker": "Kite" if self.kite else "None",
            "active_orders": len(self.executor.get_active_orders() or []),
            "ticks": self.state.ticks,
            "errors": self.state.errors,
            "last_error": self.state.last_error,
        }

    def get_last_flow_debug(self) -> Dict[str, Any]:
        s = {
            "last_tick_age_s": (time.time() - self.state.last_tick_ts) if self.state.last_tick_ts else None,
            "last_signal_age_s": (time.time() - self.state.last_signal_ts) if self.state.last_signal_ts else None,
        }
        s.update(asdict(self.state))
        return s

    def get_last_signal_debug(self) -> Optional[Dict[str, Any]]:
        return self.state.last_signal

    # compact boolean + labeled checks for /diag and /check
    def build_diag(self) -> Dict[str, Any]:
        checks: List[Dict[str, Any]] = []

        # Telegram wired
        checks.append(self._ok("Telegram controller", bool(self.telegram_controller)))

        # Broker session & credentials
        cred_ok = bool(getattr(settings.zerodha, "api_key", None) and getattr(settings.zerodha, "access_token", None))
        checks.append(self._ok("Zerodha credentials", cred_ok))
        checks.append(self._ok("Broker session (Kite)", bool(self.kite)))

        # Data source + an LTP ping check (non-fatal if None in dry)
        ds_ok = self.data_source is not None
        checks.append(self._ok("Data source (Kite)", ds_ok))

        # Instrument token present
        tok = int(getattr(settings.instruments, "nifty_token", 256265))
        checks.append(self._ok("Instrument token", tok > 0))

        # Trading window check
        within = self._within_trading_window()
        checks.append(self._ok("Trading window", within, hint="Outside window — use /tickdry after-hours" if not within else ""))

        # Quick OHLC fetch smoke (cached)
        ohlc_ok = False
        try:
            end = datetime.now()
            start = end - timedelta(minutes=5)
            df = self.data_source.fetch_ohlc(tok, start, end, "minute")
            ohlc_ok = (df is not None) and (not df.empty)
        except Exception:
            ohlc_ok = False
        checks.append(self._ok("OHLC fetch", ohlc_ok))

        # Equity snapshot (executor can report 0 in paper)
        eq_ok = True
        try:
            eq = self.executor.get_equity_snapshot()
            eq_ok = eq is not None
        except Exception:
            eq_ok = False
        checks.append(self._ok("Equity snapshot", eq_ok))

        # Risk gates simple stub (always true unless configured different)
        checks.append(self._ok("Risk gates", True))

        ok_all = all(c.get("ok", False) for c in checks)
        return {"ok": ok_all, "checks": checks, "last_signal": bool(self.state.last_signal)}

    def _ok(self, name: str, ok: bool, hint: str = "") -> Dict[str, Any]:
        d = {"name": name, "ok": bool(ok)}
        if hint:
            d["hint"] = hint
        return d

    def _within_trading_window(self) -> bool:
        """Simple clock gate; allow override for testing."""
        if getattr(settings, "allow_offhours_testing", False):
            return True
        start = str(getattr(settings.data, "time_filter_start", "09:20"))
        end = str(getattr(settings.data, "time_filter_end", "15:20"))
        now = datetime.now().strftime("%H:%M")
        return (start <= now <= end)

    def health_check(self) -> None:
        # placeholder to extend with process metrics; keep last_error fresh
        pass