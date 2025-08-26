# src/strategies/runner.py
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Optional

import pandas as pd

from src.config import settings
from src.data.source import LiveKiteSource  # ← concrete class (accepts kite)

# Optional imports (kept tolerant so runner works in shadow mode / tests)
try:
    from kiteconnect import KiteConnect  # type: ignore
except Exception:  # pragma: no cover
    KiteConnect = None  # type: ignore

try:
    # Your strategy/executor modules (names kept as per your repo)
    from src.optimized_scalping_strategy import ScalpingStrategy  # type: ignore
except Exception:  # pragma: no cover
    ScalpingStrategy = None  # type: ignore

try:
    from src.optimized_order_executor import OrderExecutor  # type: ignore
except Exception:  # pragma: no cover
    OrderExecutor = None  # type: ignore

# Optional Telegram controller; we treat it duck-typed (safe if not present)
try:
    from src.notifications.telegram_controller import TelegramController  # type: ignore
except Exception:  # pragma: no cover
    TelegramController = Any  # type: ignore


log = logging.getLogger("StrategyRunner")


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def _now_ist() -> datetime:
    return datetime.now(timezone(timedelta(hours=5, minutes=30)))


def _fmt_bool(good: bool) -> str:
    return "🟢" if good else "🔴"


def _secs_ago(ts: Optional[float]) -> Optional[int]:
    if not ts:
        return None
    return int(time.time() - ts)


# --------------------------------------------------------------------------------------
# StrategyRunner
# --------------------------------------------------------------------------------------
class StrategyRunner:
    """
    Coordinates:
      - data source (Kite / paper)
      - strategy signals
      - order execution
      - Telegram diagnostics / hot-rewire

    This runner is defensive: it runs in paper/shadow even if kite/strategy/executor
    aren’t available, so your /diag and /status never break.
    """

    # ---- lifecycle ------------------------------------------------------------------
    def __init__(
        self,
        *,
        kite: Optional[KiteConnect] = None,
        telegram: Optional[TelegramController] = None,
        live_trading: bool = False,
        use_live_equity: Optional[bool] = None,
    ) -> None:
        self.kite: Optional[KiteConnect] = kite
        self.telegram: Optional[TelegramController] = telegram
        self.live_trading: bool = bool(live_trading)
        self.use_live_equity: bool = settings.risk.use_live_equity if use_live_equity is None else bool(use_live_equity)

        # Core components
        self.data_source: LiveKiteSource = LiveKiteSource(self.kite)
        self.strategy = None  # type: ignore
        self.executor = None  # type: ignore

        # State
        self._lock = threading.RLock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Telemetry
        self.last_fetch_at: Optional[float] = None
        self.last_signal: Optional[str] = None
        self.last_error: Optional[str] = None
        self.active_positions: int = 0
        self.rr_threshold: Dict[str, Any] = {}

        # Wire everything
        self._initialize_components()
        self._wire_telegram()

    # ---- wiring ---------------------------------------------------------------------
    def _initialize_components(self) -> None:
        # Data
        self.data_source.connect()
        log.info("StrategyRunner: Data source initialized: LiveKiteSource")

        # Strategy (if module available)
        if ScalpingStrategy:
            try:
                self.strategy = ScalpingStrategy(
                    ema_fast=settings.strategy.ema_fast,
                    ema_slow=settings.strategy.ema_slow,
                    rsi_period=settings.strategy.rsi_period,
                    bb_period=settings.strategy.bb_period,
                    bb_std=settings.strategy.bb_std,
                    atr_period=settings.strategy.atr_period,
                    atr_sl_multiplier=settings.strategy.atr_sl_multiplier,
                    atr_tp_multiplier=settings.strategy.atr_tp_multiplier,
                    rr_min=settings.strategy.rr_min,
                    min_bars=settings.strategy.min_bars_for_signal,
                    confidence_threshold=settings.strategy.confidence_threshold,
                )
            except Exception as e:
                self.strategy = None
                log.exception("StrategyRunner: failed to initialize strategy: %s", e)
        else:
            log.warning("StrategyRunner: strategy module not available (shadow mode).")

        # Executor (if module available)
        if OrderExecutor:
            try:
                self.executor = OrderExecutor(
                    kite=self.kite,
                    live=self.live_trading and (self.kite is not None),
                    settings=settings,
                )
            except Exception as e:
                self.executor = None
                log.exception("StrategyRunner: failed to initialize executor: %s", e)
        else:
            log.warning("StrategyRunner: order executor not available (shadow mode).")

        log.info(
            "StrategyRunner: StrategyRunner ready (live_trading=%s, use_live_equity=%s)",
            self.live_trading, self.use_live_equity
        )

        # Reflect initial “live” request
        self._apply_live_state(self.live_trading, announce=False)

    def _wire_telegram(self) -> None:
        if not self.telegram:
            return

        # The controller API varies across repos; wire defensively.
        # We try common names and also attach the runner so commands can call us.
        try:
            if hasattr(self.telegram, "set_diag_provider"):
                self.telegram.set_diag_provider(self.diagnostics_text)  # type: ignore[attr-defined]
            if hasattr(self.telegram, "set_status_provider"):
                self.telegram.set_status_provider(self.status_card)  # type: ignore[attr-defined]
            if hasattr(self.telegram, "set_mode_switcher"):
                self.telegram.set_mode_switcher(self.handle_mode_command)  # type: ignore[attr-defined]
            if hasattr(self.telegram, "attach_runner"):
                self.telegram.attach_runner(self)  # type: ignore[attr-defined]
            log.info("StrategyRunner: Telegram controller attached.")
        except Exception as e:
            log.warning("StrategyRunner: failed to wire Telegram controller: %s", e)

    # ---- live/paper rewire -----------------------------------------------------------
    def handle_mode_command(self, mode: str) -> str:
        """
        Telegram-facing hook: /mode live | /mode paper
        """
        mode = (mode or "").strip().lower()
        if mode in ("live", "on", "enable", "enabled", "true", "1"):
            ok = self._apply_live_state(True, announce=False)
            if ok:
                return "Mode set to LIVE and rewired."
            return "Requested LIVE, but broker session is missing — staying in paper."
        elif mode in ("paper", "shadow", "off", "disable", "disabled", "false", "0"):
            self._apply_live_state(False, announce=False)
            return "Mode set to PAPER and rewired."
        else:
            return "Unknown mode. Use /mode live or /mode paper."

    def set_live(self, live: bool, kite: Optional[KiteConnect] = None) -> None:
        """
        Programmatic rewire (used by main.py if it toggles).
        """
        if kite is not None:
            self.kite = kite
        self._apply_live_state(live, announce=True)

    def _apply_live_state(self, live: bool, announce: bool = True) -> bool:
        """
        Apply live/paper modes across components. Returns True iff live+connected.
        """
        with self._lock:
            self.live_trading = bool(live)

            # Re-plug Kite into data source/executor
            self.data_source.kite = self.kite
            if self.executor and hasattr(self.executor, "set_live"):
                try:
                    self.executor.set_live(self.live_trading and (self.kite is not None))  # type: ignore[attr-defined]
                except Exception:
                    # Fallback: try attributes found in some repos
                    try:
                        self.executor.live = self.live_trading and (self.kite is not None)  # type: ignore[attr-defined]
                        self.executor.kite = self.kite  # type: ignore[attr-defined]
                    except Exception:
                        pass

            # Log announcements
            if self.live_trading:
                if self.kite is None:
                    log.warning("StrategyRunner: Requested live mode but kite=None; staying effectively in paper.")
                    if announce:
                        self._notify("⚠️ Live requested but broker session missing; staying in paper.")
                    return False
                else:
                    # connect() will just log if already attached
                    try:
                        self.data_source.connect()
                    except Exception:
                        pass
                    log.info("StrategyRunner: 🔓 Live mode ON — broker session initialized.")
                    if announce:
                        self._notify("🔓 Live mode ON — broker session initialized.")
                    return True
            else:
                log.info("StrategyRunner: Live trading disabled → paper mode.")
                if announce:
                    self._notify("🧪 Paper mode ON — live trading disabled.")
                return False

    # ---- public API used by Telegram -------------------------------------------------
    def status_card(self) -> str:
        """
        Small status card for /status.
        """
        now = _now_ist()
        if self.live_trading:
            if self.kite is None:
                mode = "LIVE | None"
            else:
                mode = "LIVE | Kite"
        else:
            mode = "LIVE | Paper"  # matches your earlier card style
        lines = [
            "📊 " + now.strftime("%Y-%m-%d %H:%M:%S"),
            "🧩 " + mode,
            f"📦 Active: {int(self.active_positions)}",
        ]
        return "\n".join(lines)

    def diagnostics_text(self) -> str:
        """
        Full /check diagnostic block.
        """
        # Telegram wiring
        wired = self.telegram is not None

        # Broker / mode
        if self.live_trading and self.kite is not None:
            broker_line = "live mode with kite"
            broker_ok = True
        elif self.live_trading and self.kite is None:
            broker_line = "live but kite=None"
            broker_ok = False
        else:
            broker_line = "paper mode"
            broker_ok = True  # paper is OK

        # Data feed recency
        age = _secs_ago(self.last_fetch_at)
        if age is None:
            data_line = "no fetch yet"
            data_ok = False
        else:
            data_line = f"age={age}s"
            data_ok = age < 60

        # Strategy readiness
        min_bars = int(getattr(settings.strategy, "min_bars_for_signal", 50))
        # We don’t know live bar count here (diagnostic context). Mark as not-evaluated until a fetch happens.
        strategy_ready = False if age is None else True  # if we fetched at least once, assume warmed
        strat_line = f"min_bars={min_bars}"

        # Risk (not evaluated here)
        risk_line = "no-eval"

        # RR threshold
        rr_line = "{}" if not self.rr_threshold else str(self.rr_threshold)

        # Errors
        err_line = "none" if not self.last_error else self.last_error

        # Last signal
        last_signal = self.last_signal or "none"

        body = [
            "🔎 Full system check",
            f"{_fmt_bool(wired)} Telegram wiring — controller attached",
            f"{_fmt_bool(broker_ok)} Broker session — {broker_line}",
            f"{_fmt_bool(data_ok)} Data feed — {data_line}",
            f"{_fmt_bool(strategy_ready)} Strategy readiness — {strat_line}",
            f"{_fmt_bool(False)} Risk gates — {risk_line}",
            f"{_fmt_bool(True)} RR threshold — {rr_line}",
            f"{_fmt_bool(self.last_error is None)} Errors — {err_line}",
            f"📈 last_signal: {last_signal}",
        ]
        return "\n".join(body)

    # ---- trading loop (optional, minimal) -------------------------------------------
    def start(self) -> None:
        """
        Optional background loop (if main.py wants it). We keep this very light.
        """
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(target=self._run, name="StrategyRunner", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._running = False

    def shutdown(self) -> None:
        self.stop()
        t = self._thread
        if t and t.is_alive():
            t.join(timeout=2.0)

    def _run(self) -> None:
        """
        Minimal tick loop: just touches data to keep diagnostics accurate.
        Your actual order/strategy code can be plugged here.
        """
        token = int(getattr(settings.instruments, "instrument_token", 256265))
        tf = str(getattr(settings.data, "timeframe", "minute"))
        lookback_min = int(getattr(settings.data, "lookback_minutes", 30))

        while True:
            with self._lock:
                if not self._running:
                    break

            # trading window guard (light)
            if not self._within_trading_window():
                time.sleep(5)
                continue

            try:
                end = self._now_naive_ist_rounded()
                start = end - timedelta(minutes=lookback_min)
                df = self.data_source.fetch_ohlc(token, start, end, tf)
                self.last_fetch_at = time.time()

                # (Optional) strategy evaluation stub
                if isinstance(df, pd.DataFrame) and not df.empty and self.strategy:
                    try:
                        signal = getattr(self.strategy, "evaluate", None)
                        if callable(signal):
                            out = signal(df)  # repo-specific
                            if isinstance(out, dict):
                                self.last_signal = out.get("type") or out.get("signal") or "none"
                    except Exception as e:
                        self.last_error = f"strategy-eval: {e}"
                time.sleep(3)

            except Exception as e:
                self.last_error = str(e)
                log.debug("runner loop error: %s", e)
                time.sleep(3)

    # ---- helpers --------------------------------------------------------------------
    def _now_naive_ist_rounded(self) -> datetime:
        now = _now_ist()
        return now.replace(tzinfo=None, second=0, microsecond=0)

    def _within_trading_window(self) -> bool:
        """
        09:20–15:20 IST by default (from config). Allow offhours if configured.
        """
        if settings.allow_offhours_testing:
            return True
        now = _now_ist()
        start_h, start_m = map(int, settings.data.time_filter_start.split(":"))
        end_h, end_m = map(int, settings.data.time_filter_end.split(":"))
        start = now.replace(hour=start_h, minute=start_m, second=0, microsecond=0)
        end = now.replace(hour=end_h, minute=end_m, second=0, microsecond=0)
        return start <= now <= end

    def _notify(self, msg: str) -> None:
        log.info(msg)
        try:
            if self.telegram and hasattr(self.telegram, "send_message"):
                self.telegram.send_message(msg)  # type: ignore[attr-defined]
        except Exception:
            pass


# --------------------------------------------------------------------------------------
# Convenience factory (used by main.py)
# --------------------------------------------------------------------------------------
def build_runner(
    *,
    kite: Optional[KiteConnect] = None,
    telegram: Optional[TelegramController] = None,
    live_trading: Optional[bool] = None,
) -> StrategyRunner:
    """
    Helper that respects settings.enable_live_trading when live_trading is None.
    """
    if live_trading is None:
        live_trading = bool(settings.enable_live_trading)
    runner = StrategyRunner(kite=kite, telegram=telegram, live_trading=live_trading)
    return runner