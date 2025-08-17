from __future__ import annotations

"""
Real-time trader orchestrator.

What this file does
- Streams market data (Kite provider when available, stub fallback).
- Receives Telegram commands via TelegramController (runs in its own worker thread).
- Generates signals with EnhancedScalpingStrategy and manages orders via OrderExecutor.
- Enforces risk via PositionSizing (RISK_PER_TRADE × equity; loss streak cooldown; day ladders).
- Handles auto regime detection (TREND/RANGE) and auto "quality" switching (confidence bump).
- Keeps a dynamic loop cadence, trading-hour gates, and optional time bucket/event windows.
- Persists active state and writes CSV trade logs.

Key changes (this version)
- Clean, explicit Quality control:
  * quality_setting: "AUTO" | "ON" | "OFF" (default "AUTO")
  * quality_effective: True/False (used to bump confidence)
  * quality_reason: human-readable reason; shown in Telegram /status
- Regime control kept distinct:
  * regime_mode: "AUTO" | "TREND" | "RANGE" | "OFF" (default "AUTO")
  * current_regime: effective TREND/RANGE used to bias TP/SL
  * regime_reason: why the regime was chosen (bbw/adx/slope)
- New Telegram commands supported in _handle_control:
  * /quality auto|on|off
  * /regime auto|trend|range|off
  * /risk <fraction or percent>  (e.g., "0.5" or "0.5%" ⇒ 0.5%)
  * /pause <minutes>  (defaults handled in TelegramController; trader uses same cooldown path)
  * /resume
- Safer Telegram polling: background worker started once here; the controller self-stops on 409 conflicts.
- Status now exposes uptime_sec, quality_* and effective regime so the controller can render a compact card.

How to use
- Construct RealTimeTrader() and call .run() in your main process.
- Start/stop trading with Telegram: /start /stop
- Switch live/shadow: /mode live | /mode shadow
- Quality: /quality auto|on|off  (AUTO by default; recommended)
- Regime: /regime auto|trend|range|off (AUTO by default)
- Risk: /risk 0.5  (means 0.5%)
"""

import atexit
import csv
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, asdict
from datetime import date, datetime, time as dtime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import schedule

from src.config import Config
from src.notifications.telegram_controller import TelegramController
from src.strategies.scalping_strategy import EnhancedScalpingStrategy
from src.utils.indicators import calculate_ema

# ----- providers / executor / risk ----------------------------------------

class DataProvider:
    """Minimal stub when live provider isn't available."""
    def get_ohlc(self, symbol: str, minutes: int, timeframe: str = "minute") -> pd.DataFrame:
        return pd.DataFrame()

    def get_last_price(self, symbol: str) -> Optional[float]:
        return None

try:
    from src.data_providers.kite_data_provider import KiteDataProvider  # type: ignore
except Exception:  # pragma: no cover
    KiteDataProvider = None  # type: ignore

try:
    from src.execution.order_executor import OrderExecutor
except Exception:  # pragma: no cover
    OrderExecutor = object  # type: ignore

try:
    from src.risk.position_sizing import PositionSizing, get_live_account_balance
except Exception:  # pragma: no cover
    class PositionSizing:  # minimal fallback
        def __init__(self, account_size: float = 0.0) -> None:
            self.equity = float(account_size)
            self.min_lots = int(getattr(Config, "MIN_LOTS", 1))
            self.max_lots = int(getattr(Config, "MAX_LOTS", 15))
            self._risk_per_trade = float(getattr(Config, "RISK_PER_TRADE", 0.02))

        def set_equity(self, x: float) -> None:
            self.equity = float(x)

        def set_risk_per_trade(self, value: float) -> None:
            try:
                self._risk_per_trade = max(1e-6, float(value))
            except Exception:
                pass

        def calculate_position_size(
            self,
            entry_price: float,
            stop_loss: float,
            signal_confidence: float,
            market_volatility: float = 0.0,
            lot_size: Optional[int] = None
        ) -> Optional[Dict[str, int]]:
            risk_amt = float(self._risk_per_trade) * float(self.equity or 0)
            L = int(lot_size or getattr(Config, "NIFTY_LOT_SIZE", 75))
            per_lot_risk = max(0.01, abs(float(entry_price) - float(stop_loss))) * L
            lots = int(max(0, risk_amt // per_lot_risk))
            lots = max(self.min_lots, min(lots, self.max_lots))
            return {"quantity": lots} if lots > 0 else None

    def get_live_account_balance() -> float:
        return float(getattr(Config, "ACCOUNT_SIZE", 0.0))

try:
    from src.utils.strike_selector import fetch_cached_instruments, get_instrument_tokens
except Exception:  # pragma: no cover
    fetch_cached_instruments = None
    get_instrument_tokens = None

logger = logging.getLogger(__name__)

# ============================== time helpers ============================== #

def _ist_now() -> datetime:
    return datetime.utcnow() + timedelta(hours=5, minutes=30)

def _now() -> datetime:
    return _ist_now() if getattr(Config, "USE_IST_CLOCK", True) else datetime.now()

def _between(now_t: dtime, start_hm: str, end_hm: str) -> bool:
    try:
        s = datetime.strptime(start_hm, "%H:%M").time()
        e = datetime.strptime(end_hm, "%H:%M").time()
        return s <= now_t <= e
    except Exception:
        return True

def _within_any_windows(now_t: dtime, windows: List[Tuple[str, str]]) -> bool:
    for a, b in (windows or []):
        try:
            if _between(now_t, a, b):
                return True
        except Exception:
            continue
    return False

def _round_to_tick(x: float, tick: float) -> float:
    if tick <= 0:
        return float(x)
    return round(float(x) / tick) * tick

# ============================== models ==================================== #

@dataclass
class ActiveTrade:
    entry_id: str
    symbol: str
    direction: str          # BUY / SELL
    quantity: int           # contracts
    entry_price: float
    stop_loss: float
    target: float
    atr: float
    opened_ts: float
    status: str = "OPEN"    # OPEN / CLOSED
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None

# ============================ RealTimeTrader ============================== #

class RealTimeTrader:
    # ---- Config mirrors ----
    MAX_CONCURRENT = int(getattr(Config, "MAX_CONCURRENT_POSITIONS", 1))
    WARMUP_BARS = int(getattr(Config, "WARMUP_BARS", 25))
    DATA_LOOKBACK_MIN = int(getattr(Config, "DATA_LOOKBACK_MINUTES", 45))
    HIST_TF = str(getattr(Config, "HISTORICAL_TIMEFRAME", "minute"))

    SLIPPAGE_BPS = float(getattr(Config, "SLIPPAGE_BPS", 4.0))
    FEES_PER_LOT = float(getattr(Config, "FEES_PER_LOT", 25.0))
    MAX_DD_PCT = float(getattr(Config, "MAX_DAILY_DRAWDOWN_PCT", 0.05))
    TRAILING_ENABLE = bool(getattr(Config, "TRAILING_ENABLE", True))
    WORKER_INTERVAL = int(getattr(Config, "WORKER_INTERVAL_SEC", 4))
    LOG_FILE = str(getattr(Config, "LOG_FILE", "logs/trades.csv"))

    LOT_SIZE = int(getattr(Config, "NIFTY_LOT_SIZE", 75))
    MIN_LOTS = int(getattr(Config, "MIN_LOTS", 1))
    MAX_LOTS = int(getattr(Config, "MAX_LOTS", 15))
    RISK_PER_TRADE = float(getattr(Config, "RISK_PER_TRADE", 0.025))
    MAX_TRADES_DAY = int(getattr(Config, "MAX_TRADES_PER_DAY", 30))
    LOSS_COOLDOWN_MIN = int(getattr(Config, "LOSS_COOLDOWN_MIN", 2))
    LOSS_LIMIT = int(getattr(Config, "CONSECUTIVE_LOSS_LIMIT", 3))
    LOSS_STREAK_PAUSE = int(getattr(Config, "LOSS_STREAK_PAUSE_MIN", 20))

    DAY_STOP_POS_R = float(getattr(Config, "DAY_STOP_AFTER_POS_R", 4.0))
    DAY_HALF_POS_R = float(getattr(Config, "DAY_HALF_SIZE_AFTER_POS_R", 2.0))
    DAY_STOP_NEG_R = float(getattr(Config, "DAY_STOP_AFTER_NEG_R", -3.0))

    TIME_START = str(getattr(Config, "TIME_FILTER_START", "09:20"))
    TIME_END = str(getattr(Config, "TIME_FILTER_END", "15:20"))
    SKIP_FIRST_MIN = int(getattr(Config, "SKIP_FIRST_MIN", 5))
    ENABLE_BUCKETS = bool(getattr(Config, "ENABLE_TIME_BUCKETS", False))
    BUCKETS = list(getattr(Config, "TIME_BUCKETS", []))
    ENABLE_EVENTS = bool(getattr(Config, "ENABLE_EVENT_WINDOWS", False))
    EVENTS = list(getattr(Config, "EVENT_WINDOWS", []))
    AUTO_EXIT_TIME = str(getattr(Config, "SESSION_AUTO_EXIT_TIME", getattr(Config, "TIME_FILTER_END", "15:20")))

    IDEMP_TTL = int(getattr(Config, "IDEMP_TTL_SEC", 60))

    # Auto-quality / regime thresholds
    QUALITY_AUTO_ENABLE = bool(getattr(Config, "QUALITY_AUTO_ENABLE", True))
    QUALITY_AUTO_ADX_MIN = float(getattr(Config, "QUALITY_AUTO_ADX_MIN", 20.0))
    QUALITY_AUTO_BB_WIDTH_MAX = float(getattr(Config, "QUALITY_AUTO_BB_WIDTH_MAX", 0.012))
    QUALITY_AUTO_SLOPE_MIN = float(getattr(Config, "QUALITY_AUTO_SLOPE_MIN", 0.0))

    ADX_PERIOD = int(getattr(Config, "ADX_PERIOD", 14))
    ADX_MIN_TREND = float(getattr(Config, "ADX_MIN_TREND", 18.0))
    BB_WINDOW = int(getattr(Config, "BB_WINDOW", 20))
    BB_WIDTH_MIN = float(getattr(Config, "BB_WIDTH_MIN", 0.006))
    BB_WIDTH_MAX = float(getattr(Config, "BB_WIDTH_MAX", 0.02))

    # Cadence controls (NEW)
    # If POLL_SEC is set (>0), it forces a fixed cadence.
    # Otherwise we use peak/off-peak with 5s defaults.
    try:
        FORCED_POLL_SEC = int(getattr(Config, "POLL_SEC", 0) or 0)
    except Exception:
        FORCED_POLL_SEC = 0
    try:
        PEAK_POLL_SEC = int(getattr(Config, "PEAK_POLL_SEC", getattr(Config, "DEFAULT_POLL_SEC", 5)))
    except Exception:
        PEAK_POLL_SEC = 5
    try:
        OFFPEAK_POLL_SEC = int(getattr(Config, "OFFPEAK_POLL_SEC", getattr(Config, "DEFAULT_POLL_SEC", 5)))
    except Exception:
        OFFPEAK_POLL_SEC = 5

    # Files
    STATE_DIR = "state"
    ACTIVE_JSON = os.path.join(STATE_DIR, "active_trades.json")

    def __init__(self, data: Optional[DataProvider] = None) -> None:
        self._lock = threading.RLock()
        self._start_ts = time.time()

        # Trading toggles / modes
        self.is_trading = False
        self.live_mode = bool(getattr(Config, "ENABLE_LIVE_TRADING", False))

        # Quality (control vs effect)
        self.quality_setting: str = "AUTO"  # "AUTO" | "ON" | "OFF"
        if not self.QUALITY_AUTO_ENABLE:
            self.quality_setting = "ON" if bool(getattr(Config, "QUALITY_MODE_DEFAULT", False)) else "OFF"
        self.quality_effective: bool = False
        self.quality_reason: str = "boot"

        # Regime control
        self.regime_mode: str = str(getattr(Config, "REGIME_MODE", "AUTO")).upper()  # AUTO|TREND|RANGE|OFF
        self.current_regime: str = "RANGE"
        self.regime_reason: str = "boot"

        # Session & risk state
        self.daily_start_equity: float = float(get_live_account_balance() or 0.0)
        self.session_date: date = _now().date()
        self.daily_pnl: float = 0.0
        self.session_R: float = 0.0
        self.trades_closed_today: int = 0
        self.loss_streak: int = 0
        self._cooldown_until: float = 0.0   # used for both loss streak & /pause
        self._halt_for_day: bool = False

        # Orders / state
        self.trades: List[Dict[str, Any]] = []
        self.active: Dict[str, ActiveTrade] = {}
        self._recent_keys: Dict[str, float] = {}

        # Components
        self.data = data or (self._build_kite_provider() if self.live_mode and KiteDataProvider else DataProvider())
        try:
            self.risk = PositionSizing(account_size=self.daily_start_equity)  # type: ignore
        except Exception:
            self.risk = PositionSizing()  # type: ignore
        try:
            self.risk.set_equity(float(self.daily_start_equity or 0.0))  # type: ignore
        except Exception:
            pass

        self.strategy = EnhancedScalpingStrategy(
            base_stop_loss_points=Config.BASE_STOP_LOSS_POINTS,
            base_target_points=Config.BASE_TARGET_POINTS,
            confidence_threshold=Config.CONFIDENCE_THRESHOLD,
            min_score_threshold=int(Config.MIN_SIGNAL_SCORE),
        )
        self.executor = self._init_executor()

        # Telegram
        self.tg: Optional[TelegramController] = None
        self._init_telegram()

        # Background workers
        self._trailing_evt = threading.Event()
        self._oco_evt = threading.Event()
        threading.Thread(target=self._trailing_worker, name="TrailingWorker", daemon=True).start()
        threading.Thread(target=self._oco_worker, name="OcoWorker", daemon=True).start()

        # Scheduler
        self._data_job = None
        self._setup_scheduler()

        # Files
        self._prepare_trade_log()
        self._maybe_restore_state()
        atexit.register(self.shutdown)

        logger.info("✅ RealTimeTrader initialized (live=%s).", self.live_mode)

    # ---------------- components ------------------------------------------

    def _build_kite_provider(self) -> DataProvider:
        try:
            return KiteDataProvider.build_from_env()  # type: ignore
        except Exception as e:
            logger.error("KiteDataProvider init failed, using stub DataProvider: %s", e, exc_info=True)
            return DataProvider()

    def _init_executor(self) -> OrderExecutor:
        if not self.live_mode:
            logger.info("Live trading disabled → simulation executor.")
            return OrderExecutor()  # type: ignore
        try:
            from kiteconnect import KiteConnect
            api_key = getattr(Config, "ZERODHA_API_KEY", "")
            access_token = getattr(Config, "KITE_ACCESS_TOKEN", "") or getattr(Config, "ZERODHA_ACCESS_TOKEN", "")
            if not api_key or not access_token:
                raise RuntimeError("ZERODHA_API_KEY or KITE_ACCESS_TOKEN missing")
            kite = KiteConnect(api_key=api_key)
            kite.set_access_token(access_token)
            logger.info("🟢 Live executor created (KiteConnect).")
            return OrderExecutor(kite=kite)  # type: ignore
        except Exception as exc:
            logger.error("Live init failed, switching to simulation: %s", exc, exc_info=True)
            self.live_mode = False
            return OrderExecutor()  # type: ignore

    def _init_telegram(self) -> None:
        if not getattr(Config, "ENABLE_TELEGRAM", True):
            return
        try:
            self.tg = TelegramController(
                status_callback=self.get_status,
                control_callback=self._handle_control,
                summary_callback=self.get_summary,
            )
            self.tg.send_startup_alert()
            # Run the controller's long-poll in a dedicated worker thread
            threading.Thread(target=self.tg.start_polling, name="TelegramPollingStart", daemon=True).start()
            logger.info("📡 Telegram polling worker started.")
        except Exception as e:
            logger.warning("Telegram init failed: %s", e)

    # --------------- scheduler / cadence -----------------------------------

    def _setup_scheduler(self) -> None:
        try:
            schedule.clear()
        except Exception:
            pass
        schedule.every(5).seconds.do(self._ensure_cadence)
        schedule.every(max(60, int(getattr(Config, "BALANCE_LOG_INTERVAL_MIN", 30)) * 60)).seconds.do(
            self.refresh_account_balance
        )
        schedule.every(30).seconds.do(self._roll_daily_if_needed)
        schedule.every(20).seconds.do(self._auto_exit_guard)

    def _current_poll_seconds(self) -> int:
        """
        Cadence order:
        1) If POLL_SEC>0, use it (fixed cadence).
        2) Else use PEAK_POLL_SEC vs OFFPEAK_POLL_SEC with 5s defaults.
        """
        try:
            if int(self.FORCED_POLL_SEC) > 0:
                return int(self.FORCED_POLL_SEC)
        except Exception:
            pass
        try:
            now_t = _now().time()
            in_peak = (dtime(9, 20) <= now_t <= dtime(11, 30)) or (dtime(13, 30) <= now_t <= dtime(15, 5))
            return int(self.PEAK_POLL_SEC) if in_peak else int(self.OFFPEAK_POLL_SEC)
        except Exception:
            # final fallback: 5s
            return 5

    def _ensure_cadence(self) -> None:
        """Keep the data loop running at a dynamic cadence."""
        try:
            sec = self._current_poll_seconds()
            if self._data_job and (getattr(self._data_job, "interval", None) == sec) and \
               (getattr(self._data_job, "unit", None) == "seconds"):
                return
            if self._data_job:
                schedule.cancel_job(self._data_job)
            self._data_job = schedule.every(sec).seconds.do(self._smart_tick)
            logger.info("📈 Data loop cadence: every %ds", sec)
        except Exception as e:
            logger.debug("Cadence error: %s", e)

    # ---------------- lifecycle -------------------------------------------

    def run(self) -> None:
        logger.info("🟢 RealTimeTrader.run() started.")
        while True:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                logger.error("Run loop error: %s", e, exc_info=True)
                time.sleep(2)

    def start(self) -> bool:
        with self._lock:
            self.is_trading = True
            logger.info("✅ Trading STARTED.")
            if self.tg:
                self.tg.send_realtime_session_alert("START")
            return True

    def stop(self) -> bool:
        with self._lock:
            self.is_trading = False
            logger.info("🛑 Trading STOPPED.")
            if self.tg:
                self.tg.send_realtime_session_alert("STOP")
            return True

    def enable_live(self) -> bool:
        self.live_mode = True
        logger.info("Mode switched to LIVE.")
        return True

    def disable_live(self) -> bool:
        self.live_mode = False
        logger.info("Mode switched to SHADOW.")
        return True

    # ---------------- telegram handler ------------------------------------

    def _handle_control(self, command: str, arg: str = "") -> bool:
        cmd = (command or "").strip().lower()
        arg = (arg or "").strip()
        logger.info("Command: /%s %s", cmd, arg)
        try:
            if cmd == "start":
                return self.start()
            if cmd == "stop":
                return self.stop()

            if cmd == "mode":
                low = arg.strip().lower()
                if low in ("live", "l"):
                    return self.enable_live()
                if low in ("shadow", "paper", "sim", "s"):
                    return self.disable_live()
                return False

            if cmd == "quality":
                low = arg.strip().lower() or "auto"
                if low not in ("auto", "on", "off"):
                    return False
                self.quality_setting = low.upper()
                if self.quality_setting == "ON":
                    self.quality_effective, self.quality_reason = True, "manual"
                elif self.quality_setting == "OFF":
                    self.quality_effective, self.quality_reason = False, "manual"
                else:
                    self.quality_reason = "auto-enabled"
                logger.info("Quality setting → %s", self.quality_setting)
                return True

            if cmd == "regime":
                low = arg.strip().lower() or "auto"
                if low in ("auto", "trend", "range", "off"):
                    self.regime_mode = low.upper()
                    self.regime_reason = "manual"
                    logger.info("Regime mode → %s", self.regime_mode)
                    return True
                return False

            if cmd == "risk":
                try:
                    val = float(arg)
                except Exception:
                    return False
                self.RISK_PER_TRADE = max(1e-6, float(val))
                try:
                    self.risk.set_risk_per_trade(self.RISK_PER_TRADE)  # type: ignore
                except Exception:
                    pass
                logger.info("Risk per trade set to %.4f (fraction)", self.RISK_PER_TRADE)
                return True

            if cmd == "pause":
                mins = 1
                try:
                    mins = max(1, int(float(arg or "1")))
                except Exception:
                    mins = 1
                self._cooldown_until = time.time() + mins * 60
                logger.info("Paused entries for %d minutes.", mins)
                return True

            if cmd == "resume":
                self._cooldown_until = 0.0
                logger.info("Resumed entries.")
                return True

            if cmd == "refresh":
                self.refresh_account_balance()
                return True

            if cmd == "health":
                ok = not self._circuit_tripped()
                if self.tg:
                    self.tg.send_message("System health: OK" if ok else "System health: CIRCUIT")
                return True

            if cmd == "emergency":
                try:
                    self.executor.cancel_all_orders()  # type: ignore
                except Exception:
                    pass
                self.active.clear()
                if self.tg:
                    self.tg.send_message("🛑 Emergency: All orders canceled, positions cleared (best-effort).")
                return True

        except Exception as e:
            logger.error("Control error: %s", e, exc_info=True)
        return False

    # ---------------- status / summary ------------------------------------

    def get_status(self) -> Dict[str, Any]:
        uptime = max(0.0, time.time() - self._start_ts)
        regime_to_show = self.current_regime if self.regime_mode == "AUTO" else self.regime_mode
        return {
            "is_trading": self.is_trading,
            "live_mode": self.live_mode,
            "quality_mode": self.quality_setting,
            "quality_auto": self.quality_setting == "AUTO",
            "quality_reason": self.quality_reason,
            "regime_mode": regime_to_show,
            "regime_reason": self.regime_reason,
            "open_positions": len(self.active),
            "closed_today": self.trades_closed_today,
            "daily_pnl": round(self.daily_pnl, 2),
            "account_size": round(self.daily_start_equity + self.daily_pnl, 2),
            "session_date": str(self.session_date),
            "uptime_sec": round(uptime, 1),
        }

    def get_summary(self) -> str:
        lines = [
            "<b>Daily Summary</b>",
            f"Date: {self.session_date}",
            f"Trades closed: {self.trades_closed_today}",
            f"PnL: ₹{self.daily_pnl:.2f}",
            f"Active: {len(self.active)}",
        ]
        return "\n".join(lines)

    def refresh_account_balance(self) -> None:
        try:
            bal = float(get_live_account_balance() or 0.0)
            if bal > 0:
                self.daily_start_equity = bal - self.daily_pnl
                if hasattr(self.risk, "set_equity"):
                    self.risk.set_equity(bal)  # type: ignore
            logger.info("💰 Equity refresh: %.2f", bal)
        except Exception as e:
            logger.debug("Balance refresh error: %s", e)

    # ---------------- main tick -------------------------------------------

    def _smart_tick(self) -> None:
        if not self.is_trading:
            return
        now = _now()
        now_t = now.time()

        # Trading hours + optional windows
        if not _between(now_t, self.TIME_START, self.TIME_END):
            return
        if self.ENABLE_BUCKETS and not _within_any_windows(now_t, self.BUCKETS):
            return
        if self.ENABLE_EVENTS and _within_any_windows(now_t, self.EVENTS):
            return

        # Skip opening chop
        if (now.hour == 9 and now.minute < max(16, self.SKIP_FIRST_MIN + 15)):
            return

        # Loss streak cooldown or /pause
        if time.time() < self._cooldown_until:
            return

        self._prune_recent_keys()

        spot_symbol = getattr(Config, "SPOT_SYMBOL", "NSE:NIFTY 50")
        df = self.data.get_ohlc(spot_symbol, self.DATA_LOOKBACK_MIN, self.HIST_TF)
        if df is None or df.empty or len(df) < max(self.WARMUP_BARS, 25):
            return

        # HTF gate
        if not self._htf_ok(df):
            return

        # Determine regime & quality effect each tick
        try:
            if self.regime_mode == "AUTO":
                regime, why, bbw, adxv, slope = self._infer_regime(df)
                self.current_regime = regime
                self.regime_reason = f"{why} (bbw={bbw:.4f}, adx={adxv:.2f}, slope={slope:.2f})"
            elif self.regime_mode == "OFF":
                regime, _why, _bbw, _adxv, _slope = self._infer_regime(df)
                self.current_regime = regime
            else:
                self.current_regime = self.regime_mode
        except Exception:
            self.current_regime, self.regime_reason = "RANGE", "calc-failed"

        # Quality
        if self.quality_setting == "AUTO" and self.QUALITY_AUTO_ENABLE:
            self._maybe_auto_quality(df)
        elif self.quality_setting == "ON":
            self.quality_effective, self.quality_reason = True, "manual"
        else:
            self.quality_effective, self.quality_reason = False, "manual"

        try:
            last_price = float(df["close"].iloc[-1])
        except Exception:
            return

        # Strategy
        sig = self._generate_spot_signal(df, last_price)
        if sig:
            try:
                if self.current_regime == "TREND":
                    tp_mult = float(getattr(Config, "REGIME_TREND_TP_MULT", 3.4))
                    sl_mult = float(getattr(Config, "REGIME_TREND_SL_MULT", 1.6))
                else:
                    tp_mult = float(getattr(Config, "REGIME_RANGE_TP_MULT", 2.4))
                    sl_mult = float(getattr(Config, "REGIME_RANGE_SL_MULT", 1.3))
                atr = float(sig.get("atr", sig.get("market_volatility", 0.0)) or 0.0)
                if atr > 0:
                    dir_mult = 1 if str(sig["signal"]).upper() == "BUY" else -1
                    sig["target"] = last_price + (tp_mult * atr) * dir_mult
                    sig["stop_loss"] = last_price - (sl_mult * atr) * dir_mult
            except Exception:
                pass

            self._place_trade(spot_symbol, sig, last_price)
            return

        # Options flow (optional hook)
        if get_instrument_tokens and fetch_cached_instruments:
            self._maybe_try_options_flow(last_price)

        # Check for exits and book P&L
        self._process_active_trades()

    # ---------------- regime / quality helpers -----------------------------

    def _infer_regime(self, df: pd.DataFrame) -> Tuple[str, str, float, float, float]:
        """
        Return (regime, reason, bb_width, adx_value, ema_slope).
        Simplified proxy:
          - BB width (% of price) from 20-period band
          - pseudo-ADX from mean absolute returns
          - EMA slope as directional hint
        """
        try:
            closes = df["close"].astype(float)
            if len(closes) < max(self.BB_WINDOW + 2, self.ADX_PERIOD + 2):
                return "RANGE", "insufficient-data", 0.0, 0.0, 0.0

            # BB width (% of price)
            roll = closes.rolling(self.BB_WINDOW)
            ma = roll.mean()
            std = roll.std(ddof=0)
            upper = ma + 2 * std
            lower = ma - 2 * std
            bb_width = float(((upper - lower) / ma).iloc[-1])

            # Pseudo ADX: mean absolute returns × 100
            ret = closes.pct_change().abs()
            adx_val = float((ret.rolling(self.ADX_PERIOD).mean() * 100).iloc[-1])

            # EMA slope (HTF-style) for trend bias
            ema = calculate_ema(df, int(getattr(Config, "HTF_EMA_PERIOD", 20)))
            slope = float(ema.iloc[-1] - ema.iloc[-3]) if (ema is not None and len(ema) >= 3) else 0.0

            if bb_width < self.BB_WIDTH_MIN and adx_val >= self.ADX_MIN_TREND:
                return "TREND", "narrow+strong-dm", bb_width, adx_val, slope
            if bb_width > self.BB_WIDTH_MAX and adx_val < self.ADX_MIN_TREND:
                return "RANGE", "wide+weak-dm", bb_width, adx_val, slope

            if adx_val >= self.ADX_MIN_TREND or abs(slope) >= self.QUALITY_AUTO_SLOPE_MIN:
                return "TREND", "slope/dir", bb_width, adx_val, slope
            return "RANGE", "default", bb_width, adx_val, slope
        except Exception as e:
            logger.debug("regime calc failed: %s", e)
            return "RANGE", "error", 0.0, 0.0, 0.0

    def _maybe_auto_quality(self, df: pd.DataFrame) -> None:
        try:
            regime, _why, bbw, adxv, slope = self._infer_regime(df)
            good = (adxv >= self.QUALITY_AUTO_ADX_MIN) and (bbw <= self.QUALITY_AUTO_BB_WIDTH_MAX)
            if abs(slope) >= self.QUALITY_AUTO_SLOPE_MIN:
                good = True
            new_q = bool(good and regime == "TREND")
            if new_q != self.quality_effective:
                self.quality_effective = new_q
                self.quality_reason = f"auto (bbw={bbw:.4f}, adx={adxv:.2f}, slope={slope:.2f}, regime={regime})"
                logger.info("✨ Auto-quality → %s | %s", self.quality_effective, self.quality_reason)
                if self.tg:
                    self.tg.send_message(
                        f"✨ Quality <b>{'ON' if self.quality_effective else 'OFF'}</b> "
                        f"(bbw={bbw:.4f}, adx={adxv:.2f}, slope={slope:.2f}, regime={regime})",
                        parse_mode="HTML",
                    )
        except Exception as e:
            logger.debug("auto-quality failed: %s", e)

    # ---------------- gates / strategy -------------------------------------

    def _htf_ok(self, df: pd.DataFrame) -> bool:
        try:
            ema = calculate_ema(df, int(getattr(Config, "HTF_EMA_PERIOD", 20)))
            if ema is None or len(ema) < 3:
                return True
            slope = float(ema.iloc[-1] - ema.iloc[-3])
            min_slope = float(getattr(Config, "HTF_MIN_SLOPE", 0.0))
            return abs(slope) >= min_slope
        except Exception:
            return True

    def _generate_spot_signal(self, df: pd.DataFrame, last_price: float) -> Optional[Dict[str, Any]]:
        try:
            sig = self.strategy.generate_signal(df, last_price)
            if sig and self.quality_effective:
                sig["confidence"] = float(sig.get("confidence", 5.0)) + float(getattr(Config, "QUALITY_SCORE_BUMP", 1.0))
            return sig
        except Exception as e:
            logger.debug("spot signal error: %s", e)
            return None

    def _maybe_try_options_flow(self, spot_last: float) -> None:
        # Hook intentionally empty. Wire your options leg builder here if needed.
        pass

    # ---------------- hygiene / circuit ------------------------------------

    def _entry_guard(self, key: str) -> bool:
        now = time.time()
        exp = self._recent_keys.get(key)
        if exp and exp > now:
            return False
        self._recent_keys[key] = now + float(self.IDEMP_TTL)
        return True

    def _prune_recent_keys(self) -> None:
        now = time.time()
        self._recent_keys = {k: v for k, v in self._recent_keys.items() if v > now}

    def _circuit_tripped(self) -> bool:
        eq = float(self.daily_start_equity or 0.0)
        cur = eq + float(self.daily_pnl or 0.0)
        if eq <= 0:
            return False
        drawdown = (cur - eq) / eq
        return drawdown <= -abs(self.MAX_DD_PCT) or self._halt_for_day

    # ---------------- execution path ---------------------------------------

    def _place_trade(self, symbol: str, sig: Dict[str, Any], mark_price: float) -> None:
        try:
            direction = str(sig["signal"]).upper()        # BUY / SELL
            entry = float(sig.get("entry_price", mark_price))
            sl = float(sig.get("stop_loss"))
            tp = float(sig.get("target"))
            conf = float(sig.get("confidence", 5.0))
            atr = float(sig.get("market_volatility", sig.get("atr", 0.0) or 0.0))
        except Exception:
            return

        if self._circuit_tripped() or (time.time() < self._cooldown_until):
            return
        if len(self.active) >= self.MAX_CONCURRENT:
            return
        if self.trades_closed_today >= self.MAX_TRADES_DAY:
            return

        key = f"{symbol}|{direction}|{round(entry,2)}|{self.session_date}"
        if not self._entry_guard(key):
            return

        qty_lots = None
        try:
            res = self.risk.calculate_position_size(entry_price=entry, stop_loss=sl,
                                                    signal_confidence=conf,
                                                    market_volatility=atr,
                                                    lot_size=self.LOT_SIZE)  # type: ignore
            qty_lots = int(res["quantity"]) if (isinstance(res, dict) and "quantity" in res) else None
        except Exception:
            pass
        if qty_lots is None:
            qty_lots = max(self.MIN_LOTS, 1)
        qty_lots = max(self.MIN_LOTS, min(int(qty_lots), self.MAX_LOTS))
        qty = qty_lots * self.LOT_SIZE
        if qty <= 0:
            return

        try:
            entry_id = self.executor.place_entry_order(  # type: ignore
                symbol, direction=direction, quantity=qty
            )
            if not entry_id:
                logger.warning("Entry placement returned no order id.")
                return
        except Exception as e:
            logger.error("Entry order failed: %s", e, exc_info=True)
            return

        try:
            self.executor.setup_gtt_orders(  # type: ignore
                entry_order_id=str(entry_id),
                symbol=symbol,
                direction=direction,
                entry_price=entry,
                stop_loss=sl,
                target=tp,
                lot_size=self.LOT_SIZE,
            )
        except Exception as e:
            logger.warning("setup_gtt_orders error: %s", e)

        at = ActiveTrade(
            entry_id=str(entry_id),
            symbol=symbol,
            direction=direction,
            quantity=qty,
            entry_price=entry,
            stop_loss=sl,
            target=tp,
            atr=atr,
            opened_ts=time.time(),
        )
        self.active[at.entry_id] = at
        self._persist_state()

        if self.tg:
            self.tg.send_signal_alert(token=len(self.trades) + len(self.active), signal=sig,
                                      position={"quantity": qty})

        logger.info("▶️ Placed %s x%d @ %.2f SL %.2f TP %.2f", direction, qty, entry, sl, tp)

    # ---------------- workers ---------------------------------------------

    def _trailing_worker(self) -> None:
        while not self._trailing_evt.is_set():
            try:
                if self.TRAILING_ENABLE and self.is_trading and not self._circuit_tripped():
                    self._trailing_tick()
            except Exception as e:
                logger.debug("Trailing worker error: %s", e)
            self._trailing_evt.wait(self.WORKER_INTERVAL)

    def _oco_worker(self) -> None:
        while not self._oco_evt.is_set():
            try:
                if self.is_trading:
                    self._oco_tick()
            except Exception as e:
                logger.debug("OCO worker error: %s", e)
            self._oco_evt.wait(self.WORKER_INTERVAL)

    def _trailing_tick(self) -> None:
        for at in list(self.active.values()):
            try:
                cur = None
                try:
                    cur = self.data.get_last_price(at.symbol)
                except Exception:
                    cur = None
                if cur is None:
                    try:
                        cur = self.executor.get_last_price(at.symbol)  # type: ignore
                    except Exception:
                        cur = None
                if cur is None:
                    continue
                self.executor.update_trailing_stop(at.entry_id, float(cur), float(at.atr))  # type: ignore
            except Exception:
                pass

    def _oco_tick(self) -> None:
        try:
            filled = self.executor.sync_and_enforce_oco()  # type: ignore
            if filled:
                self._finalize_fills(filled)
        except Exception:
            pass

    def _process_active_trades(self) -> None:
        try:
            filled = self.executor.sync_and_enforce_oco()  # type: ignore
            if filled:
                self._finalize_fills(filled)
        except Exception:
            pass

    def _finalize_fills(self, filled: List[Tuple[str, float]]) -> None:
        closed_ids: List[str] = []
        for entry_id, exit_px in filled:
            at = self.active.get(str(entry_id))
            if not at:
                continue
            at.status = "CLOSED"
            at.exit_price = float(exit_px)
            exit_reason = None
            try:
                rec = getattr(self.executor, "orders", {}).get(str(entry_id))  # type: ignore
                exit_reason = getattr(rec, "exit_reason", None) if rec else None
            except Exception:
                exit_reason = None
            at.exit_reason = exit_reason or "exit-filled"

            direction_mult = 1 if at.direction == "BUY" else -1
            pnl = (at.exit_price - at.entry_price) * direction_mult * at.quantity
            self.daily_pnl += float(pnl)
            self.trades_closed_today += 1
            self.trades.append(asdict(at))
            self._log_trade(at, pnl)
            closed_ids.append(at.entry_id)

            logger.info("⏹ Closed %s @ %.2f (%s) PnL=%.2f",
                        at.symbol, at.exit_price, at.exit_reason, pnl)

            if pnl < 0:
                self.loss_streak += 1
                if self.loss_streak >= self.LOSS_LIMIT:
                    self._cooldown_until = time.time() + self.LOSS_STREAK_PAUSE * 60
            else:
                self.loss_streak = 0

            r = pnl / max(1.0, abs(at.entry_price - at.stop_loss) * at.quantity)
            self.session_R += r
            if self.session_R >= self.DAY_STOP_POS_R:
                self._halt_for_day = True
            elif self.session_R >= self.DAY_HALF_POS_R:
                try:
                    self.MAX_LOTS = max(self.MIN_LOTS, max(1, self.MAX_LOTS // 2))
                except Exception:
                    pass

        for oid in closed_ids:
            self.active.pop(oid, None)
        if closed_ids:
            self._persist_state()

    # ---------------- daily / auto-exit ------------------------------------

    def _roll_daily_if_needed(self) -> None:
        if _now().date() != self.session_date:
            self.session_date = _now().date()
            self.daily_start_equity = float(get_live_account_balance() or (self.daily_start_equity + self.daily_pnl))
            try:
                self.risk.set_equity(self.daily_start_equity)  # type: ignore
            except Exception:
                pass
            self.daily_pnl = 0.0
            self.session_R = 0.0
            self.trades_closed_today = 0
            self.loss_streak = 0
            self._cooldown_until = 0.0
            self._halt_for_day = False
            logger.info("🔄 New session: %s", self.session_date)

    def _auto_exit_guard(self) -> None:
        try:
            t = _now().time()
            if _between(t, self.AUTO_EXIT_TIME, self.AUTO_EXIT_TIME):
                try:
                    self.executor.cancel_all_orders()  # type: ignore
                except Exception:
                    pass
                for at in list(self.active.values()):
                    try:
                        self.executor.exit_order(at.entry_id, exit_reason="auto-exit")  # type: ignore
                    except Exception:
                        pass
                self.active.clear()
                self._persist_state()
        except Exception:
            pass

    # ---------------- files / persistence ----------------------------------

    def _prepare_trade_log(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.LOG_FILE) or ".", exist_ok=True)
            if not os.path.exists(self.LOG_FILE):
                with open(self.LOG_FILE, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["ts", "symbol", "dir", "qty", "entry", "exit", "reason", "pnl"])
        except Exception as e:
            logger.debug("log prep error: %s", e)

    def _log_trade(self, at: ActiveTrade, pnl: float) -> None:
        try:
            with open(self.LOG_FILE, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    datetime.utcnow().isoformat(), at.symbol, at.direction, at.quantity,
                    round(at.entry_price, 2), round(at.exit_price or 0.0, 2),
                    at.exit_reason or "", round(float(pnl), 2)
                ])
        except Exception:
            pass

    def _persist_state(self) -> None:
        try:
            os.makedirs(self.STATE_DIR, exist_ok=True)
            with open(self.ACTIVE_JSON, "w") as f:
                json.dump({k: asdict(v) for k, v in self.active.items()}, f, indent=2)
        except Exception:
            pass

    def _maybe_restore_state(self) -> None:
        try:
            if not os.path.exists(self.ACTIVE_JSON):
                return
            with open(self.ACTIVE_JSON, "r") as f:
                raw = json.load(f)
            self.active = {k: ActiveTrade(**v) for k, v in (raw or {}).items()}
            if getattr(Config, "PERSIST_REATTACH_ON_START", True):
                try:
                    self.executor.reattach(self.active)  # type: ignore
                except Exception:
                    pass
        except Exception:
            self.active = {}

    # ---------------- shutdown ---------------------------------------------

    def shutdown(self) -> None:
        try:
            self._trailing_evt.set()
            self._oco_evt.set()
            self._persist_state()
            if self.tg:
                self.tg.stop_polling()
        except Exception:
            pass