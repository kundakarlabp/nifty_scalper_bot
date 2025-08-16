# src/data_streaming/realtime_trader.py
from __future__ import annotations

"""
Real-time trader orchestrator.

Highlights
- Data provider: KiteDataProvider (LTP + historical OHLC) with tiny cache & retry
- Telegram control (/start /stop /mode live|shadow /mode quality auto|on|off /status /summary /health /emergency)
- Adaptive loop cadence (peak/off-peak), warmup, trading-hour gates + optional buckets/events
- Strategy: EnhancedScalpingStrategy for spot/futures, with regime-aware (TREND/RANGE) SL/TP tweaks
- Auto "quality" switch (AUTO/ON/OFF) using ADX + Bollinger Band Width; align entries with HTF trend when ON
- Risk sizing via PositionSizing (RISK_PER_TRADE × equity; ATR/SL aware)
- Exits managed by OrderExecutor (GTT/Regular + partials + trailing)
- Loss cooldown, trade/day cap, session ladders, CSV log, state persistence
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

# Indicators
try:
    # Prefer shared utils if present
    from src.utils.indicators import calculate_ema, calculate_adx  # type: ignore
except Exception:
    # Fallbacks (very lightweight)
    def calculate_ema(df: pd.DataFrame, period: int = 20) -> Optional[pd.Series]:
        try:
            return df["close"].ewm(span=max(1, int(period)), adjust=False).mean()
        except Exception:
            return None

    def calculate_adx(df: pd.DataFrame, period: int = 14) -> Optional[pd.Series]:
        try:
            high = df["high"].astype(float)
            low = df["low"].astype(float)
            close = df["close"].astype(float)

            up_move = high.diff()
            down_move = -low.diff()

            plus_dm = ((up_move > down_move) & (up_move > 0)).astype(float) * up_move.clip(lower=0)
            minus_dm = ((down_move > up_move) & (down_move > 0)).astype(float) * down_move.clip(lower=0)

            tr1 = (high - low).abs()
            tr2 = (high - close.shift()).abs()
            tr3 = (low - close.shift()).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            atr = tr.ewm(alpha=1 / max(1, period), adjust=False).mean()
            pdi = 100 * (plus_dm.ewm(alpha=1 / max(1, period), adjust=False).mean() / atr).replace([pd.NA, pd.NaT], 0.0)
            mdi = 100 * (minus_dm.ewm(alpha=1 / max(1, period), adjust=False).mean() / atr).replace([pd.NA, pd.NaT], 0.0)

            dx = (abs(pdi - mdi) / (pdi + mdi).replace(0, pd.NA)).fillna(0.0) * 100.0
            adx = dx.ewm(alpha=1 / max(1, period), adjust=False).mean()
            return adx
        except Exception:
            return None

# ---- providers / executor / risk ----

# Minimal stub provider (simulation)
class DataProvider:
    def get_ohlc(self, symbol: str, minutes: int, timeframe: str = "minute") -> pd.DataFrame:
        return pd.DataFrame()

    def get_last_price(self, symbol: str) -> Optional[float]:
        return None

# Shared, dedicated provider (de-duplicated)
try:
    from src.data_providers.kite_data_provider import KiteDataProvider  # type: ignore
except Exception:  # pragma: no cover
    KiteDataProvider = None  # type: ignore

# Order executor
try:
    from src.execution.order_executor import OrderExecutor
except Exception:  # pragma: no cover
    OrderExecutor = object  # type: ignore

# Risk manager
try:
    from src.risk.position_sizing import PositionSizing, get_live_account_balance
except Exception:  # pragma: no cover
    class PositionSizing:  # minimal fallback
        def __init__(self, account_size: float = 0.0) -> None:
            self.equity = float(account_size)
            self.min_lots = int(getattr(Config, "MIN_LOTS", 1))
            self.max_lots = int(getattr(Config, "MAX_LOTS", 15))

        def set_equity(self, x: float) -> None:
            self.equity = float(x)

        def calculate_position_size(self, entry_price: float, stop_loss: float,
                                    signal_confidence: float, market_volatility: float = 0.0,
                                    lot_size: Optional[int] = None) -> Optional[Dict[str, int]]:
            risk_amt = float(getattr(Config, "RISK_PER_TRADE", 0.02)) * float(self.equity or 0)
            L = int(lot_size or getattr(Config, "NIFTY_LOT_SIZE", 75))
            per_lot_risk = max(0.01, abs(entry_price - stop_loss)) * L
            lots = int(max(0, risk_amt // per_lot_risk))
            lots = max(self.min_lots, min(lots, self.max_lots))
            return {"quantity": lots} if lots > 0 else None

    def get_live_account_balance() -> float:
        return float(getattr(Config, "ACCOUNT_SIZE", 0.0))

# Optional strike utils (kept best-effort)
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


# ============================== models ============================== #

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


# ============================ RealTimeTrader ============================ #

class RealTimeTrader:
    # ---- config mirrors ----
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

    # Regime / quality thresholds (safe defaults if not in Config)
    ADX_PERIOD = int(getattr(Config, "ADX_PERIOD", 14))
    ADX_MIN_TREND = float(getattr(Config, "ADX_MIN_TREND", 18.0))
    BB_WINDOW = int(getattr(Config, "BB_WINDOW", 20))
    BB_WIDTH_MIN = float(getattr(Config, "BB_WIDTH_MIN", 0.006))
    BB_WIDTH_MAX = float(getattr(Config, "BB_WIDTH_MAX", 0.02))

    REGIME_TREND_TP_MULT = float(getattr(Config, "REGIME_TREND_TP_MULT", 3.4))
    REGIME_TREND_SL_MULT = float(getattr(Config, "REGIME_TREND_SL_MULT", 1.6))
    REGIME_RANGE_TP_MULT = float(getattr(Config, "REGIME_RANGE_TP_MULT", 2.4))
    REGIME_RANGE_SL_MULT = float(getattr(Config, "REGIME_RANGE_SL_MULT", 1.3))

    QUALITY_SCORE_BUMP = float(getattr(Config, "QUALITY_SCORE_BUMP", 1.0))
    # AUTO quality toggle (default OFF unless added in .env)
    QUALITY_MODE_AUTO = bool(str(getattr(Config, "QUALITY_MODE_AUTO", "false")).lower() in ("1", "true", "yes", "on"))

    # HTF alignment used when quality is ON
    HTF_EMA_PERIOD = int(getattr(Config, "HTF_EMA_PERIOD", 20))

    # files
    STATE_DIR = "state"
    ACTIVE_JSON = os.path.join(STATE_DIR, "active_trades.json")

    def __init__(self, data: Optional[DataProvider] = None) -> None:
        self._lock = threading.RLock()

        self.is_trading = False
        self.live_mode = bool(getattr(Config, "ENABLE_LIVE_TRADING", False))
        self.quality_mode = bool(getattr(Config, "QUALITY_MODE_DEFAULT", False))
        self.quality_auto = bool(self.QUALITY_MODE_AUTO)
        self._quality_reason: str = "manual" if not self.quality_auto else "auto-init"

        # regime cache
        self._regime: str = "UNKNOWN"       # "TREND" | "RANGE" | "UNKNOWN"
        self._regime_reason: str = ""

        # session
        self.daily_start_equity: float = float(get_live_account_balance() or 0.0)
        self.session_date: date = _now().date()
        self.daily_pnl: float = 0.0
        self.session_R: float = 0.0
        self.trades_closed_today: int = 0
        self.loss_streak: int = 0
        self._cooldown_until: float = 0.0
        self._halt_for_day: bool = False

        # state
        self.trades: List[Dict[str, Any]] = []
        self.active: Dict[str, ActiveTrade] = {}
        self._recent_keys: Dict[str, float] = {}

        # components
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

        # telegram
        self.tg: Optional[TelegramController] = None
        self._init_telegram()

        # workers
        self._trailing_evt = threading.Event()
        self._oco_evt = threading.Event()
        threading.Thread(target=self._trailing_worker, daemon=True).start()
        threading.Thread(target=self._oco_worker, daemon=True).start()

        # scheduler
        self._data_job = None
        self._setup_scheduler()

        # files
        self._prepare_trade_log()
        self._maybe_restore_state()
        atexit.register(self.shutdown)

        logger.info("✅ RealTimeTrader initialized (live=%s).", self.live_mode)

    # ---------------- components ---------------- #

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
            threading.Thread(target=self.tg.start_polling, daemon=True).start()
            logger.info("📡 Telegram polling started.")
        except Exception as e:
            logger.warning("Telegram init failed: %s", e)

    # --------------- scheduler / cadence --------------- #

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
        try:
            now_t = _now().time()
            in_peak = (dtime(9, 20) <= now_t <= dtime(11, 30)) or (dtime(13, 30) <= now_t <= dtime(15, 5))
            return int(getattr(Config, "PEAK_POLL_SEC", 12)) if in_peak else int(getattr(Config, "OFFPEAK_POLL_SEC", 25))
        except Exception:
            return int(getattr(Config, "OFFPEAK_POLL_SEC", 25))

    def _ensure_cadence(self) -> None:
        """Keep the data loop running at a dynamic cadence."""
        try:
            sec = self._current_poll_seconds()
            # Avoid touching schedule internals that vary across versions
            if self._data_job:
                try:
                    same = (getattr(self._data_job, "interval", None) == sec) and \
                           (str(getattr(self._data_job, "unit", "")) == "seconds")
                except Exception:
                    same = False
                if same:
                    return
                schedule.cancel_job(self._data_job)
            self._data_job = schedule.every(sec).seconds.do(self._smart_tick)
            logger.info("📈 Data loop cadence: every %ds", sec)
        except Exception as e:
            logger.debug("Cadence error: %s", e)

    # ---------------- lifecycle ---------------- #

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

    # ---------------- telegram handler ---------------- #

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
                if low.startswith("quality"):
                    # accepted: "quality auto", "quality on", "quality off"
                    if "auto" in low:
                        self.quality_auto = True
                        self._quality_reason = "auto-enabled"
                        logger.info("Quality mode set to AUTO")
                    elif ("on" in low) or low.endswith("1"):
                        self.quality_auto = False
                        self.quality_mode = True
                        self._quality_reason = "manual:on"
                        logger.info("Quality mode set to ON")
                    elif ("off" in low) or low.endswith("0"):
                        self.quality_auto = False
                        self.quality_mode = False
                        self._quality_reason = "manual:off"
                        logger.info("Quality mode set to OFF")
                    else:
                        return False
                    return True
                return False
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

    # ---------------- status / summary ---------------- #

    def get_status(self) -> Dict[str, Any]:
        qm = "AUTO" if self.quality_auto else ("ON" if self.quality_mode else "OFF")
        return {
            "is_trading": self.is_trading,
            "live_mode": self.live_mode,
            "quality_mode": f"{qm} ({self._quality_reason})",
            "regime": f"{self._regime} ({self._regime_reason})" if self._regime != "UNKNOWN" else "UNKNOWN",
            "open_positions": len(self.active),
            "closed_today": self.trades_closed_today,
            "daily_pnl": round(self.daily_pnl, 2),
            "account_size": round(self.daily_start_equity + self.daily_pnl, 2),
            "session_date": str(self.session_date),
        }

    def get_summary(self) -> str:
        lines = [
            "<b>Daily Summary</b>",
            f"Date: {self.session_date}",
            f"Trades closed: {self.trades_closed_today}",
            f"PnL: ₹{self.daily_pnl:.2f}",
            f"Active: {len(self.active)}",
            f"Regime: {self._regime} ({self._regime_reason})",
            f"Quality: {'AUTO' if self.quality_auto else ('ON' if self.quality_mode else 'OFF')} ({self._quality_reason})",
        ]
        return "\n".join(lines)

    def refresh_account_balance(self) -> None:
        try:
            bal = float(get_live_account_balance() or 0.0)
            if bal > 0:
                # keep daily_start_equity such that account_size = start + pnl
                self.daily_start_equity = bal - self.daily_pnl
                if hasattr(self.risk, "set_equity"):
                    self.risk.set_equity(bal)  # type: ignore
            logger.info("💰 Equity refresh: %.2f", bal)
        except Exception as e:
            logger.debug("Balance refresh error: %s", e)

    # ---------------- regime / quality ---------------- #

    def _bb_width(self, df: pd.DataFrame, window: int) -> Optional[pd.Series]:
        try:
            close = df["close"].astype(float)
            mid = close.rolling(window).mean()
            std = close.rolling(window).std(ddof=0)
            upper = mid + 2 * std
            lower = mid - 2 * std
            width = (upper - lower) / mid.replace(0, pd.NA)
            return width.replace([pd.NA, pd.NaT], 0.0)
        except Exception:
            return None

    def _detect_regime(self, df: pd.DataFrame) -> Tuple[str, str]:
        """
        Return ("TREND"|"RANGE"|"UNKNOWN", reason string)
        Heuristic:
          - TREND if ADX >= ADX_MIN_TREND OR BB width >= BB_WIDTH_MAX
          - RANGE if ADX < ADX_MIN_TREND AND BB width between [BB_WIDTH_MIN, BB_WIDTH_MAX)
          - else UNKNOWN
        """
        try:
            adx = calculate_adx(df, self.ADX_PERIOD)
        except Exception:
            adx = None
        bbw = self._bb_width(df, self.BB_WINDOW)

        last_adx = float(adx.iloc[-1]) if (isinstance(adx, pd.Series) and len(adx) > 0) else None
        last_bbw = float(bbw.iloc[-1]) if (isinstance(bbw, pd.Series) and len(bbw) > 0) else None

        if last_adx is not None and last_adx >= self.ADX_MIN_TREND:
            return "TREND", f"ADX {last_adx:.1f}≥{self.ADX_MIN_TREND}"
        if last_bbw is not None and last_bbw >= self.BB_WIDTH_MAX:
            return "TREND", f"BBw {last_bbw:.3f}≥{self.BB_WIDTH_MAX:.3f}"
        if (last_adx is not None and last_adx < self.ADX_MIN_TREND) and \
           (last_bbw is not None and self.BB_WIDTH_MIN <= last_bbw < self.BB_WIDTH_MAX):
            return "RANGE", f"ADX {last_adx:.1f} & BBw {last_bbw:.3f}"
        if last_bbw is not None and last_bbw < self.BB_WIDTH_MIN:
            return "RANGE", f"BBw {last_bbw:.3f}<{self.BB_WIDTH_MIN:.3f}"
        return "UNKNOWN", "insufficient data"

    def _auto_quality_if_enabled(self, df: pd.DataFrame) -> None:
        """
        AUTO: turn quality ON when conditions are favorable (decent trend or
        clean ranges) and OFF during choppy/noisy periods.
        """
        if not self.quality_auto:
            return

        regime, reason = self._detect_regime(df)
        self._regime, self._regime_reason = regime, reason

        # Simple rule:
        # - Quality ON for TREND
        # - Quality OFF when UNKNOWN or very narrow/noisy ranges
        # - For RANGE with moderate BB width, keep ON but require HTF alignment on entries
        if regime == "TREND":
            self.quality_mode = True
            self._quality_reason = f"auto:trend ({reason})"
        elif regime == "RANGE":
            self.quality_mode = True
            self._quality_reason = f"auto:range ({reason})"
        else:
            self.quality_mode = False
            self._quality_reason = f"auto:off ({reason})"

    # ---------------- main tick ---------------- #

    def _smart_tick(self) -> None:
        if not self.is_trading:
            return
        now = _now()
        now_t = now.time()

        # Trading hours filter (+ optional buckets/events)
        if not _between(now_t, self.TIME_START, self.TIME_END):
            return
        if self.ENABLE_BUCKETS and not _within_any_windows(now_t, self.BUCKETS):
            return
        if self.ENABLE_EVENTS and _within_any_windows(now_t, self.EVENTS):
            return

        # skip first minutes of day
        if (now.hour == 9 and now.minute < max(16, self.SKIP_FIRST_MIN + 15)):
            return

        # Clean recent keys
        self._prune_recent_keys()

        # pull spot data
        spot_symbol = getattr(Config, "SPOT_SYMBOL", "NSE:NIFTY 50")
        df = self.data.get_ohlc(spot_symbol, self.DATA_LOOKBACK_MIN, self.HIST_TF)
        if df is None or df.empty or len(df) < max(self.WARMUP_BARS, 25):
            return

        # AUTO quality & regime detection
        self._auto_quality_if_enabled(df)

        # simple HTF gate (EMA slope)
        if not self._htf_ok(df):
            return

        try:
            last_price = float(df["close"].iloc[-1])
        except Exception:
            return

        sig = self._generate_spot_signal(df, last_price)
        if sig:
            self._place_trade(spot_symbol, sig, last_price)
            return

        # options flow (left best-effort / optional)
        if get_instrument_tokens and fetch_cached_instruments:
            self._maybe_try_options_flow(last_price)

        # check for exits and book PnL
        self._process_active_trades()

    # ---------------- gates / strategy ---------------- #

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

    def _apply_regime_and_quality(self, sig: Dict[str, Any], df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Adjust SL/TP to suit detected regime and quality mode.
        Also enforce HTF alignment when quality ON.
        """
        try:
            direction = str(sig["signal"]).upper()
            entry = float(sig.get("entry_price"))
            sl = float(sig.get("stop_loss"))
            tp = float(sig.get("target"))
        except Exception:
            return None

        # Regime (use last detected; re-check quickly if unknown)
        if self._regime == "UNKNOWN":
            self._regime, self._regime_reason = self._detect_regime(df)

        # Quality ON: require alignment with HTF EMA (simple filter)
        if self.quality_mode:
            ema = calculate_ema(df, self.HTF_EMA_PERIOD)
            if isinstance(ema, pd.Series) and len(ema) > 0:
                ema_last = float(ema.iloc[-1])
                px_last = float(df["close"].iloc[-1])
                if direction == "BUY" and px_last < ema_last:
                    return None  # skip counter-trend long in quality mode
                if direction == "SELL" and px_last > ema_last:
                    return None  # skip counter-trend short in quality mode

        # Distances
        sl_pts = abs(entry - sl)
        tp_pts = abs(tp - entry)
        if sl_pts <= 0 or tp_pts <= 0:
            return None

        if self._regime == "TREND":
            sl_pts *= self.REGIME_TREND_SL_MULT
            tp_pts *= self.REGIME_TREND_TP_MULT
        elif self._regime == "RANGE":
            sl_pts *= self.REGIME_RANGE_SL_MULT
            tp_pts *= self.REGIME_RANGE_TP_MULT
        # UNKNOWN: leave as is

        # Rebuild SL/TP around entry
        if direction == "BUY":
            sl = entry - sl_pts
            tp = entry + tp_pts
        else:
            sl = entry + sl_pts
            tp = entry - tp_pts

        # Optional: bump confidence a bit when quality ON
        conf = float(sig.get("confidence", 5.0))
        if self.quality_mode:
            conf += float(self.QUALITY_SCORE_BUMP or 0.0)

        sig = dict(sig)
        sig["stop_loss"] = _round_to_tick(sl, float(getattr(Config, "TICK_SIZE", 0.05)))
        sig["target"] = _round_to_tick(tp, float(getattr(Config, "TICK_SIZE", 0.05)))
        sig["confidence"] = conf
        sig["regime"] = self._regime
        return sig

    def _generate_spot_signal(self, df: pd.DataFrame, last_price: float) -> Optional[Dict[str, Any]]:
        try:
            base = self.strategy.generate_signal(df, last_price)
            if not base:
                return None
            return self._apply_regime_and_quality(base, df)
        except Exception as e:
            logger.debug("spot signal error: %s", e)
            return None

    def _maybe_try_options_flow(self, spot_last: float) -> None:
        # Hook left intentionally light; extend as you wire proper option candles.
        pass

    # ---------------- hygiene / circuit ---------------- #

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

    # ---------------- execution path ---------------- #

    def _place_trade(self, symbol: str, sig: Dict[str, Any], mark_price: float) -> None:
        try:
            direction = str(sig["signal"]).upper()        # BUY / SELL
            entry = float(sig.get("entry_price", mark_price))
            sl = float(sig.get("stop_loss"))
            tp = float(sig.get("target"))
            conf = float(sig.get("confidence", 5.0))
            atr = float(sig.get("market_volatility", 0.0))
        except Exception:
            return

        # circuit/loss cooldown/limits
        if self._circuit_tripped() or (time.time() < self._cooldown_until):
            return
        if len(self.active) >= self.MAX_CONCURRENT:
            return
        if self.trades_closed_today >= self.MAX_TRADES_DAY:
            return

        # idempotency
        key = f"{symbol}|{direction}|{round(entry,2)}|{self.session_date}"
        if not self._entry_guard(key):
            return

        # size by risk (lots)
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

        # place entry (MARKET by default)
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

        # set up exits (TP/SL/partials or GTT)
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

        # track active
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

    # ---------------- workers ---------------- #

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
                # get best current price
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
        # Let executor detect fills and convert to a list of (entry_id, exit_price)
        try:
            filled = self.executor.sync_and_enforce_oco()  # type: ignore
            if filled:
                self._finalize_fills(filled)
        except Exception:
            pass

    def _process_active_trades(self) -> None:
        """Periodic close detection (rely on executor’s OCO sync)."""
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
            # attempt to infer exit_reason from executor’s internal record if available
            exit_reason = None
            try:
                rec = getattr(self.executor, "orders", {}).get(str(entry_id))  # type: ignore
                exit_reason = getattr(rec, "exit_reason", None) if rec else None
            except Exception:
                exit_reason = None
            at.exit_reason = exit_reason or "exit-filled"

            # PnL (direction-aware)
            direction_mult = 1 if at.direction == "BUY" else -1
            pnl = (at.exit_price - at.entry_price) * direction_mult * at.quantity
            self.daily_pnl += float(pnl)
            self.trades_closed_today += 1
            self.trades.append(asdict(at))
            self._log_trade(at, pnl)
            closed_ids.append(at.entry_id)

            logger.info("⏹ Closed %s @ %.2f (%s) PnL=%.2f",
                        at.symbol, at.exit_price, at.exit_reason, pnl)

            # loss streak / cooldown
            if pnl < 0:
                self.loss_streak += 1
                if self.loss_streak >= self.LOSS_LIMIT:
                    self._cooldown_until = time.time() + self.LOSS_STREAK_PAUSE * 60
            else:
                self.loss_streak = 0

            # session ladders (R-based)
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

    # ---------------- daily / auto-exit ---------------- #

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
                # best-effort flatten
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

    # ---------------- files / persistence ---------------- #

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
                    # allow executor to reconcile with broker/sim state if it supports it
                    self.executor.reattach(self.active)  # type: ignore
                except Exception:
                    pass
        except Exception:
            self.active = {}

    # ---------------- shutdown ---------------- #

    def shutdown(self) -> None:
        try:
            self._trailing_evt.set()
            self._oco_evt.set()
            self._persist_state()
            if self.tg:
                self.tg.stop_polling()
        except Exception:
            pass