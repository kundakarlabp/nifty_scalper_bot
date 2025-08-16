# src/data_streaming/realtime_trader.py
"""
Real-time trader with:
- Telegram control (daemon polling)
- Adaptive main loop (peak/off-peak cadence)
- Risk-based lot sizing (RISK_PER_TRADE × equity; ATR-/SL-aware)
- Warmup & trading-hours filters (IST aware, optional time windows)
- Options strike resolution (ATM ± range via cached instruments)
- Strategy signals (spot-driven; options execution)
- Spread guard (RANGE or LTP_MID; optional dynamic scaling)
- Partial TP, breakeven hop, trailing SL (delegated to OrderExecutor)
- Daily circuit breaker + loss cooldown + trade/day cap + streak controls
- CSV trade log + daily rollover + idempotent entry protection
- Multi-timeframe (MTF) gate (e.g., 5m trend filter)
- Regime filters (ADX / BB-width) with TREND/RANGE/AUTO modes
- 3-loss shutdown at trader level
- Session auto-exit at Config.SESSION_AUTO_EXIT_TIME

Telegram toggles (supported now via controller’s command router):
  /start, /stop, /mode live|shadow
  /mode quality on|off     ← quality mode toggle
  /refresh                 ← refresh instruments cache
  /status, /summary, /health, /emergency

NOTE: Controller help will be updated when you upload telegram_controller.py for edit.
"""

from __future__ import annotations

import csv
import logging
import os
import threading
import atexit
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import schedule
from datetime import datetime, timedelta, date, time as dtime
import time

from src.config import Config
from src.strategies.scalping_strategy import EnhancedScalpingStrategy
from src.risk.position_sizing import PositionSizing, get_live_account_balance
from src.execution.order_executor import OrderExecutor
from src.notifications.telegram_controller import TelegramController
from src.utils.strike_selector import (
    get_instrument_tokens,
    fetch_cached_instruments,
)
from src.utils.atr_helper import compute_atr_df
from src.utils.indicators import (
    calculate_ema,
    calculate_adx,
    calculate_bb_width,
    calculate_vwap,
)

logger = logging.getLogger(__name__)


# ----------------------------- time helpers ----------------------------- #

def _ist_now() -> datetime:
    """IST clock without pytz (UTC+5:30)."""
    return datetime.utcnow() + timedelta(hours=5, minutes=30)


def _between(now_t: dtime, start_hm: str, end_hm: str) -> bool:
    s = datetime.strptime(start_hm, "%H:%M").time()
    e = datetime.strptime(end_hm, "%H:%M").time()
    return s <= now_t <= e


def _within_any_windows(now_t: dtime, windows: List[Tuple[str, str]]) -> bool:
    for a, b in windows or []:
        try:
            if _between(now_t, a, b):
                return True
        except Exception:
            continue
    return False


def _safe_df(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


# ============================== Trader ============================= #

class RealTimeTrader:
    """
    Production-grade runner coordinating strategy, risk, execution and IO.
    """

    # ---- knobs mirrored from Config / defaults ----
    MAX_CONCURRENT_TRADES = int(getattr(Config, "MAX_CONCURRENT_POSITIONS", 1))
    WARMUP_BARS = int(getattr(Config, "WARMUP_BARS", 25))
    DATA_LOOKBACK_MINUTES = int(getattr(Config, "DATA_LOOKBACK_MINUTES", 45))
    HIST_TIMEFRAME = str(getattr(Config, "HISTORICAL_TIMEFRAME", "minute"))

    # Spread guard
    SPREAD_GUARD_MODE = str(getattr(Config, "SPREAD_GUARD_MODE", "LTP_MID")).upper()  # RANGE|LTP_MID
    SPREAD_GUARD_BA_MAX = float(getattr(Config, "SPREAD_GUARD_BA_MAX", 0.015))
    SPREAD_GUARD_LTPMID_MAX = float(getattr(Config, "SPREAD_GUARD_LTPMID_MAX", 0.015))
    SPREAD_GUARD_PCT = float(getattr(Config, "SPREAD_GUARD_PCT", 0.03))
    DYNAMIC_SPREAD_GUARD = bool(getattr(Config, "DYNAMIC_SPREAD_GUARD", True))
    SPREAD_VOL_LOOKBACK = int(getattr(Config, "SPREAD_VOL_LOOKBACK", 20))

    # Costs / risk / trailing
    SLIPPAGE_BPS = float(getattr(Config, "SLIPPAGE_BPS", 5.0))
    FEES_PER_LOT = float(getattr(Config, "FEES_PER_LOT", 25.0))
    MAX_DAILY_DRAWDOWN_PCT = float(getattr(Config, "MAX_DAILY_DRAWDOWN_PCT", 0.03))
    CIRCUIT_RELEASE_PCT = float(getattr(Config, "CIRCUIT_RELEASE_PCT", 0.015))
    TRAILING_ENABLE = bool(getattr(Config, "TRAILING_ENABLE", True))
    TRAIL_ATR_MULTIPLIER = float(getattr(Config, "ATR_SL_MULTIPLIER", 1.5))
    WORKER_INTERVAL_SEC = int(getattr(Config, "WORKER_INTERVAL_SEC", 10))
    LOG_TRADE_FILE = str(getattr(Config, "LOG_FILE", "logs/trades.csv"))

    LOT_SIZE = int(getattr(Config, "NIFTY_LOT_SIZE", 75))
    MIN_LOTS = int(getattr(Config, "MIN_LOTS", 1))
    MAX_LOTS = int(getattr(Config, "MAX_LOTS", 10))

    RISK_PER_TRADE = float(getattr(Config, "RISK_PER_TRADE", 0.01))
    MAX_TRADES_PER_DAY = int(getattr(Config, "MAX_TRADES_PER_DAY", 30))
    LOSS_COOLDOWN_MIN = int(getattr(Config, "LOSS_COOLDOWN_MIN", 2))
    PEAK_POLL_SEC = int(getattr(Config, "PEAK_POLL_SEC", 12))
    OFFPEAK_POLL_SEC = int(getattr(Config, "OFFPEAK_POLL_SEC", 25))

    # streak/session ladders
    LOSS_STREAK_HALVE_SIZE = int(getattr(Config, "LOSS_STREAK_HALVE_SIZE", 3))
    LOSS_STREAK_PAUSE_MIN = int(getattr(Config, "LOSS_STREAK_PAUSE_MIN", 20))
    DAY_STOP_AFTER_POS_R = float(getattr(Config, "DAY_STOP_AFTER_POS_R", 4.0))
    DAY_HALF_SIZE_AFTER_POS_R = float(getattr(Config, "DAY_HALF_SIZE_AFTER_POS_R", 2.0))
    DAY_STOP_AFTER_NEG_R = float(getattr(Config, "DAY_STOP_AFTER_NEG_R", -3.0))

    # hours / filters
    USE_IST_CLOCK = bool(getattr(Config, "USE_IST_CLOCK", True))
    TIME_FILTER_START = str(getattr(Config, "TIME_FILTER_START", "09:15"))
    TIME_FILTER_END = str(getattr(Config, "TIME_FILTER_END", "15:20"))
    SKIP_FIRST_MIN = int(getattr(Config, "SKIP_FIRST_MIN", 10))
    ENABLE_TIME_BUCKETS = bool(getattr(Config, "ENABLE_TIME_BUCKETS", False))
    TIME_BUCKETS = list(getattr(Config, "TIME_BUCKETS", []))
    ENABLE_EVENT_WINDOWS = bool(getattr(Config, "ENABLE_EVENT_WINDOWS", False))
    EVENT_WINDOWS = list(getattr(Config, "EVENT_WINDOWS", []))

    STRIKE_RANGE = int(getattr(Config, "STRIKE_RANGE", 3))  # ± around ATM
    IDEMP_TTL_SEC = int(getattr(Config, "IDEMP_TTL_SEC", 60))

    # MTF & Regime
    ENABLE_MTF_FILTER = bool(getattr(Config, "ENABLE_MTF_FILTER", True))
    MTF_TIMEFRAME = str(getattr(Config, "MTF_TIMEFRAME", "5minute"))
    MTF_EMA_FAST = int(getattr(Config, "MTF_EMA_FAST", 21))
    MTF_EMA_SLOW = int(getattr(Config, "MTF_EMA_SLOW", 50))
    ENABLE_REGIME_FILTER = bool(getattr(Config, "ENABLE_REGIME_FILTER", True))
    REGIME_MODE = str(getattr(Config, "REGIME_MODE", "AUTO")).upper()  # AUTO|TREND|RANGE|OFF
    ADX_PERIOD = int(getattr(Config, "ADX_PERIOD", 14))
    ADX_MIN_TREND = float(getattr(Config, "ADX_MIN_TREND", 18.0))
    BB_WINDOW = int(getattr(Config, "BB_WINDOW", 20))
    BB_WIDTH_MIN = float(getattr(Config, "BB_WIDTH_MIN", 0.006))  # ~0.6%
    BB_WIDTH_MAX = float(getattr(Config, "BB_WIDTH_MAX", 0.02))   # ~2%

    # Quality toggle baseline
    QUALITY_MODE_DEFAULT = bool(getattr(Config, "QUALITY_MODE_DEFAULT", False))
    QUALITY_SCORE_BUMP = float(getattr(Config, "QUALITY_SCORE_BUMP", 1.0))  # raises min score need

    # Session auto-exit
    SESSION_AUTO_EXIT_TIME = str(getattr(Config, "SESSION_AUTO_EXIT_TIME", "15:20"))

    def __init__(self) -> None:
        self._lock = threading.RLock()

        self.is_trading: bool = False
        self.live_mode: bool = bool(getattr(Config, "ENABLE_LIVE_TRADING", False))

        # Session
        self.daily_start_equity: float = float(get_live_account_balance() or 0.0)
        self.session_date: date = self._now().date()
        self.daily_pnl: float = 0.0
        self.session_R: float = 0.0
        self.trades_closed_today: int = 0
        self.loss_streak: int = 0
        self._cooldown_until_ts: float = 0.0
        self._shutdown_after_3_losses: bool = False
        self.quality_mode: bool = self.QUALITY_MODE_DEFAULT

        # Trades state
        self.trades: List[Dict[str, Any]] = []             # closed trades for the day
        self.active_trades: Dict[str, Dict[str, Any]] = {} # entry_order_id → info

        # Idempotency (avoid duplicate entries)
        self._recent_entry_keys: Dict[str, float] = {}     # key → expiry_ts

        # Instruments cache
        self._nfo_cache: List[Dict[str, Any]] = []
        self._nse_cache: List[Dict[str, Any]] = []
        self._cache_ts: float = 0.0
        self._CACHE_TTL = 300.0
        self._cache_lock = threading.RLock()

        # Components
        self._init_components()

        # Telegram
        self._polling_thread: Optional[threading.Thread] = None
        self._start_polling()

        # Background workers
        self._trailing_evt = threading.Event()
        self._oco_evt = threading.Event()
        self._start_workers()

        # Adaptive scheduler
        self._data_job = None
        self._setup_adaptive_scheduler()

        # CSV log
        self._prepare_trade_log()

        # Shutdown hook
        atexit.register(self.shutdown)

        logger.info("RealTimeTrader initialized.")

    # -------------------- init -------------------- #

    def _init_components(self) -> None:
        try:
            self.strategy = EnhancedScalpingStrategy(
                base_stop_loss_points=Config.BASE_STOP_LOSS_POINTS,
                base_target_points=Config.BASE_TARGET_POINTS,
                confidence_threshold=Config.CONFIDENCE_THRESHOLD,
                min_score_threshold=int(Config.MIN_SIGNAL_SCORE),
            )
        except Exception as e:
            logger.warning(f"Strategy init failed: {e}")
            self.strategy = None

        try:
            self.risk = PositionSizing()
            try:
                # Some versions expose set_equity; otherwise set attribute.
                self.risk.set_equity(float(self.daily_start_equity or 0.0))  # type: ignore
            except Exception:
                setattr(self.risk, "equity", float(self.daily_start_equity or 0.0))
        except Exception as e:
            logger.warning(f"Risk manager init failed: {e}")
            self.risk = PositionSizing()

        self.executor = self._init_executor()

        try:
            self.tg = TelegramController(
                status_callback=self.get_status,
                control_callback=self._handle_control,
                summary_callback=self.get_summary,
            )
        except Exception as e:
            logger.warning(f"Telegram init failed: {e}")
            self.tg = None

    def _build_live_executor(self) -> OrderExecutor:
        from kiteconnect import KiteConnect
        api_key = getattr(Config, "ZERODHA_API_KEY", None)
        access_token = getattr(Config, "KITE_ACCESS_TOKEN", None) or getattr(
            Config, "ZERODHA_ACCESS_TOKEN", None
        )
        if not api_key or not access_token:
            raise RuntimeError("ZERODHA_API_KEY or KITE_ACCESS_TOKEN missing")

        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        logger.info("✅ Live executor created (KiteConnect).")
        return OrderExecutor(kite=kite)

    def _init_executor(self) -> OrderExecutor:
        if self.live_mode:
            try:
                return self._build_live_executor()
            except Exception as exc:
                logger.error("Live init failed, switching to simulation: %s", exc, exc_info=True)
                self.live_mode = False
        logger.info("Live trading disabled → simulation executor.")
        return OrderExecutor()

    # -------------------- csv log -------------------- #

    def _prepare_trade_log(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.LOG_TRADE_FILE) or ".", exist_ok=True)
            if not os.path.exists(self.LOG_TRADE_FILE):
                with open(self.LOG_TRADE_FILE, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(
                        ["date", "order_id", "symbol", "direction", "contracts",
                         "entry", "exit", "pnl", "fees", "net_pnl", "confidence",
                         "atr", "mode", "comment"]
                    )
        except Exception as e:
            logger.warning(f"Trade log init failed: {e}")

    def _append_trade_log(self, row: List[Any]) -> None:
        try:
            with open(self.LOG_TRADE_FILE, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)
        except Exception as e:
            logger.debug(f"Trade log append failed: {e}")

    # -------------------- scheduler -------------------- #

    def _setup_adaptive_scheduler(self) -> None:
        try:
            schedule.clear()
        except Exception:
            pass

        # Cadence + health
        schedule.every(5).seconds.do(self._ensure_cadence)
        schedule.every(max(60, int(getattr(Config, "BALANCE_LOG_INTERVAL_MIN", 30)) * 60)).seconds.do(
            self.refresh_account_balance
        )
        schedule.every(30).seconds.do(self._maybe_rollover_daily)

        # Auto-exit session
        schedule.every(15).seconds.do(self._maybe_session_auto_exit)

        logger.info("Adaptive scheduler primed.")

    def _ensure_cadence(self) -> None:
        try:
            sec = self._current_poll_seconds()
            if self._data_job and getattr(self._data_job, "interval", None) and self._data_job.interval.seconds == sec:
                return
            if self._data_job:
                schedule.cancel_job(self._data_job)
            self._data_job = schedule.every(sec).seconds.do(self._smart_tick)
            logger.info("Data loop cadence → every %ds.", sec)
        except Exception as e:
            logger.debug(f"Cadence ensure error: {e}")

    def _current_poll_seconds(self) -> int:
        try:
            now_t = self._now().time()
            in_peak = (dtime(9, 20) <= now_t <= dtime(11, 30)) or (dtime(13, 30) <= now_t <= dtime(15, 10))
            return self.PEAK_POLL_SEC if in_peak else self.OFFPEAK_POLL_SEC
        except Exception:
            return self.OFFPEAK_POLL_SEC

    def run(self) -> None:
        logger.info("🟢 RealTimeTrader.run() started.")
        while True:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                logger.error(f"Run loop error: {e}", exc_info=True)
                time.sleep(2)

    # -------------------- workers -------------------- #

    def _start_workers(self) -> None:
        t1 = threading.Thread(target=self._trailing_worker, daemon=True)
        t1.start()
        t2 = threading.Thread(target=self._oco_worker, daemon=True)
        t2.start()

    def _trailing_worker(self) -> None:
        while not self._trailing_evt.is_set():
            try:
                if self.TRAILING_ENABLE and self.is_trading and not self._is_circuit_breaker_tripped():
                    self._trailing_tick()
            except Exception as e:
                logger.debug(f"Trailing worker error: {e}")
            self._trailing_evt.wait(self.WORKER_INTERVAL_SEC)

    def _oco_worker(self) -> None:
        while not self._oco_evt.is_set():
            try:
                if self.is_trading:
                    self._oco_tick()
            except Exception as e:
                logger.debug(f"OCO worker error: {e}")
            self._oco_evt.wait(self.WORKER_INTERVAL_SEC)

    def _trailing_tick(self) -> None:
        with self._lock:
            items = list(self.active_trades.items())

        for oid, tr in items:
            if tr.get("status") != "OPEN":
                continue
            atr = float(tr.get("atr") or 0.0)
            if atr <= 0:
                continue
            get_last = getattr(self.executor, "get_last_price", None)
            ltp = get_last(tr.get("symbol")) if callable(get_last) else None
            if ltp is None:
                ltp = float(tr.get("last_close") or 0.0)
            if not ltp or ltp <= 0:
                continue
            try:
                self.executor.update_trailing_stop(oid, float(ltp), float(atr))
            except Exception:
                pass

    def _oco_tick(self) -> None:
        try:
            sync = getattr(self.executor, "sync_and_enforce_oco", None)
            filled = sync() if callable(sync) else []
        except Exception:
            filled = []

        actives_raw = self.executor.get_active_orders()
        if isinstance(actives_raw, dict):
            active_ids = set(actives_raw.keys())
        else:
            try:
                active_ids = {getattr(o, "order_id", None) for o in (actives_raw or [])} - {None}
            except Exception:
                active_ids = set()

        with self._lock:
            to_finalize: List[str] = []

            for entry_id, fill_px in filled or []:
                tr = self.active_trades.get(entry_id)
                if tr and tr.get("status") == "OPEN":
                    tr["exit_price"] = float(fill_px)
                    to_finalize.append(entry_id)

            for entry_id, tr in list(self.active_trades.items()):
                if tr.get("status") != "OPEN":
                    continue
                if entry_id not in active_ids:
                    # best-effort fallback — assume exit at closer of SL/TP
                    px = float(tr.get("exit_fallback") or tr.get("target") or tr.get("stop_loss") or 0.0)
                    tr["exit_price"] = px
                    to_finalize.append(entry_id)

            for entry_id in to_finalize:
                self._finalize_trade(entry_id)

    # -------------------- telegram -------------------- #

    def _start_polling(self) -> None:
        if self._polling_thread and self._polling_thread.is_alive():
            return
        try:
            if self.tg:
                self.tg.send_startup_alert()
        except Exception:
            pass
        try:
            if self.tg:
                self._polling_thread = threading.Thread(target=self.tg.start_polling, daemon=True)
                self._polling_thread.start()
                logger.info("✅ Telegram polling started (daemon).")
        except Exception as e:
            logger.error(f"Polling thread error: {e}")

    def _stop_polling(self) -> None:
        logger.info("🛑 Stopping Telegram polling…")
        if self.tg:
            try:
                self.tg.stop_polling()
            except Exception:
                pass
        if self._polling_thread and self._polling_thread.is_alive():
            if threading.current_thread() != self._polling_thread:
                self._polling_thread.join(timeout=3)
        self._polling_thread = None

    def start(self) -> bool:
        with self._lock:
            if self.is_trading:
                self._safe_send_message("🟢 Trader already running.")
                return True
            self.is_trading = True
            self._shutdown_after_3_losses = False  # reset when starting
        self._safe_send_alert("START")
        logger.info("✅ Trading started.")
        return True

    def stop(self) -> bool:
        with self._lock:
            if not self.is_trading:
                self._safe_send_message("🟨 Trader is already stopped.")
                return True
            self.is_trading = False
        self._safe_send_alert("STOP")
        logger.info("🛑 Trading stopped.")
        return True

    def emergency_stop_all(self) -> bool:
        try:
            self.stop()
            self.executor.cancel_all_orders()
            self._safe_send_message("🛑 Emergency stop executed. All open orders cancelled (best-effort).")
            return True
        except Exception as e:
            logger.error(f"Emergency stop failed: {e}")
            return False

    def _handle_control(self, command: str, arg: str = "") -> bool:
        command = (command or "").strip().lower()
        arg = (arg or "").strip().lower()
        logger.info(f"Command: /{command} {arg}")
        try:
            if command == "start":
                return self.start()
            if command == "stop":
                return self.stop()
            if command == "mode":
                # live/shadow OR quality toggle
                if arg in ("live", "l"):
                    return self.enable_live_trading()
                if arg in ("shadow", "paper", "sim", "s"):
                    return self.disable_live_trading()
                if arg.startswith("quality"):
                    # support: "quality on", "quality off", "quality:on", "quality:off"
                    flag = ("on" in arg) or (arg.endswith(":on"))
                    self.quality_mode = bool(flag)
                    self._safe_send_message(f"✨ Quality mode: {'ON' if self.quality_mode else 'OFF'}")
                    return True
                self._safe_send_message("⚠️ Usage: `/mode live|shadow` or `/mode quality on|off`", parse_mode="Markdown")
                return False
            if command == "refresh":
                return self._force_refresh_cache()
            if command == "status":
                return self._send_detailed_status()
            if command == "health":
                return self._run_health_check()
            if command == "emergency":
                return self.emergency_stop_all()
            self._safe_send_message(f"❌ Unknown command: `{command}`", parse_mode="Markdown")
            return False
        except Exception as e:
            logger.error(f"Control error: {e}")
            return False

    # -------------------- mode switching -------------------- #

    def enable_live_trading(self) -> bool:
        with self._lock:
            if self.is_trading:
                self._safe_send_message("🛑 Stop trading first: `/stop`", parse_mode="Markdown")
                return False
            if self.live_mode:
                self._safe_send_message("🟢 Already in LIVE mode.")
                return True
            try:
                self.executor = self._build_live_executor()
                self.live_mode = True
                self._safe_send_message("🟢 Switched to LIVE mode.")
                return True
            except Exception as e:
                self._safe_send_message(f"❌ Live mode failed: {e}")
                logger.error("enable_live_trading error: %s", e, exc_info=True)
                return False

    def disable_live_trading(self) -> bool:
        with self._lock:
            if self.is_trading:
                self._safe_send_message("🛑 Stop trading first: `/stop`", parse_mode="Markdown")
                return False
            if not self.live_mode:
                self._safe_send_message("🛡️ Already in SHADOW mode.")
                return True
            self.executor = OrderExecutor()
            self.live_mode = False
            self._safe_send_message("🛡️ Switched to SHADOW (simulation) mode.")
            return True

    # -------------------- utils -------------------- #

    def _now(self) -> datetime:
        return _ist_now() if self.USE_IST_CLOCK else datetime.utcnow()

    def _safe_send_message(self, text: str, parse_mode: Optional[str] = None) -> None:
        try:
            if self.tg:
                self.tg.send_message(text, parse_mode=parse_mode)
        except Exception:
            pass

    def _safe_send_alert(self, action: str) -> None:
        try:
            if self.tg:
                self.tg.send_alert(action)
        except Exception:
            pass

    def _get_kite(self):
        return getattr(self.executor, "kite", None)

    # -------------------- market data -------------------- #

    def _fetch_ohlc(self, instrument_token: int, timeframe: str, minutes_back: int) -> pd.DataFrame:
        """
        Fetch OHLC using Kite if available; otherwise return empty df.
        """
        kite = self._get_kite()
        if not kite or not instrument_token:
            return pd.DataFrame()
        end = self._now()
        start = end - timedelta(minutes=max(minutes_back, self.WARMUP_BARS + 5))
        try:
            data = kite.historical_data(instrument_token, start, end, interval=timeframe, oi=False) or []
            df = pd.DataFrame(data)
            if not df.empty and "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
            return df
        except Exception as e:
            logger.debug(f"_fetch_ohlc error: {e}")
            return pd.DataFrame()

    # -------------------- filters -------------------- #

    def _is_trading_window(self) -> bool:
        now = self._now()
        t = now.time()

        # Core window
        if not _between(t, self.TIME_FILTER_START, self.TIME_FILTER_END):
            return False

        # Optional buckets/events
        if self.ENABLE_TIME_BUCKETS and self.TIME_BUCKETS:
            if not _within_any_windows(t, self.TIME_BUCKETS):
                return False
        if self.ENABLE_EVENT_WINDOWS and self.EVENT_WINDOWS:
            if _within_any_windows(t, self.EVENT_WINDOWS) is False:
                return False

        # Skip first N minutes (warmup)
        if t < (datetime.strptime(self.TIME_FILTER_START, "%H:%M") + timedelta(minutes=self.SKIP_FIRST_MIN)).time():
            return False

        return True

    def _passes_mtf_filter(self, spot_token: Optional[int]) -> bool:
        if not self.ENABLE_MTF_FILTER or not spot_token:
            return True

        df5 = self._fetch_ohlc(spot_token, self.MTF_TIMEFRAME, max(60, self.DATA_LOOKBACK_MINUTES))
        if df5.empty or len(df5) < max(self.MTF_EMA_SLOW + 2, 30):
            return True  # don't block on missing data

        ema_fast = calculate_ema(df5, self.MTF_EMA_FAST)
        ema_slow = calculate_ema(df5, self.MTF_EMA_SLOW)
        trend_up = bool(ema_fast.iloc[-1] > ema_slow.iloc[-1])
        trend_down = bool(ema_fast.iloc[-1] < ema_slow.iloc[-1])

        # Optional ADX assist
        adx, di_p, di_m = calculate_adx(df5, period=self.ADX_PERIOD)
        strong = bool(adx.iloc[-1] >= self.ADX_MIN_TREND)

        # Gate: if regime is TREND_ONLY, require both slope & strength
        if self.REGIME_MODE == "TREND":
            return (trend_up or trend_down) and strong
        # In AUTO mode, only gate against contra-trend when strong trend present
        return True

    def _passes_regime_filter(self, spot_token: Optional[int]) -> bool:
        if not self.ENABLE_REGIME_FILTER or self.REGIME_MODE == "OFF" or not spot_token:
            return True

        df5 = self._fetch_ohlc(spot_token, self.MTF_TIMEFRAME, max(60, self.DATA_LOOKBACK_MINUTES))
        if df5.empty or len(df5) < max(self.BB_WINDOW + 5, 30):
            return True

        # Regime by ADX + BB width (as %)
        adx, di_p, di_m = calculate_adx(df5, period=self.ADX_PERIOD)
        upband, lowband = calculate_bb_width(df5, window=self.BB_WINDOW)
        width = ((upband - lowband) / df5["close"]).iloc[-1]

        adx_last = float(adx.iloc[-1])
        width_last = float(width)

        is_trend = (adx_last >= self.ADX_MIN_TREND) and (width_last >= self.BB_WIDTH_MIN)
        is_range = (adx_last < self.ADX_MIN_TREND) and (width_last <= self.BB_WIDTH_MAX)

        if self.REGIME_MODE == "TREND":
            return is_trend
        if self.REGIME_MODE == "RANGE":
            return is_range
        # AUTO → allow both; strategy logic should favor setups appropriate to regime.
        return True

    def _spread_guard_ok(self, symbol: str, ltp: Optional[float]) -> bool:
        """
        Basic microstructure guard. If best bid/ask available via executor, use it;
        otherwise fallback to LTP-based proxy or candle-range proxy.
        """
        try:
            get_bba = getattr(self.executor, "get_best_bid_ask", None)
            if callable(get_bba):
                bb = get_bba(symbol)
                if not bb:
                    return True
                bid = float(bb.get("bid") or 0.0)
                ask = float(bb.get("ask") or 0.0)
                if bid <= 0 or ask <= 0 or ask < bid:
                    return True
                mid = 0.5 * (bid + ask)
                ba = (ask - bid) / max(mid, 1e-9)
                return ba <= self.SPREAD_GUARD_BA_MAX

            # LTP vs mid proxy if mid available
            get_mid = getattr(self.executor, "get_mid_price", None)
            if callable(get_mid):
                mid = float(get_mid(symbol) or 0.0)
                if mid > 0 and ltp and ltp > 0:
                    diff = abs(ltp - mid) / mid
                    return diff <= self.SPREAD_GUARD_LTPMID_MAX
        except Exception:
            pass

        # Fallback RANGE proxy disabled here (needs candle). Allow trading by default.
        return True

    # -------------------- main tick -------------------- #

    def _smart_tick(self) -> None:
        if not self.is_trading:
            return
        if self._shutdown_after_3_losses:
            return
        if self._is_circuit_breaker_tripped():
            return
        if time.time() < self._cooldown_until_ts:
            return
        if not self._is_trading_window() and not getattr(Config, "ALLOW_OFFHOURS_TESTING", False):
            return

        # Refresh instruments cache
        self._maybe_refresh_cache()

        # Resolve instruments/strikes
        tokens = self._resolve_tokens()
        if not tokens:
            return

        spot_token = tokens.get("spot_token")
        ce_sym = tokens.get("ce_symbol")
        pe_sym = tokens.get("pe_symbol")
        ce_tok = tokens.get("ce_token")
        pe_tok = tokens.get("pe_token")

        # MTF & Regime gates
        if not self._passes_mtf_filter(spot_token):
            return
        if not self._passes_regime_filter(spot_token):
            return

        # Fetch spot candles for strategy
        spot_df = self._fetch_ohlc(spot_token, self.HIST_TIMEFRAME, self.DATA_LOOKBACK_MINUTES)
        if spot_df.empty or len(spot_df) < self.WARMUP_BARS:
            return

        # Generate signal
        if not self.strategy:
            return
        try:
            signal = self.strategy.generate_signal(spot_df)  # EXPECTED: dict
        except Exception as e:
            logger.debug(f"Strategy error: {e}")
            return
        if not isinstance(signal, dict) or not signal:
            return

        # Quality mode raises bar
        conf = float(signal.get("confidence", 0.0) or 0.0)
        min_score = float(getattr(self.strategy, "min_score_threshold", 0) or 0)
        if self.quality_mode:
            conf -= self.QUALITY_SCORE_BUMP  # demand higher true confidence
        if conf < float(getattr(Config, "CONFIDENCE_THRESHOLD", 6.0)):
            return

        direction = (signal.get("signal") or signal.get("direction") or "").upper()
        if direction not in ("BUY", "SELL"):
            return

        # Choose option leg
        side_symbol = ce_sym if direction == "BUY" else pe_sym
        side_token = ce_tok if direction == "BUY" else pe_tok
        if not side_symbol or not side_token:
            return

        # Compute ATR & entry/SL/TP
        atr_series = compute_atr_df(spot_df, period=Config.ATR_PERIOD)
        atr = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0

        entry_price = float(signal.get("entry_price") or 0.0)
        stop_loss = float(signal.get("stop_loss") or 0.0)
        target = float(signal.get("target") or 0.0)

        # If strategy didn’t supply option entry/SL/TP, use simple offsets from last close
        try:
            get_last = getattr(self.executor, "get_last_price", None)
            opt_ltp = get_last(side_symbol) if callable(get_last) else None
        except Exception:
            opt_ltp = None
        if opt_ltp and opt_ltp > 0 and (entry_price <= 0):
            entry_price = float(opt_ltp)

        if entry_price <= 0:
            return

        # Spread guard
        if not self._spread_guard_ok(side_symbol, entry_price):
            return

        # Fallback SL/TP if missing using ATR multiples on option premium
        if stop_loss <= 0:
            stop_loss = max(0.05 * entry_price, entry_price * (1.0 - 0.05))  # 5% default
        if target <= 0:
            target = entry_price * (1.0 + 0.15)  # 15% default

        # Idempotency
        key = f"{self.session_date}:{side_symbol}:{direction}:{int(entry_price / max(1.0, Config.TICK_SIZE))}"
        now_ts = time.time()
        self._recent_entry_keys = {k: v for k, v in self._recent_entry_keys.items() if v > now_ts}
        if key in self._recent_entry_keys:
            return
        self._recent_entry_keys[key] = now_ts + self.IDEMP_TTL_SEC

        # Position sizing
        pos = self.risk.calculate_position_size(
            entry_price=entry_price,
            stop_loss=stop_loss,
            signal_confidence=conf,
            market_volatility=float(atr / max(1e-9, spot_df["close"].iloc[-1])),
            lot_size=self.LOT_SIZE,
        )
        if not pos:
            return
        qty = int(pos.get("quantity") or 0)
        if qty <= 0:
            return

        # Place order
        try:
            order_id = self.executor.place_entry_order(
                symbol=side_symbol,
                direction=direction,
                quantity=qty * self.LOT_SIZE,
                price=entry_price,
                product=Config.DEFAULT_PRODUCT,
                order_type=Config.DEFAULT_ORDER_TYPE,
                validity=Config.DEFAULT_VALIDITY,
                live=self.live_mode,
            )
            if not order_id:
                return
        except Exception as e:
            logger.debug(f"Entry order error: {e}")
            return

        # Setup exits (OCO/GTT as configured inside executor)
        try:
            self.executor.setup_gtt_orders(
                entry_order_id=order_id,
                symbol=side_symbol,
                direction=direction,
                entry_price=entry_price,
                stop_loss=stop_loss,
                target=target,
                lot_size=self.LOT_SIZE,
                live=self.live_mode,
            )
        except Exception as e:
            logger.debug(f"setup_gtt_orders error: {e}")

        # Track active trade
        tr = {
            "status": "OPEN",
            "symbol": side_symbol,
            "direction": direction,
            "qty_lots": qty,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "target": target,
            "atr": atr,
            "confidence": conf,
            "open_ts": self._now().isoformat(),
        }
        with self._lock:
            self.active_trades[str(order_id)] = tr

        # Telegram alert
        try:
            if self.tg:
                self.tg.send_signal_alert(token=int(side_token), signal={
                    "signal": direction,
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "target": target,
                    "confidence": conf,
                }, position={"quantity": qty})
        except Exception:
            pass

    # -------------------- cache / tokens -------------------- #

    def _maybe_refresh_cache(self) -> None:
        if (time.time() - self._cache_ts) < self._CACHE_TTL:
            return
        self._force_refresh_cache()

    def _force_refresh_cache(self) -> bool:
        try:
            kite = self._get_kite()
            caches = fetch_cached_instruments(kite) if kite else {"NFO": [], "NSE": []}
            with self._cache_lock:
                self._nfo_cache = caches.get("NFO") or []
                self._nse_cache = caches.get("NSE") or []
                self._cache_ts = time.time()
            self._safe_send_message("🔄 Instruments cache refreshed.")
            return True
        except Exception as e:
            logger.debug(f"_force_refresh_cache error: {e}")
            return False

    def _resolve_tokens(self) -> Optional[Dict[str, Any]]:
        try:
            kite = self._get_kite()
            with self._cache_lock:
                nfo = self._nfo_cache
                nse = self._nse_cache
            if not kite or not nfo or not nse:
                return None
            res = get_instrument_tokens(
                symbol=getattr(Config, "TRADE_SYMBOL", "NIFTY"),
                kite_instance=kite,
                cached_nfo_instruments=nfo,
                cached_nse_instruments=nse,
                offset=0,
                strike_range=self.STRIKE_RANGE,
            )
            return res
        except Exception as e:
            logger.debug(f"_resolve_tokens error: {e}")
            return None

    # -------------------- finalize / pnl -------------------- #

    def _finalize_trade(self, entry_order_id: str) -> None:
        with self._lock:
            tr = self.active_trades.pop(entry_order_id, None)
        if not tr:
            return

        entry = float(tr.get("entry_price") or 0.0)
        exit_px = float(tr.get("exit_price") or 0.0)
        qty_lots = int(tr.get("qty_lots") or 0)
        contracts = qty_lots * self.LOT_SIZE

        direction = tr.get("direction")
        gross = (exit_px - entry) * contracts if direction == "BUY" else (entry - exit_px) * contracts

        # Fees/slippage model (simple)
        fees = (abs(entry) + abs(exit_px)) * contracts * (Config.FEES_PCT_PER_SIDE / 100.0)
        slippage = (Config.SLIPPAGE_BPS / 10000.0) * (abs(entry) + abs(exit_px)) * contracts
        net = gross - fees - slippage

        # Update session
        self.daily_pnl += net
        self.trades_closed_today += 1

        # Update risk manager & streaks
        cont = True
        try:
            cont = self.risk.update_after_trade(net)
        except Exception:
            pass

        if net < 0:
            self.loss_streak += 1
            if self.loss_streak >= 3:
                self._shutdown_after_3_losses = True
                self.is_trading = False
                self._safe_send_message("⛔ 3-loss shutdown triggered. Trading halted for the day.")
            else:
                # Cooldown
                self._cooldown_until_ts = time.time() + (self.LOSS_COOLDOWN_MIN * 60)
        else:
            self.loss_streak = 0

        # Log
        self._append_trade_log([
            self._now().strftime("%Y-%m-%d %H:%M:%S"),
            entry_order_id,
            tr.get("symbol"),
            direction,
            contracts,
            f"{entry:.2f}",
            f"{exit_px:.2f}",
            f"{gross:.2f}",
            f"{fees+slippage:.2f}",
            f"{net:.2f}",
            f"{tr.get('confidence', 0.0):.2f}",
            f"{tr.get('atr', 0.0):.2f}",
            "LIVE" if self.live_mode else "SIM",
            "",
        ])

        # Circuit breaker check after trade
        if not cont or self._is_circuit_breaker_tripped():
            self.is_trading = False
            self._safe_send_message("🛑 Circuit breaker/drawdown reached. Trading stopped.")

    # -------------------- health / status -------------------- #

    def _is_circuit_breaker_tripped(self) -> bool:
        equity_now = self.daily_start_equity + self.daily_pnl
        if self.daily_start_equity > 0:
            dd = (self.daily_start_equity - equity_now) / self.daily_start_equity
            return dd >= self.MAX_DAILY_DRAWDOWN_PCT
        return False

    def _run_health_check(self) -> bool:
        try:
            kite = self._get_kite()
            if not kite:
                self._safe_send_message("Health: Kite client not initialized (SIM mode).")
                return True
            # Spot LTP check
            try:
                spot_sym = getattr(Config, "SPOT_SYMBOL", "NSE:NIFTY 50")
                l = kite.ltp([spot_sym]) or {}
                ok = bool((l.get(spot_sym) or {}).get("last_price"))
                self._safe_send_message(f"Health: LTP {'OK' if ok else 'FAIL'} | Mode: {'LIVE' if self.live_mode else 'SIM'}")
            except Exception as e:
                self._safe_send_message(f"Health LTP FAIL: {e}")
            return True
        except Exception as e:
            logger.debug(f"_run_health_check error: {e}")
            return False

    def _send_detailed_status(self) -> bool:
        try:
            status = self.get_status()
            lines = [
                "📊 <b>Bot Status</b>",
                f"🔁 <b>Trading:</b> {'🟢 Running' if status['is_trading'] else '🔴 Stopped'}",
                f"🌐 <b>Mode:</b> {'🟢 LIVE' if status['live_mode'] else '🛡️ Shadow'}",
                f"✨ <b>Quality:</b> {'ON' if self.quality_mode else 'OFF'}",
                f"📦 <b>Open Positions:</b> {status['open_positions']}",
                f"📈 <b>Closed Today:</b> {status['closed_today']}",
                f"💰 <b>Daily P&L:</b> {status['daily_pnl']:.2f}",
                f"📅 <b>Session:</b> {status['session_date']}",
            ]
            self._safe_send_message("\n".join(lines), parse_mode="HTML")
            return True
        except Exception as e:
            logger.debug(f"_send_detailed_status error: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            open_positions = len(self.active_trades)
        return {
            "is_trading": self.is_trading,
            "live_mode": self.live_mode,
            "open_positions": open_positions,
            "closed_today": self.trades_closed_today,
            "daily_pnl": float(self.daily_pnl),
            "session_date": str(self.session_date),
            "quality_mode": self.quality_mode,
        }

    def get_summary(self) -> str:
        s = self.get_status()
        return (
            "<b>Daily Summary</b>\n"
            f"📅 Session: {s['session_date']}\n"
            f"🔁 Trading: {'Running' if s['is_trading'] else 'Stopped'}\n"
            f"🌐 Mode: {'LIVE' if s['live_mode'] else 'Shadow'}\n"
            f"✨ Quality: {'ON' if s['quality_mode'] else 'OFF'}\n"
            f"📦 Open positions: {s['open_positions']}\n"
            f"📈 Closed today: {s['closed_today']}\n"
            f"💰 Daily P&L: {s['daily_pnl']:.2f}"
        )

    def refresh_account_balance(self) -> None:
        try:
            bal = get_live_account_balance()
            # Update risk manager equity baseline if needed
            try:
                self.risk.equity = float(bal)  # type: ignore
            except Exception:
                pass
            logger.info(f"💰 Refreshed balance: ₹{bal:.2f}")
        except Exception as e:
            logger.debug(f"refresh_account_balance error: {e}")

    # -------------------- session helpers -------------------- #

    def _maybe_rollover_daily(self) -> None:
        now_d = self._now().date()
        if now_d != self.session_date:
            logger.info("🔄 New trading day detected; resetting counters.")
            self.session_date = now_d
            self.daily_start_equity = float(get_live_account_balance() or 0.0)
            self.daily_pnl = 0.0
            self.trades_closed_today = 0
            self.loss_streak = 0
            self._shutdown_after_3_losses = False
            try:
                self.risk.reset_daily_limits()
            except Exception:
                pass

    def _maybe_session_auto_exit(self) -> None:
        try:
            t = self._now().time()
            auto_t = datetime.strptime(self.SESSION_AUTO_EXIT_TIME, "%H:%M").time()
            if t >= auto_t and self.is_trading:
                # best-effort exit of all open orders/positions
                self.executor.cancel_all_orders()
                self.is_trading = False
                self._safe_send_message(f"🛑 Auto exit @ {self.SESSION_AUTO_EXIT_TIME}. Trading halted for session.")
        except Exception:
            pass

    # -------------------- shutdown -------------------- #

    def shutdown(self) -> None:
        logger.info("Shutting down RealTimeTrader…")
        try:
            self._trailing_evt.set()
            self._oco_evt.set()
        except Exception:
            pass
        try:
            self._stop_polling()
        except Exception:
            pass
        try:
            if self.is_trading:
                self.stop()
        except Exception:
            pass