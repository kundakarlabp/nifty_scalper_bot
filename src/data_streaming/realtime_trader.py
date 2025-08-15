# src/data_streaming/realtime_trader.py
"""
Real-time trader with:
- Telegram control (daemon polling)
- Adaptive main loop (peak/off-peak cadence)
- Risk-based sizing with loss-streak & daily R ladders
- Warmup & IST trading-hours + optional time buckets + event windows
- Options strike resolution via cached instruments
- Strategy signals (spot/option aware)
- Spread/microstructure guard (RANGE or LTP_MID) + min-premium + max-slip
- Partial TP, breakeven hop, trailing SL (via OrderExecutor)
- Daily circuit breaker + loss cooldown
- CSV trade log + daily rollover + idempotent entry cache
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
from math import floor

from src.config import Config
from src.strategies.scalping_strategy import EnhancedScalpingStrategy
from src.risk.position_sizing import get_live_account_balance
from src.execution.order_executor import OrderExecutor
from src.notifications.telegram_controller import TelegramController
from src.utils.strike_selector import (
    get_instrument_tokens,
    fetch_cached_instruments,
)
from src.utils.atr_helper import compute_atr_df

logger = logging.getLogger(__name__)


# ----------------------------- clock helpers ----------------------------- #

def _ist_now() -> datetime:
    return datetime.utcnow() + timedelta(hours=5, minutes=30)


def _in_time_range(now_t: dtime, start_hm: str, end_hm: str) -> bool:
    s = datetime.strptime(start_hm, "%H:%M").time()
    e = datetime.strptime(end_hm, "%H:%M").time()
    return s <= now_t <= e


def _in_any_ranges(now_t: dtime, ranges: List[Tuple[str, str]]) -> bool:
    for a, b in ranges or []:
        if _in_time_range(now_t, a, b):
            return True
    return False


# ============================== Trader ============================= #

class RealTimeTrader:
    # ---- knobs from Config (snapshots) ----
    MAX_CONCURRENT_TRADES = int(getattr(Config, "MAX_CONCURRENT_POSITIONS", 1))
    WARMUP_BARS = int(getattr(Config, "WARMUP_BARS", 25))
    DATA_LOOKBACK_MINUTES = int(getattr(Config, "DATA_LOOKBACK_MINUTES", 45))
    HIST_TIMEFRAME = str(getattr(Config, "HISTORICAL_TIMEFRAME", "minute"))

    # microstructure
    SPREAD_GUARD_MODE = str(getattr(Config, "SPREAD_GUARD_MODE", "RANGE")).upper()
    SPREAD_GUARD_BA_MAX = float(getattr(Config, "SPREAD_GUARD_BA_MAX", 0.015))
    SPREAD_GUARD_LTPMID_MAX = float(getattr(Config, "SPREAD_GUARD_LTPMID_MAX", 0.015))
    SPREAD_GUARD_PCT = float(getattr(Config, "SPREAD_GUARD_PCT", 0.03))
    DYNAMIC_SPREAD_GUARD = bool(getattr(Config, "DYNAMIC_SPREAD_GUARD", True))
    SPREAD_VOL_LOOKBACK = int(getattr(Config, "SPREAD_VOL_LOOKBACK", 20))

    SLIPPAGE_BPS = float(getattr(Config, "SLIPPAGE_BPS", 4.0))
    FEES_PER_LOT = float(getattr(Config, "FEES_PER_LOT", 25.0))
    MAX_DAILY_DRAWDOWN_PCT = float(getattr(Config, "MAX_DAILY_DRAWDOWN_PCT", 0.05))
    CIRCUIT_RELEASE_PCT = float(getattr(Config, "CIRCUIT_RELEASE_PCT", 0.015))
    TRAILING_ENABLE = bool(getattr(Config, "TRAILING_ENABLE", True))
    TRAIL_ATR_MULTIPLIER = float(getattr(Config, "ATR_SL_MULTIPLIER", 1.5))
    WORKER_INTERVAL_SEC = int(getattr(Config, "WORKER_INTERVAL_SEC", 4))
    LOG_TRADE_FILE = str(getattr(Config, "LOG_FILE", "logs/trades.csv"))

    LOT_SIZE = int(getattr(Config, "NIFTY_LOT_SIZE", 75))
    MIN_LOTS = int(getattr(Config, "MIN_LOTS", 1))
    MAX_LOTS = int(getattr(Config, "MAX_LOTS", 15))

    RISK_PER_TRADE = float(getattr(Config, "RISK_PER_TRADE", 0.025))
    MAX_TRADES_PER_DAY = int(getattr(Config, "MAX_TRADES_PER_DAY", 30))
    LOSS_COOLDOWN_MIN = int(getattr(Config, "LOSS_COOLDOWN_MIN", 2))
    PEAK_POLL_SEC = int(getattr(Config, "PEAK_POLL_SEC", 12))
    OFFPEAK_POLL_SEC = int(getattr(Config, "OFFPEAK_POLL_SEC", 25))
    PREFERRED_TIE_RULE = str(getattr(Config, "PREFERRED_TIE_RULE", "TREND")).upper()

    USE_IST_CLOCK = bool(getattr(Config, "USE_IST_CLOCK", True))
    TIME_FILTER_START = str(getattr(Config, "TIME_FILTER_START", "09:20"))
    TIME_FILTER_END = str(getattr(Config, "TIME_FILTER_END", "15:20"))
    SKIP_FIRST_MIN = int(getattr(Config, "SKIP_FIRST_MIN", 5))

    STRIKE_RANGE = int(getattr(Config, "STRIKE_RANGE", 3))

    # quality clamps / guards
    MIN_PREMIUM = float(getattr(Config, "MIN_PREMIUM", 12.0))
    MAX_ENTRY_SLIP_BPS = float(getattr(Config, "MAX_ENTRY_SLIP_BPS", 25.0))
    ORDER_RETRY_LIMIT = int(getattr(Config, "ORDER_RETRY_LIMIT", 2))
    ORDER_RETRY_TICK_OFFSET = int(getattr(Config, "ORDER_RETRY_TICK_OFFSET", 2))

    # streak/daily ladders
    LOSS_STREAK_HALVE_SIZE = int(getattr(Config, "LOSS_STREAK_HALVE_SIZE", 3))
    LOSS_STREAK_PAUSE_MIN = int(getattr(Config, "LOSS_STREAK_PAUSE_MIN", 20))
    DAY_STOP_AFTER_POS_R = float(getattr(Config, "DAY_STOP_AFTER_POS_R", 4.0))
    DAY_HALF_SIZE_AFTER_POS_R = float(getattr(Config, "DAY_HALF_SIZE_AFTER_POS_R", 2.0))
    DAY_STOP_AFTER_NEG_R = float(getattr(Config, "DAY_STOP_AFTER_NEG_R", -3.0))

    # volatility risk scaling
    ATR_TARGET = float(getattr(Config, "ATR_TARGET", 10.0))
    ATR_MIN = float(getattr(Config, "ATR_MIN", 3.0))
    VOL_RISK_CLAMP_MIN = float(getattr(Config, "VOL_RISK_CLAMP_MIN", 0.5))
    VOL_RISK_CLAMP_MAX = float(getattr(Config, "VOL_RISK_CLAMP_MAX", 1.5))

    # time buckets & events
    ENABLE_TIME_BUCKETS = bool(getattr(Config, "ENABLE_TIME_BUCKETS", False))
    TIME_BUCKETS = list(getattr(Config, "TIME_BUCKETS", []))
    ENABLE_EVENT_WINDOWS = bool(getattr(Config, "ENABLE_EVENT_WINDOWS", False))
    EVENT_WINDOWS = list(getattr(Config, "EVENT_WINDOWS", []))

    # idempotency
    IDEMP_TTL_SEC = int(getattr(Config, "IDEMP_TTL_SEC", 60))
    PERSIST_REATTACH_ON_START = bool(getattr(Config, "PERSIST_REATTACH_ON_START", True))

    def __init__(self) -> None:
        self._lock = threading.RLock()

        self.is_trading: bool = False
        self.live_mode: bool = bool(getattr(Config, "ENABLE_LIVE_TRADING", False))

        # PnL / session
        self.daily_pnl: float = 0.0
        self.daily_start_equity: float = float(get_live_account_balance() or 0.0)
        self.session_date: date = (_ist_now() if self.USE_IST_CLOCK else datetime.now()).date()
        self._closed_trades_today = 0
        self._last_closed_was_loss = False
        self._cooldown_until_ts: float = 0.0
        self._cum_r_today: float = 0.0
        self._loss_streak: int = 0

        # Trades
        self.trades: List[Dict[str, Any]] = []                    # closed trades (today)
        self.active_trades: Dict[str, Dict[str, Any]] = {}        # entry_order_id → info

        # Idempotent entries (hash→ts)
        self._entry_memo: Dict[str, float] = {}

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
        self._safe_log_account_balance()

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
                logger.error("Live init failed, using simulation: %s", exc, exc_info=True)
                self.live_mode = False
        logger.info("Live trading disabled → simulation.")
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
                         "entry", "exit", "pnl", "fees", "net_pnl", "confidence", "atr", "mode"]
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
        schedule.every(5).seconds.do(self._ensure_cadence)
        logger.info("Adaptive scheduler primed.")

    def _ensure_cadence(self) -> None:
        try:
            sec = self._current_poll_seconds()
            if self._data_job and self._data_job.interval.seconds == sec:
                return
            if self._data_job:
                schedule.cancel_job(self._data_job)
            self._data_job = schedule.every(sec).seconds.do(self._smart_fetch_and_process)
            logger.info("Data loop cadence → every %ds.", sec)
        except Exception as e:
            logger.debug(f"Cadence ensure error: {e}")

    def _current_poll_seconds(self) -> int:
        try:
            now = (_ist_now() if self.USE_IST_CLOCK else datetime.now()).time()
            in_peak = (dtime(9, 20) <= now <= dtime(11, 30)) or (dtime(13, 30) <= now <= dtime(15, 10))
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
                    if tr["direction"] == "BUY":
                        px = float(tr.get("target") or tr.get("stop_loss") or 0.0)
                    else:
                        px = float(tr.get("stop_loss") or tr.get("target") or 0.0)
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
                if arg in ("live", "l"):
                    return self.enable_live_trading()
                if arg in ("shadow", "paper", "sim", "s"):
                    return self.disable_live_trading()
                self._safe_send_message("⚠️ Usage: `/mode live` or `/mode shadow`", parse_mode="Markdown")
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
            try:
                self.executor = self._build_live_executor()
                self.live_mode = True
                self._refresh_instruments_cache(force=True)
                logger.info("🟢 Switched to LIVE mode.")
                self._safe_send_message("🟢 Switched to *LIVE* mode.", parse_mode="Markdown")
                return True
            except Exception as exc:
                logger.error("Enable LIVE failed: %s", exc, exc_info=True)
                self.executor = OrderExecutor()
                self.live_mode = False
                self._safe_send_message(
                    f"❌ Failed to enable LIVE: `{exc}`\nReverted to SHADOW.", parse_mode="Markdown"
                )
                return False

    def disable_live_trading(self) -> bool:
        with self._lock:
            if self.is_trading:
                self._safe_send_message("🛑 Stop trading first: `/stop`", parse_mode="Markdown")
                return False
            self.executor = OrderExecutor()
            self.live_mode = False
        logger.info("🛡️ Switched to SHADOW (simulation) mode.")
        self._safe_send_message("🛡️ Switched to *SHADOW* (simulation) mode.", parse_mode="Markdown")
        return True

    # -------------------- balance & guards -------------------- #

    def refresh_account_balance(self) -> None:
        try:
            new_bal = float(get_live_account_balance() or 0.0)
            if new_bal > 0:
                if (_ist_now() if self.USE_IST_CLOCK else datetime.now()).date() != self.session_date:
                    self.daily_start_equity = new_bal
                logger.info("💰 Live balance (approx): ₹%.2f", new_bal)
        except Exception as e:
            logger.debug(f"Balance refresh failed: {e}")

    def _safe_log_account_balance(self) -> None:
        try:
            bal = float(get_live_account_balance() or 0.0)
            if bal > 0:
                logger.info("💰 Live balance (approx): ₹%.2f", bal)
        except Exception:
            pass

    def _is_trading_hours(self, now: Optional[datetime] = None) -> bool:
        if not self.USE_IST_CLOCK:
            now = now or datetime.now()
        else:
            now = now or _ist_now()
        t = now.time()

        # base window
        if not _in_time_range(t, self.TIME_FILTER_START, self.TIME_FILTER_END):
            return False

        # optional time buckets
        if self.ENABLE_TIME_BUCKETS and self.TIME_BUCKETS:
            if not _in_any_ranges(t, self.TIME_BUCKETS):
                return False

        # block events
        if self.ENABLE_EVENT_WINDOWS and self.EVENT_WINDOWS:
            if _in_any_ranges(t, self.EVENT_WINDOWS):
                return False

        # skip first SKIP_FIRST_MIN of session
        if t < (dtime(9, 15) if self.USE_IST_CLOCK else dtime(0, 0)):
            return False
        if t <= dtime(9, 15) or (t <= dtime(9, 15 + self.SKIP_FIRST_MIN)):
            # crude “first N minutes” gate for IST
            if _in_time_range(t, "09:15", f"09:{15 + self.SKIP_FIRST_MIN:02d}"):
                return False
        return True

    def _is_circuit_breaker_tripped(self) -> bool:
        if self.daily_start_equity <= 0:
            return False
        dd = -self.daily_pnl / self.daily_start_equity
        if dd >= self.MAX_DAILY_DRAWDOWN_PCT:
            return True
        # R ladder stops
        if self._cum_r_today >= self.DAY_STOP_AFTER_POS_R:
            return True
        if self._cum_r_today <= self.DAY_STOP_AFTER_NEG_R:
            return True
        return False

    def _in_loss_cooldown(self) -> bool:
        return time.time() < self._cooldown_until_ts

    # -------------------- caching & data -------------------- #

    def _refresh_instruments_cache(self, force: bool = False) -> None:
        now = time.time()
        with self._cache_lock:
            if (not force) and (now - self._cache_ts < self._CACHE_TTL) and self._nfo_cache and self._nse_cache:
                return
            if not self.live_mode or not getattr(self.executor, "kite", None):
                # keep empty (shadow mode); caller should handle gracefully
                self._nfo_cache, self._nse_cache = [], []
                self._cache_ts = now
                return
            try:
                caches = fetch_cached_instruments(self.executor.kite)
                self._nfo_cache = caches.get("NFO", []) or []
                self._nse_cache = caches.get("NSE", []) or []
                self._cache_ts = now
                logger.info("📦 Instruments cache refreshed (NFO=%d, NSE=%d).",
                            len(self._nfo_cache), len(self._nse_cache))
            except Exception as e:
                logger.warning(f"Instruments cache refresh failed: {e}")

    # === NOTE ===
    # The following _fetch_* functions are *placeholders* that should be wired to your
    # data layer. They intentionally return empty DataFrames in shadow/offline mode so
    # the file runs without crashing. Replace with your historical fetcher.
    def _fetch_ohlc(self, symbol: str, minutes: int) -> pd.DataFrame:
        # TODO: integrate your data provider to return a DataFrame with open, high, low, close, (volume optional).
        return pd.DataFrame()

    def _fetch_option_ohlc(self, exchange: str, tradingsymbol: str, minutes: int) -> pd.DataFrame:
        return pd.DataFrame()

    # -------------------- main loop -------------------- #

    def _smart_fetch_and_process(self) -> None:
        try:
            now = _ist_now() if self.USE_IST_CLOCK else datetime.now()
            if not self._is_trading_hours(now) and not getattr(Config, "ALLOW_OFFHOURS_TESTING", False):
                if int(time.time()) % 300 < 2:
                    logger.info("⏳ Market closed. Skipping fetch.")
                return

            # session rollover
            if now.date() != self.session_date:
                self._rollover_day(now.date())

            if not self.is_trading:
                return

            if self._is_circuit_breaker_tripped():
                logger.warning("🚫 Circuit breaker active — trading paused.")
                return

            if self._in_loss_cooldown():
                return

            self.fetch_and_process_data()

        except Exception as e:
            logger.error(f"Error in smart fetch and process: {e}", exc_info=True)

    def fetch_and_process_data(self) -> None:
        """One pass: resolve strikes → get data → signal → guards → size → execute."""
        self._refresh_instruments_cache(force=False)

        # Resolve ATM CE/PE for current expiry (offset 0)
        if not self._nfo_cache or not self._nse_cache:
            logger.debug("No instrument cache; skipping.")
            return

        try:
            inf = get_instrument_tokens(
                symbol="NIFTY",
                kite_instance=self.executor.kite if self.live_mode else None,
                cached_nfo_instruments=self._nfo_cache,
                cached_nse_instruments=self._nse_cache,
                offset=0,
                strike_range=self.STRIKE_RANGE,
            )
        except Exception as e:
            logger.debug(f"Strike resolution failed: {e}")
            return

        if not inf or not (inf.get("ce_symbol") or inf.get("pe_symbol")):
            return

        # Choose which side to evaluate first – prefer trend
        sides: List[Tuple[str, str]] = []
        if inf.get("ce_symbol"):
            sides.append(("CE", inf["ce_symbol"]))
        if inf.get("pe_symbol"):
            sides.append(("PE", inf["pe_symbol"]))

        # Pull OHLC (option + spot if needed)
        for opt_type, tsym in sides:
            opt_df = self._fetch_option_ohlc("NFO", tsym, self.DATA_LOOKBACK_MINUTES)
            if opt_df is None or opt_df.empty or len(opt_df) < self.WARMUP_BARS:
                continue

            # Microstructure sanity: min premium
            try:
                last_px = float(opt_df["close"].iloc[-1])
            except Exception:
                last_px = 0.0
            if last_px < self.MIN_PREMIUM:
                continue

            # Dynamic spread proxy using range / realized vol to scale guard
            if self.SPREAD_GUARD_MODE == "RANGE":
                try:
                    w = min(self.SPREAD_VOL_LOOKBACK, len(opt_df))
                    ltp = float(opt_df["close"].iloc[-1])
                    hi = float(opt_df["high"].iloc[-w:]).mean()
                    lo = float(opt_df["low"].iloc[-w:]).mean()
                    rng = (hi - lo) / max(ltp, 1e-6)
                    guard = self.SPREAD_GUARD_PCT * (1.0 + (rng * 0.5)) if self.DYNAMIC_SPREAD_GUARD else self.SPREAD_GUARD_PCT
                    # if last close too far from bar mid, skip (rough spread/impact proxy)
                    mid = (hi + lo) / 2.0
                    if abs(ltp - mid) / max(mid, 1e-6) > guard:
                        continue
                except Exception:
                    pass
            # (LTP_MID mode guard can be added when using depth quotes)

            # Strategy signal (options-aware)
            signal = None
            try:
                signal = self.strategy.generate_options_signal(
                    options_ohlc=opt_df,
                    spot_ohlc=pd.DataFrame(),  # plug if available
                    strike_info={"type": opt_type},
                    current_option_price=last_px,
                )
            except Exception as e:
                logger.debug(f"Strategy error: {e}")
                continue

            if not signal:
                continue

            # Entry/SL/TP
            direction = signal["signal"]
            entry = float(signal["entry_price"])
            stop = float(signal["stop_loss"])
            target = float(signal["target"])
            conf = float(signal.get("confidence", 0.0))
            atr = float(signal.get("market_volatility", 0.0))

            # Max slip guard vs recent close
            try:
                prev_close = float(opt_df["close"].iloc[-2])
            except Exception:
                prev_close = entry
            slip = abs(entry - prev_close) / max(prev_close, 1e-6) * 10000  # bps
            if slip > self.MAX_ENTRY_SLIP_BPS:
                logger.debug("Entry slip %.1fbps > max %.1fbps — skip", slip, self.MAX_ENTRY_SLIP_BPS)
                continue

            # Size by risk (lots)
            lots = self._size_by_risk(entry, stop, atr)
            if lots <= 0:
                continue
            qty = lots * self.LOT_SIZE

            # Idempotent dedup (avoid duplicate entries on same bar/price)
            key = f"{tsym}:{direction}:{round(entry,2)}"
            if not self._idempotent_claim(key):
                continue

            # Place entry
            entry_side = "BUY" if direction == "BUY" else "SELL"
            oid = self.executor.place_entry_order(
                symbol=tsym, exchange="NFO", transaction_type=entry_side, quantity=qty
            )
            if not oid:
                logger.warning("Entry order rejected for %s", tsym)
                continue

            # Set exits (partials/breakeven handled in executor per Config)
            ok = self.executor.setup_gtt_orders(
                entry_order_id=oid,
                entry_price=entry,
                stop_loss_price=stop,
                target_price=target,
                symbol=tsym,
                exchange="NFO",
                quantity=qty,
                transaction_type=entry_side,
            )
            if not ok:
                logger.warning("Exit placement failed for %s", tsym)

            # Register active trade
            with self._lock:
                self.active_trades[oid] = {
                    "status": "OPEN",
                    "symbol": tsym,
                    "direction": entry_side,
                    "quantity": qty,
                    "entry_price": entry,
                    "stop_loss": stop,
                    "target": target,
                    "confidence": conf,
                    "atr": atr,
                    "last_close": last_px,
                    "open_ts": time.time(),
                }

            self._safe_send_message(
                f"🚀 Entered {entry_side} {tsym} x{qty} @ {entry:.2f} | SL {stop:.2f} | TP {target:.2f} | conf {conf:.1f}"
            )
            # single-position policy
            break

    # -------------------- sizing / idempotency -------------------- #

    def _size_by_risk(self, entry: float, stop: float, atr: float) -> int:
        if entry <= 0 or stop <= 0 or entry == stop:
            return 0
        per_contract_risk = abs(entry - stop)
        # volatility risk scaling
        scale = 1.0
        if atr and atr > 0:
            scale = max(self.VOL_RISK_CLAMP_MIN, min(self.VOL_RISK_CLAMP_MAX, self.ATR_TARGET / max(atr, self.ATR_MIN)))
        equity = float(self.daily_start_equity or 0.0)
        risk_budget = equity * self.RISK_PER_TRADE * scale if equity > 0 else 0.0
        if risk_budget <= 0:
            risk_budget = 1_000.0  # safe fallback
        lots = floor(risk_budget / max(per_contract_risk * self.LOT_SIZE, 1e-6))
        if self._loss_streak >= self.LOSS_STREAK_HALVE_SIZE:
            lots = max(1, lots // 2)
        # daily positive ladder halve
        if self._cum_r_today >= self.DAY_HALF_SIZE_AFTER_POS_R:
            lots = max(1, lots // 2)
        lots = int(max(self.MIN_LOTS, min(self.MAX_LOTS, lots)))
        return lots

    def _idempotent_claim(self, key: str) -> bool:
        """Return True if this entry key is new (and record it), False if recently seen."""
        now = time.time()
        # purge old
        for k, ts in list(self._entry_memo.items()):
            if now - ts > self.IDEMP_TTL_SEC:
                self._entry_memo.pop(k, None)
        if key in self._entry_memo:
            return False
        self._entry_memo[key] = now
        return True

    # -------------------- finalize & bookkeeping -------------------- #

    def _finalize_trade(self, entry_id: str) -> None:
        tr = self.active_trades.pop(entry_id, None)
        if not tr:
            return
        tr["status"] = "CLOSED"
        exit_px = float(tr.get("exit_price") or tr.get("target") or tr.get("stop_loss") or tr["entry_price"])
        entry_px = float(tr["entry_price"])
        qty = int(tr["quantity"])
        lots = qty // max(1, self.LOT_SIZE)
        direction_mult = 1 if tr["direction"] == "BUY" else -1
        gross = (exit_px - entry_px) * direction_mult * qty
        fees = self.FEES_PER_LOT * max(1, lots)
        net = gross - fees

        # Update pnl & R
        self.daily_pnl += net
        r_den = max(abs(entry_px - float(tr["stop_loss"])), 1e-6) * qty
        r_val = net / r_den if r_den > 0 else 0.0
        self._cum_r_today += r_val
        self._closed_trades_today += 1

        # loss streak & cooldown
        self._last_closed_was_loss = net < 0
        if self._last_closed_was_loss:
            self._loss_streak += 1
            if self._loss_streak >= self.LOSS_STREAK_HALVE_SIZE:
                self._cooldown_until_ts = time.time() + self.LOSS_STREAK_PAUSE_MIN * 60
        else:
            self._loss_streak = 0  # reset on win

        self.trades.append(
            dict(
                date=str((_ist_now() if self.USE_IST_CLOCK else datetime.now()).date()),
                order_id=entry_id,
                symbol=tr["symbol"],
                direction=tr["direction"],
                contracts=qty,
                entry=round(entry_px, 2),
                exit=round(exit_px, 2),
                pnl=round(gross, 2),
                fees=round(fees, 2),
                net_pnl=round(net, 2),
                confidence=float(tr.get("confidence", 0.0)),
                atr=float(tr.get("atr", 0.0)),
                mode=("LIVE" if self.live_mode else "SHADOW"),
            )
        )
        self._append_trade_log(
            [
                str((_ist_now() if self.USE_IST_CLOCK else datetime.now()).date()),
                entry_id,
                tr["symbol"],
                tr["direction"],
                qty,
                round(entry_px, 2),
                round(exit_px, 2),
                round(gross, 2),
                round(fees, 2),
                round(net, 2),
                tr.get("confidence", 0.0),
                tr.get("atr", 0.0),
                ("LIVE" if self.live_mode else "SHADOW"),
            ]
        )

        self._safe_send_message(
            f"🏁 Closed {tr['symbol']} | {tr['direction']} x{qty} | {entry_px:.2f} → {exit_px:.2f} | "
            f"net ₹{net:.0f} | R={r_val:.2f} | dayP&L ₹{self.daily_pnl:.0f} (ΣR {self._cum_r_today:.2f})"
        )

    def _rollover_day(self, new_date: date) -> None:
        logger.info("📅 Rollover to new session: %s → %s", self.session_date, new_date)
        self.session_date = new_date
        self.daily_pnl = 0.0
        self._cum_r_today = 0.0
        self._closed_trades_today = 0
        self._loss_streak = 0
        self._cooldown_until_ts = 0.0
        self._entry_memo.clear()

    # -------------------- status / health / messaging -------------------- #

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "is_trading": self.is_trading,
                "live_mode": self.live_mode,
                "open_orders": len(self.active_trades),
                "trades_today": self._closed_trades_today,
                "daily_pnl": round(self.daily_pnl, 2),
                "risk_level": f"{self.RISK_PER_TRADE:.3f}",
                "cum_r": round(self._cum_r_today, 2),
                "loss_streak": self._loss_streak,
            }

    def get_summary(self) -> str:
        s = self.get_status()
        return (
            "📊 <b>Summary</b>\n"
            f"Trading: {'🟢' if s['is_trading'] else '🔴'}  | Mode: {'LIVE' if s['live_mode'] else 'SHADOW'}\n"
            f"Open: {s['open_orders']}  | Trades Today: {s['trades_today']}\n"
            f"P&L: ₹{s['daily_pnl']:.0f}  | ΣR: {s['cum_r']:.2f}  | Streak(L): {s['loss_streak']}"
        )

    def _send_detailed_status(self) -> bool:
        st = self.get_status()
        txt = (
            "📊 <b>Bot Status</b>\n"
            f"🔁 Trading: {'🟢 Running' if st['is_trading'] else '🔴 Stopped'}\n"
            f"🌐 Mode: {'🟢 LIVE' if st['live_mode'] else '🛡️ Shadow'}\n"
            f"📦 Open Orders: {st['open_orders']}\n"
            f"📈 Trades Today: {st['trades_today']}\n"
            f"💰 Daily P&L: {st['daily_pnl']:.2f}\n"
            f"⚖️ Risk/Trade: {st['risk_level']}\n"
            f"ΣR: {st['cum_r']:.2f} | Loss Streak: {st['loss_streak']}"
        )
        return self._safe_send_message(txt, parse_mode="HTML")

    def _run_health_check(self) -> bool:
        ok = True
        probs = []
        if self.live_mode and not getattr(self.executor, "kite", None):
            ok = False; probs.append("Kite not attached in LIVE")
        if self.LOG_TRADE_FILE and not os.path.exists(os.path.dirname(self.LOG_TRADE_FILE) or "."):
            ok = False; probs.append("Logs directory missing")
        msg = "✅ Health OK" if ok else f"❌ Health issues: {', '.join(probs)}"
        return self._safe_send_message(msg)

    def _safe_send_message(self, text: str, parse_mode: Optional[str] = None) -> bool:
        try:
            if self.tg and bool(getattr(Config, "ENABLE_TELEGRAM", True)):
                return self.tg.send_message(text, parse_mode=parse_mode)
        except Exception:
            pass
        return False

    def _safe_send_alert(self, action: str) -> None:
        try:
            if self.tg and bool(getattr(Config, "ENABLE_TELEGRAM", True)):
                self.tg.send_realtime_session_alert(action)
        except Exception:
            pass

    def _force_refresh_cache(self) -> bool:
        try:
            self._refresh_instruments_cache(force=True)
            self._safe_send_message("🔄 Instruments cache refreshed.")
            return True
        except Exception:
            return False

    # -------------------- shutdown -------------------- #

    def shutdown(self) -> None:
        try:
            self._trailing_evt.set()
            self._oco_evt.set()
            self._stop_polling()
        except Exception:
            pass
        logger.info("👋 RealTimeTrader shutdown complete.")