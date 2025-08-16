# src/data_streaming/realtime_trader.py
"""
Real-time trader with:
- Telegram control (daemon polling)
- Adaptive main loop (peak/off-peak cadence)
- Risk-based lot sizing (R% of equity; lot-aware)
- Warmup & trading-hours filters (IST aware)
- Options strike resolution (ATM ± range via cached instruments)
- Strategy signals (spot/option aware)
- Spread guard (RANGE or LTP_MID) with safe fallbacks
- Partial TP, breakeven hop, trailing SL (via OrderExecutor)
- Daily circuit breaker + loss cooldown
- CSV trade log + daily rollover
- Robust SIM path (no Kite required), safe error handling

Notes:
- Long-options bias (BUY CE on bullish, BUY PE on bearish). Shorting is disabled by default.
- Requires OrderExecutor for exits; partials managed in REGULAR mode (GTT when partials disabled).
- In SIM mode (no Kite), data fetchers return empty -> no trades, but app stays healthy.

Public callbacks exposed to TelegramController:
- status_callback -> get_status()
- control_callback -> _handle_control()
- summary_callback -> get_summary()
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

logger = logging.getLogger(__name__)


# ----------------------------- time helpers ----------------------------- #

def _ist_now() -> datetime:
    """IST clock without pytz (UTC+5:30)."""
    return datetime.utcnow() + timedelta(hours=5, minutes=30)


def _between_ist(start_hm: str, end_hm: str) -> bool:
    now = _ist_now().time()
    s = datetime.strptime(start_hm, "%H:%M").time()
    e = datetime.strptime(end_hm, "%H:%M").time()
    return s <= now <= e


# ----------------------------- dataframe guard ----------------------------- #

def _safe_df(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


# ============================== Trader ============================= #

class RealTimeTrader:
    # ---- knobs from .env (with safe fallbacks) ----
    MAX_CONCURRENT_TRADES = int(getattr(Config, "MAX_CONCURRENT_POSITIONS", 1))
    WARMUP_BARS = int(getattr(Config, "WARMUP_BARS", 30))
    DATA_LOOKBACK_MINUTES = int(getattr(Config, "DATA_LOOKBACK_MINUTES", 60))
    HIST_TIMEFRAME = str(getattr(Config, "HISTORICAL_TIMEFRAME", "minute"))

    SPREAD_GUARD_MODE = str(getattr(Config, "SPREAD_GUARD_MODE", "RANGE")).upper()  # RANGE|LTP_MID
    SPREAD_GUARD_BA_MAX = float(getattr(Config, "SPREAD_GUARD_BA_MAX", 0.012))
    SPREAD_GUARD_LTPMID_MAX = float(getattr(Config, "SPREAD_GUARD_LTPMID_MAX", 0.015))
    SPREAD_GUARD_PCT = float(getattr(Config, "SPREAD_GUARD_PCT", 0.02))

    SLIPPAGE_BPS = float(getattr(Config, "SLIPPAGE_BPS", 4.0))
    FEES_PER_LOT = float(getattr(Config, "FEES_PER_LOT", 25.0))
    MAX_DAILY_DRAWDOWN_PCT = float(getattr(Config, "MAX_DAILY_DRAWDOWN_PCT", 0.05))
    CIRCUIT_RELEASE_PCT = float(getattr(Config, "CIRCUIT_RELEASE_PCT", 0.015))
    TRAILING_ENABLE = bool(getattr(Config, "TRAILING_ENABLE", True))
    TRAIL_ATR_MULTIPLIER = float(getattr(Config, "ATR_SL_MULTIPLIER", 1.5))
    WORKER_INTERVAL_SEC = int(getattr(Config, "WORKER_INTERVAL_SEC", 5))
    LOG_TRADE_FILE = str(getattr(Config, "LOG_FILE", "logs/trades.csv"))

    LOT_SIZE = int(getattr(Config, "NIFTY_LOT_SIZE", 75))
    MIN_LOTS = int(getattr(Config, "MIN_LOTS", 1))
    MAX_LOTS = int(getattr(Config, "MAX_LOTS", 10))

    RISK_PER_TRADE = float(getattr(Config, "RISK_PER_TRADE", 0.02))
    MAX_TRADES_PER_DAY = int(getattr(Config, "MAX_TRADES_PER_DAY", 20))
    LOSS_COOLDOWN_MIN = int(getattr(Config, "LOSS_COOLDOWN_MIN", 2))
    PEAK_POLL_SEC = int(getattr(Config, "PEAK_POLL_SEC", 15))
    OFFPEAK_POLL_SEC = int(getattr(Config, "OFFPEAK_POLL_SEC", 30))
    PREFERRED_TIE_RULE = str(getattr(Config, "PREFERRED_TIE_RULE", "TREND")).upper()

    USE_IST_CLOCK = bool(getattr(Config, "USE_IST_CLOCK", True))
    TIME_FILTER_START = str(getattr(Config, "TIME_FILTER_START", "09:20"))
    TIME_FILTER_END = str(getattr(Config, "TIME_FILTER_END", "15:20"))
    SKIP_FIRST_MIN = int(getattr(Config, "SKIP_FIRST_MIN", 5))

    STRIKE_RANGE = int(getattr(Config, "STRIKE_RANGE", 2))  # ± range around ATM

    # enable/disable taking long options only
    ENABLE_LONG_OPTIONS = bool(getattr(Config, "ENABLE_LONG_OPTIONS", True))

    def __init__(self) -> None:
        self._lock = threading.RLock()

        self.is_trading: bool = False
        self.live_mode: bool = bool(getattr(Config, "ENABLE_LIVE_TRADING", False))

        # PnL / session
        self.daily_pnl: float = 0.0
        self.daily_start_equity: float = float(get_live_account_balance() or 0.0)
        self.session_date: date = _ist_now().date()
        self._closed_trades_today = 0
        self._last_closed_was_loss = False
        self._cooldown_until_ts: float = 0.0

        # Trades
        self.trades: List[Dict[str, Any]] = []                    # closed trades (for the day)
        self.active_trades: Dict[str, Dict[str, Any]] = {}        # entry_order_id → info

        # Instruments cache
        self._nfo_cache: List[Dict[str, Any]] = []
        self._nse_cache: List[Dict[str, Any]] = []
        self._cache_ts: float = 0.0
        self._CACHE_TTL = 300.0
        self._cache_lock = threading.RLock()

        # Circuit breaker state
        self._circuit_tripped: bool = False
        self._circuit_trip_equity: float = self.daily_start_equity

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

        try:
            self.risk = PositionSizing()
            try:
                self.risk.set_equity(float(self.daily_start_equity or 0.0))
            except Exception:
                setattr(self.risk, "equity", float(self.daily_start_equity or 0.0))
        except Exception as e:
            logger.warning(f"Risk manager init failed: {e}")
            self.risk = PositionSizing()

        self.executor = self._init_executor()

        try:
            if bool(getattr(Config, "ENABLE_TELEGRAM", True)):
                self.tg = TelegramController(
                    status_callback=self.get_status,
                    control_callback=self._handle_control,
                    summary_callback=self.get_summary,
                )
            else:
                self.tg = None
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
        # balance refresh & daily rollover checks
        schedule.every(getattr(Config, "BALANCE_LOG_INTERVAL_MIN", 30)).minutes.do(self.refresh_account_balance)
        schedule.every(60).seconds.do(self._maybe_rollover_daily)
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
            now = _ist_now().time()
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
                # keep original day's equity unless new day
                if _ist_now().date() != self.session_date:
                    self.daily_start_equity = new_bal
                try:
                    self.risk.set_equity(new_bal)
                except Exception:
                    setattr(self.risk, "equity", new_bal)
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
            return True
        return _between_ist(self.TIME_FILTER_START, self.TIME_FILTER_END)

    def _maybe_rollover_daily(self) -> None:
        """End-of-day reset: finalize state & start fresh counters on new IST date."""
        today = _ist_now().date()
        if today != self.session_date:
            logger.info("📅 Session rollover: %s → %s", self.session_date, today)
            self.session_date = today
            self.daily_pnl = 0.0
            self._closed_trades_today = 0
            self._last_closed_was_loss = False
            self._cooldown_until_ts = 0.0
            self._circuit_tripped = False
            self._circuit_trip_equity = float(get_live_account_balance() or self.daily_start_equity)
            # clear internal active-trade registry (safety)
            with self._lock:
                self.active_trades.clear()

    def _is_circuit_breaker_tripped(self) -> bool:
        if self._circuit_tripped:
            # unlock only after recovery by CIRCUIT_RELEASE_PCT of start equity
            recovery = self.CIRCUIT_RELEASE_PCT * max(1.0, self.daily_start_equity)
            return self.daily_pnl < -max(0.0, self.MAX_DAILY_DRAWDOWN_PCT * self.daily_start_equity) + recovery

        dd = -self.daily_pnl  # drawdown as positive
        if dd >= self.MAX_DAILY_DRAWDOWN_PCT * max(1.0, self.daily_start_equity):
            logger.warning("🚨 Circuit breaker TRIPPED. DD=%.2f", dd)
            self._circuit_tripped = True
            return True
        return False

    def _in_loss_cooldown(self) -> bool:
        return time.time() < self._cooldown_until_ts

    # -------------------- cache & data -------------------- #

    def _force_refresh_cache(self) -> bool:
        try:
            self._refresh_instruments_cache(force=True)
            self._safe_send_message("🔄 Instruments cache refreshed.")
            return True
        except Exception as e:
            self._safe_send_message(f"⚠️ Cache refresh failed: {e}")
            return False

    def _refresh_instruments_cache(self, force: bool = False) -> None:
        if not self.live_mode or not getattr(self.executor, "kite", None):
            return
        with self._cache_lock:
            if not force and (time.time() - self._cache_ts) < self._CACHE_TTL and self._nfo_cache and self._nse_cache:
                return
            try:
                caches = fetch_cached_instruments(self.executor.kite)
                self._nfo_cache = caches.get("NFO", []) or []
                self._nse_cache = caches.get("NSE", []) or []
                self._cache_ts = time.time()
                logger.info("📦 Instruments cache refreshed: NFO=%d, NSE=%d", len(self._nfo_cache), len(self._nse_cache))
            except Exception as e:
                logger.warning(f"Instruments cache refresh failed: {e}")

    # -------------------- main loop -------------------- #

    def _smart_fetch_and_process(self) -> None:
        try:
            now = _ist_now()
            if not self._is_trading_hours(now) and not getattr(Config, "ALLOW_OFFHOURS_TESTING", False):
                # heartbeat occasionally
                if int(time.time()) % 300 < 2:
                    logger.info("⏳ Market closed. Skipping fetch.")
                return

            if not self.is_trading:
                return

            if self._is_circuit_breaker_tripped():
                logger.warning("🚫 Circuit breaker is active — trading paused.")
                return

            if self._in_loss_cooldown():
                return

            self.fetch_and_process_data()

        except Exception as e:
            logger.error(f"Error in smart fetch and process: {e}", exc_info=True)

    # -------------------- data & trade pipeline -------------------- #

    def fetch_and_process_data(self) -> None:
        """
        Core loop:
        1) Ensure instruments cache
        2) Resolve CE/PE for current expiry near ATM
        3) Pull minimal OHLC for options + spot (if available)
        4) Generate signal (spot-aware)
        5) Spread guard
        6) Size & execute via OrderExecutor
        """
        # guard: single-position policy
        with self._lock:
            open_count = sum(1 for t in self.active_trades.values() if t.get("status") == "OPEN")
        if open_count >= self.MAX_CONCURRENT_TRADES:
            return

        # live path required for quotes & tokens
        if not self.live_mode or not getattr(self.executor, "kite", None):
            logger.debug("Live disabled or Kite missing → shadow idle.")
            return

        # ensure cache
        self._refresh_instruments_cache(force=False)
        if not self._nfo_cache or not self._nse_cache:
            logger.debug("Instrument cache empty → skip loop.")
            return

        # resolve instruments around ATM
        toks = get_instrument_tokens(
            symbol="NIFTY",
            kite_instance=self.executor.kite,
            cached_nfo_instruments=self._nfo_cache,
            cached_nse_instruments=self._nse_cache,
            offset=0,
            strike_range=int(getattr(Config, "STRIKE_RANGE", 3)),
        )
        if not toks:
            logger.debug("Token resolution failed → skip.")
            return

        ce_sym, ce_tok = toks.get("ce_symbol"), toks.get("ce_token")
        pe_sym, pe_tok = toks.get("pe_symbol"), toks.get("pe_token")
        spot_price = float(toks.get("spot_price") or 0.0)
        if not ce_sym or not pe_sym:
            logger.debug("Missing CE/PE in token set → skip.")
            return

        # fetch last quotes (ltp + depth mid if available)
        ce_ltp, pe_ltp = self._safe_ltp(ce_sym), self._safe_ltp(pe_sym)
        if not ce_ltp or not pe_ltp:
            logger.debug("Missing LTPs for CE/PE → skip.")
            return

        # minimal OHLC to drive strategy (last N bars)
        lookback_min = max(10, self.DATA_LOOKBACK_MINUTES)
        ce_df = self._fetch_ohlc_by_token(ce_tok, lookback_min)
        pe_df = self._fetch_ohlc_by_token(pe_tok, lookback_min)
        spot_df = self._fetch_spot_ohlc(lookback_min)

        if ce_df.empty or pe_df.empty or spot_df.empty or len(spot_df) < self.WARMUP_BARS:
            logger.debug("OHLC insufficient (CE/PE/Spot).")
            return

        # generate spot-driven direction
        spot_close = float(spot_df["close"].iloc[-1])
        spot_signal = self.strategy.generate_signal(spot_df, spot_close) if self.strategy else None
        if not spot_signal:
            return

        direction = spot_signal.get("signal")
        confidence = float(spot_signal.get("confidence", 0.0))
        atr_val = float(spot_signal.get("market_volatility", 0.0))

        # choose option leg: BUY CE if bullish; BUY PE if bearish
        if direction == "BUY":
            opt_sym, opt_tok, opt_df, opt_ltp, opt_type = ce_sym, ce_tok, ce_df, ce_ltp, "CE"
        elif direction == "SELL":
            # long-put for bearish bias
            opt_sym, opt_tok, opt_df, opt_ltp, opt_type = pe_sym, pe_tok, pe_df, pe_ltp, "PE"
            direction = "BUY"  # we BUY PE for bearish view
        else:
            return

        # option-level breakout confirmation (lightweight)
        opt_sig = self.strategy.generate_options_signal(
            options_ohlc=opt_df,
            spot_ohlc=spot_df,
            strike_info={"type": opt_type},
            current_option_price=float(opt_ltp),
        )
        if not opt_sig or opt_sig.get("signal") != "BUY":
            return

        # spread guard
        if not self._passes_spread_guard(opt_sym, opt_df):
            return

        # compute SL/TP in premium terms (from options signal)
        entry_px = float(opt_sig.get("entry_price", opt_ltp))
        stop_loss = float(opt_sig.get("stop_loss", max(0.05, entry_px * 0.95)))
        target = float(opt_sig.get("target", entry_px * 1.15))

        # risk-based lot sizing
        qty = self._compute_quantity(entry_px, stop_loss)
        if qty <= 0:
            return

        # final double-check single-position policy
        with self._lock:
            open_count = sum(1 for t in self.active_trades.values() if t.get("status") == "OPEN")
            if open_count >= self.MAX_CONCURRENT_TRADES:
                return

        # place entry
        order_id = self.executor.place_entry_order(
            symbol=opt_sym,
            exchange=getattr(Config, "TRADE_EXCHANGE", "NFO"),
            transaction_type="BUY",
            quantity=int(qty),
            product=getattr(Config, "DEFAULT_PRODUCT", "MIS"),
            order_type=getattr(Config, "DEFAULT_ORDER_TYPE", "MARKET"),
            validity=getattr(Config, "DEFAULT_VALIDITY", "DAY"),
        )
        if not order_id:
            return

        # set exits via executor
        ok = self.executor.setup_gtt_orders(
            entry_order_id=order_id,
            entry_price=float(entry_px),
            stop_loss_price=float(stop_loss),
            target_price=float(target),
            symbol=opt_sym,
            exchange=getattr(Config, "TRADE_EXCHANGE", "NFO"),
            quantity=int(qty),
            transaction_type="BUY",
        )
        if not ok:
            logger.warning("Exit setup failed; attempting emergency exit setup may be required later.")

        # register active trade
        with self._lock:
            self.active_trades[order_id] = {
                "status": "OPEN",
                "symbol": opt_sym,
                "direction": "BUY",
                "quantity": int(qty),
                "entry_price": float(entry_px),
                "stop_loss": float(stop_loss),
                "target": float(target),
                "confidence": float(confidence),
                "atr": float(atr_val),
                "last_close": float(opt_df["close"].iloc[-1]),
                "leg": opt_type,
            }

        self._safe_send_signal(order_id, opt_sym, direction="BUY", entry=entry_px, sl=stop_loss, tp=target, conf=confidence)

    # -------------------- helpers: quotes & OHLC -------------------- #

    def _safe_ltp(self, tradingsymbol: str) -> Optional[float]:
        if not self.live_mode or not getattr(self.executor, "kite", None):
            return None
        try:
            q = self.executor.kite.ltp([f"{getattr(Config, 'TRADE_EXCHANGE', 'NFO')}:{tradingsymbol}"])
            v = q.get(f"{getattr(Config, 'TRADE_EXCHANGE', 'NFO')}:{tradingsymbol}", {}).get("last_price")
            return float(v) if v is not None else None
        except Exception:
            return None

    def _fetch_ohlc_by_token(self, token: int, lookback_min: int) -> pd.DataFrame:
        """Use Kite historical_data if available. Returns empty DF on failure."""
        if not self.live_mode or not getattr(self.executor, "kite", None):
            return pd.DataFrame()
        try:
            to_dt = _ist_now()
            from_dt = to_dt - timedelta(minutes=int(lookback_min))
            tf = self.HIST_TIMEFRAME or "minute"
            candles = self.executor.kite.historical_data(
                instrument_token=int(token),
                from_date=from_dt,
                to_date=to_dt,
                interval=tf,
                continuous=False,
                oi=False,
            )
            if not candles:
                return pd.DataFrame()
            df = pd.DataFrame(candles)
            # normalize columns -> date, open, high, low, close, volume
            if "date" in df.columns:
                df.set_index("date", inplace=True)
            return df[["open", "high", "low", "close", "volume"]].dropna(how="any")
        except Exception:
            return pd.DataFrame()

    def _fetch_spot_ohlc(self, lookback_min: int) -> pd.DataFrame:
        """Fetch spot/futures OHLC. Prefer NSE:NIFTY 50 token if available."""
        if not self.live_mode or not getattr(self.executor, "kite", None):
            return pd.DataFrame()
        try:
            # Prefer index token from env, else fallback to Config.INSTRUMENT_TOKEN
            token = int(getattr(Config, "INSTRUMENT_TOKEN", 256265))
            return self._fetch_ohlc_by_token(token, lookback_min)
        except Exception:
            return pd.DataFrame()

    # -------------------- spread guard -------------------- #

    def _passes_spread_guard(self, tradingsymbol: str, df: pd.DataFrame) -> bool:
        mode = self.SPREAD_GUARD_MODE
        if mode == "RANGE":
            # simple last-bar range proxy
            try:
                last_row = df.iloc[-1]
                rng = float(last_row["high"] - last_row["low"])
                mid = float((last_row["high"] + last_row["low"]) / 2.0)
                if mid <= 0:
                    return True
                pct = rng / mid
                ok = pct <= self.SPREAD_GUARD_PCT
                if not ok:
                    logger.debug("Spread guard RANGE failed (%.4f > %.4f) for %s", pct, self.SPREAD_GUARD_PCT, tradingsymbol)
                return ok
            except Exception:
                return True
        elif mode == "LTP_MID":
            # try quote depth; fallback to pass
            if not self.live_mode or not getattr(self.executor, "kite", None):
                return True
            try:
                q = self.executor.kite.quote([f"{getattr(Config, 'TRADE_EXCHANGE', 'NFO')}:{tradingsymbol}"])
                item = q.get(f"{getattr(Config, 'TRADE_EXCHANGE', 'NFO')}:{tradingsymbol}", {})
                bids = item.get("depth", {}).get("buy", []) or []
                asks = item.get("depth", {}).get("sell", []) or []
                if not bids or not asks:
                    return True
                best_bid = float(bids[0].get("price") or 0.0)
                best_ask = float(asks[0].get("price") or 0.0)
                if best_bid <= 0 or best_ask <= 0 or best_ask <= best_bid:
                    return True
                mid = (best_bid + best_ask) / 2.0
                ba = (best_ask - best_bid) / mid
                # also check LTP dev from mid if available
                ltp = self._safe_ltp(tradingsymbol) or mid
                ltp_dev = abs(ltp - mid) / mid if mid > 0 else 0.0
                ok = (ba <= self.SPREAD_GUARD_BA_MAX) and (ltp_dev <= self.SPREAD_GUARD_LTPMID_MAX)
                if not ok:
                    logger.debug("Spread guard LTP_MID failed (ba=%.4f, dev=%.4f) for %s", ba, ltp_dev, tradingsymbol)
                return ok
            except Exception:
                return True
        return True

    # -------------------- sizing -------------------- #

    def _compute_quantity(self, entry_px: float, stop_loss_px: float) -> int:
        """
        Risk per lot = (entry - stop) * LOT_SIZE.
        Target risk = RISK_PER_TRADE * equity.
        Clamp to [MIN_LOTS, MAX_LOTS].
        """
        try:
            risk_per_lot = max(0.0, float(entry_px - stop_loss_px)) * max(1, self.LOT_SIZE)
            if risk_per_lot <= 0:
                return 0
            equity = float(getattr(self.risk, "equity", self.daily_start_equity) or self.daily_start_equity or 0.0)
            target_risk = max(0.0, self.RISK_PER_TRADE * equity)
            lots = int(max(self.MIN_LOTS, min(self.MAX_LOTS, target_risk // risk_per_lot)))
            qty = int(lots * max(1, self.LOT_SIZE))
            return qty
        except Exception:
            return 0

    # -------------------- finalize trade -------------------- #

    def _finalize_trade(self, entry_id: str) -> None:
        with self._lock:
            tr = self.active_trades.pop(entry_id, None)

        if not tr:
            return

        side = tr.get("direction", "BUY")
        qty = int(tr.get("quantity", 0))
        entry_px = float(tr.get("entry_price", 0.0))
        exit_px = float(tr.get("exit_price", 0.0))
        leg = tr.get("leg", "CE")

        if qty <= 0 or entry_px <= 0 or exit_px <= 0:
            return

        # long options PnL
        gross = (exit_px - entry_px) * qty
        # fees: per lot round-trip (count only once here)
        lots = qty // max(1, self.LOT_SIZE)
        fees = float(self.FEES_PER_LOT) * max(1, lots)
        net = gross - fees

        # update PnL & stats
        self.daily_pnl += net
        self._closed_trades_today += 1
        if net < 0:
            self._last_closed_was_loss = True
            self._cooldown_until_ts = time.time() + (self.LOSS_COOLDOWN_MIN * 60)
        else:
            self._last_closed_was_loss = False

        # log row
        row = [
            _ist_now().strftime("%Y-%m-%d"),
            entry_id,
            tr.get("symbol"),
            side,
            qty,
            f"{entry_px:.2f}",
            f"{exit_px:.2f}",
            f"{gross:.2f}",
            f"{fees:.2f}",
            f"{net:.2f}",
            f"{float(tr.get('confidence', 0.0)):.2f}",
            f"{float(tr.get('atr', 0.0)):.2f}",
            "LIVE" if self.live_mode else "SIM",
        ]
        self._append_trade_log(row)

        # notify
        self._safe_send_message(
            f"✅ Closed {leg} {tr.get('symbol')} x{qty}\n"
            f"Entry {entry_px:.2f} → Exit {exit_px:.2f}\n"
            f"Net P&L: ₹{net:.2f} | Day P&L: ₹{self.daily_pnl:.2f}"
        )

    # -------------------- telegram helpers -------------------- #

    def _safe_send_message(self, text: str, parse_mode: Optional[str] = None) -> None:
        try:
            if self.tg:
                self.tg.send_message(text, parse_mode=parse_mode)
        except Exception:
            pass

    def _safe_send_alert(self, action: str) -> None:
        try:
            if self.tg:
                self.tg.send_realtime_session_alert(action)
        except Exception:
            pass

    def _safe_send_signal(self, order_id: str, symbol: str, direction: str, entry: float, sl: float, tp: float, conf: float) -> None:
        try:
            if not self.tg:
                return
            payload = {
                "signal": direction,
                "entry_price": float(entry),
                "stop_loss": float(sl),
                "target": float(tp),
                "confidence": float(conf),
            }
            pos = {"quantity": 0}  # we already placed; this is just a notification
            self.tg.send_signal_alert(token=hash(order_id) % 10_000, signal=payload, position=pos)
        except Exception:
            pass

    def _send_detailed_status(self) -> bool:
        st = self.get_status()
        msg = (
            "📊 *Status*\n"
            f"Trading: {'🟢' if st.get('is_trading') else '🔴'}\n"
            f"Mode: {'LIVE' if st.get('live_mode') else 'SIM'}\n"
            f"Open Orders: {st.get('open_orders', 0)}\n"
            f"Trades Today: {st.get('trades_today', 0)} / {self.MAX_TRADES_PER_DAY}\n"
            f"Daily P&L: ₹{st.get('daily_pnl', 0.0):.2f}\n"
            f"Circuit: {'TRIPPED' if st.get('circuit_tripped') else 'OK'}\n"
        )
        self._safe_send_message(msg, parse_mode="Markdown")
        return True

    def _run_health_check(self) -> bool:
        ok_live = bool(self.live_mode and getattr(self.executor, "kite", None))
        ok_cache = bool(self._nfo_cache and self._nse_cache)
        msg = (
            "🧪 *Health*\n"
            f"Kite: {'OK' if ok_live else 'N/A'}\n"
            f"Cache: {'OK' if ok_cache else 'EMPTY'}\n"
            f"Workers: trailing={'up' if not self._trailing_evt.is_set() else 'down'}, "
            f"oco={'up' if not self._oco_evt.is_set() else 'down'}"
        )
        self._safe_send_message(msg, parse_mode="Markdown")
        return True

    # -------------------- public status/summary -------------------- #

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            open_orders = len([1 for t in self.active_trades.values() if t.get("status") == "OPEN"])
        return {
            "is_trading": self.is_trading,
            "live_mode": self.live_mode,
            "open_orders": open_orders,
            "trades_today": self._closed_trades_today,
            "daily_pnl": float(self.daily_pnl),
            "risk_level": f"{self.RISK_PER_TRADE*100:.1f}%",
            "circuit_tripped": self._circuit_tripped,
        }

    def get_summary(self) -> str:
        return (
            "📒 <b>Daily Summary</b>\n"
            f"• Trades: <b>{self._closed_trades_today}</b>\n"
            f"• P&L: <b>₹{self.daily_pnl:.2f}</b>\n"
            f"• Mode: <b>{'LIVE' if self.live_mode else 'SIM'}</b>"
        )

    # -------------------- shutdown -------------------- #

    def shutdown(self) -> None:
        try:
            self._trailing_evt.set()
            self._oco_evt.set()
        except Exception:
            pass
        try:
            self._stop_polling()
        except Exception:
            pass
        logger.info("🔻 RealTimeTrader shutdown complete.")