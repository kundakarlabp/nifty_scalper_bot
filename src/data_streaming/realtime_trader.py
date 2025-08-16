# src/data_streaming/realtime_trader.py
"""
Real-time trader with:
- Telegram control (daemon polling)
- Adaptive main loop (peak/off-peak cadence)
- Risk-based lot sizing
- Warmup & IST trading-hours filters (skip first minutes)
- Instrument cache + robust strike resolution (ATM ± range, Greeks-ready)
- Option-centric signal generation (fallback to spot)
- Spread guard (RANGE or LTP_MID via quote depth)
- Partial TP, breakeven hop, trailing SL, time-expiry (via OrderExecutor)
- Daily circuit breaker + loss cooldown
- CSV trade log + daily rollover

Degrades gracefully in SHADOW mode (no live broker).
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

# Robust ATR helper
try:
    from src.utils.atr_helper import compute_atr_df
except Exception:
    compute_atr_df = None  # safe fallback

logger = logging.getLogger(__name__)


# ----------------------------- helpers ----------------------------- #

def _ist_now() -> datetime:
    """IST clock without pytz (UTC+5:30)."""
    return datetime.utcnow() + timedelta(hours=5, minutes=30)


def _safe_df(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


def _round2(x: Any) -> float:
    try:
        return round(float(x), 2)
    except Exception:
        return 0.0


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
    NFO_FREEZE_QTY = int(getattr(Config, "NFO_FREEZE_QTY", 1800))

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

    STRIKE_RANGE = int(getattr(Config, "STRIKE_RANGE", 3))  # ± range around ATM
    INSTRUMENT_TOKEN = int(getattr(Config, "INSTRUMENT_TOKEN", 256265))  # NIFTY spot token

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
        self._circuit_tripped = False

        # Trades
        self.trades: List[Dict[str, Any]] = []                    # closed trades (for the day)
        self.active_trades: Dict[str, Dict[str, Any]] = {}        # entry_order_id → info

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
        # ensure cadence job + status/rollover
        schedule.every(5).seconds.do(self._ensure_cadence)
        schedule.every(getattr(Config, "BALANCE_LOG_INTERVAL_MIN", 30)).minutes.do(
            self.refresh_account_balance
        )
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
        while True:
            try:
                if (
                    self.TRAILING_ENABLE
                    and self.is_trading
                    and not self._is_circuit_breaker_tripped()
                ):
                    self._trailing_tick()
            except Exception as e:
                logger.debug(f"Trailing worker error: {e}")
            time.sleep(self.WORKER_INTERVAL_SEC)

    def _oco_worker(self) -> None:
        while True:
            try:
                if self.is_trading:
                    self._oco_tick()
            except Exception as e:
                logger.debug(f"OCO worker error: {e}")
            time.sleep(self.WORKER_INTERVAL_SEC)

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
            ltp = get_last(f"{tr.get('exchange','NFO')}:{tr.get('symbol')}") if callable(get_last) else None
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
        now = now or _ist_now()
        t = now.time()
        try:
            s = datetime.strptime(self.TIME_FILTER_START, "%H:%M").time()
            e = datetime.strptime(self.TIME_FILTER_END, "%H:%M").time()
        except Exception:
            s, e = dtime(9, 20), dtime(15, 20)
        return s <= t <= e

    def _in_loss_cooldown(self) -> bool:
        return time.time() < float(self._cooldown_until_ts or 0.0)

    def _is_circuit_breaker_tripped(self) -> bool:
        if self._circuit_tripped:
            # Release when recovered by release pct
            recover = -self.MAX_DAILY_DRAWDOWN_PCT * self.daily_start_equity + \
                      (self.CIRCUIT_RELEASE_PCT * self.daily_start_equity)
            if self.daily_pnl >= recover:
                self._circuit_tripped = False
        else:
            if self.daily_pnl <= -self.MAX_DAILY_DRAWDOWN_PCT * self.daily_start_equity:
                self._circuit_tripped = True
        return self._circuit_tripped

    def _maybe_rollover_daily(self) -> None:
        today = _ist_now().date()
        if today != self.session_date:
            logger.info("📅 Session rollover.")
            self.session_date = today
            self.daily_pnl = 0.0
            self._closed_trades_today = 0
            self._last_closed_was_loss = False
            self._cooldown_until_ts = 0.0
            self._circuit_tripped = False
            self.trades.clear()
            # refresh equity base
            self.refresh_account_balance()

    # -------------------- instruments cache -------------------- #

    def _refresh_instruments_cache(self, force: bool = False) -> bool:
        if not self.live_mode or not getattr(self.executor, "kite", None):
            return False
        with self._cache_lock:
            now = time.time()
            if (not force) and (now - self._cache_ts) < self._CACHE_TTL:
                return True
            try:
                data = fetch_cached_instruments(self.executor.kite)
                self._nfo_cache = data.get("NFO", []) or []
                self._nse_cache = data.get("NSE", []) or []
                self._cache_ts = now
                logger.info("🔄 Instruments cache refreshed. NFO:%d NSE:%d", len(self._nfo_cache), len(self._nse_cache))
                return True
            except Exception as e:
                logger.warning(f"Instruments cache refresh failed: {e}")
                return False

    def _force_refresh_cache(self) -> bool:
        ok = self._refresh_instruments_cache(force=True)
        self._safe_send_message("🔄 Instruments cache refreshed." if ok else "⚠️ Cache refresh failed.")
        return ok

    # -------------------- data fetch -------------------- #

    def _fetch_ohlc(self, instrument_token: int, minutes: int) -> pd.DataFrame:
        """Fetch minute OHLC using Kite historical API."""
        if not self.live_mode or not getattr(self.executor, "kite", None):
            return pd.DataFrame()
        try:
            to_dt = _ist_now()
            frm_dt = to_dt - timedelta(minutes=int(minutes) + 5)
            data = self.executor.kite.historical_data(
                instrument_token=instrument_token,
                from_date=frm_dt,
                to_date=to_dt,
                interval=self.HIST_TIMEFRAME,
                continuous=False,
                oi=False,
            )
            if not data:
                return pd.DataFrame()
            df = pd.DataFrame(data)
            # normalize
            if "date" in df.columns:
                df.set_index(pd.to_datetime(df["date"]), inplace=True)
                df.drop(columns=["date"], inplace=True)
            for c in ("open", "high", "low", "close"):
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            if "volume" in df.columns:
                df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
            return df.dropna(subset=["open", "high", "low", "close"]).tail(max(self.DATA_LOOKBACK_MINUTES, self.WARMUP_BARS + 5))
        except Exception as e:
            logger.debug(f"historical_data error for {instrument_token}: {e}")
            return pd.DataFrame()

    def _ltp(self, exch_sym: str) -> Optional[float]:
        if not self.live_mode or not getattr(self.executor, "kite", None):
            return None
        try:
            q = self.executor.kite.ltp([exch_sym])
            return float(q[exch_sym]["last_price"])
        except Exception:
            return None

    # -------------------- spread guards -------------------- #

    def _range_guard_ok(self, df: pd.DataFrame) -> bool:
        if df is None or df.empty:
            return True
        last = df.iloc[-1]
        try:
            rng = float(last["high"]) - float(last["low"])
            rel = rng / max(1e-9, float(last["close"]))
            return rel <= self.SPREAD_GUARD_PCT
        except Exception:
            return True

    def _ltpmid_guard_ok(self, exch: str, symbol: str) -> bool:
        """Use quote depth to guard spreads."""
        if not self.live_mode or not getattr(self.executor, "kite", None):
            return True
        try:
            q = self.executor.kite.quote([f"{exch}:{symbol}"])
            d = q[f"{exch}:{symbol}"]
            ltp = float(d.get("last_price") or 0.0)
            depth = d.get("depth", {}) or {}
            bids = depth.get("buy", []) or []
            asks = depth.get("sell", []) or []
            if not bids or not asks:
                return True
            bid = float(bids[0].get("price") or 0.0)
            ask = float(asks[0].get("price") or 0.0)
            if bid <= 0 or ask <= 0:
                return True
            mid = 0.5 * (bid + ask)
            ba_rel = (ask - bid) / max(1e-9, mid)
            dev_rel = abs(ltp - mid) / max(1e-9, mid)
            if ba_rel > self.SPREAD_GUARD_BA_MAX:
                return False
            if dev_rel > self.SPREAD_GUARD_LTPMID_MAX:
                return False
            return True
        except Exception:
            return True

    def _spread_guard_ok(self, exch: str, symbol: str, df: pd.DataFrame) -> bool:
        mode = self.SPREAD_GUARD_MODE
        if mode == "LTP_MID":
            return self._ltpmid_guard_ok(exch, symbol)
        return self._range_guard_ok(df)

    # -------------------- main loop -------------------- #

    def _smart_fetch_and_process(self) -> None:
        try:
            now = _ist_now()
            if not self._is_trading_hours(now) and not getattr(Config, "ALLOW_OFFHOURS_TESTING", False):
                if int(time.time()) % 300 < 2:
                    logger.info("⏳ Market closed. Skipping fetch.")
                return

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
        """Core loop: get tokens → fetch OHLC → generate signal → size → execute."""
        # 0) Instruments cache
        self._refresh_instruments_cache()

        # 1) Respect single-position policy
        with self._lock:
            open_cnt = sum(1 for v in self.active_trades.values() if v.get("status") == "OPEN")
        if open_cnt >= self.MAX_CONCURRENT_TRADES:
            return

        # 2) Tokens (CE/PE + spot)
        if not self._nfo_cache or not self._nse_cache:
            logger.debug("No instrument cache yet; skipping this tick.")
            return

        # get best CE/PE around ATM (offset 0)
        try:
            tok_info = get_instrument_tokens(
                symbol="NIFTY",
                kite_instance=self.executor.kite if self.live_mode else None,
                cached_nfo_instruments=self._nfo_cache,
                cached_nse_instruments=self._nse_cache,
                offset=0,
                strike_range=self.STRIKE_RANGE,
            )
        except Exception as e:
            logger.debug(f"get_instrument_tokens error: {e}")
            tok_info = None

        if not tok_info or not (tok_info.get("ce_token") or tok_info.get("pe_token")):
            return

        ce_token, pe_token = tok_info.get("ce_token"), tok_info.get("pe_token")
        ce_symbol, pe_symbol = tok_info.get("ce_symbol"), tok_info.get("pe_symbol")
        expiry_iso = tok_info.get("expiry") or ""
        spot_token = self.INSTRUMENT_TOKEN

        # 3) Fetch OHLCs (option + spot)
        lookback = max(self.DATA_LOOKBACK_MINUTES, self.WARMUP_BARS + 5)
        ce_df = self._fetch_ohlc(int(ce_token), lookback) if ce_token else pd.DataFrame()
        pe_df = self._fetch_ohlc(int(pe_token), lookback) if pe_token else pd.DataFrame()
        spot_df = self._fetch_ohlc(int(spot_token), lookback) if spot_token else pd.DataFrame()

        # If we can't fetch enough data, skip
        if len(ce_df) < self.WARMUP_BARS and len(pe_df) < self.WARMUP_BARS:
            return

        # 4) Current prices
        ce_ltp = self._ltp(f"NFO:{ce_symbol}") if ce_symbol else None
        pe_ltp = self._ltp(f"NFO:{pe_symbol}") if pe_symbol else None

        # 5) Option signals (prefer option-led, fallback to spot strategy on option DF)
        signal = None
        chosen_side = None     # "CE" or "PE"
        chosen_df = None
        chosen_symbol = None
        chosen_token = None
        chosen_ltp = None

        # try CE
        if ce_symbol and not ce_df.empty and ce_ltp:
            sig_ce = self.strategy.generate_options_signal(
                options_ohlc=ce_df, spot_ohlc=spot_df, strike_info={"type": "CE"}, current_option_price=float(ce_ltp)
            )
            if not sig_ce:
                sig_ce = self.strategy.generate_signal(ce_df, float(ce_ltp))
            if sig_ce and sig_ce.get("signal") == "BUY":
                signal = sig_ce
                chosen_side = "CE"
                chosen_df = ce_df
                chosen_symbol = ce_symbol
                chosen_token = ce_token
                chosen_ltp = float(ce_ltp)

        # try PE if CE not chosen
        if not signal and pe_symbol and not pe_df.empty and pe_ltp:
            sig_pe = self.strategy.generate_options_signal(
                options_ohlc=pe_df, spot_ohlc=spot_df, strike_info={"type": "PE"}, current_option_price=float(pe_ltp)
            )
            if not sig_pe:
                sig_pe = self.strategy.generate_signal(pe_df, float(pe_ltp))
            if sig_pe and sig_pe.get("signal") == "BUY":
                signal = sig_pe
                chosen_side = "PE"
                chosen_df = pe_df
                chosen_symbol = pe_symbol
                chosen_token = pe_token
                chosen_ltp = float(pe_ltp)

        if not signal:
            return  # nothing to do

        # 6) Warmup and guards
        if len(chosen_df) < max(self.WARMUP_BARS, 10):
            return

        # skip first SKIP_FIRST_MIN mins after open
        tnow = _ist_now().time()
        if dtime(9, 15) <= tnow <= dtime(9, 15 + self.SKIP_FIRST_MIN):
            return

        # spread guard
        if not self._spread_guard_ok("NFO", chosen_symbol, chosen_df):
            logger.debug("Spread guard blocked trade for %s", chosen_symbol)
            return

        # 7) Size position (contracts)
        entry = float(signal.get("entry_price") or chosen_ltp)
        stop = float(signal.get("stop_loss") or (entry - 10.0))
        target = float(signal.get("target") or (entry + 20.0))
        conf = float(signal.get("confidence") or 5.0)

        per_contract_risk = max(0.5, abs(entry - stop))
        equity = float(getattr(self.risk, "equity", self.daily_start_equity))
        rupees_risk = equity * float(self.RISK_PER_TRADE)
        contracts = int(rupees_risk // per_contract_risk)
        # lot integrity + bounds
        if self.LOT_SIZE > 0:
            lots = max(self.MIN_LOTS, min(self.MAX_LOTS, contracts // self.LOT_SIZE))
            qty = int(lots * self.LOT_SIZE)
        else:
            qty = max(1, contracts)
            lots = 0

        if qty <= 0:
            return

        # 8) place order
        exch = "NFO"
        order_side = "BUY"
        entry_order_id = self.executor.place_entry_order(
            symbol=chosen_symbol,
            exchange=exch,
            transaction_type=order_side,
            quantity=int(qty),
        )
        if not entry_order_id:
            return

        # exits + internal record
        ok = self.executor.setup_gtt_orders(
            entry_order_id=entry_order_id,
            entry_price=entry,
            stop_loss_price=stop,
            target_price=target,
            symbol=chosen_symbol,
            exchange=exch,
            quantity=int(qty),
            transaction_type=order_side,
        )
        if not ok:
            logger.warning("Exits setup failed for %s %s", exch, chosen_symbol)

        # ATR for trailing (on option series)
        atr_val = 0.0
        try:
            if compute_atr_df and not chosen_df.empty:
                atr_s = compute_atr_df(chosen_df, period=int(getattr(Config, "ATR_PERIOD", 14)), method="rma")
                if isinstance(atr_s, pd.Series) and len(atr_s) > 0:
                    atr_val = float(atr_s.iloc[-1] or 0.0)
        except Exception:
            atr_val = max(1.0, abs(entry - stop))

        with self._lock:
            self.active_trades[str(entry_order_id)] = {
                "status": "OPEN",
                "order_id": str(entry_order_id),
                "symbol": chosen_symbol,
                "exchange": exch,
                "direction": order_side,
                "contracts": int(qty),
                "entry_price": float(entry),
                "stop_loss": float(stop),
                "target": float(target),
                "confidence": float(conf),
                "atr": float(atr_val),
                "expiry": expiry_iso,
                "side": chosen_side,
                "last_close": _round2(chosen_df["close"].iloc[-1]) if not chosen_df.empty else 0.0,
            }

        # notify
        self._safe_send_signal_alert(
            token=int(chosen_token or 0),
            signal=dict(
                signal="BUY",
                entry_price=_round2(entry),
                stop_loss=_round2(stop),
                target=_round2(target),
                confidence=float(conf),
            ),
            position=dict(quantity=int(qty)),
        )

    # -------------------- finalize / pnl -------------------- #

    def _finalize_trade(self, entry_id: str) -> None:
        with self._lock:
            tr = self.active_trades.pop(entry_id, None)

        if not tr:
            return

        entry = float(tr.get("entry_price") or 0.0)
        exit_px = float(tr.get("exit_price") or 0.0)
        qty = int(tr.get("contracts") or 0)
        direction = tr.get("direction", "BUY")
        lots = (qty // self.LOT_SIZE) if self.LOT_SIZE else 0

        gross = (exit_px - entry) * qty if direction == "BUY" else (entry - exit_px) * qty
        fees = float(self.FEES_PER_LOT) * max(0, lots)
        net = gross - fees

        self.daily_pnl += net
        self.trades.append(
            dict(entry_id=entry_id, symbol=tr.get("symbol"), qty=qty, entry=entry, exit=exit_px, pnl=net)
        )
        self._closed_trades_today += 1
        self._last_closed_was_loss = net < 0

        # loss cooldown
        if net < 0 and self.LOSS_COOLDOWN_MIN > 0:
            self._cooldown_until_ts = time.time() + self.LOSS_COOLDOWN_MIN * 60

        # CSV log
        self._append_trade_log(
            [
                _ist_now().strftime("%Y-%m-%d %H:%M:%S"),
                entry_id,
                tr.get("symbol"),
                direction,
                qty,
                _round2(entry),
                _round2(exit_px),
                _round2(gross),
                _round2(fees),
                _round2(net),
                _round2(tr.get("confidence", 0.0)),
                _round2(tr.get("atr", 0.0)),
                "LIVE" if self.live_mode else "SHADOW",
            ]
        )

        # Telegram summary per trade
        msg = (
            f"📌 Trade closed [{tr.get('symbol')}] {direction} x{qty}\n"
            f"Entry: {_round2(entry)}  Exit: {_round2(exit_px)}\n"
            f"Gross: {_round2(gross)}  Fees: {_round2(fees)}  Net: <b>{_round2(net)}</b>\n"
            f"Daily P&L: <b>{_round2(self.daily_pnl)}</b>"
        )
        self._safe_send_message(msg, parse_mode="HTML")

    # -------------------- status / summary -------------------- #

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            open_cnt = sum(1 for v in self.active_trades.values() if v.get("status") == "OPEN")
        return {
            "is_trading": self.is_trading,
            "live_mode": self.live_mode,
            "open_orders": open_cnt,
            "trades_today": self._closed_trades_today,
            "daily_pnl": float(self.daily_pnl),
            "risk_level": getattr(self.risk, "risk_level", "N/A"),
        }

    def get_summary(self) -> str:
        with self._lock:
            tcount = len(self.trades)
            win = sum(1 for t in self.trades if float(t.get("pnl", 0.0)) > 0)
            loss = tcount - win
            wr = (win / tcount * 100.0) if tcount else 0.0
        return (
            f"📈 <b>Daily Summary</b>\n"
            f"Trades: {tcount} (✅ {win} / ❌ {loss})  Win%: {wr:.1f}%\n"
            f"P&L: <b>{self.daily_pnl:.2f}</b>"
        )

    def _send_detailed_status(self) -> bool:
        st = self.get_status()
        text = (
            "📊 <b>Bot Status</b>\n"
            f"🔁 Trading: {'🟢 Running' if st['is_trading'] else '🔴 Stopped'}\n"
            f"🌐 Mode: {'🟢 LIVE' if st['live_mode'] else '🛡️ Shadow'}\n"
            f"📦 Open Orders: {st['open_orders']}\n"
            f"📈 Trades Today: {st['trades_today']}\n"
            f"💰 Daily P&L: {st['daily_pnl']:.2f}\n"
            f"⚖️ Risk Level: {st['risk_level']}"
        )
        self._safe_send_message(text, parse_mode="HTML")
        return True

    def _run_health_check(self) -> bool:
        ok = True
        if self.live_mode and not getattr(self.executor, "kite", None):
            ok = False
        if self.strategy is None:
            ok = False
        self._safe_send_message("✅ Health OK." if ok else "❌ Health issues detected.")
        return ok

    # -------------------- messaging -------------------- #

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

    def _safe_send_signal_alert(self, token: int, signal: Dict[str, Any], position: Dict[str, Any]) -> None:
        try:
            if self.tg:
                self.tg.send_signal_alert(token, signal, position)
        except Exception:
            pass

    # -------------------- shutdown -------------------- #

    def shutdown(self) -> None:
        logger.info("🔻 Shutting down trader…")
        try:
            self.stop()
        except Exception:
            pass
        try:
            self._stop_polling()
        except Exception:
            pass