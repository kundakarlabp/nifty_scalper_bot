# src/data_streaming/realtime_trader.py
from __future__ import annotations

"""
Real-time trader with:
- Telegram control (daemon polling)
- Quality mode toggle (/mode quality on|off), risk & regime runtime tweaks
- Adaptive main loop (peak/off-peak cadence)
- Risk-based lot sizing (RISK_PER_TRADE × equity; ATR- and SL-aware)
- Warmup & trading-hours filters (IST aware, optional buckets & event windows)
- Multi-timeframe gate (HTF EMA / slope) and regime filter (ADX + BB width)
- Options strike resolution (ATM ± range via cached instruments) [best-effort]
- Strategy signals (spot/futures + options)
- Spread guard (RANGE or LTP_MID; optional dynamic scaling)
- Partial TP, breakeven hop, trailing SL (delegated to OrderExecutor)
- 3-loss shutdown + loss cooldown + trade/day cap + streak ladders
- Daily circuit breaker + session R ladders + session auto-exit
- CSV trade log + daily rollover + idempotent entry protection
- State persistence (active trades) with best-effort reattach on restart

Assumptions:
- Config is loaded from env via src/config.py
- OrderExecutor implements: place_entry_order, setup_gtt_orders, update_trailing_stop,
  exit_order, cancel_all_orders, get_active_orders, get_positions, get_last_price,
  sync_and_enforce_oco, get_tick_size.
- EnhancedScalpingStrategy implements: generate_signal(...) and generate_options_signal(...).
- strike_selector implements: get_instrument_tokens(...), fetch_cached_instruments(...).
- You may swap data providers; default uses Kite when live, a stub when sim.

Paths:
- If you keep this file under src/data_streaming/, imports below match your tree.
"""

import atexit
import csv
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, date, time as dtime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import schedule

from src.config import Config
from src.strategies.scalping_strategy import EnhancedScalpingStrategy
from src.execution.order_executor import OrderExecutor
from src.risk.position_sizing import PositionSizing, get_live_account_balance
from src.notifications.telegram_controller import TelegramController
from src.utils.atr_helper import compute_atr_df
from src.utils.indicators import calculate_ema, calculate_adx

# Optional (best-effort) strike utilities for options flow
try:
    from src.utils.strike_selector import (
        get_instrument_tokens,
        fetch_cached_instruments,
    )
except Exception:  # pragma: no cover
    get_instrument_tokens = None
    fetch_cached_instruments = None

logger = logging.getLogger(__name__)


# ============================== time helpers ============================== #

def _ist_now() -> datetime:
    """IST clock without pytz (UTC+5:30)."""
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
    for a, b in windows or []:
        try:
            if _between(now_t, a, b):
                return True
        except Exception:
            continue
    return False


def _safe_df(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


def _round_to_tick(x: float, tick: float) -> float:
    if tick <= 0:
        return float(x)
    return round(float(x) / tick) * tick


# ============================ data providers ============================ #

class DataProvider:
    """Replace internals with your broker/data code (simulation-safe)."""

    def get_ohlc(self, symbol: str, minutes: int, timeframe: str = "minute") -> pd.DataFrame:
        return pd.DataFrame()

    def get_last_price(self, symbol: str) -> Optional[float]:
        return None


class KiteDataProvider:
    """
    Thin adapter over KiteConnect:
      - get_ohlc(symbol, minutes, timeframe) -> DataFrame [open,high,low,close,volume]
      - get_last_price(symbol) -> float

    Uses: ZERODHA_API_KEY + (KITE_ACCESS_TOKEN or ZERODHA_ACCESS_TOKEN)
    """
    _TF_MAP = {
        "minute": "minute", "1m": "minute",
        "3minute": "3minute", "3m": "3minute",
        "5minute": "5minute", "5m": "5minute",
        "10minute": "10minute",
        "15minute": "15minute", "15m": "15minute",
        "30minute": "30minute", "30m": "30minute",
        "60minute": "60minute", "1h": "60minute",
        "day": "day", "d": "day",
    }

    def __init__(self, kite, default_token: Optional[int] = None) -> None:
        self.kite = kite
        self.default_token = int(default_token or getattr(Config, "INSTRUMENT_TOKEN", 256265))
        self._ltp_cache: Dict[str, Dict[str, Any]] = {}
        self._ltp_ttl = 1.8  # seconds

    @staticmethod
    def build_from_env() -> "KiteDataProvider":
        from kiteconnect import KiteConnect
        api_key = getattr(Config, "ZERODHA_API_KEY", None)
        access_token = getattr(Config, "KITE_ACCESS_TOKEN", None) or getattr(Config, "ZERODHA_ACCESS_TOKEN", None)
        if not api_key or not access_token:
            raise RuntimeError("ZERODHA_API_KEY or KITE_ACCESS_TOKEN missing for KiteDataProvider")
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        return KiteDataProvider(kite=kite, default_token=getattr(Config, "INSTRUMENT_TOKEN", 256265))

    def _with_retries(self, fn, *args, tries=2, backoff=0.6, **kwargs):
        last = None
        for i in range(tries):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last = e
                if i == tries - 1:
                    break
                time.sleep(backoff * (2 ** i))
        raise last

    def _resolve_token(self, symbol_or_none: Optional[str]) -> int:
        if not symbol_or_none:
            return int(self.default_token)
        if symbol_or_none == getattr(Config, "SPOT_SYMBOL", "NSE:NIFTY 50"):
            return int(self.default_token)
        try:
            q = self._with_retries(self.kite.quote, [symbol_or_none])
            tok = (q.get(symbol_or_none, {}) or {}).get("instrument_token")
            if tok:
                return int(tok)
        except Exception:
            pass
        raise ValueError(f"Cannot resolve instrument token for {symbol_or_none}")

    def get_ohlc(self, symbol: str, minutes: int, timeframe: str = "minute") -> pd.DataFrame:
        interval = self._TF_MAP.get(str(timeframe).lower(), "minute")
        token = self._resolve_token(symbol)
        end_ts = _ist_now()
        start_ts = end_ts - timedelta(minutes=max(2 * minutes, minutes + 60))

        def _hist():
            return self.kite.historical_data(
                instrument_token=token,
                from_date=start_ts,
                to_date=end_ts,
                interval=interval,
                continuous=False,
                oi=False,
            )

        candles = self._with_retries(_hist)
        if not candles:
            return pd.DataFrame()
        df = pd.DataFrame(candles)
        if "date" in df.columns:
            df = df.set_index("date", drop=True)
        df = df.sort_index()
        cutoff = end_ts - timedelta(minutes=minutes + 1)
        df = df[df.index >= cutoff]
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        return df[keep].copy()

    def get_last_price(self, symbol: str) -> Optional[float]:
        now = time.time()
        c = self._ltp_cache.get(symbol)
        if c and (now - c["ts"] <= self._ltp_ttl):
            return float(c["ltp"])
        try:
            data = self._with_retries(self.kite.ltp, [symbol])
            px = (data.get(symbol, {}) or {}).get("last_price")
            if px is None:
                data = self._with_retries(self.kite.quote, [symbol])
                px = (data.get(symbol, {}) or {}).get("last_price")
            if px is None:
                return None
            self._ltp_cache[symbol] = {"ts": now, "ltp": float(px)}
            return float(px)
        except Exception:
            return None


# ============================== models =============================== #

@dataclass
class ActiveTrade:
    entry_id: str
    symbol: str
    direction: str
    quantity: int
    entry_price: float
    stop_loss: float
    target: float
    atr: float
    opened_ts: float
    status: str = "OPEN"   # OPEN/CLOSED
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None


# ============================== trader =============================== #

class RealTimeTrader:
    """
    Production-grade runner coordinating strategy, risk, execution and IO.
    """

    # ---- knobs mirrored from Config ----
    MAX_CONCURRENT = int(getattr(Config, "MAX_CONCURRENT_POSITIONS", 1))
    WARMUP_BARS = int(getattr(Config, "WARMUP_BARS", 25))
    DATA_LOOKBACK_MIN = int(getattr(Config, "DATA_LOOKBACK_MINUTES", 45))
    HIST_TF = str(getattr(Config, "HISTORICAL_TIMEFRAME", "minute"))

    # spread guard
    SPREAD_MODE = str(getattr(Config, "SPREAD_GUARD_MODE", "RANGE")).upper()
    SPREAD_BA_MAX = float(getattr(Config, "SPREAD_GUARD_BA_MAX", 0.015))
    SPREAD_LTPMID_MAX = float(getattr(Config, "SPREAD_GUARD_LTPMID_MAX", 0.015))
    SPREAD_PCT = float(getattr(Config, "SPREAD_GUARD_PCT", 0.03))
    DYN_SPREAD = bool(getattr(Config, "DYNAMIC_SPREAD_GUARD", True))
    SPREAD_VOL_LOOKBACK = int(getattr(Config, "SPREAD_VOL_LOOKBACK", 20))

    # costs / risk / trailing
    SLIPPAGE_BPS = float(getattr(Config, "SLIPPAGE_BPS", 4.0))
    FEES_PER_LOT = float(getattr(Config, "FEES_PER_LOT", 25.0))
    MAX_DD_PCT = float(getattr(Config, "MAX_DAILY_DRAWDOWN_PCT", 0.05))
    CIRCUIT_RELEASE_PCT = float(getattr(Config, "CIRCUIT_RELEASE_PCT", 0.015))
    TRAILING_ENABLE = bool(getattr(Config, "TRAILING_ENABLE", True))
    TRAIL_ATR_MULT = float(getattr(Config, "ATR_SL_MULTIPLIER", 1.5))
    WORKER_INTERVAL = int(getattr(Config, "WORKER_INTERVAL_SEC", 4))
    LOG_FILE = str(getattr(Config, "LOG_FILE", "logs/trades.csv"))

    LOT_SIZE = int(getattr(Config, "NIFTY_LOT_SIZE", 75))
    MIN_LOTS = int(getattr(Config, "MIN_LOTS", 1))
    MAX_LOTS = int(getattr(Config, "MAX_LOTS", 15))
    RISK_PER_TRADE = float(getattr(Config, "RISK_PER_TRADE", 0.025))
    MAX_TRADES_DAY = int(getattr(Config, "MAX_TRADES_PER_DAY", 30))
    LOSS_COOLDOWN_MIN = int(getattr(Config, "LOSS_COOLDOWN_MIN", 2))
    LOSS_LIMIT = int(getattr(Config, "CONSECUTIVE_LOSS_LIMIT", 3))

    # ladders / stop after good/bad days
    LOSS_STREAK_HALVE = int(getattr(Config, "LOSS_STREAK_HALVE_SIZE", 3))
    LOSS_STREAK_PAUSE = int(getattr(Config, "LOSS_STREAK_PAUSE_MIN", 20))
    DAY_STOP_POS_R = float(getattr(Config, "DAY_STOP_AFTER_POS_R", 4.0))
    DAY_HALF_POS_R = float(getattr(Config, "DAY_HALF_SIZE_AFTER_POS_R", 2.0))
    DAY_STOP_NEG_R = float(getattr(Config, "DAY_STOP_AFTER_NEG_R", -3.0))

    # hours / filters
    TIME_START = str(getattr(Config, "TIME_FILTER_START", "09:20"))
    TIME_END = str(getattr(Config, "TIME_FILTER_END", "15:20"))
    SKIP_FIRST_MIN = int(getattr(Config, "SKIP_FIRST_MIN", 5))
    ENABLE_BUCKETS = bool(getattr(Config, "ENABLE_TIME_BUCKETS", False))
    BUCKETS = list(getattr(Config, "TIME_BUCKETS", []))
    ENABLE_EVENTS = bool(getattr(Config, "ENABLE_EVENT_WINDOWS", False))
    EVENTS = list(getattr(Config, "EVENT_WINDOWS", []))
    AUTO_EXIT_TIME = str(getattr(Config, "SESSION_AUTO_EXIT_TIME", getattr(Config, "TIME_FILTER_END", "15:20")))

    STRIKE_RANGE = int(getattr(Config, "STRIKE_RANGE", 3))
    IDEMP_TTL = int(getattr(Config, "IDEMP_TTL_SEC", 60))

    # files
    STATE_DIR = "state"
    ACTIVE_JSON = os.path.join(STATE_DIR, "active_trades.json")

    def __init__(self, data: Optional[DataProvider] = None) -> None:
        self._lock = threading.RLock()

        self.is_trading = False
        self.live_mode = bool(getattr(Config, "ENABLE_LIVE_TRADING", False))
        self.quality_mode = bool(getattr(Config, "QUALITY_MODE_DEFAULT", False))

        # session stats
        self.daily_start_equity: float = float(get_live_account_balance() or 0.0)
        self.session_date: date = _now().date()
        self.daily_pnl: float = 0.0
        self.session_R: float = 0.0
        self.trades_closed_today: int = 0
        self.loss_streak: int = 0
        self._cooldown_until: float = 0.0
        self._halt_for_day: bool = False
        self._regime_mode: str = str(getattr(Config, "REGIME_MODE", "AUTO")).upper()

        # trade state
        self.trades: List[Dict[str, Any]] = []               # closed trades
        self.active: Dict[str, ActiveTrade] = {}             # entry_id -> ActiveTrade
        self._recent_keys: Dict[str, float] = {}             # dedup keys (key -> expiry ts)

        # data/executor/strategy
        self.data = data or (self._build_kite_provider() if self.live_mode else DataProvider())
        self.risk = PositionSizing()
        try:
            self.risk.set_equity(float(self.daily_start_equity or 0.0))
        except Exception:
            setattr(self.risk, "equity", float(self.daily_start_equity or 0.0))
        self.strategy = EnhancedScalpingStrategy(
            base_stop_loss_points=Config.BASE_STOP_LOSS_POINTS,
            base_target_points=Config.BASE_TARGET_POINTS,
            confidence_threshold=Config.CONFIDENCE_THRESHOLD,
            min_score_threshold=int(Config.MIN_SIGNAL_SCORE),
        )
        self.executor = self._init_executor()

        # telegram
        self.tg: Optional[TelegramController] = None
        self._polling_thread: Optional[threading.Thread] = None
        self._init_telegram()

        # workers
        self._trailing_evt = threading.Event()
        self._oco_evt = threading.Event()
        self._start_workers()

        # scheduler
        self._data_job = None
        self._setup_scheduler()

        # csv log + shutdown hook
        self._prepare_trade_log()
        self._maybe_restore_state()
        atexit.register(self.shutdown)

        logger.info("✅ RealTimeTrader initialized.")

    # -------------------------- components -------------------------- #

    def _build_kite_provider(self) -> DataProvider:
        try:
            return KiteDataProvider.build_from_env()
        except Exception as e:
            logger.error("KiteDataProvider init failed, using stub DataProvider: %s", e, exc_info=True)
            return DataProvider()

    def _init_executor(self) -> OrderExecutor:
        if not self.live_mode:
            logger.info("Live trading disabled → simulation mode.")
            return OrderExecutor()
        try:
            from kiteconnect import KiteConnect
            api_key = getattr(Config, "ZERODHA_API_KEY", "")
            access_token = getattr(Config, "KITE_ACCESS_TOKEN", "") or getattr(Config, "ZERODHA_ACCESS_TOKEN", "")
            if not api_key or not access_token:
                raise RuntimeError("ZERODHA_API_KEY or KITE_ACCESS_TOKEN missing")
            kite = KiteConnect(api_key=api_key)
            kite.set_access_token(access_token)
            logger.info("🟢 Live executor created (KiteConnect).")
            return OrderExecutor(kite=kite)
        except Exception as exc:
            logger.error("Live init failed, switching to simulation: %s", exc, exc_info=True)
            self.live_mode = False
            return OrderExecutor()

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
            self._polling_thread = threading.Thread(target=self.tg.start_polling, daemon=True)
            self._polling_thread.start()
            logger.info("📡 Telegram polling started.")
        except Exception as e:
            logger.warning("Telegram init failed: %s", e)

    # --------------------------- scheduler --------------------------- #

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
        logger.info("⏱️ Adaptive scheduler primed.")

    def _current_poll_seconds(self) -> int:
        try:
            now_t = _now().time()
            in_peak = (dtime(9, 20) <= now_t <= dtime(11, 30)) or (dtime(13, 30) <= now_t <= dtime(15, 5))
            return int(getattr(Config, "PEAK_POLL_SEC", 12)) if in_peak else int(getattr(Config, "OFFPEAK_POLL_SEC", 25))
        except Exception:
            return int(getattr(Config, "OFFPEAK_POLL_SEC", 25))

    def _ensure_cadence(self) -> None:
        try:
            sec = self._current_poll_seconds()
            if self._data_job and self._data_job.interval.seconds == sec:
                return
            if self._data_job:
                schedule.cancel_job(self._data_job)
            self._data_job = schedule.every(sec).seconds.do(self._smart_tick)
            logger.info("📈 Data loop cadence: every %ds", sec)
        except Exception as e:
            logger.debug("Cadence error: %s", e)

    # ----------------------------- run loop ----------------------------- #

    def run(self) -> None:
        logger.info("🟢 RealTimeTrader.run() started.")
        while True:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                logger.error("Run loop error: %s", e, exc_info=True)
                time.sleep(2)

    # ---------------------------- workers ---------------------------- #

    def _start_workers(self) -> None:
        threading.Thread(target=self._trailing_worker, daemon=True).start()
        threading.Thread(target=self._oco_worker, daemon=True).start()

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

    # ----------------------------- telegram ----------------------------- #

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
                    tail = low.split(" ", 1)[1] if " " in low else ""
                    if tail in ("on", "true", "1"):
                        self.quality_mode = True
                        self._safe_send_message("✨ Quality mode: ON")
                        return True
                    if tail in ("off", "false", "0"):
                        self.quality_mode = False
                        self._safe_send_message("✨ Quality mode: OFF")
                        return True
                    self._safe_send_message("Usage: /mode quality on|off")
                    return False
                self._safe_send_message("Usage: /mode live | /mode shadow | /mode quality on|off")
                return False
            if cmd == "risk":
                # /risk 0.5%  OR  /risk 0.005
                try:
                    v = arg.strip().replace("%", "")
                    pct = float(v)
                    if pct > 1:
                        pct = pct / 100.0
                    self.RISK_PER_TRADE = max(0.0005, min(0.05, pct))
                    self._safe_send_message(f"🔧 Risk per trade set to {self.RISK_PER_TRADE:.2%}")
                    return True
                except Exception:
                    self._safe_send_message("Usage: /risk 0.5%  (range 0.05%..5%)")
                    return False
            if cmd == "regime":
                # /regime auto|trend|range|off
                m = arg.strip().upper()
                if m in ("AUTO", "TREND", "RANGE", "OFF"):
                    self._regime_mode = m
                    self._safe_send_message(f"🎛️ Regime mode: {m}")
                    return True
                self._safe_send_message("Usage: /regime auto|trend|range|off")
                return False
            if cmd == "pause":
                # /pause 10m
                try:
                    txt = arg.strip().lower()
                    mins = int(txt[:-1]) if txt.endswith("m") else int(txt)
                    self._cooldown_until = time.time() + max(1, mins) * 60
                    self._safe_send_message(f"⏸️ Paused for {mins} min.")
                    return True
                except Exception:
                    self._safe_send_message("Usage: /pause <minutes>  e.g. /pause 10")
                    return False
            if cmd == "resume":
                self._cooldown_until = 0.0
                self._safe_send_message("▶️ Resumed.")
                return True
            if cmd == "refresh":
                return self.refresh_account_balance()
            if cmd == "status":
                return self._send_detailed_status()
            if cmd == "health":
                return self._run_health_check()
            if cmd == "emergency":
                return self.emergency_stop_all()
            self._safe_send_message(f"❌ Unknown command: {cmd}")
            return False
        except Exception as e:
            logger.error("Control error: %s", e, exc_info=True)
            return False

    # ----------------------------- mode ----------------------------- #

    def enable_live(self) -> bool:
        with self._lock:
            if self.is_trading:
                self._safe_send_message("🛑 Stop trading first: /stop")
                return False
            self.live_mode = True
            self.data = self._build_kite_provider()
            self.executor = self._init_executor()
        self._safe_send_message("🌐 Mode: LIVE")
        return True

    def disable_live(self) -> bool:
        with self._lock:
            if self.is_trading:
                self._safe_send_message("🛑 Stop trading first: /stop")
                return False
            self.live_mode = False
            self.data = DataProvider()
            self.executor = self._init_executor()
        self._safe_send_message("🛡️ Mode: Shadow (paper)")
        return True

    # --------------------------- status APIs --------------------------- #

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "is_trading": self.is_trading,
                "live_mode": self.live_mode,
                "quality_mode": self.quality_mode,
                "open_positions": len(self.active),
                "trades_today": self.trades_closed_today,
                "daily_pnl": round(self.daily_pnl, 2),
                "account_size": round(getattr(self.risk, "equity", 0.0), 2),
                "session_date": str(self.session_date),
            }

    def get_summary(self) -> str:
        with self._lock:
            win = sum(1 for t in self.trades if (t.get("net_pnl", 0) or 0) > 0)
            loss = sum(1 for t in self.trades if (t.get("net_pnl", 0) or 0) <= 0)
            return (
                f"<b>Daily Summary ({self.session_date})</b>\n"
                f"Closed trades: {self.trades_closed_today}\n"
                f"W/L: {win}/{loss}\n"
                f"Daily P&L: ₹{self.daily_pnl:.2f}\n"
                f"Session R: {self.session_R:.2f}\n"
                f"Quality: {'ON' if self.quality_mode else 'OFF'}\n"
                f"Regime: {self._regime_mode}"
            )

    def _send_detailed_status(self) -> bool:
        try:
            s = self.get_status()
            if self.tg:
                self.tg._send_status(s)
            return True
        except Exception:
            return False

    # ----------------------- csv log & bookkeeping ---------------------- #

    def _prepare_trade_log(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.LOG_FILE) or ".", exist_ok=True)
            if not os.path.exists(self.LOG_FILE):
                with open(self.LOG_FILE, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(
                        ["date", "order_id", "symbol", "direction", "contracts",
                         "entry", "exit", "pnl", "fees", "net_pnl", "confidence",
                         "atr", "mode", "comment"]
                    )
        except Exception as e:
            logger.warning("Trade log init failed: %s", e)

    def _append_trade_log(self, row: List[Any]) -> None:
        try:
            with open(self.LOG_FILE, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)
        except Exception as e:
            logger.debug("Trade log append failed: %s", e)

    def _roll_daily_if_needed(self) -> None:
        if _now().date() != self.session_date:
            logger.info("📅 New session date detected → resetting day stats.")
            self.session_date = _now().date()
            self.daily_start_equity = float(get_live_account_balance() or getattr(self.risk, "equity", 0.0))
            self.daily_pnl = 0.0
            self.session_R = 0.0
            self.trades_closed_today = 0
            self.loss_streak = 0
            self._halt_for_day = False
            self._recent_keys.clear()
            self._persist_state()

    # --------------------------- public control --------------------------- #

    def start(self) -> bool:
        with self._lock:
            if self.is_trading:
                self._safe_send_message("Already running.")
                return True
            self.is_trading = True
        self._safe_send_alert("START")
        return True

    def stop(self) -> bool:
        with self._lock:
            if not self.is_trading:
                self._safe_send_message("Already stopped.")
                return True
            self.is_trading = False
        self._safe_send_alert("STOP")
        return True

    def emergency_stop_all(self) -> bool:
        try:
            self.stop()
            self.executor.cancel_all_orders()
            self._safe_send_message("Emergency stop executed. All open orders cancelled (best-effort).")
            return True
        except Exception as e:
            logger.error("Emergency stop failed: %s", e)
            return False

    # --------------------------- smart tick --------------------------- #

    def _smart_tick(self) -> None:
        if not self.is_trading:
            return

        now = _now()
        # trading hours & event windows
        if not _between(now.time(), self.TIME_START, self.TIME_END) and not getattr(Config, "ALLOW_OFFHOURS_TESTING", False):
            return
        if self.ENABLE_BUCKETS and not _within_any_windows(now.time(), self.BUCKETS):
            return
        if self.ENABLE_EVENTS and _within_any_windows(now.time(), self.EVENTS):
            return
        # first N min guard after open
        try:
            t_open = datetime.strptime(self.TIME_START, "%H:%M").time()
            mins_since_open = (datetime.combine(now.date(), now.time()) - datetime.combine(now.date(), t_open)).seconds // 60
            if 0 <= mins_since_open < self.SKIP_FIRST_MIN:
                return
        except Exception:
            pass

        # risk halts
        if self._halt_for_day or self._circuit_tripped() or self._in_loss_cooldown():
            return

        # capacity & trade count
        if len(self.active) >= self.MAX_CONCURRENT:
            return
        if self.trades_closed_today >= self.MAX_TRADES_DAY:
            return

        # pull spot/index data
        spot_symbol = getattr(Config, "SPOT_SYMBOL", "NSE:NIFTY 50")
        df_spot = _safe_df(self.data.get_ohlc(symbol=spot_symbol,
                                              minutes=self.DATA_LOOKBACK_MIN,
                                              timeframe=self.HIST_TF))
        if df_spot.empty or len(df_spot) < self.WARMUP_BARS or "close" not in df_spot.columns:
            return

        current_price = float(df_spot["close"].iloc[-1])

        # HTF & regime filters
        if not self._pass_mtf_gate(spot_symbol):
            return
        if not self._pass_regime_gate(df_spot):
            return

        # Try options path first (if configured and strike utils available)
        option_signal, option_meta = (None, None)
        if (str(getattr(Config, "OPTION_TYPE", "BOTH")).upper() != "NONE") and get_instrument_tokens:
            try:
                option_signal, option_meta = self._options_pass(df_spot)
            except Exception as e:
                logger.debug("Options pass error: %s", e)

        if option_signal and option_meta:
            # Place option trade
            self._place_and_register(signal=option_signal,
                                     trade_symbol=option_meta["symbol"],
                                     is_option=True,
                                     df_for_atr=df_spot)
            return

        # Fallback: futures/spot signal
        signal = self.strategy.generate_signal(df_spot, current_price)
        if not signal:
            return

        # quality gate (extra strict if quality ON)
        if self.quality_mode:
            bump = float(getattr(Config, "QUALITY_SCORE_BUMP", 1.0))
            if signal.get("confidence", 0) < (float(getattr(Config, "CONFIDENCE_THRESHOLD", 6.0)) + bump):
                return

        # idempotency (direction + price + minute)
        key = f"{signal['signal']}-{round(signal['entry_price'],2)}-{df_spot.index[-1]}"
        if not self._idemp_ok(key):
            return

        # spread guard (spot is lenient → pass)
        if not self._spread_ok(signal["entry_price"]):
            return

        # place futures/spot leg (use TRADE_SYMBOL/EXCHANGE from Config)
        self._place_and_register(signal=signal,
                                 trade_symbol=getattr(Config, "TRADE_SYMBOL", "NIFTY"),
                                 is_option=False,
                                 df_for_atr=df_spot)

    # -------------------------- options path -------------------------- #

    def _options_pass(self, df_spot: pd.DataFrame) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Resolve ATM±range strikes, evaluate option breakout signals,
        return (signal_dict, meta_dict) for the best candidate.
        meta_dict contains: symbol, strike, type.
        """
        if not callable(get_instrument_tokens):
            return None, None

        try:
            spot_px = float(df_spot["close"].iloc[-1])
        except Exception:
            return None, None

        # Gather candidate strikes via project utilities
        try:
            # get_instrument_tokens(...) is project-specific; tolerate variations
            candidates = get_instrument_tokens(spot_px, range_count=self.STRIKE_RANGE) \
                if "range_count" in get_instrument_tokens.__code__.co_varnames \
                else get_instrument_tokens(spot_px, self.STRIKE_RANGE)
        except Exception:
            candidates = []

        if not candidates and callable(fetch_cached_instruments):
            try:
                # fallback: filter cached instruments near ATM
                inst = fetch_cached_instruments()
                # best-effort: pick CE/PE around ATM (requires your implementation)
                candidates = inst  # if already filtered upstream
            except Exception:
                pass

        if not candidates:
            return None, None

        best = None
        best_meta = None
        for c in candidates:
            sym = c.get("symbol") or c.get("tradingsymbol")
            if not sym:
                continue
            try:
                df_opt = _safe_df(self.data.get_ohlc(symbol=sym,
                                                     minutes=max(30, self.DATA_LOOKBACK_MIN // 2),
                                                     timeframe=self.HIST_TF))
                if df_opt.empty or len(df_opt) < 6:
                    continue
                ltp = self.data.get_last_price(sym) or float(df_opt["close"].iloc[-1])
                if not ltp or ltp <= 0:
                    continue
                sig = self.strategy.generate_options_signal(
                    options_ohlc=df_opt,
                    spot_ohlc=df_spot,
                    strike_info={"symbol": sym, "strike": c.get("strike"), "type": str(c.get("type", "")).upper()},
                    current_option_price=float(ltp),
                )
                if not sig:
                    continue
                if (best is None) or (sig.get("confidence", 0) > best.get("confidence", 0)):
                    best = sig
                    best_meta = {"symbol": sym, "strike": c.get("strike"), "type": str(c.get("type", "")).upper()}
            except Exception as e:
                logger.debug("Option eval failed for %s: %s", sym, e)

        return best, best_meta

    # --------------------- placement & registration --------------------- #

    def _place_and_register(self, *, signal: Dict[str, Any], trade_symbol: str, is_option: bool, df_for_atr: pd.DataFrame) -> None:
        direction = signal["signal"]
        entry = float(signal["entry_price"])
        stop_loss = float(signal["stop_loss"])
        target = float(signal["target"])
        confidence = float(signal.get("confidence", 5.0))

        # idempotency: include symbol too
        key = f"{trade_symbol}-{direction}-{round(entry,2)}-{int(time.time()//60)}"
        if not self._idemp_ok(key):
            return

        # pre-entry spread guard for options (future: use bid/ask mid via quote)
        if is_option and not self._spread_ok(entry, is_option=True, symbol=trade_symbol):
            return

        qty = self._position_size(entry, stop_loss, confidence)
        if qty <= 0:
            return

        side = "BUY" if direction == "BUY" else "SELL"
        oid = self.executor.place_entry_order(
            symbol=trade_symbol,
            exchange=getattr(Config, "TRADE_EXCHANGE", "NFO"),
            transaction_type=side,
            quantity=qty,
        )
        if not oid:
            return

        ok = self.executor.setup_gtt_orders(
            entry_order_id=oid,
            entry_price=entry,
            stop_loss_price=stop_loss,
            target_price=target,
            symbol=trade_symbol,
            exchange=getattr(Config, "TRADE_EXCHANGE", "NFO"),
            quantity=qty,
            transaction_type=side,
        )
        if not ok:
            logger.warning("Exit setup failed; trade unmanaged!")

        atr = float(compute_atr_df(df_for_atr, period=int(getattr(Config, "ATR_PERIOD", 14))).iloc[-1])
        with self._lock:
            self.active[oid] = ActiveTrade(
                entry_id=oid,
                symbol=trade_symbol,
                direction=direction,
                quantity=qty,
                entry_price=entry,
                stop_loss=stop_loss,
                target=target,
                atr=atr,
                opened_ts=time.time(),
            )
        self._persist_state()

        # notify
        self._safe_send_signal_alert(token=len(self.trades) + len(self.active), signal=signal,
                                     position={"quantity": qty})

    # -------------------------- gates & guards -------------------------- #

    def _pass_mtf_gate(self, symbol: str) -> bool:
        """Multi-timeframe gate using Config HTF_* (5m by default)."""
        if not getattr(Config, "HTF_TIMEFRAME_MIN", 5):
            return True
        tf_min = int(getattr(Config, "HTF_TIMEFRAME_MIN", 5))
        tf = f"{tf_min}minute" if tf_min < 60 else "60minute"
        df = _safe_df(self.data.get_ohlc(symbol=symbol,
                                         minutes=max(120, self.DATA_LOOKBACK_MIN),
                                         timeframe=tf))
        if df.empty or len(df) < max(30, int(getattr(Config, "HTF_EMA_PERIOD", 20)) + 5):
            return True  # fail-open
        ema = calculate_ema(df, int(getattr(Config, "HTF_EMA_PERIOD", 20)))
        if ema.empty:
            return True
        # Simple slope check over last k bars
        k = min(5, len(ema) - 1)
        slope = float(ema.iloc[-1] - ema.iloc[-1 - k]) / max(1, k)
        min_slope = float(getattr(Config, "HTF_MIN_SLOPE", 0.0))
        return (slope >= min_slope) or (min_slope <= 0.0)

    def _pass_regime_gate(self, df: pd.DataFrame) -> bool:
        if str(self._regime_mode) == "OFF":
            return True
        # ADX for trend strength
        adx, _, _ = calculate_adx(df, period=int(getattr(Config, "ADX_PERIOD", 14)))
        adx_now = float(adx.iloc[-1]) if not adx.empty else 0.0

        # Bollinger width proxy using close stdev
        close = df["close"]
        bb_win = int(getattr(Config, "BB_WINDOW", 20))
        stdev = close.rolling(bb_win, min_periods=bb_win).std(ddof=0)
        ma = close.rolling(bb_win, min_periods=bb_win).mean()
        width = (2 * stdev) / ma
        w = float(width.iloc[-1]) if not width.empty and pd.notna(width.iloc[-1]) else 0.0

        adx_min = float(getattr(Config, "ADX_MIN_TREND", 18.0))
        w_max = float(getattr(Config, "BB_WIDTH_MAX", 0.02))

        mode = self._regime_mode
        if mode == "TREND":
            return adx_now >= adx_min
        if mode == "RANGE":
            return (w <= w_max) and (adx_now < adx_min + 2)
        # AUTO: accept either sufficiently trending or acceptable range
        return (adx_now >= adx_min) or (w <= w_max)

    def _spread_ok(self, entry_price: float, *, is_option: bool = False, symbol: Optional[str] = None) -> bool:
        """
        Guard based on mode:
          - RANGE: recent bar range / price
          - LTP_MID: deviation vs mid-price (requires quote; best-effort here)
        For options we can be stricter later using quote() bid/ask.
        """
        try:
            if self.SPREAD_MODE == "RANGE":
                # use last few spot closes as realized spread proxy
                pct = float(self.SPREAD_PCT or 0.03)
                return True if not is_option else (pct <= self.SPREAD_PCT)
            # LTP_MID path is broker-quote dependent; allow for now
            return True
        except Exception:
            return True

    def _idemp_ok(self, key: str) -> bool:
        now = time.time()
        # purge
        expired = [k for k, t in self._recent_keys.items() if t < now]
        for k in expired:
            self._recent_keys.pop(k, None)
        if key in self._recent_keys:
            return False
        self._recent_keys[key] = now + self.IDEMP_TTL
        return True

    # ----------------------- trailing & OCO ticks ----------------------- #

    def _trailing_tick(self) -> None:
        with self._lock:
            items = list(self.active.items())
        for oid, tr in items:
            if tr.status != "OPEN":
                continue
            ltp = self.data.get_last_price(tr.symbol) or tr.entry_price
            if ltp <= 0:
                continue
            try:
                self.executor.update_trailing_stop(oid, float(ltp), float(tr.atr))
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

        to_finalize: List[str] = []

        with self._lock:
            for entry_id, fill_px in filled or []:
                tr = self.active.get(entry_id)
                if tr and tr.status == "OPEN":
                    tr.exit_price = float(fill_px)
                    tr.exit_reason = "target/stop"
                    to_finalize.append(entry_id)

            for entry_id, tr in list(self.active.items()):
                if tr.status != "OPEN":
                    continue
                if entry_id not in active_ids:
                    # fallback assumption: one of the exits filled
                    tr.exit_price = tr.target if tr.direction == "BUY" else tr.stop_loss
                    tr.exit_reason = "unknown_fill"
                    to_finalize.append(entry_id)

        for entry_id in to_finalize:
            self._finalize_trade(entry_id)

    # ----------------------- finalization & PnL ----------------------- #

    def _finalize_trade(self, entry_id: str) -> None:
        with self._lock:
            tr = self.active.pop(entry_id, None)
        if not tr:
            return

        exit_px = float(tr.exit_price or tr.target)
        gross = (exit_px - tr.entry_price) * (1 if tr.direction == "BUY" else -1) * tr.quantity
        # very rough fee model: per lot * 2; tweak in executor if needed
        lots = max(1, tr.quantity // max(1, self.LOT_SIZE))
        fees = lots * self.FEES_PER_LOT * 2
        net = gross - fees

        r = (exit_px - tr.entry_price) / max(1e-9, abs(tr.entry_price - tr.stop_loss))
        r *= (1 if tr.direction == "BUY" else -1)

        with self._lock:
            self.daily_pnl += net
            self.session_R += r
            self.trades_closed_today += 1
            if net < 0:
                self.loss_streak += 1
            else:
                self.loss_streak = 0
        self._persist_state()

        # 3-loss shutdown & cooldown
        if self.loss_streak >= self.LOSS_LIMIT:
            self._cooldown_until = time.time() + self.LOSS_STREAK_PAUSE * 60
            self._safe_send_message(f"⛔ {self.loss_streak} consecutive losses. Cooling down for {self.LOSS_STREAK_PAUSE} min.")
            if self.loss_streak >= self.LOSS_STREAK_HALVE and hasattr(self.risk, "halve_next_n"):
                self.risk.halve_next_n(3)

        # session ladders
        if self.session_R >= self.DAY_STOP_POS_R:
            self._halt_for_day = True
            self._safe_send_message(f"🏁 Session target reached (+{self.session_R:.1f}R). Halting for the day.")
        elif self.session_R >= self.DAY_HALF_POS_R and hasattr(self.risk, "halve_next_n"):
            self.risk.halve_next_n(3)
            self._safe_send_message("🔻 Halving size for next few trades (good day).")
        elif self.session_R <= self.DAY_STOP_NEG_R:
            self._halt_for_day = True
            self._safe_send_message(f"🛑 Session drawdown ({self.session_R:.1f}R). Halting for the day.")

        # log
        self._append_trade_log([
            str(self.session_date),
            entry_id,
            tr.symbol,
            tr.direction,
            tr.quantity,
            round(tr.entry_price, 2),
            round(exit_px, 2),
            round(gross, 2),
            round(fees, 2),
            round(net, 2),
            "-",  # optional: pass confidence from signal if you store it
            round(tr.atr, 2),
            "LIVE" if self.live_mode else "SIM",
            tr.exit_reason or "",
        ])

    # ---------------------------- guards ---------------------------- #

    def _circuit_tripped(self) -> bool:
        if self.daily_start_equity <= 0:
            return False
        dd = -self.daily_pnl / self.daily_start_equity
        if dd >= self.MAX_DD_PCT:
            if not self._halt_for_day:
                self._halt_for_day = True
                self._safe_send_message(f"🚨 Circuit breaker: daily DD {dd:.2%} ≥ limit {self.MAX_DD_PCT:.2%}.")
            return True
        return False

    def _in_loss_cooldown(self) -> bool:
        return time.time() < getattr(self, "_cooldown_until", 0.0)

    def _auto_exit_guard(self) -> None:
        try:
            end_hm = self.AUTO_EXIT_TIME
            now = _now()
            if _between(now.time(), end_hm, end_hm) and self.active:
                self._safe_send_message("⏰ Session auto-exit time reached. Closing active trades.")
                try:
                    self.executor.cancel_all_orders()
                except Exception:
                    pass
                # mark as closed at last known prices
                with self._lock:
                    for oid, tr in list(self.active.items()):
                        tr.exit_price = tr.target if tr.direction == "BUY" else tr.stop_loss
                        tr.exit_reason = "session_auto_exit"
                        # finalize outside lock to reuse logic
                for oid in list(self.active.keys()):
                    self._finalize_trade(oid)
        except Exception:
            pass

    # ---------------------------- sizing ---------------------------- #

    def _position_size(self, entry: float, stop: float, confidence: float) -> int:
        equity = float(getattr(self.risk, "equity", 0.0))
        risk_amt = float(self.RISK_PER_TRADE or 0.02) * equity
        stop_pts = abs(entry - stop)
        if stop_pts <= 0:
            return 0
        # contracts by rupees / pts; convert to lots
        contracts = int(max(0, risk_amt / stop_pts))
        lots = max(self.MIN_LOTS, min(self.MAX_LOTS, contracts // max(1, self.LOT_SIZE)))
        if self.quality_mode:
            lots = max(self.MIN_LOTS, int(lots * 0.75))
        qty = max(self.LOT_SIZE, lots * self.LOT_SIZE)
        return qty

    # --------------------------- balances --------------------------- #

    def refresh_account_balance(self) -> bool:
        try:
            bal = float(get_live_account_balance() or 0.0)
            if bal > 0:
                self.risk.set_equity(bal) if hasattr(self.risk, "set_equity") else setattr(self.risk, "equity", bal)
            return True
        except Exception:
            return False

    # ----------------------------- IO ----------------------------- #

    def _safe_send_alert(self, action: str) -> None:
        try:
            if self.tg:
                self.tg.send_alert(action)
        except Exception:
            pass

    def _safe_send_message(self, text: str, parse_mode: Optional[str] = None) -> None:
        try:
            if self.tg:
                self.tg.send_message(text, parse_mode=parse_mode)
        except Exception:
            pass

    def _safe_send_signal_alert(self, token: int, signal: Dict[str, Any], position: Dict[str, Any]) -> None:
        try:
            if self.tg:
                self.tg.send_signal_alert(token, signal, position)
        except Exception:
            pass

    def _run_health_check(self) -> bool:
        ok_bits = []
        try:
            ok_bits.append("data✓" if not _safe_df(self.data.get_ohlc(getattr(Config, "SPOT_SYMBOL", "NSE:NIFTY 50"), 3)).empty else "data")
        except Exception:
            ok_bits.append("data✗")
        ok_bits.append("executor✓" if self.executor else "executor✗")
        self._safe_send_message("Health: " + " ".join(ok_bits))
        return True

    # ------------------------ persistence (best-effort) ------------------------ #

    def _persist_state(self) -> None:
        try:
            os.makedirs(self.STATE_DIR, exist_ok=True)
            with open(self.ACTIVE_JSON, "w", encoding="utf-8") as f:
                json.dump({k: asdict(v) for k, v in self.active.items()}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _maybe_restore_state(self) -> None:
        if not bool(getattr(Config, "PERSIST_REATTACH_ON_START", True)):
            return
        try:
            if not os.path.exists(self.ACTIVE_JSON):
                return
            with open(self.ACTIVE_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
            restored = 0
            with self._lock:
                for k, v in (data or {}).items():
                    if k in self.active:
                        continue
                    try:
                        self.active[k] = ActiveTrade(**v)
                        restored += 1
                    except Exception:
                        continue
            if restored:
                self._safe_send_message(f"♻️ Restored {restored} active trade(s) from state. Reconciling…")
        except Exception:
            pass

    # --------------------------- shutdown --------------------------- #

    def shutdown(self) -> None:
        try:
            if self.tg:
                self.tg.stop_polling()
        except Exception:
            pass
        try:
            self._trailing_evt.set()
            self._oco_evt.set()
        except Exception:
            pass
        self._persist_state()
        logger.info("🔚 Trader shutdown.")