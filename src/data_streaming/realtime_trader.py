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

# Optional strike utilities (best-effort)
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

    # ---- knobs from Config ----
    MAX_CONCURRENT = int(getattr(Config, "MAX_CONCURRENT_POSITIONS", 1))
    WARMUP_BARS = int(getattr(Config, "WARMUP_BARS", 25))
    DATA_LOOKBACK_MIN = int(getattr(Config, "DATA_LOOKBACK_MINUTES", 45))
    HIST_TF = str(getattr(Config, "HISTORICAL_TIMEFRAME", "minute"))

    SPREAD_MODE = str(getattr(Config, "SPREAD_GUARD_MODE", "RANGE")).upper()
    SPREAD_BA_MAX = float(getattr(Config, "SPREAD_GUARD_BA_MAX", 0.015))
    SPREAD_LTPMID_MAX = float(getattr(Config, "SPREAD_GUARD_LTPMID_MAX", 0.015))
    SPREAD_PCT = float(getattr(Config, "SPREAD_GUARD_PCT", 0.03))
    DYN_SPREAD = bool(getattr(Config, "DYNAMIC_SPREAD_GUARD", True))
    SPREAD_VOL_LOOKBACK = int(getattr(Config, "SPREAD_VOL_LOOKBACK", 20))

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

    LOSS_STREAK_HALVE = int(getattr(Config, "LOSS_STREAK_HALVE_SIZE", 3))
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

    STRIKE_RANGE = int(getattr(Config, "STRIKE_RANGE", 3))
    IDEMP_TTL = int(getattr(Config, "IDEMP_TTL_SEC", 60))

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
                    on = any(x in low for x in ("on", "true", "1"))
                    self.quality_mode = on
                    logger.info("Quality mode: %s", "ON" if on else "OFF")
                    if self.tg:
                        self.tg.send_message(f"Quality mode: {'ON' if on else 'OFF'}")
                    return True
                if self.tg:
                    self.tg.send_message("Usage: /mode live | /mode shadow | /mode quality on|off")
                return True
            if cmd == "refresh":
                self.refresh_account_balance()
                if self.tg:
                    self.tg.send_message("Refreshed account balance & instruments cache.")
                return True
            if cmd == "health":
                ok = not self._circuit_tripped()
                if self.tg:
                    self.tg.send_message(f"Health: {'OK' if ok else 'CIRCUIT TRIPPED'}")
                return True
            if cmd == "emergency":
                logger.warning("🚨 EMERGENCY STOP triggered via Telegram.")
                self.stop()
                try:
                    self.executor.cancel_all_orders()
                except Exception:
                    pass
                if self.tg:
                    self.tg.send_message("🛑 Emergency stop: trading halted and orders cancelled.")
                return True
            return False
        except Exception as e:
            logger.error("Control error: %s", e, exc_info=True)
            return False

    # ----------------------------- lifecycle ----------------------------- #

    def start(self) -> bool:
        with self._lock:
            if self.is_trading:
                return True
            self.is_trading = True
            if self.tg:
                self.tg.send_alert("START")
            logger.info("Trading STARTED.")
            return True

    def stop(self) -> bool:
        with self._lock:
            if not self.is_trading:
                return True
            self.is_trading = False
            if self.tg:
                self.tg.send_alert("STOP")
            logger.info("Trading STOPPED.")
            return True

    def enable_live(self) -> bool:
        if self.live_mode:
            return True
        self.live_mode = True
        self.executor = self._init_executor()
        if self.tg:
            self.tg.send_message("Switched to 🟢 LIVE mode.")
        return True

    def disable_live(self) -> bool:
        if not self.live_mode:
            return True
        self.live_mode = False
        self.executor = OrderExecutor()  # sim
        if self.tg:
            self.tg.send_message("Switched to 🛡️ Shadow (simulation) mode.")
        return True

    # ------------------------------ status/summary ------------------------------ #

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            st = {
                "is_trading": self.is_trading,
                "live_mode": self.live_mode,
                "open_positions": len(self.active),
                "closed_today": self.trades_closed_today,
                "daily_pnl": round(self.daily_pnl, 2),
                "account_size": round(float(self.risk.equity or 0.0), 2),
                "session_date": str(self.session_date),
            }
            return st

    def get_summary(self) -> str:
        s = self.get_status()
        lines = [
            "<b>Daily Summary</b>",
            f"📅 Session: <code>{s['session_date']}</code>",
            f"🔁 Trading: {'🟢 Running' if s['is_trading'] else '🔴 Stopped'}",
            f"🌐 Mode: {'🟢 LIVE' if s['live_mode'] else '🛡️ Shadow'}",
            f"📦 Open: {s['open_positions']} | 📈 Closed today: {s['closed_today']}",
            f"💰 P&L: <b>{s['daily_pnl']:.2f}</b> | 🏦 Equity: <b>{s['account_size']:.2f}</b>",
        ]
        return "\n".join(lines)

    # ------------------------------ daily/session ------------------------------ #

    def refresh_account_balance(self) -> None:
        try:
            eq = float(get_live_account_balance() or 0.0)
            if eq > 0:
                self.risk.set_equity(eq)
                logger.info("Refreshed equity: %.2f", eq)
        except Exception as e:
            logger.debug("Balance refresh failed: %s", e)

    def _roll_daily_if_needed(self) -> None:
        today = _now().date()
        if today != self.session_date:
            logger.info("🔄 Rollover: %s → %s", self.session_date, today)
            self.session_date = today
            self.daily_start_equity = float(self.risk.equity or self.daily_start_equity)
            self.daily_pnl = 0.0
            self.session_R = 0.0
            self.trades_closed_today = 0
            self.loss_streak = 0
            self._cooldown_until = 0.0
            self._halt_for_day = False

    def _auto_exit_guard(self) -> None:
        try:
            now_t = _now().time()
            if _between(now_t, self.AUTO_EXIT_TIME, self.AUTO_EXIT_TIME):
                if self.active:
                    logger.info("⏰ Auto-exit time; closing open positions.")
                    for oid, rec in list(self.active.items()):
                        try:
                            self.executor.exit_order(oid, exit_reason="session_auto_exit")
                        except Exception:
                            pass
                    self.active.clear()
        except Exception:
            pass

    # ------------------------------ gates/guards ------------------------------ #

    def _circuit_tripped(self) -> bool:
        # daily dd
        eq0 = float(self.daily_start_equity or 0.0)
        if eq0 > 0 and (self.daily_pnl / eq0) <= -abs(self.MAX_DD_PCT):
            return True
        # day R ladders
        if self.session_R <= self.DAY_STOP_NEG_R:
            return True
        if self.session_R >= self.DAY_STOP_POS_R:
            return True
        # halt flags
        if self._halt_for_day:
            return True
        # cooldown after loss bursts
        if time.time() < self._cooldown_until:
            return True
        return False

    def _time_allowed(self) -> bool:
        now = _now()
        if not _between(now.time(), self.TIME_START, self.TIME_END):
            return False
        # bucketed sessions
        if self.ENABLE_BUCKETS and self.BUCKETS:
            if not _within_any_windows(now.time(), self.BUCKETS):
                return False
        # known event windows (block)
        if self.ENABLE_EVENTS and self.EVENTS:
            if _within_any_windows(now.time(), self.EVENTS):
                return False
        # skip first few minutes
        if now.time() < (datetime.strptime(self.TIME_START, "%H:%M") + timedelta(minutes=self.SKIP_FIRST_MIN)).time():
            return False
        return True

    # ------------------------------ main data tick ------------------------------ #

    def _smart_tick(self) -> None:
        if not self.is_trading:
            return
        if self._circuit_tripped():
            logger.debug("Circuit guard active; skipping tick.")
            return
        if not self._time_allowed():
            logger.debug("Outside trading windows; skipping tick.")
            return

        try:
            self._expire_idempotency()
            self._signal_and_maybe_enter()
        except Exception as e:
            logger.error("smart_tick error: %s", e, exc_info=True)

    def _load_data(self) -> Optional[pd.DataFrame]:
        try:
            lookback_min = max(self.DATA_LOOKBACK_MIN, self.WARMUP_BARS + 10)
            sym = getattr(Config, "SPOT_SYMBOL", "NSE:NIFTY 50")
            df = self.data.get_ohlc(sym, minutes=lookback_min, timeframe=self.HIST_TF)
            if df is None or df.empty or len(df) < max(self.WARMUP_BARS, 30):
                return None
            return df
        except Exception as e:
            logger.debug("Data load failed: %s", e)
            return None

    def _signal_and_maybe_enter(self) -> None:
        df = self._load_data()
        if df is None:
            return

        current_price = float(df["close"].iloc[-1])
        signal = self.strategy.generate_signal(df, current_price=current_price)
        if not signal:
            return

        direction = signal["signal"]
        entry_price = float(signal["entry_price"])
        stop_loss = float(signal["stop_loss"])
        target = float(signal["target"])

        # idempotency: key by direction+bar timestamp rounded
        key = f"{direction}@{df.index[-1]}"
        if key in self._recent_keys:
            logger.debug("Idempotent skip for %s", key)
            return

        # risk sizing (contracts)
        r_per_trade = float(self.RISK_PER_TRADE or 0.02)
        self.risk.set_risk_per_trade(r_per_trade)
        qty = int(self.risk.size_by_stop(entry_price, stop_loss, lot_size=self.LOT_SIZE))
        if qty <= 0:
            logger.debug("Sizer returned zero qty; skip.")
            return
        if len(self.active) >= self.MAX_CONCURRENT:
            logger.debug("Max concurrent positions reached.")
            return

        # place entry
        side = "BUY" if direction == "BUY" else "SELL"
        entry_id = self.executor.place_entry_order(
            symbol=getattr(Config, "TRADE_SYMBOL", "NIFTY"),
            exchange=getattr(Config, "TRADE_EXCHANGE", "NFO"),
            transaction_type=side,
            quantity=qty,
        )
        if not entry_id:
            logger.warning("Entry placement failed.")
            return

        ok = self.executor.setup_gtt_orders(
            entry_order_id=entry_id,
            entry_price=entry_price,
            stop_loss_price=stop_loss,
            target_price=target,
            symbol=getattr(Config, "TRADE_SYMBOL", "NIFTY"),
            exchange=getattr(Config, "TRADE_EXCHANGE", "NFO"),
            quantity=qty,
            transaction_type=side,
        )
        if not ok:
            logger.warning("Exit legs setup failed; proceeding but watch risk.")

        atr_series = compute_atr_df(df, period=int(getattr(Config, "ATR_PERIOD", 14)), method="rma")
        atr_val = float(atr_series.iloc[-1]) if (atr_series is not None and not atr_series.empty) else 0.0

        self.active[entry_id] = ActiveTrade(
            entry_id=entry_id,
            symbol=getattr(Config, "TRADE_SYMBOL", "NIFTY"),
            direction=direction,
            quantity=qty,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target=target,
            atr=atr_val,
            opened_ts=time.time(),
        )
        self._recent_keys[key] = time.time() + self.IDEMP_TTL
        logger.info("📥 Entered %s x%d @ %.2f | SL %.2f | TP %.2f", direction, qty, entry_price, stop_loss, target)

        if self.tg:
            self.tg.send_signal_alert(token=len(self.trades) + len(self.active), signal=signal, position={
                "quantity": qty
            })

    def _expire_idempotency(self) -> None:
        now = time.time()
        for k in list(self._recent_keys):
            if self._recent_keys[k] <= now:
                del self._recent_keys[k]

    # ------------------------------ trailing / oco ------------------------------ #

    def _trailing_tick(self) -> None:
        try:
            tick = self.executor.get_tick_size()
        except Exception:
            tick = float(getattr(Config, "TICK_SIZE", 0.05))

        for oid, rec in list(self.active.items()):
            try:
                # simplistic ATR-based trail: hop every N*ATR in favor
                ltp = self.executor.get_last_price(rec.symbol) or rec.entry_price
                if rec.direction == "BUY":
                    if ltp - rec.entry_price >= self.TRAIL_ATR_MULT * max(rec.atr, tick * 10):
                        new_sl = max(rec.stop_loss, _round_to_tick(rec.entry_price + tick, tick))
                        self.executor.update_trailing_stop(oid, current_price=ltp, atr=rec.atr)
                        rec.stop_loss = new_sl
                else:
                    if rec.entry_price - ltp >= self.TRAIL_ATR_MULT * max(rec.atr, tick * 10):
                        new_sl = min(rec.stop_loss, _round_to_tick(rec.entry_price - tick, tick))
                        self.executor.update_trailing_stop(oid, current_price=ltp, atr=rec.atr)
                        rec.stop_loss = new_sl
            except Exception as e:
                logger.debug("Trailing error for %s: %s", oid, e)

    def _oco_tick(self) -> None:
        try:
            fills = self.executor.sync_and_enforce_oco()  # returns [(entry_id, fill_price)]
        except Exception:
            fills = []
        if not fills:
            return
        for oid, px in fills:
            rec = self.active.pop(oid, None)
            if not rec:
                continue
            rec.status = "CLOSED"
            rec.exit_price = float(px or rec.target)
            rec.exit_reason = "tp_or_sl"
            self._on_close(rec)

    # ------------------------------ accounting ------------------------------ #

    def _on_close(self, rec: ActiveTrade) -> None:
        pnl = (rec.exit_price - rec.entry_price) if rec.direction == "BUY" else (rec.entry_price - rec.exit_price)
        pnl_points = float(pnl or 0.0)
        # points × lot size × contracts per lot (here quantity already contracts)
        gross = pnl_points * rec.quantity
        net = gross - (self.FEES_PER_LOT * max(1, rec.quantity // self.LOT_SIZE))
        self.daily_pnl += net
        r_value = (pnl_points / max(1e-6, abs(rec.entry_price - rec.stop_loss)))
        self.session_R += r_value
        self.trades_closed_today += 1

        if net < 0:
            self.loss_streak += 1
            if self.loss_streak >= self.LOSS_LIMIT:
                self._cooldown_until = time.time() + (self.LOSS_COOLDOWN_MIN * 60)
                if self.tg:
                    self.tg.send_message(f"⚠️ Loss streak {self.loss_streak}. Cooldown for {self.LOSS_COOLDOWN_MIN} min.")
        else:
            self.loss_streak = 0

        self._log_trade(rec, net, r_value)
        if self.tg:
            self.tg.send_message(
                f"✅ Closed {rec.direction} @ {rec.exit_price:.2f} | PnL: {net:.2f} | dayPnL: {self.daily_pnl:.2f}"
            )

        # session ladders
        if self.session_R >= self.DAY_STOP_POS_R:
            self._halt_for_day = True
            if self.tg:
                self.tg.send_message("🛑 Day stop after positive R reached.")
        if self.session_R <= self.DAY_STOP_NEG_R:
            self._halt_for_day = True
            if self.tg:
                self.tg.send_message("🛑 Day stop after negative R reached.")

    # ------------------------------ persistence & logs ------------------------------ #

    def _prepare_trade_log(self) -> None:
        os.makedirs(os.path.dirname(self.LOG_FILE), exist_ok=True)
        if not os.path.exists(self.LOG_FILE):
            with open(self.LOG_FILE, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["ts", "symbol", "dir", "qty", "entry", "exit", "pnl_net", "r", "reason"])

    def _log_trade(self, rec: ActiveTrade, pnl_net: float, r_value: float) -> None:
        with open(self.LOG_FILE, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                datetime.now().isoformat(timespec="seconds"),
                rec.symbol, rec.direction, rec.quantity,
                f"{rec.entry_price:.2f}", f"{rec.exit_price:.2f}",
                f"{pnl_net:.2f}", f"{r_value:.2f}", rec.exit_reason or ""
            ])

    def _maybe_restore_state(self) -> None:
        try:
            os.makedirs(self.STATE_DIR, exist_ok=True)
            if not os.path.exists(self.ACTIVE_JSON):
                return
            with open(self.ACTIVE_JSON, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
            for oid, d in data.items():
                self.active[oid] = ActiveTrade(**d)
            if self.active:
                logger.info("Restored %d active trades from state.", len(self.active))
        except Exception as e:
            logger.debug("State restore failed: %s", e)

    def _persist_state(self) -> None:
        try:
            os.makedirs(self.STATE_DIR, exist_ok=True)
            payload = {oid: asdict(rec) for oid, rec in self.active.items()}
            with open(self.ACTIVE_JSON, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            logger.debug("State persist failed: %s", e)

    def shutdown(self) -> None:
        try:
            self._persist_state()
        except Exception:
            pass
        try:
            if self.tg:
                self.tg.stop_polling()
        except Exception:
            pass
        self._trailing_evt.set()
        self._oco_evt.set()
        logger.info("👋 Trader shutdown complete.")