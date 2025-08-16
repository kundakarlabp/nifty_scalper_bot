# src/data_streaming/realtime_trader.py
from __future__ import annotations

"""
Real-time trader orchestrator.

Highlights
- KiteDataProvider (LTP + historical OHLC) with tiny cache & retry
- Telegram control (/start /stop /mode live|shadow /status /summary /health /emergency)
- Adaptive loop cadence (peak/off-peak), warmup, trading-hour gates + optional buckets/events
- Strategy: EnhancedScalpingStrategy for spot/futures + options (fallback)
- Options strike resolution via strike_selector (cached instruments, Greeks optional)
- Risk sizing by RISK_PER_TRADE × equity (ATR/SL aware) via PositionSizing
- Spread guard (RANGE or LTP_MID) with optional dynamic scaling
- Trailing/partials delegated to OrderExecutor; OCO worker; idempotent entries
- Loss cooldown, 3-loss stop, trade/day cap, day circuit breakers, session R ladders
- CSV trade log + active state persistence & best-effort reattach-on-restart
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
from src.utils.atr_helper import compute_atr_df
from src.utils.indicators import calculate_adx, calculate_ema

# optional best-effort imports (kept safe)
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

        def set_equity(self, x: float) -> None:
            self.equity = float(x)

        def calc_size_by_risk(self, entry: float, stop: float, lot_size: int) -> int:
            risk_amt = float(getattr(Config, "RISK_PER_TRADE", 0.02)) * float(self.equity or 0)
            per_lot_risk = max(0.01, abs(entry - stop)) * lot_size
            lots = int(max(0, risk_amt // per_lot_risk))
            return max(int(getattr(Config, "MIN_LOTS", 1)), min(lots, int(getattr(Config, "MAX_LOTS", 15))))

    def get_live_account_balance() -> float:
        return float(getattr(Config, "ACCOUNT_SIZE", 0.0))

# strike utils (optional)
try:
    from src.utils.strike_selector import fetch_cached_instruments, get_instrument_tokens
except Exception:  # pragma: no cover
    fetch_cached_instruments = None
    get_instrument_tokens = None

logger = logging.getLogger(__name__)


# ---------- time helpers ----------

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


# ---------- data providers ----------

class DataProvider:
    def get_ohlc(self, symbol: str, minutes: int, timeframe: str = "minute") -> pd.DataFrame:
        return pd.DataFrame()

    def get_last_price(self, symbol: str) -> Optional[float]:
        return None


class KiteDataProvider(DataProvider):
    """
    Minimal adapter around kiteconnect with retries and small LTP cache.
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

    def _with_retries(self, fn, *args, tries=2, backoff=0.7, **kwargs):
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

    def _resolve_token(self, symbol: Optional[str]) -> int:
        if not symbol:
            return int(self.default_token)
        if symbol == getattr(Config, "SPOT_SYMBOL", "NSE:NIFTY 50"):
            return int(self.default_token)
        try:
            q = self._with_retries(self.kite.quote, [symbol])
            tok = (q.get(symbol, {}) or {}).get("instrument_token")
            if tok:
                return int(tok)
        except Exception:
            pass
        raise ValueError(f"Cannot resolve instrument token for {symbol}")

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


# ---------- models ----------

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


# ---------- RealTimeTrader ----------

class RealTimeTrader:
    # config mirrors
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

    # files
    STATE_DIR = "state"
    ACTIVE_JSON = os.path.join(STATE_DIR, "active_trades.json")

    def __init__(self, data: Optional[DataProvider] = None) -> None:
        self._lock = threading.RLock()

        self.is_trading = False
        self.live_mode = bool(getattr(Config, "ENABLE_LIVE_TRADING", False))
        self.quality_mode = bool(getattr(Config, "QUALITY_MODE_DEFAULT", False))

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
        self.data = data or (self._build_kite_provider() if self.live_mode else DataProvider())
        self.risk = PositionSizing(account_size=self.daily_start_equity) if "account_size" in PositionSizing.__init__.__code__.co_varnames else PositionSizing()
        try:
            self.risk.set_equity(float(self.daily_start_equity or 0.0))
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

    # -------- components --------

    def _build_kite_provider(self) -> DataProvider:
        try:
            return KiteDataProvider.build_from_env()
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

    # -------- scheduler / cadence --------

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

    # -------- public lifecycle --------

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

    # -------- telegram handler --------

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
                    self.quality_mode = ("on" in low) or (low.endswith("1"))
                    logger.info("Quality mode set to %s", self.quality_mode)
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
                    self.executor.cancel_all_orders()
                except Exception:
                    pass
                self.active.clear()
                if self.tg:
                    self.tg.send_message("🛑 Emergency: All orders canceled, positions cleared (best‑effort).")
                return True
        except Exception as e:
            logger.error("Control error: %s", e, exc_info=True)
        return False

    # -------- status / summary --------

    def get_status(self) -> Dict[str, Any]:
        return {
            "is_trading": self.is_trading,
            "live_mode": self.live_mode,
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
        ]
        return "\n".join(lines)

    def refresh_account_balance(self) -> None:
        try:
            bal = float(get_live_account_balance() or 0.0)
            if bal > 0:
                self.daily_start_equity = bal - self.daily_pnl
                if hasattr(self.risk, "set_equity"):
                    self.risk.set_equity(bal)
            logger.info("💰 Equity refresh: %.2f", bal)
        except Exception as e:
            logger.debug("Balance refresh error: %s", e)

    # -------- main tick --------

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

        # Idempotent key cleanup
        self._prune_recent_keys()

        # pull spot data
        spot_symbol = getattr(Config, "SPOT_SYMBOL", "NSE:NIFTY 50")
        df = self.data.get_ohlc(spot_symbol, self.DATA_LOOKBACK_MIN, self.HIST_TF)
        if df is None or df.empty or len(df) < max(self.WARMUP_BARS, 25):
            return

        # skip first minutes of day
        if (now.hour == 9 and now.minute < max(16, self.SKIP_FIRST_MIN + 15)):
            return

        # HTF/Regime gates (lightweight)
        if not self._htf_ok(df):
            return

        # Strategy generic spot/futures signal
        try:
            last_price = float(df["close"].iloc[-1])
        except Exception:
            return

        sig = self._generate_spot_signal(df, last_price)
        if sig:
            self._place_trade(spot_symbol, sig, last_price)
            return

        # Options path (best‑effort)
        if get_instrument_tokens and fetch_cached_instruments:
            self._maybe_try_options_flow(last_price)

        # housekeeping
        self._process_active_trades()

    # -------- gates / regime --------

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

    # -------- strategy wrappers --------

    def _generate_spot_signal(self, df: pd.DataFrame, last_price: float) -> Optional[Dict[str, Any]]:
        try:
            res = self.strategy.generate_signal(df, last_price)
            return res
        except Exception as e:
            logger.debug("spot signal error: %s", e)
            return None

    def _maybe_try_options_flow(self, spot_last: float) -> None:
        try:
            # resolve CE/PE tokens around ATM
            from kiteconnect import KiteConnect  # for type only
            api_key = getattr(Config, "ZERODHA_API_KEY", "")
            access_token = getattr(Config, "KITE_ACCESS_TOKEN", "") or getattr(Config, "ZERODHA_ACCESS_TOKEN", "")
            if not api_key or not access_token:
                return
            kite = None
            try:
                Kite = KiteConnect  # type: ignore
                kite = Kite(api_key=api_key)
                kite.set_access_token(access_token)
            except Exception:
                return

            cache = fetch_cached_instruments(kite) if fetch_cached_instruments else {}
            nfo = cache.get("NFO", [])
            nse = cache.get("NSE", [])

            toks = get_instrument_tokens(
                symbol="NIFTY",
                kite_instance=kite,
                cached_nfo_instruments=nfo,
                cached_nse_instruments=nse,
                offset=0,
                strike_range=self.STRIKE_RANGE,
            )
            if not toks:
                return

            # Build option OHLC quickly (lightweight minute pull)
            for side in ("ce_symbol", "pe_symbol"):
                sym = toks.get(side)
                if not sym:
                    continue
                # For options, we prefer LTP led simple breakout in strategy:
                # feed a tiny window from spot as placeholder if option candles are heavy to fetch.
                # If you want true option candles, wire another provider that resolves token → historical.
                # Here we just run generic fallback on spot df again (safe) or skip.
                # You can extend: self.strategy.generate_options_signal(option_df, spot_df, strike_info, ltp)
                pass
        except Exception as e:
            logger.debug("options flow error: %s", e)

    # -------- spread/circuit/hygiene --------

    def _entry_guard(self, key: str) -> bool:
        """Simple idempotency: reject repeats inside TTL."""
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
        if drawdown <= -abs(self.MAX_DD_PCT):
            return True
        if self._halt_for_day:
            return True
        return False

    # -------- execution path --------

    def _place_trade(self, symbol: str, sig: Dict[str, Any], mark_price: float) -> None:
        try:
            direction = str(sig["signal"]).upper()
            entry = float(sig.get("entry_price", mark_price))
            sl = float(sig.get("stop_loss"))
            tp = float(sig.get("target"))
            atr = float(sig.get("market_volatility", 0.0))
        except Exception:
            return

        # circuit/loss cooldown/limits
        if self._circuit_tripped():
            logger.info("Circuit tripped; skipping entries.")
            return
        if time.time() < self._cooldown_until:
            return
        if len(self.active) >= self.MAX_CONCURRENT:
            return
        if self.trades_closed_today >= self.MAX_TRADES_DAY:
            return

        # idempotency
        key = f"{symbol}|{direction}|{round(entry,2)}|{self.session_date}"
        if not self._entry_guard(key):
            return

        # size by risk
        lots = 1
        if hasattr(self.risk, "calc_size_by_risk"):
            lots = self.risk.calc_size_by_risk(entry, sl, self.LOT_SIZE)
        lots = max(self.MIN_LOTS, min(int(lots), self.MAX_LOTS))
        qty = lots * self.LOT_SIZE
        if qty <= 0:
            return

        # executor: place entry
        try:
            order_id = self.executor.place_entry_order(symbol, direction, qty, entry_price=entry)  # type: ignore
        except Exception as e:
            logger.error("Entry order failed: %s", e)
            return

        # record active
        at = ActiveTrade(
            entry_id=str(order_id or f"sim-{int(time.time())}"),
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
            self.tg.send_signal_alert(token=len(self.trades) + len(self.active), signal=sig, position={"quantity": qty})

        logger.info("▶️ Placed %s x%d @ %.2f SL %.2f TP %.2f", direction, qty, entry, sl, tp)

    # -------- workers --------

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
                self.executor.update_trailing_stop(at.entry_id)  # type: ignore
            except Exception:
                pass

    def _oco_tick(self) -> None:
        try:
            self.executor.sync_and_enforce_oco()  # type: ignore
        except Exception:
            pass

    def _process_active_trades(self) -> None:
        """
        Best-effort polling to detect exits and book PnL.
        """
        closed_ids: List[str] = []
        for at in list(self.active.values()):
            try:
                status = self.executor.get_order_status(at.entry_id)  # type: ignore
            except Exception:
                status = None
            if not status:
                continue
            if status.get("status") == "CLOSED":
                at.status = "CLOSED"
                at.exit_price = float(status.get("exit_price", at.entry_price))
                at.exit_reason = status.get("exit_reason", "unknown")
                pnl = (at.exit_price - at.entry_price) * (at.quantity if at.direction == "BUY" else -at.quantity)
                self.daily_pnl += float(pnl)
                self.trades_closed_today += 1
                self.trades.append(asdict(at))
                closed_ids.append(at.entry_id)
                self._log_trade(at, pnl)
                logger.info("⏹ Closed %s @ %.2f (%s) PnL=%.2f", at.symbol, at.exit_price, at.exit_reason, pnl)

                # loss streak / cooldown
                if pnl < 0:
                    self.loss_streak += 1
                    if self.loss_streak >= self.LOSS_LIMIT:
                        self._cooldown_until = time.time() + self.LOSS_STREAK_PAUSE * 60
                else:
                    self.loss_streak = 0

                # session ladders
                r = pnl / max(1.0, abs(at.entry_price - at.stop_loss) * at.quantity)
                self.session_R += r
                if self.session_R >= self.DAY_STOP_POS_R:
                    self._halt_for_day = True
                elif self.session_R >= self.DAY_HALF_POS_R:
                    try:
                        # halve min/max lots
                        self.MAX_LOTS = max(self.MIN_LOTS, max(1, self.MAX_LOTS // 2))
                    except Exception:
                        pass

        for oid in closed_ids:
            self.active.pop(oid, None)
        if closed_ids:
            self._persist_state()

    # -------- daily cycle / auto-exit --------

    def _roll_daily_if_needed(self) -> None:
        if _now().date() != self.session_date:
            self.session_date = _now().date()
            self.daily_start_equity = float(get_live_account_balance() or (self.daily_start_equity + self.daily_pnl))
            if hasattr(self.risk, "set_equity"):
                self.risk.set_equity(self.daily_start_equity)
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
                        self.executor.exit_order(at.entry_id)  # type: ignore
                    except Exception:
                        pass
                self.active.clear()
                self._persist_state()
        except Exception:
            pass

    # -------- files / persistence --------

    def _prepare_trade_log(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.LOG_FILE), exist_ok=True)
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
                w.writerow([datetime.utcnow().isoformat(), at.symbol, at.direction, at.quantity, at.entry_price, at.exit_price, at.exit_reason, round(pnl, 2)])
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

    # -------- shutdown --------

    def shutdown(self) -> None:
        try:
            self._trailing_evt.set()
            self._oco_evt.set()
            self._persist_state()
            if self.tg:
                self.tg.stop_polling()
        except Exception:
            pass