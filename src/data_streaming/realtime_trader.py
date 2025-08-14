# src/data_streaming/realtime_trader.py
"""
Real-time trader with:
- Telegram control (daemon polling)
- Adaptive main loop (peak/off-peak cadence)
- Risk-based lot sizing
- Warmup & trading-hours filters (IST aware)
- Options strike resolution (ATM ± range via cached instruments)
- Strategy signals (spot/option aware)
- Spread guard (RANGE or LTP_MID)
- Partial TP, breakeven hop, trailing SL (via OrderExecutor)
- Daily circuit breaker + loss cooldown
- CSV trade log + daily rollover

This module keeps external dependencies limited to our own utils + KiteConnect
(when live). In shadow mode it degrades gracefully (no orders placed).
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
# Optional ATR helper (if available)
try:
    from src.utils.atr_helper import compute_atr_df  # type: ignore
except Exception:  # keep module import-safe if helper absent
    compute_atr_df = None  # type: ignore

logger = logging.getLogger(__name__)


# ----------------------------- helpers ----------------------------- #

def _ist_now() -> datetime:
    """IST clock without pytz (UTC+5:30)."""
    return datetime.utcnow() + timedelta(hours=5, minutes=30)


def _between_ist(start_hm: str, end_hm: str) -> bool:
    """True if IST now is between HH:MM ranges inclusive."""
    now = _ist_now().time()
    s = datetime.strptime(start_hm, "%H:%M").time()
    e = datetime.strptime(end_hm, "%H:%M").time()
    return s <= now <= e


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
        """IST market hours filter + first N minutes skip."""
        try:
            if not self.USE_IST_CLOCK:
                return True
            now = _ist_now()
            if not _between_ist(self.TIME_FILTER_START, self.TIME_FILTER_END):
                return False
            # skip first N minutes after start
            s = datetime.strptime(self.TIME_FILTER_START, "%H:%M").time()
            start_dt = now.replace(hour=s.hour, minute=s.minute, second=0, microsecond=0)
            if (now - start_dt).total_seconds() < self.SKIP_FIRST_MIN * 60:
                return False
            return True
        except Exception:
            return True

    def _in_loss_cooldown(self) -> bool:
        return time.time() < float(self._cooldown_until_ts or 0.0)

    def _trip_loss_cooldown(self) -> None:
        self._cooldown_until_ts = time.time() + self.LOSS_COOLDOWN_MIN * 60
        self._safe_send_message(f"🧊 Cooling down {self.LOSS_COOLDOWN_MIN}m after loss.")

    def _is_circuit_breaker_tripped(self) -> bool:
        try:
            if self.daily_start_equity <= 0:
                return False
            dd = -float(self.daily_pnl) / float(self.daily_start_equity)
            return dd >= self.MAX_DAILY_DRAWDOWN_PCT
        except Exception:
            return False

    # -------------------- cache / instruments -------------------- #

    def _refresh_instruments_cache(self, force: bool = False) -> None:
        try:
            with self._cache_lock:
                now = time.time()
                if not force and (now - self._cache_ts) < self._CACHE_TTL:
                    return
                kite = getattr(self.executor, "kite", None)
                if not kite:
                    # simulation: leave empty but no crash
                    self._nfo_cache = self._nfo_cache or []
                    self._nse_cache = self._nse_cache or []
                    self._cache_ts = now
                    return
                data = fetch_cached_instruments(kite)
                self._nfo_cache = data.get("NFO", []) or []
                self._nse_cache = data.get("NSE", []) or []
                self._cache_ts = now
            logger.info("📦 Instruments cache refreshed.")
        except Exception as e:
            logger.warning(f"Instruments cache refresh failed: {e}")

    def _force_refresh_cache(self) -> bool:
        self._refresh_instruments_cache(force=True)
        return True

    # -------------------- main loop -------------------- #

    def _smart_fetch_and_process(self) -> None:
        try:
            now = _ist_now()
            self._rollover_if_needed(now.date())

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
            logger.error(f"Smart fetch/process error: {e}", exc_info=True)

    def fetch_and_process_data(self) -> None:
        """Core one-tick decision path: resolve strikes → ask strategy → size → execute."""
        # 1) Ensure instruments cache
        self._refresh_instruments_cache()

        kite = getattr(self.executor, "kite", None)
        if not kite:
            logger.debug("No live Kite instance (simulation). Proceeding with logic but no orders.")

        # 2) Resolve CE/PE tokens around ATM (offset 0 by default)
        try:
            toks = get_instrument_tokens(
                symbol=str(getattr(Config, "TRADE_SYMBOL", "NIFTY")),
                kite_instance=kite,
                cached_nfo_instruments=self._nfo_cache,
                cached_nse_instruments=self._nse_cache,
                offset=0,
                strike_range=int(self.STRIKE_RANGE),
            )
        except Exception as e:
            logger.error(f"Strike resolution failed: {e}", exc_info=True)
            toks = None

        if not toks or (not toks.get("ce_symbol") and not toks.get("pe_symbol")):
            return

        spot = float(toks.get("spot_price") or 0.0)
        atm = int(toks.get("atm_strike") or 0)

        # 3) Get ATR if helper available (optional)
        atr_val = 0.0
        try:
            if compute_atr_df and kite and toks.get("ce_symbol"):
                # Example: compute ATR on spot for context (you may switch to option LTP if desired)
                df = compute_atr_df(kite, instrument="NSE:NIFTY 50", period=14, lookback_minutes=self.DATA_LOOKBACK_MINUTES)  # type: ignore
                if isinstance(df, pd.DataFrame) and not df.empty and "ATR" in df.columns:
                    atr_val = float(df["ATR"].iloc[-1])  # last ATR on spot scale (points)
        except Exception:
            atr_val = 0.0

        # 4) Ask strategy for a signal
        signal: Optional[Dict[str, Any]] = None
        try:
            # Strategy interface should be designed to accept basic context;
            # keep this lenient to avoid runtime errors if strategy signature differs.
            context = {
                "spot": spot,
                "atm": atm,
                "time": _ist_now(),
                "atr": atr_val,
                "tokens": toks,
            }
            # Common return shape expected:
            #  {'direction': 'LONG'|'SHORT', 'score': float, 'confidence': float,
            #   'entry_price': float, 'stop_loss': float, 'target': float}
            signal = self.strategy.generate_signal(context) if self.strategy else None  # type: ignore
        except TypeError:
            # very defensive: call with no args if signature differs
            try:
                signal = self.strategy.generate_signal() if self.strategy else None  # type: ignore
            except Exception:
                signal = None
        except Exception:
            signal = None

        if not signal:
            return

        # Quick filters
        score = float(signal.get("score", 0.0))
        conf = float(signal.get("confidence", 0.0))
        if score < float(getattr(Config, "MIN_SIGNAL_SCORE", 2)) or conf < float(getattr(Config, "CONFIDENCE_THRESHOLD", 5.2)):
            return

        # 5) Choose side + tradingsymbol
        dir_raw = str(signal.get("direction", "LONG")).upper()
        is_long = dir_raw in ("LONG", "BUY", "BULL", "CALL", "CE")
        side_symbol = toks.get("ce_symbol") if is_long else toks.get("pe_symbol")
        side_token = toks.get("ce_token") if is_long else toks.get("pe_token")
        if not side_symbol or not side_token:
            return

        # 6) Prices (fallbacks)
        entry = float(signal.get("entry_price") or 0.0)
        stop = float(signal.get("stop_loss") or 0.0)
        target = float(signal.get("target") or 0.0)

        if entry <= 0 or stop <= 0 or target <= 0:
            # if strategy didn't set, infer rough prices from LTP with Config base points
            try:
                ltp_map = kite.ltp([f"NFO:{side_symbol}"]) if kite else {}
                ltp = float(ltp_map.get(f"NFO:{side_symbol}", {}).get("last_price") or 0.0)
            except Exception:
                ltp = 0.0
            entry = entry or ltp or 1.0
            base_sl_pts = float(getattr(Config, "BASE_STOP_LOSS_POINTS", 20.0))
            base_tp_pts = float(getattr(Config, "BASE_TARGET_POINTS", 40.0))
            if is_long:
                stop = stop or max(0.5, entry - base_sl_pts)
                target = target or (entry + base_tp_pts)
            else:
                stop = stop or (entry + base_sl_pts)
                target = target or max(0.5, entry - base_tp_pts)

        # 7) Position sizing (risk per trade of equity)
        lots = self._compute_lots(entry, stop)
        if lots <= 0:
            return
        lots = max(self.MIN_LOTS, min(lots, self.MAX_LOTS))
        qty = lots * self.LOT_SIZE

        # Respect max concurrent
        with self._lock:
            open_count = sum(1 for t in self.active_trades.values() if t.get("status") == "OPEN")
        if open_count >= self.MAX_CONCURRENT_TRADES:
            return

        # 8) Place entry (simulation ok)
        txn_side = "BUY" if is_long else "SELL"
        oid = self.executor.place_entry_order(
            symbol=side_symbol,
            exchange=str(getattr(Config, "TRADE_EXCHANGE", "NFO")),
            transaction_type=txn_side,
            quantity=int(qty),
        )
        if not oid:
            return

        # 9) Exits
        ok = self.executor.setup_gtt_orders(
            entry_order_id=oid,
            entry_price=float(entry),
            stop_loss_price=float(stop),
            target_price=float(target),
            symbol=side_symbol,
            exchange=str(getattr(Config, "TRADE_EXCHANGE", "NFO")),
            quantity=int(qty),
            transaction_type=txn_side,
        )
        if not ok:
            logger.warning("Exits could not be set for %s; trade may be unmanaged.", oid)

        # 10) Register active trade
        with self._lock:
            self.active_trades[oid] = {
                "status": "OPEN",
                "symbol": side_symbol,
                "direction": txn_side,
                "qty": int(qty),
                "entry_price": float(entry),
                "stop_loss": float(stop),
                "target": float(target),
                "atr": float(atr_val or 0.0),
                "score": score,
                "confidence": conf,
                "opened_at": _ist_now().isoformat(timespec="seconds"),
            }

        self._safe_send_message(
            f"🚀 Opened {txn_side} {side_symbol} x{lots} lots | "
            f"entry {entry:.2f} SL {stop:.2f} TP {target:.2f}\n"
            f"score={score:.2f} conf={conf:.2f}"
        )

    # -------------------- sizing / pnl -------------------- #

    def _compute_lots(self, entry: float, stop: float) -> int:
        """Risk per trade sizing in lots: equity * RPT / (risk_per_lot)."""
        try:
            eq = float(getattr(self.risk, "equity", 0.0) or 0.0) or float(self.daily_start_equity or 0.0)
            if eq <= 0:
                return 0
            per_trade = float(self.RISK_PER_TRADE or 0.02) * eq
            risk_pts = abs(float(entry) - float(stop))
            if risk_pts <= 0:
                return 0
            risk_per_lot = risk_pts * self.LOT_SIZE
            lots = int(per_trade // risk_per_lot)
            return max(0, lots)
        except Exception:
            return 0

    # -------------------- finalization / logging -------------------- #

    def _finalize_trade(self, entry_id: str) -> None:
        tr = self.active_trades.get(entry_id)
        if not tr:
            return

        entry = float(tr.get("entry_price") or 0.0)
        exit_px = float(tr.get("exit_price") or 0.0)
        qty = int(tr.get("qty") or 0)
        direction = tr.get("direction") or "BUY"

        if entry <= 0 or qty <= 0:
            self.active_trades.pop(entry_id, None)
            return

        # PnL (approx; net of simple fees model)
        gross = (exit_px - entry) * qty if direction == "BUY" else (entry - exit_px) * qty
        fees = (qty / max(1, self.LOT_SIZE)) * float(self.FEES_PER_LOT)
        net = gross - fees

        with self._lock:
            self.daily_pnl += float(net)
            self._closed_trades_today += 1
            self._last_closed_was_loss = net < 0
            self.active_trades.pop(entry_id, None)

        self._append_trade_log([
            _ist_now().strftime("%Y-%m-%d"),
            entry_id,
            tr.get("symbol"),
            direction,
            qty,
            round(entry, 2),
            round(exit_px, 2),
            round(gross, 2),
            round(fees, 2),
            round(net, 2),
            round(float(tr.get("confidence", 0.0)), 2),
            round(float(tr.get("atr", 0.0)), 2),
            "LIVE" if self.live_mode else "SHADOW",
        ])

        self._safe_send_message(
            f"{'✅' if net>=0 else '❌'} Closed {direction} {tr.get('symbol')} x{qty//self.LOT_SIZE} lots | "
            f"entry {entry:.2f} exit {exit_px:.2f} | net ₹{net:.0f}"
        )

        if self._last_closed_was_loss:
            self._trip_loss_cooldown()

    # -------------------- day roll / status -------------------- #

    def _rollover_if_needed(self, today: date) -> None:
        if today == self.session_date:
            return
        # new day
        self.session_date = today
        self.daily_pnl = 0.0
        self._closed_trades_today = 0
        self._cooldown_until_ts = 0.0
        self.refresh_account_balance()
        self._safe_send_message("🔄 New session started. Stats reset.")

    # -------------------- status / summary / health -------------------- #

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            open_n = sum(1 for t in self.active_trades.values() if t.get("status") == "OPEN")
            return {
                "is_trading": self.is_trading,
                "live_mode": self.live_mode,
                "open_orders": open_n,
                "trades_today": self._closed_trades_today,
                "daily_pnl": round(self.daily_pnl, 2),
                "risk_level": f"{int(self.RISK_PER_TRADE*100)}%/trade",
            }

    def _send_detailed_status(self) -> bool:
        st = self.get_status()
        msg = (
            "📊 <b>Bot Status</b>\n"
            f"🔁 <b>Trading:</b> {'🟢 Running' if st['is_trading'] else '🔴 Stopped'}\n"
            f"🌐 <b>Mode:</b> {'🟢 LIVE' if st['live_mode'] else '🛡️ Shadow'}\n"
            f"📦 <b>Open:</b> {st['open_orders']}\n"
            f"📈 <b>Trades Today:</b> {st['trades_today']}\n"
            f"💰 <b>Daily P&L:</b> {st['daily_pnl']:.2f}\n"
            f"⚖️ <b>Risk:</b> {st['risk_level']}"
        )
        self._safe_send_message(msg, parse_mode="HTML")
        return True

    def get_summary(self) -> str:
        return (
            f"🗒️ Summary\n"
            f"Date: {_ist_now().date().isoformat()}\n"
            f"Trades closed: {self._closed_trades_today}\n"
            f"Daily P&L: ₹{round(self.daily_pnl,2)}\n"
            f"Mode: {'LIVE' if self.live_mode else 'SHADOW'}"
        )

    def _run_health_check(self) -> bool:
        ok = True
        issues: List[str] = []
        if self.live_mode and not getattr(self.executor, "kite", None):
            ok = False
            issues.append("live_mode but no Kite instance")
        with self._cache_lock:
            if not self._nfo_cache:
                issues.append("NFO cache empty")
            if not self._nse_cache:
                issues.append("NSE cache empty")
        if issues:
            self._safe_send_message("⚠️ Health issues: " + "; ".join(issues))
        else:
            self._safe_send_message("✅ Health OK.")
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

    # -------------------- shutdown -------------------- #

    def shutdown(self) -> None:
        try:
            self._trailing_evt.set()
            self._oco_evt.set()
            self._stop_polling()
        except Exception:
            pass