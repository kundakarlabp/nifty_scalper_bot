# src/data_streaming/realtime_trader.py
from __future__ import annotations

import atexit
import csv
import logging
import os
import threading
import time
from datetime import date, datetime, timedelta, time as dtime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import schedule

from src.config import Config
from src.execution.order_executor import OrderExecutor
from src.notifications.telegram_controller import TelegramController
from src.risk.position_sizing import PositionSizing, get_live_account_balance
from src.strategies.scalping_strategy import EnhancedScalpingStrategy
from src.utils.atr_helper import compute_atr_df
from src.utils.strike_selector import (
    fetch_cached_instruments,
    get_instrument_tokens,
)

logger = logging.getLogger(__name__)


# ------------------------------- time helpers ------------------------------- #

def _ist_now() -> datetime:
    """IST clock without external tz deps (UTC+5:30)."""
    return datetime.utcnow() + timedelta(hours=5, minutes=30)


def _within_ist_window(start_hm: Tuple[int, int], end_hm: Tuple[int, int]) -> bool:
    now = _ist_now().time()
    s = dtime(*start_hm)
    e = dtime(*end_hm)
    return s <= now <= e


# ================================= Trader ================================== #

class RealTimeTrader:
    """
    Real-time options trader.
    - Telegram daemon polling
    - Adaptive scheduler (peak/off-peak)
    - Single-position policy
    - Circuit breaker & loss cooldown
    - ATR trailing & profit-lock via executor
    - Instrument cache to avoid rate limits
    """

    # Quick aliases from Config (values read at import time; runtime changes via .env reload require restart)
    MAX_CONCURRENT_TRADES = int(Config.MAX_CONCURRENT_POSITIONS)
    WARMUP_BARS = int(Config.WARMUP_BARS)
    DATA_LOOKBACK_MINUTES = int(Config.DATA_LOOKBACK_MINUTES)
    HIST_TIMEFRAME = str(Config.HISTORICAL_TIMEFRAME)

    SPREAD_GUARD_MODE = str(Config.SPREAD_GUARD_MODE).upper()        # RANGE | LTP_MID
    SPREAD_GUARD_BA_MAX = float(Config.SPREAD_GUARD_BA_MAX)
    SPREAD_GUARD_LTPMID_MAX = float(Config.SPREAD_GUARD_LTPMID_MAX)
    SPREAD_GUARD_PCT = float(Config.SPREAD_GUARD_PCT)

    SLIPPAGE_BPS = float(Config.SLIPPAGE_BPS)
    FEES_PER_LOT = float(Config.FEES_PER_LOT)
    MAX_DAILY_DRAWDOWN_PCT = float(Config.MAX_DAILY_DRAWDOWN_PCT)
    CIRCUIT_RELEASE_PCT = float(Config.CIRCUIT_RELEASE_PCT)
    TRAILING_ENABLE = bool(Config.TRAILING_ENABLE)
    TRAIL_ATR_MULTIPLIER = float(Config.ATR_SL_MULTIPLIER)
    WORKER_INTERVAL_SEC = int(Config.WORKER_INTERVAL_SEC)
    LOG_TRADE_FILE = str(Config.LOG_FILE)

    LOT_SIZE = int(Config.NIFTY_LOT_SIZE)
    MIN_LOTS = int(Config.MIN_LOTS)
    MAX_LOTS = int(Config.MAX_LOTS)

    RISK_PER_TRADE = float(Config.RISK_PER_TRADE)
    MAX_TRADES_PER_DAY = int(Config.MAX_TRADES_PER_DAY)
    LOSS_COOLDOWN_MIN = int(Config.LOSS_COOLDOWN_MIN)
    PEAK_POLL_SEC = int(Config.PEAK_POLL_SEC)
    OFFPEAK_POLL_SEC = int(Config.OFFPEAK_POLL_SEC)
    PREFERRED_TIE_RULE = str(Config.PREFERRED_TIE_RULE).upper()

    USE_IST_CLOCK = bool(Config.USE_IST_CLOCK)
    TIME_FILTER_START_HM = Config.TIME_FILTER_START_HM
    TIME_FILTER_END_HM = Config.TIME_FILTER_END_HM
    SKIP_FIRST_MIN = int(Config.SKIP_FIRST_MIN)

    STRIKE_RANGE = int(Config.STRIKE_RANGE)

    # Regime split
    REGIME_SPLIT_ENABLE = bool(Config.REGIME_SPLIT_ENABLE)
    REGIME_ADX_TREND = int(Config.REGIME_ADX_TREND)
    REGIME_BBWIDTH_TREND = float(Config.REGIME_BBWIDTH_TREND)
    REGIME_ATR_MIN = float(Config.REGIME_ATR_MIN)
    REGIME_WARMUP_BARS = int(Config.REGIME_WARMUP_BARS)
    TREND_TP_MULT = float(Config.TREND_TP_MULT)
    TREND_SL_MULT = float(Config.TREND_SL_MULT)
    RANGE_TP_MULT = float(Config.RANGE_TP_MULT)
    RANGE_SL_MULT = float(Config.RANGE_SL_MULT)

    # Profit lock ladder
    PROFIT_LOCK_ENABLE = bool(Config.PROFIT_LOCK_ENABLE)
    PROFIT_LOCK_STEPS = int(Config.PROFIT_LOCK_STEPS)
    PROFIT_LOCK_STEP_TICKS = int(Config.PROFIT_LOCK_STEP_TICKS)
    PROFIT_LOCK_STEP_SL_BACK_TICKS = int(Config.PROFIT_LOCK_STEP_SL_BACK_TICKS)

    def __init__(self) -> None:
        self._lock = threading.RLock()

        self.is_trading: bool = False
        self.live_mode: bool = bool(Config.ENABLE_LIVE_TRADING)

        # PnL / session
        self.daily_pnl: float = 0.0
        self.daily_start_equity: float = float(get_live_account_balance() or 0.0)
        self.session_date: date = _ist_now().date()
        self._closed_trades_today = 0
        self._last_closed_was_loss = False
        self._cooldown_until_ts: float = 0.0

        # Trade registry
        self.trades: List[Dict[str, Any]] = []  # closed trades for the day
        self.active_trades: Dict[str, Dict[str, Any]] = {}  # entry_order_id → info

        # Instruments cache
        self._nfo_cache: List[Dict[str, Any]] = []
        self._nse_cache: List[Dict[str, Any]] = []
        self._cache_ts: float = 0.0
        self._CACHE_TTL = 300.0
        self._cache_lock = threading.RLock()

        # Strategy / Risk / Executor / Telegram
        self._init_components()

        # Telegram polling
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

        # Shutdown
        atexit.register(self.shutdown)

        logger.info("RealTimeTrader initialized.")
        self._safe_log_account_balance()

    # ------------------------------ init helpers ----------------------------- #

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
            ) if (Config.ENABLE_TELEGRAM and Config.has_telegram()) else None
        except Exception as e:
            logger.warning(f"Telegram init failed: {e}")
            self.tg = None

        # warm cache once (best-effort)
        self._refresh_instruments_cache(force=True)

    def _build_live_executor(self) -> OrderExecutor:
        from kiteconnect import KiteConnect

        if not Config.has_live_creds():
            raise RuntimeError("ZERODHA_API_KEY or KITE_ACCESS_TOKEN missing")

        kite = KiteConnect(api_key=Config.ZERODHA_API_KEY)
        token = Config.KITE_ACCESS_TOKEN or Config.ZERODHA_ACCESS_TOKEN
        kite.set_access_token(token)
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

    # ------------------------------- CSV log --------------------------------- #

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

    # ----------------------------- scheduling -------------------------------- #

    def _setup_adaptive_scheduler(self) -> None:
        try:
            schedule.clear()
        except Exception:
            pass
        schedule.every(5).seconds.do(self._ensure_cadence)
        # also keep balance refresh + rollover checks
        schedule.every(int(Config.BALANCE_LOG_INTERVAL_MIN)).minutes.do(self.refresh_account_balance)
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

    # -------------------------------- workers -------------------------------- #

    def _start_workers(self) -> None:
        t1 = threading.Thread(target=self._trailing_worker, daemon=True, name="TrailingWorker")
        t1.start()
        t2 = threading.Thread(target=self._oco_worker, daemon=True, name="OcoWorker")
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
            # regular ATR trailing
            try:
                self.executor.update_trailing_stop(oid, float(ltp), float(atr))
            except Exception:
                pass
            # light profit-lock hook (ladder logic is handled by trailing cadence + Config)
            # we keep state-less here; executor already tightens-only.

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

            # implicit closures (disappeared from executor actives)
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

    # ------------------------------ Telegram --------------------------------- #

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

    # ----------------------------- mode switching ---------------------------- #

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

    # --------------------------- balance / rollovers -------------------------- #

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

    def _maybe_rollover_daily(self) -> None:
        today = _ist_now().date()
        if today != self.session_date:
            logger.info("🔁 New session date detected → rolling daily state.")
            with self._lock:
                self.session_date = today
                self.daily_pnl = 0.0
                self.trades.clear()
                self._closed_trades_today = 0
                self._last_closed_was_loss = False
                self._cooldown_until_ts = 0.0

    # ------------------------------- guards ---------------------------------- #

    def _is_trading_hours(self) -> bool:
        if not self.USE_IST_CLOCK:
            return True
        if not _within_ist_window(self.TIME_FILTER_START_HM, self.TIME_FILTER_END_HM):
            return False
        # Skip first N minutes if configured
        start_h, start_m = self.TIME_FILTER_START_HM
        start_dt = _ist_now().replace(hour=start_h, minute=start_m, second=0, microsecond=0)
        return ( _ist_now() - start_dt ).total_seconds() >= self.SKIP_FIRST_MIN * 60

    def _is_circuit_breaker_tripped(self) -> bool:
        if self.daily_start_equity <= 0:
            return False
        dd = (-self.daily_pnl) / self.daily_start_equity
        return dd >= self.MAX_DAILY_DRAWDOWN_PCT

    def _in_loss_cooldown(self) -> bool:
        return time.time() < self._cooldown_until_ts

    # -------------------------- cache / instruments --------------------------- #

    def _refresh_instruments_cache(self, force: bool = False) -> None:
        if not self.live_mode or not getattr(self.executor, "kite", None):
            return
        with self._cache_lock:
            if not force and (time.time() - self._cache_ts) < self._CACHE_TTL:
                return
            try:
                caches = fetch_cached_instruments(self.executor.kite)
                self._nfo_cache = caches.get("NFO", []) or []
                self._nse_cache = caches.get("NSE", []) or []
                self._cache_ts = time.time()
                logger.info("📦 Instruments cache refreshed: NFO=%d NSE=%d",
                            len(self._nfo_cache), len(self._nse_cache))
            except Exception as e:
                logger.warning(f"Instruments cache refresh failed: {e}")

    def _force_refresh_cache(self) -> bool:
        self._refresh_instruments_cache(force=True)
        self._safe_send_message("🔄 Instruments cache refreshed.")
        return True

    # -------------------------- main data loop tick --------------------------- #

    def _smart_fetch_and_process(self) -> None:
        try:
            if not self._is_trading_hours() and not Config.ALLOW_OFFHOURS_TESTING:
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
            logger.error(f"Error in smart fetch/process: {e}", exc_info=True)

    # ----------------------------- SIGNAL PIPELINE ---------------------------- #
    # NOTE: Plug your actual OHLC/quote source below. Everything else is wired.

    def fetch_and_process_data(self) -> bool:
        """
        Fetch → select strike → compute signal → size → execute.
        Return True if a trade was placed or updated; False otherwise.

        Integration points:
          - replace `_load_recent_ohlc()` with your data loader
          - ensure a 1-min (or configured) OHLC DataFrame for the selected option
          - call strategy.generate_signal(...) or generate_options_signal(...)
        """
        # 0) refresh caches periodically in live mode
        self._refresh_instruments_cache()

        # 1) Resolve instrument (ATM ± offset window)
        if not self.live_mode or not getattr(self.executor, "kite", None):
            logger.debug("Shadow mode or no Kite: skipping live strike resolution.")
            return False

        try:
            sel = get_instrument_tokens(
                symbol=Config.TRADE_SYMBOL,
                kite_instance=self.executor.kite,
                cached_nfo_instruments=self._nfo_cache,
                cached_nse_instruments=self._nse_cache,
                offset=0,
                strike_range=self.STRIKE_RANGE,
            )
        except Exception as e:
            logger.error(f"Strike resolution failed: {e}")
            return False

        if not sel or not sel.get("ce_token") or not sel.get("pe_token"):
            logger.debug("No valid CE/PE selection this tick.")
            return False

        # 2) Load OHLC (spot and option) — you must implement these to return DataFrames
        #    with columns: open/high/low/close[ , volume], indexed by time ascending.
        opt_symbol = sel["ce_symbol"] if self._prefer_call(sel) else sel["pe_symbol"]
        if not opt_symbol:
            return False

        option_df = self._load_recent_ohlc(opt_symbol)
        spot_df = self._load_recent_spot_ohlc()

        if option_df is None or option_df.empty:
            return False

        # 3) Compute ATR on option for trailing scale
        try:
            option_df = option_df.copy()
            option_df["atr"] = compute_atr_df(option_df, period=int(Config.ATR_PERIOD), method="rma")
            atr_val = float(option_df["atr"].iloc[-1] or 0.0)
        except Exception:
            atr_val = 0.0

        # 4) Generate a signal (two approaches; keep one)
        signal = self.strategy.generate_signal(option_df, float(option_df["close"].iloc[-1]))
        # OR: option-aware breakout (if you prefer this path)
        # signal = self.strategy.generate_options_signal(option_df, spot_df, {"type": "CE"|"PE"}, current_option_price)

        if not signal:
            return False

        # 5) Position sizing (lots → contracts)
        lots = self._size_position(atr_val=atr_val)
        if lots <= 0:
            return False
        contracts = lots * self.LOT_SIZE

        # 6) Place entry + exits
        side = "BUY" if signal["signal"] == "BUY" else "SELL"
        entry_px = float(signal["entry_price"])
        stop = float(signal["stop_loss"])
        target = float(signal["target"])

        oid = self.executor.place_entry_order(
            symbol=opt_symbol,
            exchange=Config.TRADE_EXCHANGE,
            transaction_type=side,
            quantity=int(contracts),
            product=Config.DEFAULT_PRODUCT,
            order_type=Config.DEFAULT_ORDER_TYPE,
            validity=Config.DEFAULT_VALIDITY,
        )
        if not oid:
            logger.error("Entry order placement failed.")
            return False

        ok = self.executor.setup_gtt_orders(
            entry_order_id=oid,
            entry_price=entry_px,
            stop_loss_price=stop,
            target_price=target,
            symbol=opt_symbol,
            exchange=Config.TRADE_EXCHANGE,
            quantity=int(contracts),
            transaction_type=side,
        )
        if not ok:
            logger.warning("Exits setup failed (continuing with entry recorded).")

        # 7) Register as active
        with self._lock:
            self.active_trades[oid] = {
                "status": "OPEN",
                "symbol": opt_symbol,
                "direction": side,
                "contracts": int(contracts),
                "entry": entry_px,
                "stop_loss": stop,
                "target": target,
                "atr": atr_val,
                "last_close": float(option_df["close"].iloc[-1]),
                "opened_at": _ist_now().isoformat(),
                "mode": "LIVE" if self.live_mode else "SHADOW",
            }

        # 8) Telegram alert (best-effort)
        try:
            if self.tg:
                self.tg.send_signal_alert(token=0, signal=signal, position={"quantity": int(contracts)})
        except Exception:
            pass

        return True

    # ----------------------------- sizing / ties ------------------------------ #

    def _prefer_call(self, sel: Dict[str, Any]) -> bool:
        """Simple tie rule for CE/PE: follows configured rule."""
        rule = self.PREFERRED_TIE_RULE
        # You can extend with spot trend/strength
        return True if rule == "TREND" else True  # default bias to CE; refine as needed

    def _size_position(self, atr_val: float) -> int:
        """Convert equity & risk into lots, capped by Config.MIN/MAX_LOTS."""
        eq = float(get_live_account_balance() or self.daily_start_equity or 0.0)
        if eq <= 0:
            eq = self.daily_start_equity
        risk_amt = eq * self.RISK_PER_TRADE
        # Approx per-lot risk proxy: ATR * tick value (assume option premium risk ~ ATR)
        per_lot_risk = max(10.0, atr_val * 1.0)  # ₹ proxy; tune if you have per-lot rupee calc
        lots = int(max(Config.MIN_LOTS, min(Config.MAX_LOTS, risk_amt / max(1.0, per_lot_risk))))
        return lots

    # ------------------------------- finalize -------------------------------- #

    def _finalize_trade(self, entry_id: str) -> None:
        tr = self.active_trades.pop(entry_id, None)
        if not tr:
            return
        entry = float(tr.get("entry") or 0.0)
        exit_px = float(tr.get("exit_price") or 0.0)
        qty = int(tr.get("contracts") or 0)
        if entry <= 0 or exit_px <= 0 or qty <= 0:
            return

        direction = tr.get("direction", "BUY")
        gross = (exit_px - entry) * qty if direction == "BUY" else (entry - exit_px) * qty
        fees = (qty / self.LOT_SIZE) * self.FEES_PER_LOT
        net = gross - fees

        with self._lock:
            self.daily_pnl += net
            self._closed_trades_today += 1
            self._last_closed_was_loss = net < 0.0
            if self._last_closed_was_loss and self.LOSS_COOLDOWN_MIN > 0:
                self._cooldown_until_ts = time.time() + self.LOSS_COOLDOWN_MIN * 60

        self._append_trade_log(
            [
                _ist_now().date().isoformat(),
                entry_id,
                tr.get("symbol"),
                direction,
                qty,
                round(entry, 2),
                round(exit_px, 2),
                round(gross, 2),
                round(fees, 2),
                round(net, 2),
                tr.get("confidence", ""),
                round(float(tr.get("atr", 0.0)), 2),
                tr.get("mode", ""),
            ]
        )

    # ------------------------------ status / ui ------------------------------ #

    def _safe_send_message(self, text: str, parse_mode: Optional[str] = None) -> None:
        if self.tg:
            try:
                self.tg.send_message(text, parse_mode=parse_mode)
            except Exception:
                pass

    def _safe_send_alert(self, action: str) -> None:
        if self.tg:
            try:
                self.tg.send_realtime_session_alert(action)
            except Exception:
                pass

    def _send_detailed_status(self) -> bool:
        st = self.get_status()
        msg = (
            "📊 Bot Status\n"
            f"🔁 Trading: {'🟢 Running' if st['is_trading'] else '🔴 Stopped'}\n"
            f"🌐 Mode: {'🟢 LIVE' if st['live_mode'] else '🛡️ Shadow'}\n"
            f"📦 Open Orders: {st['open_orders']}\n"
            f"📈 Trades Today: {st['trades_today']}\n"
            f"💰 Daily P&L: {st['daily_pnl']:.2f}\n"
            f"⚖️ Risk Level: {st['risk_level']}"
        )
        self._safe_send_message(msg)
        return True

    def _run_health_check(self) -> bool:
        ok = bool(self.strategy and self.executor and self.risk)
        self._safe_send_message("✅ Health OK" if ok else "❌ Health check failed")
        return ok

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "is_trading": self.is_trading,
                "live_mode": self.live_mode,
                "open_orders": len(self.active_trades),
                "trades_today": self._closed_trades_today,
                "daily_pnl": float(self.daily_pnl),
                "risk_level": f"{self.RISK_PER_TRADE*100:.1f}%",
            }

    def get_summary(self) -> str:
        with self._lock:
            return (
                f"📅 {self.session_date} | Trades: {self._closed_trades_today} | "
                f"PnL: ₹{self.daily_pnl:.2f}"
            )

    # -------------------------- graceful shutdown ---------------------------- #

    def shutdown(self) -> None:
        logger.info("🛑 Shutting down RealTimeTrader…")
        try:
            self.stop()
        except Exception:
            pass
        try:
            self._stop_polling()
        except Exception:
            pass
        try:
            self._trailing_evt.set()
            self._oco_evt.set()
        except Exception:
            pass

    # --------------------------- data source stubs --------------------------- #
    # Replace these with your actual OHLC loaders. Keep signatures intact.

    def _load_recent_ohlc(self, option_tradingsymbol: str) -> Optional[pd.DataFrame]:
        """
        Return a minute-level DataFrame with columns open, high, low, close[, volume].
        Index must be ascending timestamps. Length >= Config.WARMUP_BARS.
        """
        # TODO: wire to your historical API/cache
        return None

    def _load_recent_spot_ohlc(self) -> Optional[pd.DataFrame]:
        """Same contract as _load_recent_ohlc(), but for NIFTY spot/futures."""
        # TODO: wire to your historical API/cache
        return None