# src/data_streaming/realtime_trader.py
"""
Real-time trader with:
- Telegram control (daemon polling)
- Scheduled main loop (fetch -> select -> signal -> size -> execute)
- ATR trailing worker (background)
- Circuit breaker (daily drawdown)
- Single-position policy
- Warmup filter
- Spread guard:
    • RANGE mode (old, candle-range proxy)
    • LTP_MID mode (new, uses bid/ask depth mid via quote())
- Slippage + fees model
- Rate limit safety (bulk quote)
- CSV trade log persistence
- Daily session rollover
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
from datetime import datetime, timedelta, date
import time
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.config import Config
from src.strategies.scalping_strategy import EnhancedScalpingStrategy
from src.risk.position_sizing import PositionSizing, get_live_account_balance
from src.execution.order_executor import OrderExecutor
from src.notifications.telegram_controller import TelegramController
from src.utils.strike_selector import (
    _get_spot_ltp_symbol,
    get_instrument_tokens,
    get_next_expiry_date,
    health_check as _health_check,
)

logger = logging.getLogger(__name__)


class RealTimeTrader:
    """
    Real-time options trader.
    - Telegram polling runs in its own daemon thread from controller
    - Data polling uses `schedule` in `run()`
    - Trailing/OCO worker ticks independently
    """

    # sensible defaults if Config misses keys
    MAX_CONCURRENT_TRADES = int(getattr(Config, "MAX_CONCURRENT_TRADES", 1))
    WARMUP_BARS = int(getattr(Config, "WARMUP_BARS", 60))

    # --- Spread guard config ---
    # Mode: "LTP_MID" (use bid/ask depth) or "RANGE" (candle-range proxy)
    SPREAD_GUARD_MODE = str(getattr(Config, "SPREAD_GUARD_MODE", "LTP_MID")).upper()
    # Max relative bid/ask spread (ask-bid)/mid
    SPREAD_GUARD_BA_MAX = float(getattr(Config, "SPREAD_GUARD_BA_MAX", 0.012))  # 1.2%
    # Max relative LTP vs MID deviation |ltp-mid|/mid
    SPREAD_GUARD_LTPMID_MAX = float(getattr(Config, "SPREAD_GUARD_LTPMID_MAX", 0.015))  # 1.5%
    # Legacy RANGE guard (kept for fallback)
    SPREAD_GUARD_PCT = float(getattr(Config, "SPREAD_GUARD_PCT", 0.02))  # 2% last-bar proxy

    SLIPPAGE_BPS = float(getattr(Config, "SLIPPAGE_BPS", 4.0))  # 4 bps per side
    FEES_PER_LOT = float(getattr(Config, "FEES_PER_LOT", 25.0))  # ₹ per lot round trip
    MAX_DAILY_DRAWDOWN_PCT = float(getattr(Config, "MAX_DAILY_DRAWDOWN_PCT", 0.03))  # 3%
    CIRCUIT_RELEASE_PCT = float(getattr(Config, "CIRCUIT_RELEASE_PCT", 0.015))  # 1.5% recovery required
    TRAILING_ENABLE = bool(getattr(Config, "TRAILING_ENABLE", True))
    TRAIL_ATR_MULTIPLIER = float(getattr(Config, "ATR_SL_MULTIPLIER", 1.5))
    WORKER_INTERVAL_SEC = int(getattr(Config, "WORKER_INTERVAL_SEC", 10))
    LOG_TRADE_FILE = getattr(Config, "LOG_FILE", "logs/trades.csv")
    HIST_TIMEFRAME = getattr(Config, "HISTORICAL_TIMEFRAME", "minute")

    def __init__(self) -> None:
        self._lock = threading.RLock()

        self.is_trading: bool = False
        self.live_mode: bool = bool(getattr(Config, "ENABLE_LIVE_TRADING", False))

        # PnL / session
        self.daily_pnl: float = 0.0
        self.daily_start_equity: float = get_live_account_balance()  # safe fallback inside
        self.session_date: date = datetime.now().date()

        # Trade registry
        self.trades: List[Dict[str, Any]] = []  # closed trades for the day
        self.active_trades: Dict[str, Dict[str, Any]] = {}  # order_id → info

        # Instrument cache (for Kite rate-limits)
        self._nfo_instruments_cache: Optional[List[Dict]] = None
        self._nse_instruments_cache: Optional[List[Dict]] = None
        self._instruments_cache_timestamp: float = 0
        self._INSTRUMENT_CACHE_DURATION: int = 300  # seconds
        self._cache_lock = threading.RLock()

        # Strategy / Risk / Executor / Telegram
        self._init_components()

        # Telegram polling thread
        self._polling_thread: Optional[threading.Thread] = None
        self._start_polling()

        # Background workers
        self._trailing_worker_stop = threading.Event()
        self._oco_worker_stop = threading.Event()
        self._start_workers()

        # Scheduling
        self._setup_smart_scheduling()

        # CSV log setup
        self._prepare_trade_log()

        # Graceful shutdown
        atexit.register(self.shutdown)

        logger.info("RealTimeTrader initialized.")
        self._safe_log_account_balance()

    # ---------- Init helpers ----------

    def _init_components(self) -> None:
        try:
            self.strategy = EnhancedScalpingStrategy(
                base_stop_loss_points=Config.BASE_STOP_LOSS_POINTS,
                base_target_points=Config.BASE_TARGET_POINTS,
                confidence_threshold=Config.CONFIDENCE_THRESHOLD,
                min_score_threshold=int(Config.MIN_SIGNAL_SCORE),
            )
        except Exception as e:
            logger.warning(f"Failed to initialize strategy: {e}")
            self.strategy = None

        try:
            self.risk_manager = PositionSizing()
        except Exception as e:
            logger.warning(f"Failed to initialize risk manager: {e}")
            self.risk_manager = PositionSizing()

        self.order_executor = self._init_order_executor()

        try:
            self.telegram_controller = TelegramController(
                status_callback=self.get_status,
                control_callback=self._handle_control,
                summary_callback=self.get_summary,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize Telegram controller: {e}")
            raise

    def _build_live_executor(self) -> OrderExecutor:
        from kiteconnect import KiteConnect

        api_key = getattr(Config, "ZERODHA_API_KEY", None)
        access_token = getattr(Config, "KITE_ACCESS_TOKEN", None) or getattr(Config, "ZERODHA_ACCESS_TOKEN", None)
        if not api_key or not access_token:
            raise RuntimeError("ZERODHA_API_KEY or KITE_ACCESS_TOKEN missing")

        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        logger.info("✅ Live executor created (KiteConnect).")
        return OrderExecutor(kite=kite)

    def _init_order_executor(self) -> OrderExecutor:
        if self.live_mode:
            try:
                return self._build_live_executor()
            except Exception as exc:
                logger.error("Live init failed, falling back to simulation: %s", exc, exc_info=True)
                self.live_mode = False
        logger.info("Live trading disabled → simulation mode.")
        return OrderExecutor()

    # ---------- CSV log ----------

    def _prepare_trade_log(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.LOG_TRADE_FILE) or ".", exist_ok=True)
            if not os.path.exists(self.LOG_TRADE_FILE):
                with open(self.LOG_TRADE_FILE, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow([
                        "date", "order_id", "symbol", "direction", "contracts",
                        "entry", "exit", "pnl", "fees", "net_pnl", "confidence",
                        "atr", "mode"
                    ])
        except Exception as e:
            logger.warning(f"Trade log init failed: {e}")

    def _append_trade_log(self, row: List[Any]) -> None:
        try:
            with open(self.LOG_TRADE_FILE, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(row)
        except Exception as e:
            logger.debug(f"Trade log append failed: {e}")

    # ---------- Scheduling / main loop ----------

    def _setup_smart_scheduling(self) -> None:
        try:
            schedule.clear()
        except Exception:
            pass

        # data loop: every 30s inside market hours
        schedule.every(30).seconds.do(self._smart_fetch_and_process)
        # balance refresh
        schedule.every(getattr(Config, "BALANCE_LOG_INTERVAL_MIN", 30)).minutes.do(
            self.refresh_account_balance
        )
        # daily rollover check
        schedule.every(60).seconds.do(self._maybe_rollover_daily)

        logger.info("Scheduled fetch/process every 30s (market hours only).")

    def run(self) -> None:
        """Main non-blocking scheduler loop. Call this in a thread or main process."""
        logger.info("🟢 RealTimeTrader.run() loop started.")
        while True:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error in run loop: {e}", exc_info=True)
                time.sleep(2)

    def _smart_fetch_and_process(self) -> None:
        try:
            now = datetime.now()
            if not self._is_trading_hours(now) and not getattr(Config, "ALLOW_OFFHOURS_TESTING", False):
                # heartbeat every ~5 minutes
                if int(time.time()) % 300 < 2:
                    logger.info("⏳ Market closed. Skipping fetch.")
                return

            if not self.is_trading:
                return

            # respect circuit breaker
            if self._is_circuit_breaker_tripped():
                logger.warning("🚫 Circuit breaker is active — trading paused.")
                return

            self.fetch_and_process_data()

        except Exception as e:
            logger.error(f"Error in smart fetch and process: {e}")

    # ---------- Background workers ----------

    def _start_workers(self) -> None:
        # Trailing worker
        t1 = threading.Thread(target=self._trailing_worker, daemon=True)
        t1.start()
        self._tw = t1

        # OCO & housekeeping worker
        t2 = threading.Thread(target=self._oco_and_housekeeping_worker, daemon=True)
        t2.start()
        self._ow = t2

    def _trailing_worker(self) -> None:
        """Periodically trail SL for active trades (uses ATR multiplier)."""
        while not self._trailing_worker_stop.is_set():
            try:
                if self.TRAILING_ENABLE and self.is_trading and not self._is_circuit_breaker_tripped():
                    self._trailing_tick()
            except Exception as e:
                logger.debug(f"Trailing worker error: {e}")
            self._trailing_worker_stop.wait(self.WORKER_INTERVAL_SEC)

    def _oco_and_housekeeping_worker(self) -> None:
        while not self._oco_worker_stop.is_set():
            try:
                if self.is_trading:
                    self._oco_and_housekeeping_tick()
            except Exception as e:
                logger.debug(f"OCO worker error: {e}")
            self._oco_worker_stop.wait(self.WORKER_INTERVAL_SEC)

    def _trailing_tick(self) -> None:
        # Iterate active orders and attempt trailing
        with self._lock:
            items = list(self.active_trades.items())

        for oid, tr in items:
            if tr.get("status") != "OPEN":
                continue

            symbol = tr.get("symbol")
            atr = float(tr.get("atr", 0.0)) or 0.0
            if atr <= 0:
                continue

            # Executor should expose a best-effort last price (SIM: may be None)
            get_last = getattr(self.order_executor, "get_last_price", None)
            ltp = get_last(symbol) if callable(get_last) else None
            if ltp is None:
                # fallback to recent close we already fetched (kept in trade)
                ltp = float(tr.get("last_close", 0.0) or 0.0)
            if not ltp or ltp <= 0:
                continue

            try:
                self.order_executor.update_trailing_stop(oid, float(ltp), float(atr))
            except Exception:
                pass

    def _oco_and_housekeeping_tick(self) -> None:
        # 1) Best-effort sync fills & enforce OCO
        filled = []
        try:
            sync = getattr(self.order_executor, "sync_and_enforce_oco", None)
            filled = sync() if callable(sync) else []
        except Exception:
            filled = []

        # 2) Normalize actives view from executor (dict or list)
        actives_raw = self.order_executor.get_active_orders()
        if isinstance(actives_raw, dict):
            active_ids = set(actives_raw.keys())
        else:
            # list of records having order_id
            try:
                active_ids = {getattr(o, "order_id", None) for o in (actives_raw or [])} - {None}
            except Exception:
                active_ids = set()

        with self._lock:
            to_finalize = []

            # from explicit fills
            for entry_id, fill_px in filled or []:
                tr = self.active_trades.get(entry_id)
                if tr and tr.get("status") == "OPEN":
                    tr["exit_price"] = float(fill_px)
                    to_finalize.append(entry_id)

            # implicit closures (disappeared)
            for entry_id, tr in list(self.active_trades.items()):
                if tr.get("status") != "OPEN":
                    continue
                if entry_id not in active_ids:
                    # No exact exit; fall back to known target/SL depending on side
                    if tr["direction"] == "BUY":
                        fallback_px = float(tr.get("target") or tr.get("stop_loss", 0.0))
                    else:
                        fallback_px = float(tr.get("stop_loss") or tr.get("target", 0.0))
                    tr["exit_price"] = fallback_px
                    to_finalize.append(entry_id)

            for entry_id in to_finalize:
                self._finalize_trade(entry_id)

    # ---------- Telegram control ----------

    def _start_polling(self) -> None:
        if self._polling_thread and self._polling_thread.is_alive():
            return
        try:
            self.telegram_controller.send_startup_alert()
        except Exception:
            pass
        try:
            self._polling_thread = threading.Thread(
                target=self.telegram_controller.start_polling, daemon=True
            )
            self._polling_thread.start()
            logger.info("✅ Telegram polling started (daemon).")
        except Exception as e:
            logger.error(f"Failed to start polling thread: {e}")

    def _stop_polling(self) -> None:
        logger.info("🛑 Stopping Telegram polling (app shutdown)...")
        if self.telegram_controller:
            try:
                self.telegram_controller.stop_polling()
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
        """Stop trading and cancel open orders (best-effort)."""
        try:
            self.stop()
            self.order_executor.cancel_all_orders()
            self._safe_send_message("🛑 Emergency stop executed. All open orders cancelled (best-effort).")
            return True
        except Exception as e:
            logger.error(f"Emergency stop failed: {e}")
            return False

    def _handle_control(self, command: str, arg: str = "") -> bool:
        command = (command or "").strip().lower()
        arg = (arg or "").strip().lower()
        logger.info(f"Received command: /{command} {arg}")
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
            logger.error(f"Error handling control command: {e}")
            return False

    # ---------- Mode switching (runtime) ----------

    def enable_live_trading(self) -> bool:
        with self._lock:
            if self.is_trading:
                self._safe_send_message("🛑 Stop trading first: `/stop`", parse_mode="Markdown")
                return False
            try:
                self.order_executor = self._build_live_executor()
                self.live_mode = True
                self._refresh_instruments_cache(force=True)
                logger.info("🟢 Switched to LIVE mode.")
                self._safe_send_message("🟢 Switched to *LIVE* mode.", parse_mode="Markdown")
                return True
            except Exception as exc:
                logger.error("Enable LIVE failed: %s", exc, exc_info=True)
                self.order_executor = OrderExecutor()
                self.live_mode = False
                self._safe_send_message(
                    f"❌ Failed to enable LIVE: `{exc}`\nReverted to SHADOW.",
                    parse_mode="Markdown",
                )
                return False

    def disable_live_trading(self) -> bool:
        with self._lock:
            if self.is_trading:
                self._safe_send_message("🛑 Stop trading first: `/stop`", parse_mode="Markdown")
                return False
            self.order_executor = OrderExecutor()
            self.live_mode = False
        logger.info("🛡️ Switched to SHADOW (simulation) mode.")
        self._safe_send_message("🛡️ Switched to *SHADOW* (simulation) mode.", parse_mode="Markdown")
        return True

    # ---------- Balance/session helpers ----------

    def refresh_account_balance(self) -> None:
        """Pull live balance and push into risk manager (logs result)."""
        try:
            new_bal = get_live_account_balance()
            self.risk_manager.account_size = new_bal
            self.risk_manager.equity = new_bal
            self.risk_manager.equity_peak = max(self.risk_manager.equity_peak, new_bal)
            logger.info(f"💳 Refreshed account balance: ₹{new_bal:.2f}")
        except Exception as e:
            logger.warning(f"Balance refresh failed: {e}")

    def _safe_log_account_balance(self) -> None:
        try:
            logger.info(f"💰 Account size (cached): ₹{self.risk_manager.account_size:.2f}")
        except Exception:
            pass

    def _maybe_rollover_daily(self) -> None:
        """At day change or after market close, roll daily stats."""
        now = datetime.now()
        market_closed = now.hour > 15 or (now.hour == 15 and now.minute >= 31)
        if now.date() != self.session_date or market_closed:
            if self.trades or self.daily_pnl != 0:
                logger.info(f"📘 Daily roll — Trades: {len(self.trades)}, PnL: ₹{self.daily_pnl:.2f}")
            self.trades = []
            self.active_trades = {}
            self.daily_pnl = 0.0
            self.session_date = now.date()
            self.daily_start_equity = self.risk_manager.account_size

    # ---------- Market hours ----------

    def _is_trading_hours(self, current_time: datetime) -> bool:
        if getattr(Config, "ALLOW_OFFHOURS_TESTING", False):
            return True
        hour = current_time.hour
        minute = current_time.minute
        if hour < 9 or (hour == 9 and minute < 15):
            return False
        if hour > 15 or (hour == 15 and minute > 30):
            return False
        return True

    # ---------- Circuit breaker ----------

    def _is_circuit_breaker_tripped(self) -> bool:
        # daily drawdown relative to start of day equity
        eq0 = max(1.0, float(self.daily_start_equity or 1.0))
        dd = -self.daily_pnl / eq0
        return dd >= self.MAX_DAILY_DRAWDOWN_PCT

    # ---------- Cache handling ----------

    def _force_refresh_cache(self) -> bool:
        try:
            self._refresh_instruments_cache(force=True)
            self._safe_send_message("🔄 Instrument cache refreshed successfully.")
            return True
        except Exception as e:
            logger.error(f"Error refreshing cache: {e}")
            return False

    def _refresh_instruments_cache(self, force: bool = False) -> None:
        with self._cache_lock:
            kite = getattr(self.order_executor, "kite", None)
            if not kite:
                self._nfo_instruments_cache = []
                self._nse_instruments_cache = []
                return

            current_time = time.time()
            needs_refresh = (
                force
                or self._nfo_instruments_cache is None
                or self._nse_instruments_cache is None
                or (current_time - self._instruments_cache_timestamp) > self._INSTRUMENT_CACHE_DURATION
            )

            if needs_refresh:
                try:
                    with ThreadPoolExecutor(max_workers=2) as ex:
                        nfo_f = ex.submit(kite.instruments, "NFO")
                        nse_f = ex.submit(kite.instruments, "NSE")
                        self._nfo_instruments_cache = nfo_f.result(timeout=12)
                        self._nse_instruments_cache = nse_f.result(timeout=12)
                    self._instruments_cache_timestamp = current_time
                    logger.info("✅ Instruments cache refreshed.")
                except Exception as e:
                    logger.error(f"Failed to refresh instruments cache: {e}")
                    self._nfo_instruments_cache = self._nfo_instruments_cache or []
                    self._nse_instruments_cache = self._nse_instruments_cache or []

    def _get_cached_instruments(self) -> Tuple[List[Dict], List[Dict]]:
        with self._cache_lock:
            return self._nfo_instruments_cache or [], self._nse_instruments_cache or []

    @lru_cache(maxsize=256)
    def _get_cached_atm_strike(self, spot_price: float, bucket: int) -> int:
        """Round to nearest 50; `bucket` should be int(time.time()//30) for TTL-like caching."""
        return int(round(spot_price / 50.0) * 50)

    # ---------- Health/Status ----------

    def _send_detailed_status(self) -> bool:
        try:
            status = self.get_status()
            cache_age = (time.time() - self._instruments_cache_timestamp) / 60 if self._instruments_cache_timestamp else -1
            status_msg = (
                "📊 **Detailed Status**\n"
                f"🔄 Trading: {'✅ Active' if status['is_trading'] else '❌ Stopped'}\n"
                f"🎯 Mode: {'🟢 LIVE' if self.live_mode else '🛡️ SHADOW'}\n"
                f"📈 Open Orders: {status.get('open_orders', 0)}\n"
                f"💼 Trades Today: {status['trades_today']}\n"
                f"🕐 Cache Age: {cache_age:.1f} min\n"
                f"💰 Daily PnL: ₹{self.daily_pnl:.2f}\n"
            )
            self._safe_send_message(status_msg, parse_mode="Markdown")
            return True
        except Exception as e:
            logger.error(f"Error sending detailed status: {e}")
            return False

    def _run_health_check(self) -> bool:
        try:
            kite = getattr(self.order_executor, "kite", None)
            health_result = _health_check(kite) if kite else {"overall_status": "ERROR", "message": "No Kite instance"}
            health_msg = (
                "🏥 **System Health Check**\n"
                f"📊 Overall: {health_result.get('overall_status', 'UNKNOWN')}\n"
                f"🔄 Trading: {'✅ ACTIVE' if self.is_trading else '⏹️ STOPPED'}\n"
                f"🎯 Mode: {'🟢 LIVE' if self.live_mode else '🛡️ SHADOW'}\n"
                f"💼 Trades: {len(self.trades)}\n"
                f"📱 Telegram: {'✅ ACTIVE' if self._polling_thread and self._polling_thread.is_alive() else '❌ INACTIVE'}\n"
            )
            self._safe_send_message(health_msg, parse_mode="Markdown")
            return True
        except Exception as e:
            logger.error(f"Error running health check: {e}")
            return False

    # ---------- Fetch & process ----------

    def fetch_and_process_data(self) -> None:
        t0 = time.time()
        try:
            kite = getattr(self.order_executor, "kite", None)
            if not kite:
                logger.error("KiteConnect instance not found. Is live mode enabled?")
                return

            # single-position policy
            actives_raw = self.order_executor.get_active_orders()
            active_count = len(actives_raw) if isinstance(actives_raw, dict) else len(actives_raw or [])
            if active_count >= self.MAX_CONCURRENT_TRADES:
                logger.debug("Single-position policy: already at max concurrent trades.")
                return

            # 1) Refresh instruments if needed
            self._refresh_instruments_cache()
            cached_nfo, cached_nse = self._get_cached_instruments()
            if not cached_nfo and not cached_nse:
                logger.error("Instrument cache is empty. Cannot proceed.")
                return

            # 2) Spot price and base instruments
            with ThreadPoolExecutor(max_workers=2) as ex:
                spot_future = ex.submit(self._fetch_spot_price)
                instr_future = ex.submit(self._get_instruments_data, cached_nfo, cached_nse)
                spot_price = spot_future.result(timeout=10)
                instruments_data = instr_future.result(timeout=10)

            if not spot_price or not instruments_data:
                logger.debug("Missing spot price or instruments data.")
                return

            atm_strike = instruments_data["atm_strike"]
            spot_token = instruments_data.get("spot_token")
            end_time = datetime.now()
            lookback = int(getattr(Config, "DATA_LOOKBACK_MINUTES", 30))
            start_time = end_time - timedelta(minutes=lookback)

            # 3) Fetch historical data in parallel for spot + a window of option strikes
            spot_df, options_data = self._fetch_all_data_parallel(
                spot_token, atm_strike, start_time, end_time, cached_nfo, cached_nse
            )
            if spot_df.empty:
                logger.debug("No spot candles.")
                return

            # Build list of candidate option symbols
            candidate_symbols = list(options_data.keys())

            # 3b) Bulk quotes for spread guard (rate-limit friendly)
            quotes = self._bulk_quote(candidate_symbols) if candidate_symbols else {}

            # 4) Pick strikes for processing (pass quotes for LTP_MID mode)
            if options_data:
                selected_strikes_info = self._analyze_options_data_optimized(float(spot_price), options_data, quotes)
            else:
                selected_strikes_info = self._get_fallback_strikes(atm_strike, cached_nfo, cached_nse)

            if not selected_strikes_info:
                logger.debug("No strikes selected for processing.")
                return

            # 5) Process the chosen strikes
            self._process_selected_strikes(selected_strikes_info, options_data, spot_df)

            logger.debug(f"Data fetch+process in {time.time() - t0:.2f}s")
        except Exception as e:
            logger.error(f"Error in fetch_and_process_data: {e}", exc_info=True)

    def _fetch_spot_price(self) -> Optional[float]:
        try:
            sym = _get_spot_ltp_symbol()
            ltp = self.order_executor.kite.ltp([sym])
            price = ltp.get(sym, {}).get("last_price")
            return float(price) if price is not None else None
        except Exception as e:
            logger.error(f"Exception fetching spot price: {e}")
            return None

    def _get_instruments_data(self, cached_nfo: List[Dict], cached_nse: List[Dict]) -> Optional[Dict]:
        try:
            return get_instrument_tokens(
                symbol=Config.SPOT_SYMBOL,
                kite_instance=self.order_executor.kite,
                cached_nfo_instruments=cached_nfo,
                cached_nse_instruments=cached_nse,
            )
        except Exception as e:
            logger.error(f"Error getting instruments data: {e}")
            return None

    def _fetch_historical_data(self, token: int, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Wraps kite.historical_data; returns DataFrame with DatetimeIndex & ['open','high','low','close','volume'].""" 
        try:
            k = self.order_executor.kite
            tf = self.HIST_TIMEFRAME
            candles = k.historical_data(token, start_time, end_time, tf, oi=False)
            if not candles:
                return pd.DataFrame()
            df = pd.DataFrame(candles)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
            cols = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
            return df[cols].copy()
        except Exception as e:
            logger.debug(f"_fetch_historical_data error for {token}: {e}")
            return pd.DataFrame()

    def _fetch_all_data_parallel(
        self,
        spot_token: Optional[int],
        atm_strike: int,
        start_time: datetime,
        end_time: datetime,
        cached_nfo: List[Dict],
        cached_nse: List[Dict],
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        spot_df = pd.DataFrame()
        options_data: Dict[str, pd.DataFrame] = {}
        tasks: List[Tuple[str, int]] = []

        if spot_token:
            tasks.append(("SPOT", spot_token))

        try:
            rng = int(getattr(Config, "STRIKE_RANGE", 2))
            for offset in range(-rng, rng + 1):
                info = get_instrument_tokens(
                    symbol=Config.SPOT_SYMBOL,
                    offset=offset,
                    kite_instance=self.order_executor.kite,
                    cached_nfo_instruments=cached_nfo,
                    cached_nse_instruments=cached_nse,
                )
                if not info:
                    continue
                for opt_type, token_key, symbol_key in (("CE", "ce_token", "ce_symbol"), ("PE", "pe_token", "pe_symbol")):
                    token = info.get(token_key)
                    symbol = info.get(symbol_key)
                    if token and symbol:
                        tasks.append((symbol, token))
        except Exception as e:
            logger.error(f"Error preparing fetch tasks: {e}")

        futures = {}
        with ThreadPoolExecutor(max_workers=8) as ex:
            for symbol, token in tasks:
                futures[submit := ex.submit(self._fetch_historical_data, token, start_time, end_time)] = symbol
            for fut in as_completed(futures):
                symbol = futures[fut]
                try:
                    df = fut.result(timeout=15)
                    if symbol == "SPOT":
                        spot_df = df
                    else:
                        options_data[symbol] = df
                except Exception as e:
                    logger.debug(f"Fetch failed for {symbol}: {e}")

        # cache last close for trailing fallback
        for sym, df in options_data.items():
            try:
                if not df.empty:
                    last_close = float(df["close"].iloc[-1])
                    with self._lock:
                        for tr in self.active_trades.values():
                            if tr.get("symbol") == sym and tr.get("status") == "OPEN":
                                tr["last_close"] = last_close
            except Exception:
                pass

        return spot_df, options_data

    # ---------- Quotes / mid helpers ----------

    def _bulk_quote(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch quote() for a list of symbols in one call. Returns symbol → quote dict."""
        if not symbols:
            return {}
        kite = getattr(self.order_executor, "kite", None)
        if not kite:
            return {}
        try:
            quotes = kite.quote(symbols)
            return quotes or {}
        except Exception as e:
            logger.debug(f"bulk quote failed: {e}")
            return {}

    @staticmethod
    def _mid_from_depth(q: Dict[str, Any]) -> Optional[float]:
        """Compute mid from top-of-book depth if available."""
        try:
            depth = q.get("depth") or {}
            buy = (depth.get("buy") or [])
            sell = (depth.get("sell") or [])
            best_bid = float(buy[0]["price"]) if buy else None
            best_ask = float(sell[0]["price"]) if sell else None
            if best_bid and best_ask and best_ask > 0:
                return (best_bid + best_ask) / 2.0
            return None
        except Exception:
            return None

    # ---------- Selection / processing ----------

    def _analyze_options_data_optimized(
        self,
        spot_price: float,
        options_data: Dict[str, pd.DataFrame],
        quotes: Dict[str, Dict[str, Any]] | None = None,
    ) -> List[Dict]:
        """
        Selection logic:
        - Warmup filter
        - Spread guard
            * LTP_MID: use bid/ask mid + ltp-mid deviation and raw bid/ask spread
            * RANGE: last bar range proxy (legacy)
        - Rank by absolute % change over last bar; pick top CE and PE
        """
        latest: List[Tuple[str, float]] = []
        use_mid_mode = (self.SPREAD_GUARD_MODE == "LTP_MID")
        quotes = quotes or {}

        for sym, df in options_data.items():
            if df is None or df.empty or "close" not in df.columns:
                continue
            if len(df) < max(2, self.WARMUP_BARS):
                continue

            last = df.iloc[-1]
            c = float(last["close"])
            if c <= 0:
                continue

            # --- Spread guard ---
            if use_mid_mode:
                q = quotes.get(sym, {}) if quotes else {}
                ltp = float((q.get("last_price") or c))
                mid = self._mid_from_depth(q)
                if mid is None or mid <= 0:
                    # if depth missing, skip this symbol to be conservative
                    logger.debug(f"Depth unavailable for {sym}; skipping in LTP_MID mode.")
                    continue

                # raw bid/ask spread (ask - bid)/mid
                try:
                    depth = q.get("depth") or {}
                    buy = (depth.get("buy") or [])
                    sell = (depth.get("sell") or [])
                    best_bid = float(buy[0]["price"]) if buy else None
                    best_ask = float(sell[0]["price"]) if sell else None
                except Exception:
                    best_bid = None
                    best_ask = None

                if not best_bid or not best_ask or best_bid <= 0 or best_ask <= 0:
                    logger.debug(f"Top-of-book missing for {sym}; skipping.")
                    continue

                ba_spread = (best_ask - best_bid) / mid
                ltp_mid_dev = abs(ltp - mid) / mid

                if ba_spread > self.SPREAD_GUARD_BA_MAX or ltp_mid_dev > self.SPREAD_GUARD_LTPMID_MAX:
                    logger.debug(f"Spread guard fail {sym}: BA={ba_spread:.4f}, DEV={ltp_mid_dev:.4f}")
                    continue

                prev = float(df["close"].iloc[-2])
                if prev <= 0:
                    continue
                chg = abs((c - prev) / prev)
                latest.append((sym, chg))

            else:
                # Legacy RANGE proxy
                h = float(last["high"])
                l = float(last["low"])
                proxy_spread = (h - l) / c
                if proxy_spread > self.SPREAD_GUARD_PCT:
                    continue

                prev = float(df["close"].iloc[-2])
                if prev <= 0:
                    continue
                chg = abs((c - prev) / prev)
                latest.append((sym, chg))

        if not latest:
            return []

        ce_list = [t for t in latest if t[0].endswith("CE")]
        pe_list = [t for t in latest if t[0].endswith("PE")]
        ce_list.sort(key=lambda x: x[1], reverse=True)
        pe_list.sort(key=lambda x: x[1], reverse=True)

        picks: List[Dict] = []
        if ce_list:
            picks.append({"symbol": ce_list[0][0]})
        if pe_list:
            picks.append({"symbol": pe_list[0][0]})
        return picks

    def _get_fallback_strikes(self, atm_strike: int, cached_nfo: List[Dict], cached_nse: List[Dict]) -> List[Dict]:
        """Fallback: just use ATM CE and PE symbols."""
        try:
            info = get_instrument_tokens(
                symbol=Config.SPOT_SYMBOL,
                offset=0,
                kite_instance=self.order_executor.kite,
                cached_nfo_instruments=cached_nfo,
                cached_nse_instruments=cached_nse,
            )
            picks: List[Dict] = []
            if info and info.get("ce_symbol"):
                picks.append({"symbol": info["ce_symbol"]})
            if info and info.get("pe_symbol"):
                picks.append({"symbol": info["pe_symbol"]})
            return picks
        except Exception as e:
            logger.error(f"Fallback strike selection failed: {e}")
            return []

    def _process_selected_strikes(
        self,
        selected: List[Dict],
        options_data: Dict[str, pd.DataFrame],
        spot_df: pd.DataFrame,
    ) -> None:
        """Run strategy → position sizing → place orders + alerts."""
        if not self.strategy:
            logger.warning("Strategy not initialized.")
            return

        L = int(getattr(Config, "NIFTY_LOT_SIZE", 50))  # contracts per lot

        for pick in selected:
            sym = pick["symbol"]
            df = options_data.get(sym, pd.DataFrame())
            if df.empty or len(df) < max(self.WARMUP_BARS, 30) or not isinstance(df.index, pd.DatetimeIndex):
                logger.debug(f"Insufficient data for {sym}")
                continue

            current_price = float(df["close"].iloc[-1])

            # Strategy → signal
            try:
                signal = self.strategy.generate_signal(df, current_price)
            except Exception as e:
                logger.debug(f"Strategy failed for {sym}: {e}")
                continue

            if not signal or float(signal.get("confidence", 0.0)) < float(Config.CONFIDENCE_THRESHOLD):
                logger.debug(f"No valid signal for {sym}")
                continue

            # single-position policy (again just before entry)
            actives_raw = self.order_executor.get_active_orders()
            active_count = len(actives_raw) if isinstance(actives_raw, dict) else len(actives_raw or [])
            if active_count >= self.MAX_CONCURRENT_TRADES:
                logger.debug("Single-position policy: already at max concurrent trades (pre-entry).")
                break

            # Position sizing (returns lots)
            try:
                pos = self.risk_manager.calculate_position_size(
                    entry_price=signal.get("entry_price", current_price),
                    stop_loss=signal.get("stop_loss", current_price),
                    signal_confidence=signal.get("confidence", 0.0),
                    market_volatility=signal.get("market_volatility", 0.0),
                )
            except Exception as e:
                logger.debug(f"Position sizing failed for {sym}: {e}")
                continue

            lots = int(pos.get("quantity", 0))
            if lots <= 0:
                logger.debug(f"Zero/negative lots for {sym}")
                continue
            qty_contracts = lots * L

            # Send signal alert
            token = len(self.trades) + len(self.active_trades) + 1
            try:
                self.telegram_controller.send_signal_alert(token, signal, {"quantity": qty_contracts})
            except Exception:
                pass

            txn_type = (signal.get("signal") or signal.get("direction", "")).upper()
            if txn_type not in ("BUY", "SELL"):
                logger.debug(f"No direction for {sym}")
                continue

            # --- slippage + fees model (approx) ---
            slip_perc = self.SLIPPAGE_BPS / 10000.0  # bps to fraction
            entry_p = float(signal.get("entry_price", current_price))
            entry_effective = entry_p * (1.0 + slip_perc if txn_type == "BUY" else 1.0 - slip_perc)
            fees = lots * float(self.FEES_PER_LOT)

            # Place entry
            try:
                exchange = getattr(Config, "TRADE_EXCHANGE", "NFO")
                order_id = self.order_executor.place_entry_order(
                    symbol=sym,
                    exchange=exchange,
                    transaction_type=txn_type,
                    quantity=qty_contracts,
                )
                if not order_id:
                    logger.warning(f"Entry order failed for {sym}")
                    continue
            except Exception as e:
                logger.warning(f"Entry order error for {sym}: {e}")
                continue

            # Bracket (GTT) orders — keep your executor signature
            try:
                self.order_executor.setup_gtt_orders(
                    entry_order_id=order_id,
                    entry_price=entry_p,
                    stop_loss_price=signal.get("stop_loss", current_price),
                    target_price=signal.get("target", current_price),
                    symbol=sym,
                    exchange=getattr(Config, "TRADE_EXCHANGE", "NFO"),
                    quantity=qty_contracts,
                    transaction_type=txn_type,
                )
            except Exception as e:
                logger.warning(f"GTT/exit setup failed for {sym}: {e}")

            # Record active trade
            with self._lock:
                self.active_trades[order_id] = {
                    "order_id": order_id,
                    "symbol": sym,
                    "direction": txn_type,
                    "quantity": qty_contracts,
                    "lots": lots,
                    "entry_price": entry_effective,   # include slippage
                    "raw_entry_price": entry_p,
                    "stop_loss": float(signal.get("stop_loss", current_price)),
                    "target": float(signal.get("target", current_price)),
                    "confidence": float(signal.get("confidence", 0.0)),
                    "atr": float(signal.get("market_volatility", 0.0)),
                    "fees": float(fees),
                    "status": "OPEN",
                    "ts": datetime.now().isoformat(timespec="seconds"),
                }

            logger.info(f"✅ Trade opened: {txn_type} {qty_contracts}c ({lots} lot) {sym} @ {entry_effective:.2f}")

    # ---------- Finalize / PnL ----------

    def _finalize_trade(self, entry_id: str) -> None:
        with self._lock:
            tr = self.active_trades.pop(entry_id, None)

        if not tr:
            return

        # Obtain exit price if not present (fallback to ltp)
        exit_price = float(tr.get("exit_price", 0.0))
        if exit_price <= 0:
            get_last = getattr(self.order_executor, "get_last_price", None)
            px = get_last(tr["symbol"]) if callable(get_last) else None
            if px is not None:
                exit_price = float(px)
            else:
                exit_price = tr["target"] if tr["direction"] == "SELL" else tr["stop_loss"]

        # Apply slippage on exit as well
        slip_perc = self.SLIPPAGE_BPS / 10000.0
        if tr["direction"] == "BUY":
            exit_effective = exit_price * (1.0 - slip_perc)
            pnl_per_contract = exit_effective - tr["entry_price"]
        else:
            exit_effective = exit_price * (1.0 + slip_perc)
            pnl_per_contract = tr["entry_price"] - exit_effective

        pnl_gross = pnl_per_contract * tr["quantity"]
        net_pnl = pnl_gross - tr.get("fees", 0.0)

        # Update day PnL and risk
        self.daily_pnl += net_pnl
        try:
            self.risk_manager.update_after_trade(net_pnl)
        except Exception:
            pass

        self.trades.append(
            {
                "order_id": tr["order_id"],
                "symbol": tr["symbol"],
                "direction": tr["direction"],
                "quantity": tr["quantity"],
                "entry": tr["entry_price"],
                "exit": exit_effective,
                "pnl": net_pnl,
                "confidence": tr["confidence"],
                "atr": tr["atr"],
                "ts": tr["ts"],
            }
        )

        # Append CSV log
        self._append_trade_log([
            datetime.now().strftime("%Y-%m-%d"),
            tr["order_id"],
            tr["symbol"],
            tr["direction"],
            tr["quantity"],
            f"{tr['entry_price']:.2f}",
            f"{exit_effective:.2f}",
            f"{pnl_gross:.2f}",
            f"{tr.get('fees', 0.0):.2f}",
            f"{net_pnl:.2f}",
            f"{tr['confidence']:.2f}",
            f"{tr['atr']:.4f}",
            "LIVE" if self.live_mode else "SHADOW",
        ])

        logger.info(
            "🏁 Trade closed %s %s x%d | entry %.2f → exit %.2f | net PnL ₹%.2f",
            tr["direction"], tr["symbol"], tr["quantity"], tr["entry_price"], exit_effective, net_pnl
        )

    # ---------- Status / summary ----------

    def get_status(self) -> Dict[str, Any]:
        actives_raw = self.order_executor.get_active_orders()
        open_orders = len(actives_raw) if isinstance(actives_raw, dict) else len(actives_raw or [])
        status: Dict[str, Any] = {
            "is_trading": self.is_trading,
            "open_orders": open_orders,
            "trades_today": len(self.trades),
            "live_mode": self.live_mode,
        }
        status.update(self.risk_manager.get_risk_status())
        return status

    def get_summary(self) -> str:
        lines = [
            "📊 <b>Daily Summary</b>",
            f"🟢 Mode: {'LIVE' if self.live_mode else 'SHADOW'}",
            f"🔁 <b>Total trades:</b> {len(self.trades)}",
            f"💰 <b>PNL:</b> {self.daily_pnl:.2f}",
            "────────────────────",
        ]
        for t in self.trades:
            lines.append(
                f"{t['direction']} {t['quantity']} {t.get('symbol','')} @ {float(t['entry']):.2f} "
                f"→ {float(t['exit']):.2f} (PnL ₹{float(t['pnl']):.2f})"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"<RealTimeTrader trading={self.is_trading} live_mode={self.live_mode} trades={len(self.trades)}>"

    # ---------- Safe Telegram helpers ----------

    def _safe_send_message(self, text: str, parse_mode: Optional[str] = None) -> None:
        try:
            self.telegram_controller.send_message(text, parse_mode=parse_mode)
        except Exception:
            pass

    def _safe_send_alert(self, kind: str) -> None:
        try:
            self.telegram_controller.send_realtime_session_alert(kind)
        except Exception:
            pass

    # ---------- Shutdown ----------

    def shutdown(self) -> None:
        logger.info("👋 Shutting down RealTimeTrader...")
        try:
            self.stop()
        except Exception:
            pass
        self._trailing_worker_stop.set()
        self._oco_worker_stop.set()
        self._stop_polling()
        logger.info("✅ RealTimeTrader shutdown complete.")
