# src/data_streaming/realtime_trader.py
import logging
import threading
import atexit
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import schedule
from datetime import datetime, timedelta, date
import time
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import os

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
    Real-time options trader with:
      - Telegram long-poll (daemon thread)
      - Smart scheduler (market-hours aware)
      - ATR trailing worker + OCO hygiene
      - Circuit breaker, warmup, spread guard, single-position policy
      - PnL logging and daily rollover
    """

    # ---------- Tunables (sane defaults, override in .env/Config) ----------
    WARMUP_BARS = getattr(Config, "WARMUP_BARS", 40)
    MAX_SPREAD_PCT = float(getattr(Config, "MAX_SPREAD_PCT", 0.025))  # 2.5% candle range proxy
    MAX_CONCURRENT_TRADES = int(getattr(Config, "MAX_CONCURRENT_TRADES", 1))
    SLIPPAGE_BPS = float(getattr(Config, "SLIPPAGE_BPS", 3.0))  # 3 bps (~0.03%) per side
    FEES_PCT = float(getattr(Config, "FEES_PCT", 0.0006))       # 6 bps round numbers
    ATR_PERIOD = int(getattr(Config, "ATR_PERIOD", 14))
    TRAIL_INTERVAL_SEC = int(getattr(Config, "TRAIL_INTERVAL_SEC", 20))
    OCO_SYNC_INTERVAL_SEC = int(getattr(Config, "OCO_SYNC_INTERVAL_SEC", 25))
    FETCH_INTERVAL_SEC = int(getattr(Config, "FETCH_INTERVAL_SEC", 30))
    BALANCE_LOG_INTERVAL_MIN = int(getattr(Config, "BALANCE_LOG_INTERVAL_MIN", 30))
    ALLOW_OFFHOURS_TESTING = bool(getattr(Config, "ALLOW_OFFHOURS_TESTING", False))
    MARKET_CLOSE_HHMM = getattr(Config, "MARKET_CLOSE_HHMM", "15:30")  # 24h

    # Circuit breaker (extra, on top of PositionSizing controls)
    MAX_DAILY_DRAWDOWN_PCT = float(getattr(Config, "MAX_DAILY_DRAWDOWN_PCT", 0.03))
    MAX_CONSECUTIVE_LOSSES = int(getattr(Config, "MAX_CONSECUTIVE_LOSSES", 3))

    def __init__(self) -> None:
        self._lock = threading.RLock()

        self.is_trading: bool = False
        self.live_mode: bool = bool(getattr(Config, "ENABLE_LIVE_TRADING", False))

        # Daily state
        self.daily_pnl: float = 0.0
        self.daily_start_equity: float = 0.0
        self.consecutive_losses: int = 0
        self._trade_day: date = date.today()

        # Trades registry for PnL/accounting (order_id -> info)
        self.active_trades: Dict[str, Dict[str, Any]] = {}
        self.trades: List[Dict[str, Any]] = []  # intraday memory (optional)

        # Instrument caches (rate-limit safety)
        self._nfo_instruments_cache: Optional[List[Dict]] = None
        self._nse_instruments_cache: Optional[List[Dict]] = None
        self._instruments_cache_timestamp: float = 0
        self._INSTRUMENT_CACHE_DURATION: int = 300  # seconds
        self._cache_lock = threading.RLock()

        # ATM cache TTL (avoid spam)
        self._ATM_CACHE_DURATION: int = 30  # seconds

        # Strategy / Risk / Executor / Telegram
        self._init_components()

        # Telegram polling thread
        self._polling_thread: Optional[threading.Thread] = None
        self._start_polling()

        # Scheduling
        self._setup_smart_scheduling()

        # Graceful shutdown
        atexit.register(self.shutdown)

        logger.info("RealTimeTrader initialized.")
        self._log_account_balance()

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
            logger.warning("Failed to initialize strategy: %s", e)
            self.strategy = None

        try:
            self.risk_manager = PositionSizing()
        except Exception as e:
            logger.warning("Failed to initialize risk manager: %s", e)
            self.risk_manager = PositionSizing()

        self.order_executor = self._init_order_executor()

        try:
            self.telegram_controller = TelegramController(
                status_callback=self.get_status,
                control_callback=self._handle_control,
                summary_callback=self.get_summary,
            )
        except Exception as e:
            logger.warning("Failed to initialize Telegram controller: %s", e)
            raise

        # Daily start equity for drawdown circuit breaker
        try:
            self.daily_start_equity = float(self.risk_manager.equity or self.risk_manager.account_size or 0.0)
        except Exception:
            self.daily_start_equity = 0.0

        # Ensure log file header exists
        self._ensure_trade_log()

    def _build_live_executor(self) -> OrderExecutor:
        from kiteconnect import KiteConnect

        api_key = getattr(Config, "ZERODHA_API_KEY", None)
        access_token = getattr(Config, "KITE_ACCESS_TOKEN", None) or getattr(Config, "ZERODHA_ACCESS_TOKEN", None)
        if not api_key or not access_token:
            raise RuntimeError("ZERODHA_API_KEY or KITE_ACCESS_TOKEN missing")

        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        logger.info("Live executor created (KiteConnect).")
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

    # ---------- Scheduling / main loop ----------
    def _setup_smart_scheduling(self) -> None:
        schedule.clear()

        # Data fetch & trade logic
        schedule.every(max(5, self.FETCH_INTERVAL_SEC)).seconds.do(self._smart_fetch_and_process)

        # Balance refresh
        schedule.every(max(5, self.BALANCE_LOG_INTERVAL_MIN)).minutes.do(self.refresh_account_balance)

        # OCO hygiene & trailing workers
        schedule.every(max(10, self.OCO_SYNC_INTERVAL_SEC)).seconds.do(self._oco_and_housekeeping_tick)
        schedule.every(max(10, self.TRAIL_INTERVAL_SEC)).seconds.do(self._trailing_tick)

        # Daily rollover near market close
        try:
            hh, mm = self.MARKET_CLOSE_HHMM.split(":")
            schedule.every().day.at(f"{int(hh):02d}:{int(mm):02d}").do(self._daily_rollover)
        except Exception:
            schedule.every().day.at("15:30").do(self._daily_rollover)

        logger.info(
            "Scheduler: fetch=%ss, trail=%ss, oco=%ss, balance=%smin, rollover=%s",
            self.FETCH_INTERVAL_SEC, self.TRAIL_INTERVAL_SEC, self.OCO_SYNC_INTERVAL_SEC,
            self.BALANCE_LOG_INTERVAL_MIN, self.MARKET_CLOSE_HHMM
        )

    def run(self) -> None:
        """Main non-blocking scheduler loop. Call this in a thread or main process."""
        logger.info("RealTimeTrader.run() loop started.")
        while True:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                logger.error("Error in run loop: %s", e, exc_info=True)
                time.sleep(2)

    def _smart_fetch_and_process(self) -> None:
        try:
            now = datetime.now()
            if not self._is_trading_hours(now) and not self.ALLOW_OFFHOURS_TESTING:
                # heartbeat every ~5 minutes
                if int(time.time()) % 300 < 2:
                    logger.info("Market closed. Skipping fetch.")
                return
            self.fetch_and_process_data()
        except Exception as e:
            logger.error("Error in smart fetch and process: %s", e)

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
            logger.info("Telegram polling started (daemon).")
        except Exception as e:
            logger.error("Failed to start polling thread: %s", e)

    def _stop_polling(self) -> None:
        logger.info("Stopping Telegram polling (app shutdown)...")
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
                self._safe_send_message("Trader already running.")
                return True
            self.is_trading = True
        self._safe_send_alert("START")
        logger.info("Trading started.")
        return True

    def stop(self) -> bool:
        with self._lock:
            if not self.is_trading:
                self._safe_send_message("Trader is already stopped.")
                return True
            self.is_trading = False
        self._safe_send_alert("STOP")
        logger.info("Trading stopped.")
        return True

    def emergency_stop_all(self) -> bool:
        """Stop trading and cancel open orders (best-effort)."""
        try:
            self.stop()
            try:
                self.order_executor.cancel_all_orders()
            except Exception:
                pass
            self._safe_send_message("Emergency stop executed. All open orders cancelled (best-effort).")
            return True
        except Exception as e:
            logger.error("Emergency stop failed: %s", e)
            return False

    def _handle_control(self, command: str, arg: str = "") -> bool:
        command = (command or "").strip().lower()
        arg = (arg or "").strip().lower()
        logger.info("Received command: /%s %s", command, arg)
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
                self._safe_send_message("Usage: `/mode live` or `/mode shadow`", parse_mode="Markdown")
                return False
            if command == "refresh":
                return self._force_refresh_cache()
            if command == "status":
                return self._send_detailed_status()
            if command == "health":
                return self._run_health_check()
            if command == "emergency":
                return self.emergency_stop_all()

            self._safe_send_message(f"Unknown command: `{command}`", parse_mode="Markdown")
            return False
        except Exception as e:
            logger.error("Error handling control command: %s", e)
            return False

    # ---------- Mode switching (runtime) ----------
    def enable_live_trading(self) -> bool:
        with self._lock:
            if self.is_trading:
                self._safe_send_message("Stop trading first: `/stop`", parse_mode="Markdown")
                return False
            try:
                self.order_executor = self._build_live_executor()
                self.live_mode = True
                self._refresh_instruments_cache(force=True)
                logger.info("Switched to LIVE mode.")
                self._safe_send_message("Switched to *LIVE* mode.", parse_mode="Markdown")
                return True
            except Exception as exc:
                logger.error("Enable LIVE failed: %s", exc, exc_info=True)
                self.order_executor = OrderExecutor()
                self.live_mode = False
                self._safe_send_message(
                    f"Failed to enable LIVE: `{exc}`\nReverted to SHADOW.",
                    parse_mode="Markdown",
                )
                return False

    def disable_live_trading(self) -> bool:
        with self._lock:
            if self.is_trading:
                self._safe_send_message("Stop trading first: `/stop`", parse_mode="Markdown")
                return False
            self.order_executor = OrderExecutor()
            self.live_mode = False
        logger.info("Switched to SHADOW (simulation) mode.")
        self._safe_send_message("Switched to *SHADOW* (simulation) mode.", parse_mode="Markdown")
        return True

    # ---------- Balance helpers ----------
    def refresh_account_balance(self) -> None:
        """Pull live balance and push into risk manager (logs result)."""
        try:
            new_bal = get_live_account_balance()
            self.risk_manager.account_size = new_bal
            self.risk_manager.equity = new_bal
            self.risk_manager.equity_peak = max(self.risk_manager.equity_peak, new_bal)
            logger.info("Refreshed account balance: ₹%.2f", new_bal)
        except Exception as e:
            logger.warning("Balance refresh failed: %s", e)

    def _log_account_balance(self) -> None:
        try:
            logger.info("Account size (cached): ₹%.2f", float(self.risk_manager.account_size or 0.0))
        except Exception:
            pass

    # ---------- Market hours ----------
    def _is_trading_hours(self, current_time: datetime) -> bool:
        if self.ALLOW_OFFHOURS_TESTING:
            return True
        hour = current_time.hour
        minute = current_time.minute
        if hour < 9 or (hour == 9 and minute < 15):
            return False
        if hour > 15 or (hour == 15 and minute > 30):
            return False
        return True

    # ---------- Cache handling ----------
    def _force_refresh_cache(self) -> bool:
        try:
            self._refresh_instruments_cache(force=True)
            self._safe_send_message("Instrument cache refreshed successfully.")
            return True
        except Exception as e:
            logger.error("Error refreshing cache: %s", e)
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
                    logger.info("Instruments cache refreshed.")
                except Exception as e:
                    logger.error("Failed to refresh instruments cache: %s", e)
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
                f"📈 Open Positions: {status.get('open_orders', 0)}\n"
                f"💼 Trades Today: {status['trades_today']}\n"
                f"🕐 Cache Age: {cache_age:.1f} min\n"
                f"💰 Daily P&L: ₹{self.daily_pnl:.2f}\n"
                f"⚠️ Consecutive Losses: {self.consecutive_losses}\n"
            )
            self._safe_send_message(status_msg, parse_mode="Markdown")
            return True
        except Exception as e:
            logger.error("Error sending detailed status: %s", e)
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
            logger.error("Error running health check: %s", e)
            return False

    # ---------- Data fetch & trade pipeline ----------
    def fetch_and_process_data(self) -> None:
        if not self.is_trading:
            return
        t0 = time.time()
        try:
            kite = getattr(self.order_executor, "kite", None)
            if not kite:
                logger.error("KiteConnect instance not found. Is live mode enabled?")
                return

            # 1) Refresh instruments if needed
            self._refresh_instruments_cache()
            cached_nfo, cached_nse = self._get_cached_instruments()
            if not cached_nfo and not cached_nse:
                logger.error("Instrument cache is empty. Cannot proceed.")
                return

            # 2) Spot and instruments meta
            with ThreadPoolExecutor(max_workers=2) as ex:
                spot_future = ex.submit(self._fetch_spot_price)
                instr_future = ex.submit(self._get_instruments_data, cached_nfo, cached_nse)
                spot_price = spot_future.result(timeout=10)
                instruments_data = instr_future.result(timeout=10)

            if not spot_price or not instruments_data:
                logger.warning("Failed to fetch essential data (spot price or instruments).")
                return

            atm_strike = instruments_data["atm_strike"]
            spot_token = instruments_data.get("spot_token")
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=getattr(Config, "DATA_LOOKBACK_MINUTES", 30))

            # 3) Historical data in parallel (spot + window of option strikes)
            spot_df, options_data = self._fetch_all_data_parallel(
                spot_token, atm_strike, start_time, end_time, cached_nfo, cached_nse
            )
            if spot_df is None or spot_df.empty or len(spot_df) < max(20, self.WARMUP_BARS):
                logger.debug("Insufficient spot bars for warm-up.")
                return

            # 4) Pick strikes
            if options_data:
                selected_strikes_info = self._analyze_options_data_optimized(spot_price, options_data)
            else:
                selected_strikes_info = self._get_fallback_strikes(atm_strike, cached_nfo, cached_nse)

            if not selected_strikes_info:
                logger.info("No option strikes selected for processing.")
                return

            # 5) Process selected strikes (signal → sizing → orders)
            self._process_selected_strikes(selected_strikes_info, options_data, spot_df)

            logger.debug("Data fetch+process in %.2fs", time.time() - t0)
        except Exception as e:
            logger.error("Error in fetch_and_process_data: %s", e, exc_info=True)

    def _fetch_spot_price(self) -> Optional[float]:
        try:
            sym = _get_spot_ltp_symbol()
            k = self.order_executor.kite
            ltp = k.ltp([sym])
            price = ltp.get(sym, {}).get("last_price")
            return float(price) if price is not None else None
        except Exception as e:
            logger.error("Exception fetching spot price: %s", e)
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
            logger.error("Error getting instruments data: %s", e)
            return None

    def _fetch_historical_data(self, token: int, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Wraps kite.historical_data; returns DataFrame with DatetimeIndex & ['open','high','low','close','volume']."""
        try:
            k = self.order_executor.kite
            tf = getattr(Config, "HISTORICAL_TIMEFRAME", "minute")
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
            logger.debug("_fetch_historical_data error for %s: %s", str(token), e)
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

        # Spot minute candles
        if spot_token:
            tasks.append(("SPOT", spot_token))

        # Build a small window around ATM
        try:
            rng = getattr(Config, "STRIKE_RANGE", 2)
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
            logger.error("Error preparing fetch tasks: %s", e)

        # Parallel fetch (bounded workers)
        futures = {}
        with ThreadPoolExecutor(max_workers=8) as ex:
            for symbol, token in tasks:
                futures[ex.submit(self._fetch_historical_data, token, start_time, end_time)] = symbol
            for fut in as_completed(futures):
                symbol = futures[fut]
                try:
                    df = fut.result(timeout=15)
                    if symbol == "SPOT":
                        spot_df = df
                    else:
                        options_data[symbol] = df
                except Exception as e:
                    logger.debug("Fetch failed for %s: %s", symbol, e)

        return spot_df, options_data

    # ---------- Selection / processing ----------
    def _analyze_options_data_optimized(self, spot_price: float, options_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """
        Heuristic:
          - Warmup: need WARMUP_BARS
          - Spread guard: reject symbols with outsized bar range
          - Rank by absolute % change of last vs previous close
          - Return top CE and top PE if available
        """
        latest: List[Tuple[str, float]] = []
        for sym, df in options_data.items():
            if df is None or df.empty or "close" not in df.columns:
                continue
            if len(df) < max(20, self.WARMUP_BARS):
                continue

            # spread guard via last bar range proxy
            if self._estimate_spread_pct(df) > self.MAX_SPREAD_PCT:
                logger.debug("Spread guard: skip %s", sym)
                continue

            last = float(df["close"].iloc[-1])
            prev = float(df["close"].iloc[-2])
            if prev <= 0:
                continue
            chg = abs((last - prev) / prev)
            latest.append((sym, chg))

        if not latest:
            return []

        # Split CE / PE and pick top1 from each
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
            logger.error("Fallback strike selection failed: %s", e)
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

        # Single-position policy
        try:
            open_positions = len(self.order_executor.get_active_orders())
            if open_positions >= self.MAX_CONCURRENT_TRADES:
                logger.debug("Single-position policy: open=%d >= limit=%d; skip.",
                             open_positions, self.MAX_CONCURRENT_TRADES)
                return
        except Exception:
            pass

        # Circuit breaker (drawdown)
        if self._is_circuit_breaker_tripped():
            logger.warning("Circuit breaker tripped; no new trades.")
            return

        for pick in selected:
            sym = pick["symbol"]
            df = options_data.get(sym, pd.DataFrame())
            if df.empty or len(df) < max(20, self.WARMUP_BARS) or not isinstance(df.index, pd.DatetimeIndex):
                logger.debug("Insufficient data for %s", sym)
                continue

            # spread guard already applied in selection; recheck
            if self._estimate_spread_pct(df) > self.MAX_SPREAD_PCT:
                logger.debug("Spread guard (late) skip %s", sym)
                continue

            current_price = float(df["close"].iloc[-1])

            # Strategy → signal
            try:
                signal = self.strategy.generate_signal(df, current_price)
            except Exception as e:
                logger.debug("Strategy failed for %s: %s", sym, e)
                continue

            if not signal or float(signal.get("confidence", 0.0)) < float(Config.CONFIDENCE_THRESHOLD):
                logger.debug("No valid signal for %s", sym)
                continue

            # Position sizing
            try:
                pos = self.risk_manager.calculate_position_size(
                    entry_price=signal.get("entry_price", current_price),
                    stop_loss=signal.get("stop_loss", current_price),
                    signal_confidence=signal.get("confidence", 0.0),
                    market_volatility=signal.get("market_volatility", 0.0),
                )
            except Exception as e:
                logger.debug("Position sizing failed for %s: %s", sym, e)
                continue

            qty = int(pos.get("quantity", 0))
            if qty <= 0:
                logger.debug("Zero/negative quantity for %s", sym)
                continue

            # Signal alert
            token = len(self.trades) + 1
            try:
                self.telegram_controller.send_signal_alert(token, signal, pos)
            except Exception:
                pass

            # Place entry
            try:
                exchange = getattr(Config, "TRADE_EXCHANGE", "NFO")
                txn_type = signal.get("signal") or signal.get("direction")
                if not txn_type:
                    logger.debug("No direction for %s", sym)
                    continue

                order_id = self.order_executor.place_entry_order(
                    symbol=sym,
                    exchange=exchange,
                    transaction_type=txn_type,
                    quantity=qty,
                )
                if not order_id:
                    logger.warning("Entry order failed for %s", sym)
                    continue
            except Exception as e:
                logger.warning("Entry order error for %s: %s", sym, e)
                continue

            # Bracket (GTT/REGULAR) orders
            try:
                self.order_executor.setup_gtt_orders(
                    entry_order_id=order_id,
                    entry_price=signal.get("entry_price", current_price),
                    stop_loss_price=signal.get("stop_loss", current_price),
                    target_price=signal.get("target", current_price),
                    symbol=sym,
                    exchange=getattr(Config, "TRADE_EXCHANGE", "NFO"),
                    quantity=qty,
                    transaction_type=txn_type,
                )
            except Exception as e:
                logger.warning("Exits setup failed for %s: %s", sym, e)

            # Record trade (track ATR from bars for trailing worker)
            atr_val = self._compute_atr(df, period=self.ATR_PERIOD)
            trade_rec = {
                "order_id": order_id,
                "symbol": sym,
                "direction": txn_type,
                "quantity": qty,
                "entry_price": float(signal.get("entry_price", current_price)),
                "stop_loss": float(signal.get("stop_loss", current_price)),
                "target": float(signal.get("target", current_price)),
                "confidence": float(signal.get("confidence", 0.0)),
                "atr": float(atr_val or 0.0),
                "status": "OPEN",
                "ts": datetime.now().isoformat(timespec="seconds"),
            }
            self.trades.append(trade_rec)
            self.active_trades[order_id] = trade_rec
            logger.info("Trade recorded: %s %d x %s @ %.2f", txn_type, qty, sym, trade_rec["entry_price"])

            # Respect single-position policy: place one per tick
            break

    # ---------- ATR / spread helpers ----------
    @staticmethod
    def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
        """
        Lightweight ATR (no ta-lib). Requires 'high','low','close'.
        """
        try:
            if any(c not in df.columns for c in ("high", "low", "close")) or len(df) < period + 1:
                return 0.0
            high = df["high"].astype(float)
            low = df["low"].astype(float)
            close = df["close"].astype(float)
            prev_close = close.shift(1)

            tr1 = high - low
            tr2 = (high - prev_close).abs()
            tr3 = (low - prev_close).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period, min_periods=period).mean().iloc[-1]
            return float(atr if pd.notnull(atr) else 0.0)
        except Exception:
            return 0.0

    @staticmethod
    def _estimate_spread_pct(df: pd.DataFrame) -> float:
        """
        Proxy for wide spreads using last bar range: (high-low)/close.
        """
        try:
            if len(df) == 0:
                return 1.0
            last = df.iloc[-1]
            close = float(last.get("close", 0) or 0)
            if close <= 0:
                return 1.0
            rng = float(last.get("high", close) - last.get("low", close))
            return max(0.0, rng / close)
        except Exception:
            return 1.0

    # ---------- Workers: OCO hygiene + trailing ----------
    def _oco_and_housekeeping_tick(self) -> None:
        """
        Periodically:
          - Enforce OCO on REGULAR exits (cancel sibling when one fills)
          - Detect closures → finalize PnL, logs, circuit breaker counters
          - Session day rollover check (backup)
        """
        try:
            # Broker sync (REGULAR exits)
            self.order_executor.sync_and_enforce_oco()
        except Exception:
            pass

        # Scan for newly closed trades (internal record moved to closed by executor)
        to_finalize: List[str] = []
        with self._lock:
            for oid, rec in list(self.active_trades.items()):
                ord_rec = self.order_executor.get_active_orders().get(oid)
                if ord_rec is None:  # not found among active → assume closed
                    to_finalize.append(oid)

        for oid in to_finalize:
            self._finalize_trade(oid)

        # Day change safeguard
        if date.today() != self._trade_day:
            self._daily_rollover()

    def _trailing_tick(self) -> None:
        """
        Trailing worker:
          - For each open trade, fetch LTP and call update_trailing_stop(order_id, ltp, atr)
          - ATR taken from trade record (frozen at entry) for stability
        """
        try:
            if not self.active_trades:
                return
            kite = getattr(self.order_executor, "kite", None)
            if not kite:
                return

            # Build LTP request list
            symbols = []
            order_ids = []
            atr_map: Dict[str, float] = {}
            with self._lock:
                for oid, rec in self.active_trades.items():
                    if rec.get("status") != "OPEN":
                        continue
                    symbols.append(rec["symbol"])
                    order_ids.append(oid)
                    atr_map[oid] = float(rec.get("atr", 0.0))

            if not symbols:
                return

            # Deduplicate (Kite ltp expects exchange-prefixed symbols)
            symbols = list(dict.fromkeys(symbols))
            try:
                ltps = kite.ltp(symbols)
            except Exception as e:
                logger.debug("ltp batch fetch failed: %s", e)
                return

            # Apply trailing
            for oid in order_ids:
                rec = self.active_trades.get(oid)
                if not rec or rec.get("status") != "OPEN":
                    continue
                sym = rec["symbol"]
                info = ltps.get(sym, {})
                ltp = info.get("last_price")
                if ltp is None:
                    continue
                try:
                    self.order_executor.update_trailing_stop(oid, float(ltp), float(atr_map.get(oid, 0.0)))
                except Exception:
                    continue
        except Exception as e:
            logger.debug("Trailing tick error: %s", e)

    # ---------- Trade finalization / accounting ----------
    def _finalize_trade(self, order_id: str) -> None:
        """
        When OCO worker closes an order (or any other path), compute PnL, log, update breakers.
        """
        rec = self.active_trades.pop(order_id, None)
        if not rec:
            return

        # Decide exit side & price best-effort
        # We infer via our last known bracket prices (target/stop)
        # and apply slippage + fees model.
        direction = rec["direction"].upper()
        qty = int(rec["quantity"])
        entry = float(rec["entry_price"])
        stop = float(rec["stop_loss"])
        target = float(rec["target"])

        # If we can, peek executor record to see which leg filled
        filled = "UNKNOWN"
        try:
            # The executor removes the OrderRecord from "active" when closed.
            pass
        except Exception:
            pass

        # Use conservative assumption: if price was closer to stop recently? Hard without stream.
        # So, we’ll log BOTH potential prices only if unknown. But PnL must be single number.
        # We default to target if target > entry and direction is BUY and target far; else stop.
        if direction == "BUY":
            # Prefer target if reasonable
            exit_price = target if target > entry else stop
            filled = "TP" if exit_price == target else "SL"
            pnl_gross = (exit_price - entry) * qty
        else:
            exit_price = target if target < entry else stop
            filled = "TP" if exit_price == target else "SL"
            pnl_gross = (entry - exit_price) * qty

        # Slippage + fees (simple model: two sides slippage + fees on notional)
        slip_mult = 1.0 - (self.SLIPPAGE_BPS / 10000.0) * 2.0
        fees_cost = self.FEES_PCT * max(entry, exit_price) * qty
        pnl_net = pnl_gross * slip_mult - fees_cost

        self.daily_pnl += pnl_net
        if pnl_net < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        # Append persistent log
        self._append_trade_log({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "order_id": order_id,
            "symbol": rec["symbol"],
            "direction": direction,
            "qty": qty,
            "entry": round(entry, 2),
            "exit": round(exit_price, 2),
            "exit_leg": filled,
            "pnl_gross": round(pnl_gross, 2),
            "pnl_net": round(pnl_net, 2),
            "confidence": round(float(rec.get("confidence", 0.0)), 2),
            "atr": round(float(rec.get("atr", 0.0)), 2),
        })

        # Circuit breaker check post trade
        if self._is_circuit_breaker_tripped():
            self._safe_send_message("Circuit breaker tripped. Stopping trading for the day.")
            self.stop()

        logger.info("Trade closed: %s | %s %d @ %.2f → %.2f (%s) | PnL net=₹%.2f",
                    order_id, direction, qty, entry, exit_price, filled, pnl_net)

    def _is_circuit_breaker_tripped(self) -> bool:
        try:
            if self.MAX_CONSECUTIVE_LOSSES > 0 and self.consecutive_losses >= self.MAX_CONSECUTIVE_LOSSES:
                return True
            if self.daily_start_equity > 0:
                dd = max(0.0, -self.daily_pnl) / self.daily_start_equity
                if dd >= self.MAX_DAILY_DRAWDOWN_PCT:
                    return True
        except Exception:
            pass
        return False

    # ---------- Status / summary ----------
    def get_status(self) -> Dict[str, Any]:
        status: Dict[str, Any] = {
            "is_trading": self.is_trading,
            "open_orders": len(self.order_executor.get_active_orders()),
            "trades_today": len(self.trades),
            "live_mode": self.live_mode,
        }
        try:
            status.update(self.risk_manager.get_risk_status())
        except Exception:
            pass
        return status

    def get_summary(self) -> str:
        lines = [
            "📊 <b>Daily Summary</b>",
            f"🟢 Mode: {'LIVE' if self.live_mode else 'SHADOW'}",
            f"🔁 <b>Total trades:</b> {len(self.trades)}",
            f"💰 <b>PNL:</b> {self.daily_pnl:.2f}",
            f"⚠️ Consecutive losses: {self.consecutive_losses}",
            "────────────────────",
        ]
        # Last few trades (if any)
        for t in self.trades[-5:]:
            lines.append(
                f"{t['direction']} {t['quantity']} {t.get('symbol','')} @ {t['entry_price']:.2f} "
                f"(SL {t['stop_loss']:.2f}, TP {t['target']:.2f}, conf {t.get('confidence',0):.2f})"
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

    # ---------- Daily rollover ----------
    def _daily_rollover(self) -> None:
        """
        Called at market close or when day changes:
          - Stop trading automatically (safety)
          - Reset counters (drawdown, losses)
          - Clear intraday trades memory (persisted logs remain)
          - Refresh balance baseline for next day
        """
        logger.info("Daily rollover starting.")
        try:
            self.stop()
        except Exception:
            pass

        try:
            self.trades.clear()
            self.active_trades.clear()
            self.consecutive_losses = 0
            self._trade_day = date.today()
            self.daily_pnl = 0.0
            self.daily_start_equity = float(self.risk_manager.equity or self.risk_manager.account_size or 0.0)
        except Exception:
            pass
        logger.info("Daily rollover complete.")

    # ---------- Persistence ----------
    def _ensure_trade_log(self) -> None:
        try:
            path = getattr(Config, "LOG_FILE", "logs/trades.csv")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if not os.path.exists(path):
                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["ts", "order_id", "symbol", "direction", "qty", "entry", "exit",
                                     "exit_leg", "pnl_gross", "pnl_net", "confidence", "atr"])
        except Exception as e:
            logger.debug("Ensure log file failed: %s", e)

    def _append_trade_log(self, row: Dict[str, Any]) -> None:
        try:
            path = getattr(Config, "LOG_FILE", "logs/trades.csv")
            with open(path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    row.get("ts"), row.get("order_id"), row.get("symbol"), row.get("direction"),
                    row.get("qty"), row.get("entry"), row.get("exit"), row.get("exit_leg"),
                    row.get("pnl_gross"), row.get("pnl_net"), row.get("confidence"), row.get("atr"),
                ])
        except Exception as e:
            logger.debug("Append log failed: %s", e)

    # ---------- Shutdown ----------
    def shutdown(self) -> None:
        logger.info("Shutting down RealTimeTrader...")
        try:
            self.stop()
        except Exception:
            pass
        self._stop_polling()
        logger.info("RealTimeTrader shutdown complete.")
