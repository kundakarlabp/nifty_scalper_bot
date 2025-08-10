# src/data_streaming/realtime_trader.py
import logging
import threading
import atexit
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import schedule
from datetime import datetime, timedelta
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
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()

        self.is_trading: bool = False
        self.daily_pnl: float = 0.0
        self.trades: List[Dict[str, Any]] = []
        self.live_mode: bool = bool(getattr(Config, "ENABLE_LIVE_TRADING", False))

        # Instrument cache (for Kite rate-limits)
        self._nfo_instruments_cache: Optional[List[Dict]] = None
        self._nse_instruments_cache: Optional[List[Dict]] = None
        self._instruments_cache_timestamp: float = 0
        self._INSTRUMENT_CACHE_DURATION: int = 300  # seconds
        self._cache_lock = threading.RLock()

        # ATM cache
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
        access_token = getattr(Config, "ZERODHA_ACCESS_TOKEN", None)
        if not api_key or not access_token:
            raise RuntimeError("ZERODHA_API_KEY or ZERODHA_ACCESS_TOKEN missing")

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

    # ---------- Scheduling / main loop ----------
    def _setup_smart_scheduling(self) -> None:
        try:
            schedule.clear()
            schedule.every(30).seconds.do(self._smart_fetch_and_process)
            schedule.every(getattr(Config, "BALANCE_LOG_INTERVAL_MIN", 30)).minutes.do(
                self.refresh_account_balance
            )
            logger.info("Scheduled fetch/process every 30s (market hours only).")
        except Exception as e:
            logger.error(f"Error setting up scheduling: {e}")

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
            self.fetch_and_process_data()
        except Exception as e:
            logger.error(f"Error in smart fetch and process: {e}")

    # ---------- Telegram control ----------
    def _start_polling(self) -> None:
        if self._polling_thread and self._polling_thread.is_alive():
            return
        try:
            self.telegram_controller.send_startup_alert()
        except Exception as e:
            logger.warning(f"Failed to send startup alert: {e}")
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
            except Exception as e:
                logger.warning(f"Error stopping telegram polling: {e}")
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
            if hasattr(self.order_executor, "cancel_all"):
                self.order_executor.cancel_all()
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

    # ---------- Balance helpers ----------
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

    def _log_account_balance(self) -> None:
        try:
            logger.info(f"💰 Account size (cached): ₹{self.risk_manager.account_size:.2f}")
        except Exception as e:
            logger.debug(f"Could not log account balance: {e}")

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

            # 2) Spot price and base instruments
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
            start_time = end_time - timedelta(minutes=getattr(Config, "DATA_LOOKBACK_MINUTES", 15))

            # 3) Fetch historical data in parallel for spot + a window of option strikes
            spot_df, options_data = self._fetch_all_data_parallel(
                spot_token, atm_strike, start_time, end_time, cached_nfo, cached_nse
            )

            # 4) Pick strikes for processing
            if options_data:
                selected_strikes_info = self._analyze_options_data_optimized(spot_price, options_data)
            else:
                selected_strikes_info = self._get_fallback_strikes(atm_strike, cached_nfo, cached_nse)

            if not selected_strikes_info:
                logger.info("No strikes selected for processing.")
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
        """
        Wraps kite.historical_data; returns DataFrame with DatetimeIndex & ['open','high','low','close','volume'].
        """
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
            return df[["open", "high", "low", "close", "volume"]].copy()
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
            logger.error(f"Error preparing fetch tasks: {e}")

        # Parallel fetch
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
                    logger.debug(f"Fetch failed for {symbol}: {e}")

        return spot_df, options_data

    # ---------- Selection / processing ----------
    def _analyze_options_data_optimized(self, spot_price: float, options_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """
        Very simple heuristic:
        - Choose the latest-available candle per symbol
        - Rank by absolute % change of last close vs previous close
        - Return top CE and top PE if available
        """
        latest: List[Tuple[str, float]] = []
        for sym, df in options_data.items():
            if df is None or df.empty or "close" not in df.columns:
                continue
            if len(df) < 2:
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

        for pick in selected:
            sym = pick["symbol"]
            df = options_data.get(sym, pd.DataFrame())
            if df.empty or len(df) < 30 or not isinstance(df.index, pd.DatetimeIndex):
                logger.debug(f"Insufficient data for {sym}")
                continue

            # Use last price as current_price
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

            # Position sizing
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

            qty = int(pos.get("quantity", 0))
            if qty <= 0:
                logger.debug(f"Zero/negative quantity for {sym}")
                continue

            # Send signal alert
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
                    logger.debug(f"No direction for {sym}")
                    continue
                order_id = self.order_executor.place_entry_order(
                    symbol=sym,
                    exchange=exchange,
                    transaction_type=txn_type,
                    quantity=qty,
                )
                if not order_id:
                    logger.warning(f"Entry order failed for {sym}")
                    continue
            except Exception as e:
                logger.warning(f"Entry order error for {sym}: {e}")
                continue

            # Bracket (GTT) orders
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
                logger.warning(f"GTT setup failed for {sym}: {e}")

            # Record trade
            self.trades.append(
                {
                    "order_id": order_id,
                    "symbol": sym,
                    "direction": txn_type,
                    "quantity": qty,
                    "entry_price": signal.get("entry_price", current_price),
                    "stop_loss": signal.get("stop_loss", current_price),
                    "target": signal.get("target", current_price),
                    "confidence": signal.get("confidence", 0.0),
                }
            )
            logger.info(f"✅ Trade recorded: {txn_type} {qty} x {sym} @ {signal.get('entry_price', current_price):.2f}")

    # ---------- Status / summary ----------
    def get_status(self) -> Dict[str, Any]:
        status: Dict[str, Any] = {
            "is_trading": self.is_trading,
            "open_orders": len(self.order_executor.get_active_orders()),
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
                f"{t['direction']} {t['quantity']} {t.get('symbol','')} @ {t['entry_price']:.2f} "
                f"(SL {t['stop_loss']:.2f}, TP {t['target']:.2f})"
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
        self._stop_polling()
        logger.info("✅ RealTimeTrader shutdown complete.")